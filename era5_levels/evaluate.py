"""Distributed evaluation for the levels study.

IMPORTANT: under jigsaw the model and data are sharded across GPUs — channels
across the ``JChannel`` group and longitude across the ``JSpatial`` group — so
the full (B, C, lat, lon) field never exists on one rank and CANNOT be gathered
onto a single GPU. Evaluation therefore runs *inside the same process mesh* as
training: each rank computes metrics on its local shard, and we reduce:

  * area-weighted spatial means  ->  all_reduce(SUM) over JSpatial  (longitude shards)
  * per-variable assembly        ->  all_gather over JChannel        (channel shards)

This mirrors beast's own `reduce_spatial_mean` / `gather_*` reductions. Prefer
`beast.evaluation` once that subpackage stabilises; this is the small, study-
local version so the poster isn't blocked on it.

Latitude is NOT sharded (only longitude is), so the cos-latitude weights are the
full-height vector and identical on every rank.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from .variable_layout import build_ordered_variables


def latitude_weights(n_lat: int, device=None) -> torch.Tensor:
    """cos(latitude) weights normalised to mean 1, shape (1, 1, n_lat, 1)."""
    lat = torch.linspace(np.pi / 2, -np.pi / 2, n_lat, device=device)
    w = torch.cos(lat)
    w = w * n_lat / w.sum()
    return w.view(1, 1, n_lat, 1)


def reduce_spatial_mean(local_field: torch.Tensor, weights: torch.Tensor,
                        spatial_group) -> torch.Tensor:
    """Area-weighted mean over (batch, lat, local-lon), reduced across JSpatial.

    Returns one value per *local* channel. `local_field` is (B, C_local, lat,
    lon_local); `weights` is (1, 1, lat, 1). Identical reduction to beast's.
    """
    local_sum = (local_field * weights).sum(dim=(0, 2, 3))            # (C_local,)
    weight_map = torch.ones_like(local_field[0, 0]) * weights[0, 0]   # (lat, lon_local)
    local_w = weight_map.sum() * local_field.shape[0]
    if dist.is_initialized() and dist.get_world_size(spatial_group) > 1:
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=spatial_group)
        dist.all_reduce(local_w, op=dist.ReduceOp.SUM, group=spatial_group)
    return local_sum / local_w


def gather_channels(local_vec: torch.Tensor, channel_group) -> torch.Tensor:
    """Assemble a global per-variable vector from local channel shards.

    Concatenates in JChannel-rank order, which matches the contiguous channel
    split (rank r owns global channels [r*chunk, (r+1)*chunk)) and therefore the
    `build_ordered_variables` order.
    """
    if not dist.is_initialized() or dist.get_world_size(channel_group) <= 1:
        return local_vec
    parts = [torch.empty_like(local_vec) for _ in range(dist.get_world_size(channel_group))]
    dist.all_gather(parts, local_vec.contiguous(), group=channel_group)
    return torch.cat(parts, dim=0)


def subset_indices(levels_full, levels_subset) -> torch.Tensor:
    """Global channel indices of `levels_subset` within the `levels_full` layout.

    Pure index bookkeeping (no tensors/sharding) — used to pick the 13 standard
    levels out of the assembled 37-level RMSE vector. Surface variables (level-
    independent) are always included.
    """
    full = build_ordered_variables(levels_full)
    sub = build_ordered_variables(levels_subset)
    pos = {name: i for i, name in enumerate(full)}
    missing = [n for n in sub if n not in pos]
    if missing:
        raise ValueError(f"{len(missing)} subset variables not in full layout, "
                         f"e.g. {missing[:3]} — check the level lists/order.")
    return torch.tensor([pos[n] for n in sub], dtype=torch.long)


@torch.no_grad()
def validate_per_level(model, dataloader, levels_full, groups, device,
                       subset_levels=None, data_std=None) -> tuple[list[str], torch.Tensor]:
    """Distributed per-variable RMSE over the validation set.

    Each rank scores its local channel/longitude shard; we reduce spatially over
    JSpatial and gather over JChannel to get a global per-variable RMSE vector on
    every rank. If `subset_levels` is given (e.g. the 13 standard levels), the
    returned vector/names are restricted to that subset — this is the *free*
    13-vs-37 comparison (no retraining): run the trained 37-level model and read
    off the 13 common levels.

    Returns (variable_names, rmse_vector). RMSE is in normalised units unless
    `data_std` (per global channel) is supplied, in which case it is scaled to
    physical units.
    """
    model.eval()
    spatial_group, channel_group = groups["JSpatial"], groups["JChannel"]

    weights = None
    sq_acc = None
    n_batches = 0
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        if weights is None:
            weights = latitude_weights(pred.shape[-2], device=pred.device)
            sq_acc = torch.zeros(pred.shape[1], device=pred.device)
        sq_acc += reduce_spatial_mean((pred - y) ** 2, weights, spatial_group)
        n_batches += 1

    local_rmse = torch.sqrt(sq_acc / max(1, n_batches))     # per local channel
    global_rmse = gather_channels(local_rmse, channel_group)  # all variables
    names = build_ordered_variables(levels_full)

    if data_std is not None:
        global_rmse = global_rmse * torch.as_tensor(data_std, device=global_rmse.device)

    if subset_levels is not None:
        idx = subset_indices(levels_full, subset_levels).to(global_rmse.device)
        global_rmse = global_rmse[idx]
        names = [names[i] for i in idx.tolist()]
    return names, global_rmse
