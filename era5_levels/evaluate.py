"""Distributed evaluation for the levels study.

IMPORTANT: under jigsaw the model and data are sharded across GPUs — channels
across the ``JChannel`` group and longitude across the ``JSpatial`` group — so
the full (B, C, lat, lon) field never exists on one rank and CANNOT be gathered
onto a single GPU. Evaluation therefore runs *inside the same process mesh* as
training: each rank computes metrics on its local shard, and we reduce:

  * area-weighted spatial means  ->  all_reduce(SUM) over JSpatial  (longitude shards)
  * per-variable assembly        ->  all_gather over JChannel        (channel shards)

This mirrors beast's own reductions. Prefer ``beast.evaluation`` once that
subpackage stabilises; this is the small, study-local version so the poster
isn't blocked on it.

Latitude is NOT sharded (only longitude is), so the cos-latitude weights are the
full-height vector and identical on every rank.

What's here:
  - per-variable RMSE of the model (`validate_per_level` / `evaluate_all`);
  - cheap baselines for context on the plots — **persistence** and a self-
    computed **climatology** (no external files needed);
  - per-epoch metric logging to CSV (rank 0) so all the curves/heatmaps can be
    drawn offline;
  - a sample-field map dump (gather one variable's lat/lon field over JSpatial)
    for the poster's forecast/error maps.

NOTE: like everything beast-touching in this repo, the distributed paths here are
untested off-cluster. Two layout assumptions are flagged inline and must be
confirmed on the first real run: (a) channels are split contiguously by JChannel
rank in `build_ordered_variables` order; (b) the dataloader lays the input out as
[timestep-0 vars, timestep-1 vars, ..., constant masks], all channel-sharded the
same way as the output.
"""

from __future__ import annotations

import csv
import os

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


def gather_spatial_field(field: torch.Tensor, spatial_group) -> torch.Tensor:
    """Gather a (lat, lon_local) field into the full (lat, lon) over JSpatial.

    Used for sample-map dumps. Concatenates along the longitude (last) axis in
    JSpatial-rank order. Safe to call with a 1-process group (returns as-is).
    """
    if not dist.is_initialized() or dist.get_world_size(spatial_group) <= 1:
        return field
    parts = [torch.empty_like(field) for _ in range(dist.get_world_size(spatial_group))]
    dist.all_gather(parts, field.contiguous(), group=spatial_group)
    return torch.cat(parts, dim=-1)


def subset_indices(levels_full, levels_subset) -> torch.Tensor:
    """Global channel indices of `levels_subset` within the `levels_full` layout.

    Pure index bookkeeping (no tensors/sharding) — used to pick the 13 standard
    levels out of the assembled 37-level vectors. Surface variables are always
    included.
    """
    full = build_ordered_variables(levels_full)
    sub = build_ordered_variables(levels_subset)
    pos = {name: i for i, name in enumerate(full)}
    missing = [n for n in sub if n not in pos]
    if missing:
        raise ValueError(f"{len(missing)} subset variables not in full layout, "
                         f"e.g. {missing[:3]} — check the level lists/order.")
    return torch.tensor([pos[n] for n in sub], dtype=torch.long)


def persistence_prediction(x: torch.Tensor, n_in_timesteps: int,
                           n_local_vars: int) -> torch.Tensor:
    """The most-recent input timestep's variables, as a 'no change' forecast.

    Persistence (tomorrow = today) is the cheapest baseline; any useful model
    must beat it. Assumes the local input is laid out as
    ``[ts0 vars | ts1 vars | ... | masks]`` (all channel-sharded like the output),
    so the latest timestep's local variables are the block ending at
    ``n_in_timesteps * n_local_vars``. CONFIRM this layout on the first run.
    """
    end = n_in_timesteps * n_local_vars
    return x[:, end - n_local_vars:end]


@torch.no_grad()
def _accumulate_climatology(dataloader, device) -> torch.Tensor | None:
    """Time-mean target field per local channel over the eval set (local only).

    Each (channel, lat, lon) element is averaged over time independently, so no
    cross-rank reduction is needed. This is an in-sample climatology baseline
    (computed on the same set it scores) — fine for poster context; note it as
    such. Returns (C_local, lat, lon_local) or None if the set is empty.
    """
    total = None
    n = 0
    for batch in dataloader:
        y = batch[1].to(device)
        s = y.sum(dim=0)
        total = s if total is None else total + s
        n += y.shape[0]
    return None if total is None else total / max(1, n)


@torch.no_grad()
def evaluate_all(model, dataloader, levels_full, groups, device, *,
                 n_in_timesteps: int = 1, baselines: bool = True,
                 subset_levels=None, data_std=None) -> tuple[list[str], dict]:
    """Distributed per-variable RMSE for the model (+ optional baselines).

    Returns (variable_names, {metric: rmse_vector}) with metric in
    {"model", "persistence", "climatology"} (the latter two only if
    ``baselines``). RMSE is normalised-unit unless ``data_std`` (per global
    channel) is given, then physical units. If ``subset_levels`` is set (e.g. the
    13 standard levels), names/vectors are restricted to that subset — the free
    13-vs-37 comparison.
    """
    spatial_group, channel_group = groups["JSpatial"], groups["JChannel"]
    clim = _accumulate_climatology(dataloader, device) if baselines else None

    model.eval()
    keys = ["model"] + (["persistence", "climatology"] if baselines else [])
    weights = None
    acc = {k: None for k in keys}
    n_batches = 0
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        if weights is None:
            weights = latitude_weights(pred.shape[-2], device=pred.device)
            acc = {k: torch.zeros(pred.shape[1], device=pred.device) for k in keys}
        acc["model"] += reduce_spatial_mean((pred - y) ** 2, weights, spatial_group)
        if baselines:
            persist = persistence_prediction(x, n_in_timesteps, y.shape[1])
            acc["persistence"] += reduce_spatial_mean((persist - y) ** 2, weights, spatial_group)
            acc["climatology"] += reduce_spatial_mean((clim.unsqueeze(0) - y) ** 2, weights, spatial_group)
        n_batches += 1

    names = build_ordered_variables(levels_full)
    idx = (subset_indices(levels_full, subset_levels) if subset_levels is not None
           else None)
    result = {}
    for k in keys:
        rmse = gather_channels(torch.sqrt(acc[k] / max(1, n_batches)), channel_group)
        if data_std is not None:
            rmse = rmse * torch.as_tensor(data_std, device=rmse.device)
        if idx is not None:
            rmse = rmse[idx.to(rmse.device)]
        result[k] = rmse
    if idx is not None:
        names = [names[i] for i in idx.tolist()]
    return names, result


def validate_per_level(model, dataloader, levels_full, groups, device,
                       subset_levels=None, data_std=None) -> tuple[list[str], torch.Tensor]:
    """Back-compat wrapper: model-only per-variable RMSE (no baselines)."""
    names, result = evaluate_all(
        model, dataloader, levels_full, groups, device,
        baselines=False, subset_levels=subset_levels, data_std=data_std)
    return names, result["model"]


def write_metrics_csv(path: str, epoch: int, names: list[str], metrics: dict) -> None:
    """Append one row per variable to a long-format CSV (call on rank 0 only).

    Columns: epoch, variable, <metric_1>, <metric_2>, ... — easy to load with
    pandas and pivot for the per-level curves / improvement heatmaps.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    keys = list(metrics.keys())
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "variable", *keys])
        for i, name in enumerate(names):
            w.writerow([epoch, name, *[f"{metrics[k][i].item():.6f}" for k in keys]])


@torch.no_grad()
def dump_sample_maps(model, dataloader, levels_full, groups, device,
                     var_names: list[str], out_dir: str) -> None:
    """Save full (lat, lon) prediction/truth/error maps for a few variables.

    Runs one forward on the first validation batch, then for each requested
    variable: finds which JChannel rank owns that channel, gathers its
    (lat, lon_local) field into the full grid over JSpatial, and writes
    ``<var>_{pred,true,err}.npy`` (from the JSpatial-rank-0 of the owning channel
    shard). Feed these to matplotlib offline for the poster maps.
    """
    channel_group, spatial_group = groups["JChannel"], groups["JSpatial"]
    full_names = build_ordered_variables(levels_full)

    model.eval()
    batch = next(iter(dataloader))
    x, y = batch[0].to(device), batch[1].to(device)
    pred = model(x)
    chunk = pred.shape[1]  # local channel count

    for vname in var_names:
        if vname not in full_names:
            continue
        g = full_names.index(vname)
        owner, local = g // chunk, g % chunk
        if channel_group.rank() != owner:
            continue
        full_pred = gather_spatial_field(pred[0, local], spatial_group)  # (lat, lon)
        full_true = gather_spatial_field(y[0, local], spatial_group)
        if dist.get_world_size(spatial_group) <= 1 or spatial_group.rank() == 0:
            os.makedirs(out_dir, exist_ok=True)
            p, t = full_pred.float().cpu().numpy(), full_true.float().cpu().numpy()
            np.save(os.path.join(out_dir, f"{vname}_pred.npy"), p)
            np.save(os.path.join(out_dir, f"{vname}_true.npy"), t)
            np.save(os.path.join(out_dir, f"{vname}_err.npy"), p - t)
