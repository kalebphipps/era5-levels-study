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
    """Build cos(latitude) area weights normalised to mean 1.

    Parameters
    ----------
    n_lat : int
        Number of latitude rows (full height; latitude is never sharded).
    device : torch.device, optional
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Weights of shape ``(1, 1, n_lat, 1)``, broadcastable over
        ``(B, C, lat, lon)``.
    """
    lat = torch.linspace(np.pi / 2, -np.pi / 2, n_lat, device=device)
    w = torch.cos(lat)
    w = w * n_lat / w.sum()
    return w.view(1, 1, n_lat, 1)


def reduce_spatial_mean(local_field: torch.Tensor, weights: torch.Tensor,
                        spatial_group) -> torch.Tensor:
    """Compute an area-weighted spatial mean, reduced across JSpatial.

    The mean is taken over ``(batch, lat, local-lon)`` and summed across the
    longitude shards, giving one value per *local* channel. Identical reduction
    to beast's own.

    Parameters
    ----------
    local_field : torch.Tensor
        Local shard of shape ``(B, C_local, lat, lon_local)`` (e.g. squared
        error).
    weights : torch.Tensor
        Latitude weights of shape ``(1, 1, lat, 1)``.
    spatial_group : torch.distributed.ProcessGroup
        The ``JSpatial`` group over which the longitude shards are reduced.

    Returns
    -------
    torch.Tensor
        Area-weighted mean per local channel, shape ``(C_local,)``.
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
    split (rank ``r`` owns global channels ``[r*chunk, (r+1)*chunk)``) and
    therefore the :func:`build_ordered_variables` order.

    Parameters
    ----------
    local_vec : torch.Tensor
        This rank's per-local-channel vector, shape ``(C_local,)``.
    channel_group : torch.distributed.ProcessGroup
        The ``JChannel`` group across which channels are sharded.

    Returns
    -------
    torch.Tensor
        The assembled global vector, shape ``(C_global,)`` (returns the input
        unchanged for a 1-process group).
    """
    if not dist.is_initialized() or dist.get_world_size(channel_group) <= 1:
        return local_vec
    parts = [torch.empty_like(local_vec) for _ in range(dist.get_world_size(channel_group))]
    dist.all_gather(parts, local_vec.contiguous(), group=channel_group)
    return torch.cat(parts, dim=0)


def gather_spatial_field(field: torch.Tensor, spatial_group) -> torch.Tensor:
    """Gather a ``(lat, lon_local)`` field into the full ``(lat, lon)`` grid.

    Used for sample-map dumps. Concatenates along the longitude (last) axis in
    JSpatial-rank order.

    Parameters
    ----------
    field : torch.Tensor
        This rank's local field, shape ``(lat, lon_local)``.
    spatial_group : torch.distributed.ProcessGroup
        The ``JSpatial`` group across which longitude is sharded.

    Returns
    -------
    torch.Tensor
        The full field, shape ``(lat, lon)`` (returns the input unchanged for a
        1-process group).
    """
    if not dist.is_initialized() or dist.get_world_size(spatial_group) <= 1:
        return field
    parts = [torch.empty_like(field) for _ in range(dist.get_world_size(spatial_group))]
    dist.all_gather(parts, field.contiguous(), group=spatial_group)
    return torch.cat(parts, dim=-1)


def subset_indices(levels_full, levels_subset) -> torch.Tensor:
    """Find the global channel indices of a level subset within a full layout.

    Pure index bookkeeping (no tensors/sharding) — used to pick the 13 standard
    levels out of the assembled 37-level vectors. Surface variables are always
    included.

    Parameters
    ----------
    levels_full : list of int or str
        The full level set (e.g. the 37 levels).
    levels_subset : list of int or str
        The subset of levels to select (e.g. the 13 standard levels).

    Returns
    -------
    torch.Tensor
        1-D ``long`` tensor of channel indices of the subset within the full
        layout.

    Raises
    ------
    ValueError
        If any subset variable is absent from the full layout.
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
    """Return the most-recent input timestep's variables as a persistence forecast.

    Persistence (tomorrow = today) is the cheapest baseline; any useful model
    must beat it. Assumes the local input is laid out as
    ``[ts0 vars | ts1 vars | ... | masks]`` (all channel-sharded like the
    output), so the latest timestep's local variables are the block ending at
    ``n_in_timesteps * n_local_vars``. CONFIRM this layout on the first run.

    Parameters
    ----------
    x : torch.Tensor
        Local input shard, shape ``(B, C_in_local, lat, lon_local)``.
    n_in_timesteps : int
        Number of input timesteps stacked along the channel axis.
    n_local_vars : int
        Number of local variable channels per timestep (the output channel
        count for this shard).

    Returns
    -------
    torch.Tensor
        The latest timestep's variables, shape ``(B, n_local_vars, lat,
        lon_local)``.
    """
    end = n_in_timesteps * n_local_vars
    return x[:, end - n_local_vars:end]


@torch.no_grad()
def _accumulate_climatology(dataloader, device) -> torch.Tensor | None:
    """Compute the time-mean target field per local channel over the eval set.

    Each ``(channel, lat, lon)`` element is averaged over time independently, so
    no cross-rank reduction is needed. This is an in-sample climatology baseline
    (computed on the same set it scores) — fine for poster context, but note it
    as such.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader (its targets are averaged over time).
    device : torch.device
        Compute device.

    Returns
    -------
    torch.Tensor or None
        Time-mean field of shape ``(C_local, lat, lon_local)``, or ``None`` if
        the set is empty.
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
    """Compute distributed per-variable RMSE for the model (+ optional baselines).

    RMSE is in normalised units unless ``data_std`` (per global channel) is
    given, in which case it is rescaled to physical units. If ``subset_levels``
    is set (e.g. the 13 standard levels), the names/vectors are restricted to
    that subset — the free 13-vs-37 comparison.

    Parameters
    ----------
    model : torch.nn.Module
        The (sharded) model to score.
    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader.
    levels_full : list of int or str
        The level set the model was trained on (defines the channel layout).
    groups : dict
        Mapping of logical group name to ``torch.distributed.ProcessGroup``
        (``JSpatial`` and ``JChannel`` are used).
    device : torch.device
        Compute device.
    n_in_timesteps : int, optional
        Number of input timesteps, used for the persistence baseline (default 1).
    baselines : bool, optional
        If ``True`` (default), also compute persistence and climatology RMSE.
    subset_levels : list of int or str, optional
        If given, restrict the result to these levels' variables.
    data_std : array-like, optional
        Per-global-channel standard deviation; if given, RMSE is returned in
        physical units.

    Returns
    -------
    names : list of str
        Variable names in result order.
    result : dict
        Mapping of metric name (``"model"``, and ``"persistence"`` /
        ``"climatology"`` if ``baselines``) to its per-variable RMSE tensor.
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
    """Compute model-only per-variable RMSE (back-compat wrapper, no baselines).

    Parameters
    ----------
    model : torch.nn.Module
        The (sharded) model to score.
    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader.
    levels_full : list of int or str
        The level set defining the channel layout.
    groups : dict
        Mapping of logical group name to ``torch.distributed.ProcessGroup``.
    device : torch.device
        Compute device.
    subset_levels : list of int or str, optional
        If given, restrict the result to these levels' variables.
    data_std : array-like, optional
        Per-global-channel standard deviation for physical-unit RMSE.

    Returns
    -------
    names : list of str
        Variable names in result order.
    rmse : torch.Tensor
        Per-variable model RMSE.
    """
    names, result = evaluate_all(
        model, dataloader, levels_full, groups, device,
        baselines=False, subset_levels=subset_levels, data_std=data_std)
    return names, result["model"]


def write_metrics_csv(path: str, epoch: int, names: list[str], metrics: dict) -> None:
    """Append one row per variable to a long-format CSV (call on rank 0 only).

    Columns are ``epoch, variable, <metric_1>, <metric_2>, ...`` — easy to load
    with pandas and pivot for the per-level curves / improvement heatmaps.

    Parameters
    ----------
    path : str
        Output CSV path (created with a header if it does not exist).
    epoch : int
        Epoch index written into each row.
    names : list of str
        Variable names, one row each.
    metrics : dict
        Mapping of metric name to its per-variable tensor (aligned with
        ``names``).
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
    """Save full ``(lat, lon)`` prediction/truth/error maps for a few variables.

    Runs one forward on the first validation batch, then for each requested
    variable finds which JChannel rank owns that channel, gathers its
    ``(lat, lon_local)`` field into the full grid over JSpatial, and writes
    ``<var>_{pred,true,err}.npy`` (from the JSpatial-rank-0 of the owning channel
    shard). Feed these to matplotlib offline for the poster maps.

    Parameters
    ----------
    model : torch.nn.Module
        The (sharded) model.
    dataloader : torch.utils.data.DataLoader
        Validation dataloader (the first batch is used).
    levels_full : list of int or str
        The level set defining the channel layout.
    groups : dict
        Mapping of logical group name to ``torch.distributed.ProcessGroup``
        (``JChannel`` and ``JSpatial`` are used).
    device : torch.device
        Compute device.
    var_names : list of str
        Variable names to dump (silently skips names not in the layout).
    out_dir : str
        Output directory for the ``.npy`` dumps.
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
