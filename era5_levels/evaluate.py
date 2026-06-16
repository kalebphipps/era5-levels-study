"""Distributed evaluation for the levels study. Uses the beast evaluation."""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.distributed as dist

from . import beast_api
from .variable_layout import (
    PRESSURE_VARIABLES,
    SURFACE_VARIABLES,
    build_ordered_variables,
)


def _global_lat_weighted_mean(ev, se_field: torch.Tensor, spatial_group) -> torch.Tensor:
    """Latitude-weighted spatial mean of a local field, reduced across JSpatial.

    Parameters
    ----------
    ev : module
        The ``beast.evaluation`` package.
    se_field : torch.Tensor
        Local per-channel field, shape ``(C_local, lat, lon_local)``.
    spatial_group : torch.distributed.ProcessGroup
        The ``JSpatial`` group across which longitude is sharded.

    Returns
    -------
    torch.Tensor
        Global lat-weighted mean per local channel, shape ``(C_local,)``.
    """
    local = ev.latitude_weighted_average(se_field, lat_dim=-2, lon_dim=-1)
    if dist.is_initialized() and dist.get_world_size(spatial_group) > 1:
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=spatial_group)
        local = local / dist.get_world_size(spatial_group)
    return local


def _std_for_names(ev, names: list[str], levels_full, data_std, device) -> torch.Tensor:
    """Select the per-channel std for ``names`` from a full-layout std array.

    Parameters
    ----------
    ev : module
        The ``beast.evaluation`` package.
    names : list of str
        Evaluation variable names (result order).
    levels_full : list of int or str
        The full level set defining the global channel layout.
    data_std : array-like
        Per-global-channel standard deviation.
    device : torch.device
        Device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Std values aligned with ``names``, shape ``(len(names),)``.
    """
    full_names = ev.make_variable_names(
        all_surface_variables=SURFACE_VARIABLES,
        all_pressure_variables=PRESSURE_VARIABLES,
        all_pressure_levels=list(levels_full),
    )
    pos = {n: i for i, n in enumerate(full_names)}
    std = torch.as_tensor(data_std, dtype=torch.float32)
    return torch.tensor([float(std[pos[n]]) for n in names], device=device)


def subset_indices(levels_full, levels_subset) -> torch.Tensor:
    """Find the global channel indices of a level subset within a full layout.

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

    Parameters
    ----------
    x : torch.Tensor
        Local input shard, shape.
    n_in_timesteps : int
        Number of input timesteps stacked along the channel axis.
    n_local_vars : int
        Number of local variable channels per timestep.

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

     This is not the true climatology, but a simple proxy.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader.
    device : torch.device
        Compute device.

    Returns
    -------
    torch.Tensor or None
        Time-mean field as a proxy for climatology.
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
    """Compute distributed per-variable RMSE for the model and optional baselines.

    Parameters
    ----------
    model : torch.nn.Module
        The  model to score.
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
    ev = beast_api.get_evaluation()
    spatial_group, channel_group = groups["JSpatial"], groups["JChannel"]
    eval_levels = list(subset_levels) if subset_levels is not None else list(levels_full)

    layout_kw = dict(
        all_surface_variables=SURFACE_VARIABLES,
        all_pressure_variables=PRESSURE_VARIABLES,
        all_pressure_levels=list(levels_full),
        eval_pressure_levels=eval_levels,
    )
    names = ev.make_variable_names(**layout_kw)

    clim = _accumulate_climatology(dataloader, device) if baselines else None

    model.eval()
    keys = ["model"] + (["persistence", "climatology"] if baselines else [])
    acc: dict = {k: None for k in keys}
    n_batches = 0
    for batch in dataloader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        fields = {"model": pred}
        if baselines:
            fields["persistence"] = persistence_prediction(x, n_in_timesteps, y.shape[1])
            fields["climatology"] = clim.unsqueeze(0).expand_as(y)
        for k in keys:
            se = ev.mse(fields[k], y, batch_dimension=0)
            chan_mse = _global_lat_weighted_mean(ev, se, spatial_group)
            acc[k] = chan_mse if acc[k] is None else acc[k] + chan_mse
        n_batches += 1

    result = {}
    for k in keys:
        local_mse = acc[k] / max(1, n_batches)
        local_sel = ev.select_variables(
            local_mse, dim=0,
            channel_rank=channel_group.rank(),
            channel_group_size=channel_group.size(),
            **layout_kw,
        )
        global_mse = ev.gather_along_dimension(
            local_sel, group=channel_group, dim=0, variable_size=True
        )
        assert global_mse.shape[0] == len(names), (
            f"channel gather produced {global_mse.shape[0]} values but expected "
            f"{len(names)} variables ({k}) -- channel split/order mismatch"
        )
        rmse = ev.rmse({"mse": global_mse})
        if data_std is not None:
            rmse = rmse * _std_for_names(ev, names, levels_full, data_std, rmse.device)
        result[k] = rmse
    return names, result


def validate_per_level(model, dataloader, levels_full, groups, device,
                       subset_levels=None, data_std=None) -> tuple[list[str], torch.Tensor]:
    """Compute model-only per-variable RMSE.

    Parameters
    ----------
    model : torch.nn.Module
        The model to score.
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
    """Append one row per variable to a long-format CSV.

    Parameters
    ----------
    path : str
        Output CSV path.
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
    """Save full prediction/truth/error maps for a few variables.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    dataloader : torch.utils.data.DataLoader
        Validation dataloader.
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
    ev = beast_api.get_evaluation()
    channel_group, spatial_group = groups["JChannel"], groups["JSpatial"]
    full_names = build_ordered_variables(levels_full)

    model.eval()
    batch = next(iter(dataloader))
    x, y = batch[0].to(device), batch[1].to(device)
    pred = model(x)
    chunk = pred.shape[1]

    for vname in var_names:
        if vname not in full_names:
            continue
        g = full_names.index(vname)
        owner, local = g // chunk, g % chunk
        if channel_group.rank() != owner:
            continue
        full_pred = ev.gather_along_dimension(pred[0, local], group=spatial_group, dim=-1)
        full_true = ev.gather_along_dimension(y[0, local], group=spatial_group, dim=-1)
        if dist.get_world_size(spatial_group) <= 1 or spatial_group.rank() == 0:
            os.makedirs(out_dir, exist_ok=True)
            p, t = full_pred.float().cpu().numpy(), full_true.float().cpu().numpy()
            np.save(os.path.join(out_dir, f"{vname}_pred.npy"), p)
            np.save(os.path.join(out_dir, f"{vname}_true.npy"), t)
            np.save(os.path.join(out_dir, f"{vname}_err.npy"), p - t)
