"""Study-local deterministic evaluation.

We intentionally do NOT use beast's old top-level ``evaluation.py`` (deprecated)
or the in-flux ``beast.evaluation`` subpackage — for a fast, stable poster we
keep a small, dependency-light evaluator here and swap to beast's once it
settles.

Metrics (per output channel = per variable-at-level), latitude weighted:
- RMSE  : root mean squared error of the prediction.
- ACC   : anomaly correlation vs a climatology (optional; skipped if none given).
- BIAS  : mean signed error.

Plus the headline `subset_indices` helper for the **free** 13-vs-37 comparison:
the channel indices of the 13 standard levels *within* the 37-level layout, so a
37-level model can be scored on exactly the variables/levels the 13-level model
predicts.

All reductions here assume a *single process holds the full field*. Under the
distributed (jigsaw) layout you would gather first; for the poster it is simplest
to run evaluation by loading checkpoints on one GPU at the study resolution, or
to reuse beast's distributed validate() once stable. The metric math is the
reference implementation either way.
"""

from __future__ import annotations

import numpy as np
import torch

from .variable_layout import build_ordered_variables


def latitude_weights(n_lat: int, device=None) -> torch.Tensor:
    """cos(latitude) weights normalised to mean 1, shape (1, 1, n_lat, 1)."""
    lat = torch.linspace(np.pi / 2, -np.pi / 2, n_lat, device=device)
    w = torch.cos(lat)
    w = w * n_lat / w.sum()
    return w.view(1, 1, n_lat, 1)


def _wmean(field, weights):
    num = (field * weights).sum(dim=(0, 2, 3))
    den = weights.expand_as(field).sum(dim=(0, 2, 3))
    return num / den


def rmse(pred, target, weights):
    return torch.sqrt(_wmean((pred - target) ** 2, weights))


def bias(pred, target, weights):
    return _wmean(pred - target, weights)


def acc(pred, target, climatology, weights):
    pa, ta = pred - climatology.unsqueeze(0), target - climatology.unsqueeze(0)
    num = (pa * ta * weights).sum(dim=(0, 2, 3))
    den = torch.sqrt((pa**2 * weights).sum(dim=(0, 2, 3))
                     * (ta**2 * weights).sum(dim=(0, 2, 3)) + 1e-12)
    return num / den


def subset_indices(levels_full, levels_subset) -> torch.Tensor:
    """Channel indices of `levels_subset` within the `levels_full` layout.

    Used for the free subset-eval: index a 37-level model's output down to the
    13 standard levels so it can be compared to the 13-level model on identical
    targets. Surface variables (level-independent) are always included.
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
def evaluate_fields(pred, target, levels, weights=None, climatology=None) -> dict:
    """Per-channel RMSE/BIAS(/ACC), plus the variable names for those channels.

    pred, target : (B, C, H, W) in *normalised* units (multiply by per-variable
    std afterwards to report physical units). Returns a dict of 1-D tensors keyed
    by metric, plus ``names``.
    """
    if weights is None:
        weights = latitude_weights(pred.shape[-2], device=pred.device)
    out = {"rmse": rmse(pred, target, weights), "bias": bias(pred, target, weights)}
    if climatology is not None:
        out["acc"] = acc(pred, target, climatology, weights)
    out["names"] = build_ordered_variables(levels)
    return out
