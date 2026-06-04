"""Free 13-vs-37 subset evaluation (run offline, after training).

The headline result: a 37-level model is scored on EXACTLY the 13 standard
levels (a subset of its outputs), so it is compared to the 13-level model on
identical variables/levels — no extra training.

This script is deliberately framework-light: it takes two arrays of predictions
and the matching targets (already de-normalised or in normalised units — be
consistent) and prints per-variable RMSE plus the 13-vs-37 difference. How you
*produce* the predictions depends on your eval setup (single-GPU checkpoint
reload, or beast's distributed validate once stable); this handles the scoring
and the channel bookkeeping, which is the easy-to-get-wrong part.

Example
-------
    # pred37: (B, 228, H, W) from the 37-level model
    # pred13: (B, 84,  H, W) from the 13-level model
    # truth : the ground-truth fields for each
    python -c "..."  # or import the functions below
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from era5_levels.evaluate import evaluate_fields, latitude_weights, subset_indices
from era5_levels.variable_layout import PRESSURE_LEVELS_13, PRESSURE_LEVELS_37


def subset_37_to_13(pred37: torch.Tensor) -> torch.Tensor:
    """Select the 13-standard-level channels from a 37-level prediction."""
    idx = subset_indices(PRESSURE_LEVELS_37, PRESSURE_LEVELS_13)
    return pred37[:, idx]


def compare(pred13, truth13, pred37, truth37):
    """Per-variable RMSE for both models on the shared 13 levels + their delta.

    pred37/truth37 are full 37-level fields; they are subset to the 13 levels
    internally so both models are scored on the same targets.
    """
    w = latitude_weights(pred13.shape[-2], device=pred13.device)
    p37 = subset_37_to_13(pred37)
    t37 = subset_37_to_13(truth37)

    r13 = evaluate_fields(pred13, truth13, PRESSURE_LEVELS_13, weights=w)
    r37 = evaluate_fields(p37, t37, PRESSURE_LEVELS_13, weights=w)

    names = r13["names"]
    print(f"{'variable':30s} {'rmse_13':>10s} {'rmse_37':>10s} {'Δ(37-13)':>10s}")
    for i, name in enumerate(names):
        a, b = r13["rmse"][i].item(), r37["rmse"][i].item()
        print(f"{name:30s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
    print(f"\nmean RMSE  13-level: {r13['rmse'].mean():.4f}   "
          f"37-level (on 13): {r37['rmse'].mean():.4f}")
    return r13, r37


def _demo():
    """Shape-only demo with random data so the script runs without checkpoints."""
    B, H, W = 2, 32, 64
    n13 = 6 + 6 * len(PRESSURE_LEVELS_13)
    n37 = 6 + 6 * len(PRESSURE_LEVELS_37)
    g = torch.Generator().manual_seed(0)
    compare(torch.randn(B, n13, H, W, generator=g),
            torch.randn(B, n13, H, W, generator=g),
            torch.randn(B, n37, H, W, generator=g),
            torch.randn(B, n37, H, W, generator=g))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run a random-data shape demo (no checkpoints needed)")
    args = ap.parse_args()
    if args.demo:
        _demo()
    else:
        print("Wire up your prediction loading, then call compare(...). "
              "Run with --demo to see the scoring on random data.")
