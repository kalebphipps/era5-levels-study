"""Deterministic training loss for the levels study.

The full model trains a Bayesian ELBO; we are deliberately ignoring that and
training a plain deterministic model. The objective is a **latitude-weighted
MSE**: we multiply prediction and target by sqrt(cos-latitude weights) and take
MSE, which is exactly the deterministic branch already used in the reference
training loop (`loss_fn(pred * sqrt_w, y * sqrt_w)` with `MSELoss`). Weighting by
sqrt on both sides makes the per-pixel squared error scale with the cell area.
"""

import torch
import torch.nn as nn


class LatitudeWeightedMSE(nn.Module):
    """MSE on area-weighted residuals. Pass sqrt(spatial_weights) at call time.

    Using the same `get_spatial_weights` helper as the rest of beast keeps the
    weighting identical between training loss and evaluation metrics.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sqrt_spatial_weights: torch.Tensor) -> torch.Tensor:
        return self.mse(pred * sqrt_spatial_weights, target * sqrt_spatial_weights)
