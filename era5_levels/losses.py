"""Extracted deterministic training loss from BEAST."""

import torch
import torch.nn as nn


class LatitudeWeightedMSE(nn.Module):
    """MSE on area-weighted residuals."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                sqrt_spatial_weights: torch.Tensor) -> torch.Tensor:
        """Compute the latitude-weighted MSE between prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Model prediction, shape ``(B, C, lat, lon)``.
        target : torch.Tensor
            Ground-truth field, same shape as ``pred``.
        sqrt_spatial_weights : torch.Tensor
            Square root of the per-cell area weights, broadcastable to ``pred``
            (typically ``(1, 1, lat, 1)``). Applied to both sides so the squared
            error scales with cell area.

        Returns
        -------
        torch.Tensor
            Scalar latitude-weighted MSE.
        """
        return self.mse(pred * sqrt_spatial_weights, target * sqrt_spatial_weights)
