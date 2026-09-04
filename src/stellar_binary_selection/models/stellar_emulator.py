"""Compact stellar emulator (TODO M8)."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["StellarEmulator"]


class StellarEmulator(nn.Module):
    """MLP that maps (M, log_age, feh) → 6 absolute magnitudes [G, BP, RP, J, H, Ks].

    Once trained on PARSEC isochrones the weights are frozen and the
    module is reused inside the PI-NN physics decoder (TODO M15.2).
    """

    def __init__(self, hidden: tuple[int, ...] = (64, 64), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 3
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)