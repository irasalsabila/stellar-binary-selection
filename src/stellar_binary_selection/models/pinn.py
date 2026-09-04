"""Plain MLP and Physics-Informed NN models (TODO M14–M15)."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["PlainMLP", "PhysicsInformedNN", "PINNConfig"]


def _make_mlp(in_dim: int, hidden: list[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PlainMLP(nn.Module):
    """Standard regression / classification MLP (NN-0).

    Outputs a single logit per sample for binary classification.
    """

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 128, 64), dropout: float = 0.05):
        super().__init__()
        self.backbone = _make_mlp(in_dim, list(hidden), 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Inverse Model (M15).

    Encoder maps observables → latent physical parameters.
    Decoder re-uses the frozen stellar emulator and exact flux
    addition to produce reconstructed photometry.

    Latent variables returned in raw form:
        (p_binary, q, M1, log_age, feh, distance_pc, parallax_pred)
    """

    def __init__(
        self,
        in_dim: int,
        emulator: nn.Module | None,
        hidden: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.emulator = emulator  # may be None during ablation studies
        out_dim = 7  # p_bin, q, M1, log_age, feh, distance_pc, parallax_pred
        self.encoder = _make_mlp(in_dim, list(hidden), out_dim, dropout)
        self.q_min = 0.05
        self.q_max = 1.0
        self.distance_min = 1.0
        self.distance_max = 500.0

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        # apply softplus/sigmoid bounds for q and distance
        p_bin = torch.sigmoid(z[:, 0])
        q = self.q_min + (self.q_max - self.q_min) * torch.sigmoid(z[:, 1])
        m1 = 0.5 + 1.1 * torch.sigmoid(z[:, 2])  # 0.5–1.6 Msun
        log_age = 9.0 + 1.3 * torch.sigmoid(z[:, 3])  # 9.0–10.3 (yr)
        feh = -1.2 + 1.9 * torch.sigmoid(z[:, 4])  # -1.2–0.7
        dist = self.distance_min + (self.distance_max - self.distance_min) * torch.sigmoid(z[:, 5])
        parallax_pred = 1000.0 / dist
        return {
            "p_binary": p_bin,
            "q": q,
            "m1": m1,
            "log_age": log_age,
            "feh": feh,
            "distance_pc": dist,
            "parallax_pred": parallax_pred,
            "latent": z,
        }


class PINNConfig:  # noqa: D401 — simple config container
    """Container for PI-NN training hyperparameters."""

    def __init__(
        self,
        in_dim: int,
        emulator: nn.Module | None = None,
        hidden: tuple[int, ...] = (128, 128, 64),
        dropout: float = 0.05,
    ):
        self.in_dim = in_dim
        self.emulator = emulator
        self.hidden = hidden
        self.dropout = dropout