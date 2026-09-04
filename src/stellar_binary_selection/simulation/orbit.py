"""Orbital sampling helpers (TODO M21.2)."""

from __future__ import annotations

import numpy as np

__all__ = ["sample_eccentricity", "sample_inclination", "sample_angles", "sample_phase"]


def sample_eccentricity(rng: np.random.Generator, n: int) -> np.ndarray:
    """Thermal eccentricity distribution f(e) ∝ 2e, 0 ≤ e < 1."""
    return np.sqrt(rng.uniform(0.0, 1.0, size=n))


def sample_inclination(rng: np.random.Generator, n: int) -> np.ndarray:
    """Isotropic sin(i) weighting."""
    u = rng.uniform(0.0, 1.0, size=n)
    return np.arccos(1.0 - 2.0 * u)


def sample_angles(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    omega = rng.uniform(0.0, 2.0 * np.pi, size=n)  # argument of periastron
    Omega = rng.uniform(0.0, 2.0 * np.pi, size=n)  # longitude of ascending node
    return omega, Omega


def sample_phase(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * np.pi, size=n)