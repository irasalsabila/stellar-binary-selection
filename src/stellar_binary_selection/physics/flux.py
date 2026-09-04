"""Physics: flux addition for unresolved binaries.

Implements the core binary photometry equations from PRD §18.
These functions are pure NumPy and are used both by the synthetic
population generator (M9) and as forward-model components of the
PI-NN (M15).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "absolute_mag_to_flux",
    "flux_to_absolute_mag",
    "combine_fluxes",
    "combine_magnitudes",
    "apply_distance_modulus",
    "resolve_unresolved_binary_mag",
]


def absolute_mag_to_flux(absolute_mag: np.ndarray | float) -> np.ndarray | float:
    """Convert absolute magnitudes to linear flux (arbitrary zero point).

    Uses the standard convention F = 10^(-0.4 M) so two equal components
    each contribute F₁ and the unresolved system has F = 2 F₁.
    """
    return np.power(10.0, -0.4 * np.asarray(absolute_mag))


def flux_to_absolute_mag(flux: np.ndarray | float) -> np.ndarray | float:
    return -2.5 * np.log10(np.maximum(np.asarray(flux), 1e-30))


def combine_fluxes(flux_1: np.ndarray | float, flux_2: np.ndarray | float) -> np.ndarray | float:
    """Add two band fluxes exactly. Result is always >= each component."""
    return np.asarray(flux_1) + np.asarray(flux_2)


def combine_magnitudes(
    mag_1: np.ndarray | float,
    mag_2: np.ndarray | float,
) -> np.ndarray | float:
    """Combine two absolute magnitudes via linear flux addition."""
    f1 = absolute_mag_to_flux(mag_1)
    f2 = absolute_mag_to_flux(mag_2)
    return flux_to_absolute_mag(combine_fluxes(f1, f2))


def apply_distance_modulus(absolute_mag: np.ndarray | float, distance_pc: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(absolute_mag) + 5.0 * np.log10(np.maximum(np.asarray(distance_pc), 1e-3)) - 5.0


def apply_extinction(
    absolute_mag: np.ndarray | float,
    a_band: np.ndarray | float,
) -> np.ndarray | float:
    return np.asarray(absolute_mag) + np.asarray(a_band)


def resolve_unresolved_binary_mag(
    mag_primary: np.ndarray | float,
    mag_secondary: np.ndarray | float,
    distance_pc: np.ndarray | float,
    a_band: np.ndarray | float = 0.0,
) -> np.ndarray | float:
    """End-to-end unresolved binary apparent magnitude in a single band."""
    f_total = combine_fluxes(absolute_mag_to_flux(mag_primary), absolute_mag_to_flux(mag_secondary))
    m_absolute = flux_to_absolute_mag(f_total)
    return apply_extinction(apply_distance_modulus(m_absolute, distance_pc), a_band)