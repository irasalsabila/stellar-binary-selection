"""Astrophysical primitives: flux addition, extinction, distance, isochrones."""

from .flux import (
    absolute_mag_to_flux,
    apply_distance_modulus,
    apply_extinction,
    combine_fluxes,
    combine_magnitudes,
    flux_to_absolute_mag,
    resolve_unresolved_binary_mag,
)
from .isochrones import (
    BANDS,
    EmpiricalMSGrid,
    IsochroneGrid,
    MistGrid,
    ParsecGrid,
    load_isochrone_grid,
)

__all__ = [
    "absolute_mag_to_flux",
    "apply_distance_modulus",
    "apply_extinction",
    "combine_fluxes",
    "combine_magnitudes",
    "flux_to_absolute_mag",
    "resolve_unresolved_binary_mag",
    "BANDS",
    "EmpiricalMSGrid",
    "IsochroneGrid",
    "MistGrid",
    "ParsecGrid",
    "load_isochrone_grid",
]