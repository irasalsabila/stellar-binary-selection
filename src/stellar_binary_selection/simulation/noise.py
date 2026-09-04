"""Empirical Gaia/2MASS noise model (TODO M10).

`EmpiricalNoiseModel` implements magnitude-dependent per-band sigma
curves plus Gaussian injection. `load_fitted_noise_model` reads the
JSON produced by `scripts/fit_noise_model.py`, which is calibrated
against the real Gaia DR3 + 2MASS parent catalogue. If that file is
not present, `default_noise_model()` falls back to approximate
literature values so the pipeline still runs during development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["EmpiricalNoiseModel", "default_noise_model", "load_fitted_noise_model"]


def _interp(knots_x: list[float], knots_y: list[float], x: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation with constant extrapolation at the ends."""
    x = np.asarray(x, dtype=float)
    if len(knots_x) < 2:
        return np.full_like(x, knots_y[0] if knots_y else 0.0)
    return np.interp(x, np.asarray(knots_x, dtype=float), np.asarray(knots_y, dtype=float))


@dataclass
class EmpiricalNoiseModel:
    """Per-band magnitude-dependent sigma + Gaussian injection."""

    bands: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    parallax_knots: tuple[list[float], list[float]] | None = None
    source: str = "unfitted"

    def sigma(self, band: str, mag: np.ndarray | float) -> np.ndarray:
        if band not in self.bands:
            raise KeyError(f"unknown band: {band}")
        knots_mag, knots_sigma = self.bands[band]
        return _interp(knots_mag, knots_sigma, mag)

    def inject(
        self,
        mag: np.ndarray | float,
        *,
        band: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        sigma = self.sigma(band, mag)
        return np.asarray(mag) + rng.normal(0.0, sigma)

    def parallax_sigma(self, g_mag: np.ndarray | float) -> np.ndarray:
        """Parallax uncertainty in mas as a function of apparent G."""
        if self.parallax_knots is None:
            # Rough fallback: sigma_parallax grows with magnitude.
            return 0.02 + 0.002 * (np.asarray(g_mag, dtype=float) - 10.0) ** 2
        knots_mag, knots_sigma = self.parallax_knots
        return _interp(knots_mag, knots_sigma, g_mag)

    def inject_parallax(
        self,
        parallax_mas: np.ndarray | float,
        g_mag: np.ndarray | float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        sigma = self.parallax_sigma(g_mag)
        return np.asarray(parallax_mas) + rng.normal(0.0, sigma)


def load_fitted_noise_model(path: str | Path) -> EmpiricalNoiseModel:
    """Load the JSON noise model produced by scripts/fit_noise_model.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"noise model not found: {path}")
    raw = json.loads(path.read_text())
    bands = {
        band: (vals["mag"], vals["sigma_mag"]) for band, vals in raw.get("bands", {}).items()
    }
    par = raw.get("parallax_error_vs_g")
    parallax_knots = (par["mag"], par["sigma_mas"]) if par else None
    return EmpiricalNoiseModel(
        bands=bands, parallax_knots=parallax_knots, source=raw.get("source", "fitted")
    )


def default_noise_model(
    fitted_path: str | Path | None = None,
) -> EmpiricalNoiseModel:
    """Return the fitted model if available, else a literature-ish fallback."""
    if fitted_path is not None:
        try:
            return load_fitted_noise_model(fitted_path)
        except FileNotFoundError:
            pass

    # Fallback: coarse magnitude-dependent sigma in mag.
    # Gaia bands: ~0.0002–0.003 mag for G<21; 2MASS: ~0.01–0.03 mag.
    def curve(mags, sigmas):
        return (list(mags), list(sigmas))

    return EmpiricalNoiseModel(
        bands={
            "G": curve([5, 10, 15, 21], [0.00021, 0.0005, 0.0012, 0.0022]),
            "BP": curve([5, 10, 15, 22], [0.00045, 0.0012, 0.008, 0.10]),
            "RP": curve([5, 10, 15, 20], [0.00034, 0.0008, 0.004, 0.010]),
            "J": curve([5, 10, 15, 17], [0.00002, 0.00003, 0.00005, 0.00006]),
            "H": curve([5, 10, 15, 17], [0.00002, 0.00003, 0.00005, 0.00007]),
            "Ks": curve([5, 10, 15, 17], [0.00002, 0.00004, 0.00006, 0.00008]),
        },
        parallax_knots=([10, 14, 18], [0.017, 0.06, 0.19]),
        source="fallback",
    )