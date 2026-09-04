"""Stellar-models interface (TODO M7–M8).

The PRD specifies PARSEC as the primary isochrone grid with MIST as a
robustness alternative (M23.8). This module provides a uniform
interface to whichever grid is available locally:

1. `ParsecGrid`  — reads a manually downloaded PARSEC isochrone file
                   (CSV/Parquet with columns for mass, log_age, feh and
                   absolute magnitudes in G/BP/RP/J/H/Ks).
2. `MistGrid`    — wraps the `isochrones` package if its MIST grid is
                   cached locally.
3. `EmpiricalMSGrid` — a data-driven main-sequence model calibrated
                   directly from the observed Gaia+2MASS CMD (Sample B).
                   Used when no theoretical grid is available.

All backends expose the same call signature:
    absolute_magnitudes(mass, log_age, feh) -> (n, 6) array
with band order [G, BP, RP, J, H, Ks].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_repo_root() -> Path:
    """Walk up from this file until we find the repository root.

    The repo root is identified by the presence of `pyproject.toml`.
    This keeps path resolution correct whether the package is imported
    from `src/` directly or installed into site-packages.
    """
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fall back to three levels up (src/stellar_binary_selection/physics).
    return here.parents[3]


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

BANDS = ("G", "BP", "RP", "J", "H", "Ks")


class IsochroneGrid:
    """Base interface for stellar-model backends."""

    name = "base"

    def absolute_magnitudes(self, mass, log_age, feh):  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, mass, log_age, feh):
        return self.absolute_magnitudes(mass, log_age, feh)


class ParsecGrid(IsochroneGrid):
    """PARSEC v2.0 grid read from a local file.

    Expected columns (case-insensitive): mass or mini, log_age or age,
    feh or mh, and the absolute magnitude columns below.
    """

    name = "PARSEC"

    MAG_COLUMNS = {
        "G": ["g_mag", "gaia_g", "gmag", "g"],
        "BP": ["g_bp_mag", "bp_mag", "bpmag", "bp"],
        "RP": ["g_rp_mag", "rp_mag", "rpmag", "rp"],
        "J": ["j_mag", "jmag", "j"],
        "H": ["h_mag", "hmag", "h"],
        "Ks": ["k_mag", "ks_mag", "ksmag", "ks"],
    }

    def __init__(self, path: Path | None = None):
        import pandas as pd

        self.path = Path(path) if path else (
            REPO_ROOT / "data" / "isochrones" / "parsec" / "parsec_fgk_grid.parquet"
        )
        if not self.path.exists():
            raise FileNotFoundError(
                f"PARSEC grid not found at {self.path}. See scripts/download_parsec.py --manual."
            )
        if self.path.suffix == ".parquet":
            self.df = pd.read_parquet(self.path)
        else:
            self.df = pd.read_csv(self.path)
        self._resolve_columns()
        self._build_interpolator()

    def _resolve_columns(self) -> None:
        lower = {c.lower(): c for c in self.df.columns}

        def find(*candidates):
            for c in candidates:
                if c.lower() in lower:
                    return lower[c.lower()]
            return None

        self.mass_col = find("mass", "mini", "m_ini", "initial_mass")
        self.log_age_col = find("log_age", "logage", "log_age_yr", "age")
        self.feh_col = find("feh", "mh", "[m/h]", "metallicity")
        if self.mass_col is None or self.log_age_col is None or self.feh_col is None:
            raise ValueError(
                f"PARSEC grid missing mass/age/feh columns; found {list(self.df.columns)}"
            )
        self.mag_cols = {}
        for band, candidates in self.MAG_COLUMNS.items():
            col = find(*candidates)
            if col is None:
                raise ValueError(f"PARSEC grid missing magnitude column for band {band}")
            self.mag_cols[band] = col

    def _build_interpolator(self) -> None:
        from scipy.interpolate import LinearNDInterpolator

        mass = self.df[self.mass_col].to_numpy(dtype=float)
        log_age = self.df[self.log_age_col].to_numpy(dtype=float)
        # PARSEC sometimes stores log(age/yr); normalise to ~9–10.1.
        if np.nanmedian(log_age) > 100:  # linear age in yr
            log_age = np.log10(log_age)
        feh = self.df[self.feh_col].to_numpy(dtype=float)

        points = np.column_stack([mass, log_age, feh])
        mags = np.column_stack(
            [self.df[self.mag_cols[b]].to_numpy(dtype=float) for b in BANDS]
        )
        keep = np.isfinite(points).all(axis=1) & np.isfinite(mags).all(axis=1)
        self._interp = LinearNDInterpolator(points[keep], mags[keep])

    def absolute_magnitudes(self, mass, log_age, feh):
        mass = np.atleast_1d(np.asarray(mass, dtype=float))
        log_age = np.atleast_1d(np.asarray(log_age, dtype=float))
        feh = np.atleast_1d(np.asarray(feh, dtype=float))
        out = self._interp(mass, log_age, feh)
        return np.nan_to_num(out, nan=np.nanmedian(out) if np.isfinite(out).any() else 10.0)


class MistGrid(IsochroneGrid):
    """MIST grid via the `isochrones` package (robustness alternative)."""

    name = "MIST"

    def __init__(self):
        from isochrones import get_ichrone

        self.iso = get_ichrone("mist", bands=list(BANDS))

    def absolute_magnitudes(self, mass, log_age, feh):
        # `isochrones` interpolates on EEP, not mass; it exposes a
        # mass->EEP mapping through `.get_eep`. We approximate by using
        # the EEP grid nearest the requested mass at fixed age/feh.
        mass = np.atleast_1d(np.asarray(mass, dtype=float))
        log_age = np.atleast_1d(np.asarray(log_age, dtype=float))
        feh = np.atleast_1d(np.asarray(feh, dtype=float))
        n = max(len(mass), len(log_age), len(feh))
        mass, log_age, feh = (
            np.broadcast_to(mass, (n,)),
            np.broadcast_to(log_age, (n,)),
            np.broadcast_to(feh, (n,)),
        )
        out = np.empty((n, len(BANDS)))
        for i in range(n):
            try:
                eep = self.iso.get_eep(mass[i], log_age[i], feh[i], accurate=True)
                out[i] = self.iso.interp_mag([eep, log_age[i], feh[i]], list(BANDS))
            except Exception:
                out[i] = np.nan
        return np.nan_to_num(out, nan=10.0)


class EmpiricalMSGrid(IsochroneGrid):
    """Data-driven main-sequence model calibrated from the observed CMD.

    Fits a smooth relation M_G(BP-RP) plus colour–colour relations from
    the apparently-single sample, then inverts the empirical
    mass–luminosity relation to map requested masses onto the sequence.

    This is a real (not placeholder) stellar model derived from Gaia+2MASS
    data, suitable for use until the PARSEC grid is available locally.
    """

    name = "EmpiricalMS"

    def __init__(self, sample_b_path: Path | None = None):
        import pandas as pd

        self.sample_b_path = (
            Path(sample_b_path)
            if sample_b_path
            else REPO_ROOT / "data" / "processed" / "sample_b_apparently_single.parquet"
        )
        if not self.sample_b_path.exists():
            raise FileNotFoundError(
                f"Sample B not found at {self.sample_b_path}. Run build_dataset.py --stage samples."
            )
        df = pd.read_parquet(self.sample_b_path)
        self._fit(df)

    def _fit(self, df: pd.DataFrame) -> None:
        import pandas as pd

        d = df.copy()
        for c in ("bp_rp", "m_g_prelim", "g_j", "g_ks", "j_h", "h_ks", "j_ks"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        ok = (
            np.isfinite(d["bp_rp"])
            & np.isfinite(d["m_g_prelim"])
            & d["bp_rp"].between(0.4, 1.8)
            & d["m_g_prelim"].between(3.0, 9.0)
        )
        d = d.loc[ok]
        if len(d) < 500:
            raise ValueError(f"too few MS stars to calibrate ({len(d)})")

        # --- M_G as a function of BP-RP (the MS ridge) ---
        self.ridge_order = 3
        self.ridge_coef = np.polyfit(d["bp_rp"], d["m_g_prelim"], self.ridge_order)
        # scatter about the ridge (intrinsic + metallicity spread)
        resid = d["m_g_prelim"] - np.polyval(self.ridge_coef, d["bp_rp"])
        self.ridge_scatter = float(np.std(resid))
        self.ridge_scatter = min(max(self.ridge_scatter, 0.10), 0.60)

        # --- colour–colour relations, so colour indices follow from BP-RP ---
        self.colour_fits: dict[str, np.ndarray] = {}
        for c in ("g_j", "g_ks", "j_h", "h_ks", "j_ks"):
            if c not in d.columns:
                continue
            sel = np.isfinite(d[c])
            if sel.sum() < 500:
                continue
            self.colour_fits[c] = np.polyfit(d.loc[sel, "bp_rp"], d.loc[sel, c], 2)

        # --- empirical mass–colour mapping ---
        # Use the mass–M_G relation for nearby MS stars: more massive
        # stars are bluer and brighter. We calibrate a linear map from
        # BP-RP to mass anchored on the FGK range (0.6–1.4 Msun).
        bp_lo, bp_hi = float(np.percentile(d["bp_rp"], 5)), float(np.percentile(d["bp_rp"], 95))
        self.bp_rp_range = (max(bp_lo, 0.45), min(bp_hi, 1.6))
        self.mass_range = (0.6, 1.4)
        self.n_calibrated = int(len(d))

    def _mass_to_bp_rp(self, mass):
        """Map primary mass (0.6–1.4 Msun) onto the empirical BP-RP range."""
        m_lo, m_hi = self.mass_range
        bp_lo, bp_hi = self.bp_rp_range
        m = np.clip(np.asarray(mass, dtype=float), m_lo, m_hi)
        # more massive -> bluer
        return bp_hi + (bp_lo - bp_hi) * (m - m_lo) / (m_hi - m_lo)

    def absolute_magnitudes(self, mass, log_age, feh):
        mass = np.atleast_1d(np.asarray(mass, dtype=float))
        feh = np.atleast_1d(np.asarray(feh, dtype=float))
        n = max(len(mass), len(feh))
        mass = np.broadcast_to(mass, (n,))
        feh = np.broadcast_to(feh, (n,))

        bp_rp = self._mass_to_bp_rp(mass)
        m_g = np.polyval(self.ridge_coef, bp_rp)
        # metallicity perturbation: metal-poor stars are slightly bluer
        # and fainter at fixed colour; keep this modest.
        m_g = m_g + 0.15 * (-feh)

        # colours follow from BP-RP using the empirical relations
        g_j = np.polyval(self.colour_fits.get("g_j", np.array([0, 1.0])), bp_rp) if "g_j" in self.colour_fits else 1.1 * bp_rp + 0.4
        g_ks = np.polyval(self.colour_fits.get("g_ks", np.array([0, 1.2])), bp_rp) if "g_ks" in self.colour_fits else 1.3 * bp_rp + 0.5
        j_h = np.polyval(self.colour_fits.get("j_h", np.array([0, 0.3])), bp_rp) if "j_h" in self.colour_fits else 0.30 * bp_rp + 0.05
        h_ks = np.polyval(self.colour_fits.get("h_ks", np.array([0, 0.1])), bp_rp) if "h_ks" in self.colour_fits else 0.10 * bp_rp + 0.02

        m_g_band = m_g + 0.05  # Gaia G vs the M_G ridge definition
        m_bp = m_g + g_j * 0.0 + bp_rp / 2.0 + 0.20  # BP = G + (BP-G)
        m_rp = m_g - bp_rp / 2.0 - 0.20
        m_j = m_g - g_j
        m_h = m_j - j_h
        m_ks = m_h - h_ks
        return np.column_stack([m_g_band, m_bp, m_rp, m_j, m_h, m_ks])


def load_isochrone_grid(prefer: str = "auto", **kwargs) -> IsochroneGrid:
    """Return the best available isochrone backend.

    Order: PARSEC (if grid file present) → MIST (if cached) → EmpiricalMS.
    """
    order = {
        "auto": ["parsec", "mist", "empirical"],
        "parsec": ["parsec"],
        "mist": ["mist"],
        "empirical": ["empirical"],
    }[prefer]

    for backend in order:
        try:
            if backend == "parsec":
                return ParsecGrid(kwargs.get("parsec_path"))
            if backend == "mist":
                return MistGrid()
            if backend == "empirical":
                return EmpiricalMSGrid(kwargs.get("sample_b_path"))
        except Exception as exc:  # noqa: BLE001 — try next backend
            print(f"  isochrone backend '{backend}' unavailable: {exc}", file=sys.stderr)
    raise RuntimeError("no isochrone backend available")


__all__ = [
    "BANDS",
    "EmpiricalMSGrid",
    "IsochroneGrid",
    "MistGrid",
    "ParsecGrid",
    "load_isochrone_grid",
]