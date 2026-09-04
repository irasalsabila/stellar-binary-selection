"""Empirical Gaia/2MASS noise model calibrated from the real parent sample.

Implements TODO M10. Fits magnitude-dependent sigma(band) curves using
the observed uncertainties and residual scatter in the parent catalogue,
then rewrites the default noise model used by the synthetic generator.

Run via:
    PYTHONPATH=src python scripts/fit_noise_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# (photometric column, uncertainty column, label)
BANDS = [
    ("phot_g_mean_mag", "phot_g_mean_flux_error", "G"),
    ("phot_bp_mean_mag", "phot_bp_mean_flux_error", "BP"),
    ("phot_rp_mean_mag", "phot_rp_mean_flux_error", "RP"),
    ("j_m", "j_msigcom", "J"),
    ("h_m", "h_msigcom", "H"),
    ("ks_m", "ks_msigcom", "Ks"),
]


def flux_error_to_mag_sigma(flux_err, flux):
    """Convert a flux uncertainty to a magnitude uncertainty.

    sigma_mag ≈ (2.5 / ln 10) * (sigma_F / F) = 1.0857 * (sigma_F / F)
    """
    return 1.085736 * np.asarray(flux_err) / np.abs(np.asarray(flux))


def fit_band(mag: np.ndarray, sigma_mag: np.ndarray, n_bins: int = 12):
    """Fit a smooth sigma(mag) curve using binned medians.

    Returns (knots_mag, knots_sigma) for piecewise-linear interpolation.
    """
    mag = np.asarray(mag, dtype=float)
    sigma_mag = np.asarray(sigma_mag, dtype=float)
    ok = np.isfinite(mag) & np.isfinite(sigma_mag) & (sigma_mag > 0)
    mag, sigma_mag = mag[ok], sigma_mag[ok]
    if len(mag) < 100:
        return None

    # bin on magnitude percentiles for even coverage
    edges = np.percentile(mag, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = mag.min() - 1e-6, mag.max() + 1e-6
    idx = np.clip(np.digitize(mag, edges) - 1, 0, n_bins - 1)

    knots_mag, knots_sigma = [], []
    for i in range(n_bins):
        sel = idx == i
        if sel.sum() < 20:
            continue
        knots_mag.append(float(np.median(mag[sel])))
        knots_sigma.append(float(np.median(sigma_mag[sel])))
    if len(knots_mag) < 2:
        return None
    return knots_mag, knots_sigma


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/processed/gaia_dr3_2mass_derived.parquet",
        help="Parent catalogue with photometry + uncertainties.",
    )
    p.add_argument(
        "--output",
        default="data/processed/noise_model.json",
        help="Where to save the fitted noise model.",
    )
    args = p.parse_args()

    path = REPO_ROOT / args.input
    if not path.exists():
        raise SystemExit(f"Parent catalogue not found at {path}. Run build_dataset.py first.")
    print(f"Loading {path.name}...", flush=True)
    df = pd.read_parquet(path)

    # Recover fluxes from the ADQL-selected flux columns where available.
    model: dict = {"bands": {}, "version": "1.0", "source": str(args.input)}
    print("\nFitting per-band sigma(mag) curves...", flush=True)
    print(f"{'band':>6s}  {'n':>8s}  {'mag range':>18s}  {'sigma range (mag)':>22s}")
    for mag_col, err_col, label in BANDS:
        if mag_col not in df.columns or err_col not in df.columns:
            print(f"{label:>6s}  MISSING COLUMN")
            continue
        mag = pd.to_numeric(df[mag_col], errors="coerce")

        if err_col.endswith("sigcom"):
            # 2MASS: already a magnitude uncertainty
            sigma = pd.to_numeric(df[err_col], errors="coerce") / 1000.0  # mmag → mag
        else:
            # Gaia: flux error + flux → magnitude error
            flux_col = mag_col.replace("_mean_mag", "_mean_flux")
            if flux_col not in df.columns:
                print(f"{label:>6s}  missing flux column {flux_col}")
                continue
            flux = pd.to_numeric(df[flux_col], errors="coerce")
            sigma = pd.Series(flux_error_to_mag_sigma(df[err_col], flux), index=df.index)

        fit = fit_band(mag.to_numpy(), sigma.to_numpy())
        if fit is None:
            print(f"{label:>6s}  fit failed")
            continue
        knots_mag, knots_sigma = fit
        model["bands"][label] = {"mag": knots_mag, "sigma_mag": knots_sigma}
        ok = np.isfinite(sigma.to_numpy())
        print(
            f"{label:>6s}  {int(ok.sum()):>8,}  "
            f"[{np.nanmin(mag):6.2f}, {np.nanmax(mag):6.2f}]  "
            f"[{min(knots_sigma):.5f}, {max(knots_sigma):.5f}]"
        )

    # Parallax uncertainty model: sigma_parallax as a function of G magnitude
    if "parallax_error" in df.columns and "phot_g_mean_mag" in df.columns:
        g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
        pe = pd.to_numeric(df["parallax_error"], errors="coerce")
        fit = fit_band(g.to_numpy(), pe.to_numpy(), n_bins=10)
        if fit is not None:
            model["parallax_error_vs_g"] = {"mag": fit[0], "sigma_mas": fit[1]}
            print(f"\n{'parallax_error vs G':>6s}: knots at G = "
                  f"[{min(fit[0]):.2f}, {max(fit[0]):.2f}], "
                  f"sigma = [{min(fit[1]):.5f}, {max(fit[1]):.5f}] mas")

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    print(f"  bands fitted: {sorted(model['bands'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())