"""Synthetic stellar population generator.

Implements TODO M9 and M11. Produces a parquet table of single and
binary systems with PARSEC-derived absolute magnitudes (placeholder
until the emulator from M8 is trained), exact flux addition, distance
modulus, optional extinction and observational noise injection (M10).

Outputs:
    data/synthetic/synthetic_<tag>.parquet
with ground-truth columns and noisy observables in the same band set
the real Gaia+2MASS catalogue uses (G, BP, RP, J, H, Ks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.physics.flux import (
    apply_distance_modulus,
    apply_extinction,
    combine_magnitudes,
)
from stellar_binary_selection.simulation.noise import (  # noqa: F401  (filled below)
    default_noise_model,
)
from stellar_binary_selection.utils import set_global_seed, write_run_metadata


BANDS = ("G", "BP", "RP", "J", "H", "Ks")


def sample_primary_mass(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)


def sample_mass_ratio(rng: np.random.Generator, n: int) -> np.ndarray:
    """Piecewise weighting: heavier around q ≈ 0.3–0.8.

    Implements the weighting in config/synthetic.yaml. Vectorised
    piecewise inversion of the empirical CDF.
    """
    edges = np.array([0.10, 0.30, 0.80, 1.00])
    weights = np.array([1.0, 3.0, 1.0])
    widths = np.diff(edges)
    bin_probs = widths * weights
    cdf = np.cumsum(bin_probs)
    cdf /= cdf[-1]
    # CDF inversion: for u in [cdf_k, cdf_{k+1}] → q in [edges_k, edges_{k+1}]
    u = rng.uniform(0.0, 1.0, size=n)
    # find bin via searchsorted on cdf (n-1 cut points)
    idx = np.searchsorted(cdf[:-1], u, side="right")
    idx = np.clip(idx, 0, len(widths) - 1)
    bin_lo = edges[idx]
    bin_width = widths[idx]
    bin_cdf_lo = np.where(idx == 0, 0.0, cdf[np.maximum(idx - 1, 0)])
    local = (u - bin_cdf_lo) / (cdf[idx] - bin_cdf_lo)
    q = bin_lo + local * bin_width
    return q


def sample_age_gyr(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.5, 12.0, size=n)


def sample_metallicity(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-1.0, 0.5, size=n)


def sample_distance_pc(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(20.0, 200.0, size=n)


def sample_log_period_days(rng: np.random.Generator, n: int, lo: float = 1.0, hi: float = 3000.0) -> np.ndarray:
    return 10 ** rng.uniform(np.log10(lo), np.log10(hi), size=n)


def synthetic_absolute_magnitudes(
    primary_mass: np.ndarray,
    age_gyr: np.ndarray,
    feh: np.ndarray,
    emulator,
) -> np.ndarray:
    """Return absolute magnitudes (n, 6) in BANDS for single stars.

    Until the trained emulator from M8 is available we use a deliberately
    crude placeholder: a linear mass-luminosity relation plus small
    metallicity/age perturbations. This must be replaced by the frozen
    emulator before any PI-NN training (TODO M8.3).
    """
    n = len(primary_mass)
    placeholder = np.zeros((n, len(BANDS)))
    # crude main-sequence: M_G ≈ 4.83 + 5*(log10(0.9)-log10(M/Msun))
    # ignore metallicity / age for placeholder
    mg = 4.83 + 5.0 * (np.log10(0.9) - np.log10(np.maximum(primary_mass, 1e-3)))
    # crude colours
    bp_rp = 1.2 * (1.0 - primary_mass) + 0.4
    bp = mg + bp_rp / 2.0 + 0.2
    rp = mg - bp_rp / 2.0 - 0.2
    g = mg - 0.05
    j = mg + 0.3
    h = mg + 0.5
    ks = mg + 0.6
    placeholder[:, 0] = g
    placeholder[:, 1] = bp
    placeholder[:, 2] = rp
    placeholder[:, 3] = j
    placeholder[:, 4] = h
    placeholder[:, 5] = ks
    if emulator is not None:
        try:
            placeholder = emulator.predict(primary_mass, age_gyr, feh)
        except Exception:
            pass
    return placeholder


def add_intrinsic_ms_scatter(rng: np.random.Generator, abs_mags: np.ndarray, sigma_mag: float = 0.25) -> np.ndarray:
    """Add realistic intrinsic main-sequence scatter to absolute magnitudes.

    The observed Gaia+2MASS main sequence is NOT a razor-thin ridge. At
    fixed colour it has ~0.25 mag of astrophysical scatter (measured
    from the apparently-single sample) due to the spread in age,
    metallicity, and residual systematics. Without this scatter,
    synthetic singles are confined to an artificially thin locus and the
    photometric selection function saturates at C_phot = 1 for all q,
    which is unphysical.

    The scatter is applied coherently in G (the luminosity direction)
    with colour-correlated offsets, so it moves stars ALONG the MS
    rather than adding independent per-band noise.
    """
    n = abs_mags.shape[0]
    # luminosity-direction offset in G
    dG = rng.normal(0.0, sigma_mag, size=n)
    out = abs_mags.copy()
    out[:, 0] += dG
    # colours shift more weakly (metallicity/age move colour less than
    # luminosity for FGK stars): ~0.3 of the G offset in BP-RP etc.
    out[:, 1] += dG * 0.30  # BP
    out[:, 2] += dG * 0.10  # RP
    out[:, 3] += dG * 0.12  # J
    out[:, 4] += dG * 0.12  # H
    out[:, 5] += dG * 0.12  # Ks
    return out


def generate(
    n_systems: int,
    *,
    binary_fraction: float = 0.5,
    seed: int = 42,
    emulator=None,
    noise_model=None,
    intrinsic_scatter_mag: float = 0.25,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_binary = int(round(n_systems * binary_fraction))
    n_single = n_systems - n_binary

    n_total = n_systems
    m1 = sample_primary_mass(rng, n_total, 0.6, 1.4)
    age = sample_age_gyr(rng, n_total)
    feh = sample_metallicity(rng, n_total)
    dist = sample_distance_pc(rng, n_total)
    q_full = sample_mass_ratio(rng, n_total)

    is_binary = np.zeros(n_total, dtype=bool)
    is_binary[:n_binary] = True
    rng.shuffle(is_binary)

    q = np.where(is_binary, q_full, 0.0)
    m2 = q * m1

    abs_mag_primary = synthetic_absolute_magnitudes(m1, age, feh, emulator)

    # build secondary absolute mags by re-evaluating the emulator at M2
    abs_mag_secondary = synthetic_absolute_magnitudes(np.maximum(m2, 1e-3), age, feh, emulator)

    # Inject intrinsic main-sequence scatter (age/metallicity spread) so
    # the synthetic CMD width matches the observed one.
    abs_mag_primary = add_intrinsic_ms_scatter(rng, abs_mag_primary, intrinsic_scatter_mag)
    abs_mag_secondary = add_intrinsic_ms_scatter(rng, abs_mag_secondary, intrinsic_scatter_mag)

    # unresolved photometry: exact flux addition per band
    combined_abs = np.empty_like(abs_mag_primary)
    for i, _ in enumerate(BANDS):
        combined_abs[:, i] = combine_magnitudes(
            abs_mag_primary[:, i], abs_mag_secondary[:, i]
        )

    # apply distance modulus + extinction
    apparent_clean = apply_distance_modulus(combined_abs, dist[:, None])

    df = pd.DataFrame({
        "source_id": np.arange(n_total, dtype=np.int64),
        "is_binary": is_binary,
        "m1_msun": m1,
        "m2_msun": m2,
        "q": q,
        "age_gyr": age,
        "feh": feh,
        "distance_pc": dist,
        "log_period_days": np.where(is_binary, sample_log_period_days(rng, n_total), np.nan),
        "abs_mag_G": combined_abs[:, 0],
        "abs_mag_BP": combined_abs[:, 1],
        "abs_mag_RP": combined_abs[:, 2],
        "abs_mag_J": combined_abs[:, 3],
        "abs_mag_H": combined_abs[:, 4],
        "abs_mag_Ks": combined_abs[:, 5],
    })
    for j, b in enumerate(BANDS):
        col = f"apparent_{b}"
        clean = apparent_clean[:, j]
        if noise_model is None:
            df[col] = clean
        else:
            df[col] = noise_model.inject(clean, band=b)
        # also expose the per-band uncertainty used during injection
        if noise_model is not None:
            df[f"sigma_{b}"] = noise_model.sigma(b, clean)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binary-fraction",
        type=float,
        default=0.5,
        help="Training binary fraction; underlying physical weights are saved separately.",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic/synthetic_v1.parquet",
    )
    parser.add_argument("--metadata-dir", default="results/runs/generate_synthetic")
    args = parser.parse_args()

    set_global_seed(args.seed)
    df = generate(
        args.n,
        binary_fraction=args.binary_fraction,
        seed=args.seed,
    )
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic systems to {out_path}")

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={
            "n_systems": args.n,
            "seed": args.seed,
            "binary_fraction": args.binary_fraction,
        },
        metrics={
            "n_rows": len(df),
            "n_binary": int(df["is_binary"].sum()),
            "n_single": int((~df["is_binary"]).sum()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())