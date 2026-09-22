"""Generate the first 100k synthetic population with progress reporting.

Run via:
    PYTHONPATH=src python scripts/generate_synthetic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.physics import load_isochrone_grid
from stellar_binary_selection.physics.flux import apply_distance_modulus, combine_magnitudes
from stellar_binary_selection.simulation.noise import default_noise_model
from stellar_binary_selection.simulation.population import (
    BANDS,
    add_intrinsic_ms_scatter,
    sample_age_gyr,
    sample_distance_pc,
    sample_log_period_days,
    sample_mass_ratio,
    sample_metallicity,
    sample_primary_mass,
    synthetic_absolute_magnitudes,
)
from stellar_binary_selection.utils import write_run_metadata

DEFAULT_NOISE_PATH = REPO_ROOT / "data" / "processed" / "noise_model.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--binary-fraction", type=float, default=0.5)
    p.add_argument("--output", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument("--metadata-dir", default="results/runs/generate_synthetic")
    p.add_argument(
        "--grid-backend",
        default="auto",
        choices=["auto", "parsec", "mist", "empirical"],
        help="Isochrone backend used to generate the synthetic photometry.",
    )
    p.add_argument(
        "--intrinsic-scatter",
        type=float,
        default=0.25,
        help="Intrinsic main-sequence scatter in mag (measured from real Gaia MS ~0.25).",
    )
    p.add_argument(
        "--noise-model",
        default=str(DEFAULT_NOISE_PATH),
        help="JSON noise model from fit_noise_model.py; falls back to defaults if missing.",
    )
    p.add_argument(
        "--support-policy", choices=["reject", "nearest"], default="reject",
        help="How to handle parameters outside the PARSEC convex hull.",
    )
    args = p.parse_args()

    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n:,} synthetic systems (binary_fraction={args.binary_fraction})...", flush=True)
    rng = np.random.default_rng(args.seed)
    n_total = args.n
    n_binary = int(round(n_total * args.binary_fraction))
    is_binary = np.zeros(n_total, dtype=bool)
    is_binary[:n_binary] = True
    rng.shuffle(is_binary)

    with tqdm(total=5, desc="synthetic", unit="step") as bar:
        bar.set_postfix_str("sampling primaries")
        m1 = sample_primary_mass(rng, n_total, 0.6, 1.4)
        age = sample_age_gyr(rng, n_total)
        feh = sample_metallicity(rng, n_total)
        dist = sample_distance_pc(rng, n_total)
        q_full = sample_mass_ratio(rng, n_total)
        q = np.where(is_binary, q_full, 0.0)
        m2 = q * m1
        bar.update(1)

        bar.set_postfix_str("isochrone photometry")
        # Use the real isochrone grid (not the old placeholder relation) so
        # the synthetic photometry is consistent with the frozen stellar
        # emulator used by the PI-NN photometric loss.
        grid = load_isochrone_grid(args.grid_backend)
        max_age_gyr = 12.0
        if hasattr(grid, "log_age_range"):
            max_age_gyr = min(12.0, 10 ** float(grid.log_age_range[1]) / 1e9)
            age = np.clip(age, 0.5, max_age_gyr)
            print(f"  age support used: 0.5--{max_age_gyr:.3f} Gyr", flush=True)
        log_age = np.log10(age * 1e9)
        if args.support_policy == "reject" and not hasattr(grid, "support_mask"):
            raise RuntimeError("support-policy=reject requires a grid with convex-hull support metadata")
        if args.support_policy == "reject" and hasattr(grid, "support_mask"):
            # Rejection sampling preserves the requested binary fraction while
            # ensuring both components are represented by the linear grid.
            with tqdm(total=100, desc="PARSEC support", unit="attempt", leave=True) as support_bar:
              for attempt in range(100):
                ok_p = grid.support_mask(m1, log_age, feh)
                ok_s = np.ones(n_total, dtype=bool)
                if is_binary.any():
                    ok_s[is_binary] = grid.support_mask(
                        np.maximum(m2[is_binary], 0.09), log_age[is_binary], feh[is_binary]
                    )
                ok = ok_p & ok_s
                support_bar.update(1)
                support_bar.set_postfix(valid=f"{ok.mean():.1%}")
                if ok.all():
                    break
                bad = ~ok
                m1[bad] = sample_primary_mass(rng, int(bad.sum()), 0.6, 1.4)
                age[bad] = rng.uniform(0.5, max_age_gyr, size=int(bad.sum()))
                feh[bad] = sample_metallicity(rng, int(bad.sum()))
                q_full[bad] = sample_mass_ratio(rng, int(bad.sum()))
                q[bad] = np.where(is_binary[bad], q_full[bad], 0.0)
                m2[bad] = q[bad] * m1[bad]
                log_age[bad] = np.log10(age[bad] * 1e9)
              else:
                  raise RuntimeError("Could not sample a fully PARSEC-supported population after 100 attempts")
            print(f"  support rejection attempts: {attempt + 1}", flush=True)
        abs_p = np.asarray(grid.absolute_magnitudes(m1, log_age, feh), dtype=np.float64)
        fallback_p = int(getattr(grid, "last_nearest_fallback_count", 0))
        abs_s = abs_p.copy()
        fallback_s = 0
        if is_binary.any():
            abs_s[is_binary] = np.asarray(
                grid.absolute_magnitudes(
                    np.maximum(m2[is_binary], 0.09), log_age[is_binary], feh[is_binary]
                ),
                dtype=np.float64,
            )
            fallback_s = int(getattr(grid, "last_nearest_fallback_count", 0))
        if fallback_p or fallback_s:
            print(
                f"  nearest-grid fallback rows: primary={fallback_p:,}, secondary={fallback_s:,}",
                flush=True,
            )
        abs_p = add_intrinsic_ms_scatter(rng, abs_p, args.intrinsic_scatter)
        abs_s = add_intrinsic_ms_scatter(rng, abs_s, args.intrinsic_scatter)
        bar.update(1)

        bar.set_postfix_str("flux addition")
        combined = np.empty_like(abs_p)
        for i, _ in enumerate(BANDS):
            combined[:, i] = combine_magnitudes(abs_p[:, i], abs_s[:, i])
        # A nominal single has no secondary flux; do not evaluate an
        # out-of-domain zero-mass companion or double its primary flux.
        combined[~is_binary] = abs_p[~is_binary]
        bar.update(1)

        bar.set_postfix_str("distance modulus")
        apparent_clean = apply_distance_modulus(combined, dist[:, None])
        bar.update(1)

        bar.set_postfix_str("noise injection")
        nm = default_noise_model(args.noise_model)
        print(f"  noise model source: {nm.source}", flush=True)
        df = pd.DataFrame({
            "source_id": np.arange(n_total, dtype=np.int64),
            "is_binary": is_binary,
            "m1_msun": m1, "m2_msun": m2, "q": q,
            "age_gyr": age, "feh": feh, "distance_pc": dist,
            "log_period_days": np.where(is_binary, sample_log_period_days(rng, n_total), np.nan),
            "abs_mag_G": combined[:, 0],
            "abs_mag_BP": combined[:, 1],
            "abs_mag_RP": combined[:, 2],
            "abs_mag_J": combined[:, 3],
            "abs_mag_H": combined[:, 4],
            "abs_mag_Ks": combined[:, 5],
        })
        for j, b in enumerate(BANDS):
            df[f"apparent_{b}"] = nm.inject(apparent_clean[:, j], band=b)
            df[f"sigma_{b}"] = nm.sigma(b, apparent_clean[:, j])
        # Parallax observable with empirically calibrated uncertainty.
        true_parallax = 1000.0 / dist
        df["parallax_obs"] = nm.inject_parallax(true_parallax, apparent_clean[:, 0])
        df["parallax_true"] = true_parallax
        df["parallax_error"] = nm.parallax_sigma(apparent_clean[:, 0])
        bar.update(1)

    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} rows to {out.relative_to(REPO_ROOT)}", flush=True)
    print(f"  binary = {int(df['is_binary'].sum()):,}", flush=True)
    print(f"  single = {int((~df['is_binary']).sum()):,}", flush=True)

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={"n_systems": args.n, "seed": args.seed, "binary_fraction": args.binary_fraction},
        metrics={
            "n_rows": len(df),
            "n_binary": int(df["is_binary"].sum()),
            "n_single": int((~df["is_binary"]).sum()),
            "q_min": float(df.loc[df["is_binary"], "q"].min()),
            "q_max": float(df.loc[df["is_binary"], "q"].max()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
