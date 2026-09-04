"""Photometric–astrometric complementarity analysis (TODO M22, E12).

This is the MAIN SCIENTIFIC GATE of the project. It evaluates both
detection channels across the synthetic binary population and produces
the four-regime complementarity map.

Run via:
    PYTHONPATH=src python scripts/complementarity.py
    PYTHONPATH=src python scripts/complementarity.py --fpr 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.evaluation.metrics import (  # noqa: E402
    binomial_completeness,
)
from stellar_binary_selection.selection.astrometric import (  # noqa: E402
    astro_detection_probability,
)
from stellar_binary_selection.selection.complementarity import (  # noqa: E402
    REGIME_LABELS,
    assign_regimes,
    completeness_grid,
    check_independence,
)
from stellar_binary_selection.utils import write_run_metadata  # noqa: E402


def photometric_score(df: pd.DataFrame) -> np.ndarray:
    """Photometric binary score: overluminosity above the single-star ridge.

    Uses the OBSERVED parallax (not the true distance) to build the CMD,
    because detection can only use quantities a real survey measures.
    Using the true distance would leak ground truth and drive C_phot to
    an unphysical value of 1.

    Returns the overluminosity in magnitudes; larger = brighter than the
    single-star expectation at the same observed colour.
    """
    parallax = pd.to_numeric(df["parallax_obs"], errors="coerce")
    # guard against non-positive parallax before taking logs
    dist_obs = 1000.0 / parallax.clip(lower=1e-3)

    bp_rp = df["apparent_BP"] - df["apparent_RP"]
    g_obs = pd.to_numeric(df["apparent_G"], errors="coerce")
    m_g_obs = g_obs - 5 * np.log10(dist_obs) + 5

    # Fit the single-star ridge from the synthetic singles on the SAME
    # observed CMD (i.e. with the same parallax/distance error), so the
    # comparison is like-for-like.
    singles = df[~df["is_binary"]]
    s_bp = (singles["apparent_BP"] - singles["apparent_RP"]).to_numpy()
    s_par = pd.to_numeric(singles["parallax_obs"], errors="coerce").clip(lower=1e-3)
    s_mg = (
        pd.to_numeric(singles["apparent_G"], errors="coerce")
        - 5 * np.log10(1000.0 / s_par)
        + 5
    ).to_numpy()
    ok = np.isfinite(s_bp) & np.isfinite(s_mg)
    if ok.sum() > 100:
        coef = np.polyfit(s_bp[ok], s_mg[ok], 3)
        m_g_ridge = np.polyval(coef, bp_rp.to_numpy())
    else:
        m_g_ridge = np.full(len(df), np.nanmedian(m_g_obs))

    return (m_g_ridge - m_g_obs.to_numpy())


def calibrate_threshold(single_scores: np.ndarray, fpr: float) -> float:
    """Threshold on the overluminosity score giving the requested FPR
    against the single-star (control) population."""
    return float(np.quantile(single_scores, 1.0 - fpr))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument("--fpr", type=float, default=0.01, help="Operating false-positive rate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results/tables")
    p.add_argument("--figdir", default="results/figures")
    args = p.parse_args()

    path = REPO_ROOT / args.synthetic
    if not path.exists():
        raise SystemExit(f"Synthetic catalogue not found at {path}. Run generate_synthetic.py.")
    print(f"Loading {path.name}...", flush=True)
    df = pd.read_parquet(path)
    print(f"  {len(df):,} systems ({int(df['is_binary'].sum()):,} binaries)", flush=True)

    # ---------------- Photometric channel ----------------
    print("\nPhotometric channel...", flush=True)
    df["phot_score"] = photometric_score(df)
    singles = df[~df["is_binary"]]
    thr = calibrate_threshold(singles["phot_score"].to_numpy(), args.fpr)
    df["phot_detected"] = df["phot_score"] > thr
    print(f"  threshold at FPR={args.fpr:.3f}: overluminosity > {thr:.4f} mag", flush=True)
    c_phot_all = df.loc[df["is_binary"], "phot_detected"].mean()
    print(f"  overall C_phot = {c_phot_all:.4f}", flush=True)

    # ---------------- Astrometric channel ----------------
    print("\nAstrometric channel...", flush=True)
    binaries = df["is_binary"].to_numpy()
    m1 = df["m1_msun"].to_numpy(dtype=float)
    q = df["q"].to_numpy(dtype=float)
    P = df["log_period_days"].to_numpy(dtype=float)
    d = df["distance_pc"].to_numpy(dtype=float)

    p_astro = np.zeros(len(df))
    if binaries.sum() > 0:
        p_astro[binaries] = astro_detection_probability(
            m1[binaries], q[binaries], P[binaries], d[binaries], seed=args.seed
        )
    rng = np.random.default_rng(args.seed)
    df["p_astro"] = p_astro
    df["astro_detected"] = (rng.random(len(df)) < p_astro) & binaries
    c_astro_all = df.loc[binaries, "astro_detected"].mean() if binaries.sum() else 0.0
    print(f"  overall C_astro = {c_astro_all:.4f}", flush=True)

    # ---------------- Four regimes ----------------
    print("\nAssigning four detection regimes...", flush=True)
    df["regime"] = assign_regimes(
        df["phot_detected"].to_numpy(), df["astro_detected"].to_numpy()
    )
    print(f"\n{'regime':>20s}  {'n':>9s}  {'fraction':>9s}")
    n_bin = int(binaries.sum())
    for code, name in REGIME_LABELS.items():
        n = int((df.loc[binaries, "regime"] == code).sum())
        frac = n / n_bin if n_bin else 0.0
        print(f"{name:>20s}  {n:>9,}  {frac:>9.4f}")

    # ---------------- Independence test ----------------
    print("\nIndependence test (PRD M22.3 — independence must NOT be assumed):", flush=True)
    indep = check_independence(
        df.loc[binaries, "phot_detected"].to_numpy(),
        df.loc[binaries, "astro_detected"].to_numpy(),
    )
    for k, v in indep.items():
        print(f"  {k}: {v}")

    # ---------------- Completeness grids ----------------
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print("\nComputing completeness grids...", flush=True)

    q_bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    p_bins = [(1, 10), (10, 100), (100, 1000), (1000, 3000)]

    grids = {}
    for name, spec, col in [
        ("c_phot_vs_q_M1", {"q": q_bins, "m1_msun": [(0.6, 0.9), (0.9, 1.2), (1.2, 1.4)]}, "phot_detected"),
        ("c_phot_vs_q_d", {"q": q_bins, "distance_pc": [(20, 60), (60, 100), (100, 150), (150, 200)]}, "phot_detected"),
        ("c_astro_vs_q_P", {"q": q_bins, "log_period_days": p_bins}, "astro_detected"),
        ("c_astro_vs_q_d", {"q": q_bins, "distance_pc": [(20, 60), (60, 100), (100, 150), (150, 200)]}, "astro_detected"),
    ]:
        sub = df[binaries]
        g = completeness_grid(sub, bin_spec=spec, detected_col=col)
        grids[name] = g
        out = outdir / f"{name}.csv"
        g.to_csv(out, index=False)
        print(f"  {name}: {len(g)} cells → {out.name}", flush=True)

    # ---------------- Complementarity map (q, P) ----------------
    print("\nBuilding (q, P) complementarity map...", flush=True)
    rows = []
    with tqdm(total=len(q_bins) * len(p_bins), desc="q-P cells", unit="cell") as bar:
        for q_lo, q_hi in q_bins:
            for p_lo, p_hi in p_bins:
                sel = (
                    binaries
                    & df["q"].between(q_lo, q_hi)
                    & df["log_period_days"].between(p_lo, p_hi)
                )
                n = int(sel.sum())
                bar.update(1)
                if n < 20:
                    continue
                lab = df.loc[sel, "regime"]
                row = {
                    "q_bin": (q_lo, q_hi),
                    "P_bin": (p_lo, p_hi),
                    "n": n,
                    "frac_phot_only": float((lab == 1).mean()),
                    "frac_astro_only": float((lab == 2).mean()),
                    "frac_both": float((lab == 3).mean()),
                    "frac_neither": float((lab == 0).mean()),
                    "C_phot": float(df.loc[sel, "phot_detected"].mean()),
                    "C_astro": float(df.loc[sel, "astro_detected"].mean()),
                }
                row["C_combined_empirical"] = float(
                    (df.loc[sel, "phot_detected"] | df.loc[sel, "astro_detected"]).mean()
                )
                row["C_combined_if_independent"] = (
                    1 - (1 - row["C_phot"]) * (1 - row["C_astro"])
                )
                rows.append(row)
    comp = pd.DataFrame(rows)
    comp_path = outdir / "complementarity_q_P.csv"
    comp.to_csv(comp_path, index=False)
    print(f"  {len(comp)} cells → {comp_path.name}", flush=True)
    print("\nComplementarity map (q × P):")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------------- GATE M22 ----------------
    print("\n" + "=" * 60, flush=True)
    print("GATE M22 — MAIN SCIENTIFIC GATE", flush=True)
    print("=" * 60, flush=True)
    phot_only = float((df.loc[binaries, "regime"] == 1).mean())
    astro_only = float((df.loc[binaries, "regime"] == 2).mean())
    both = float((df.loc[binaries, "regime"] == 3).mean())
    neither = float((df.loc[binaries, "regime"] == 0).mean())

    # The gate: each channel must recover a non-trivial population that
    # the other misses, i.e. complementary regimes exist.
    complementary_fraction = phot_only + astro_only
    gate_passed = phot_only > 0.02 and astro_only > 0.02
    print(f"  photometric-only recovered : {phot_only:.4f}", flush=True)
    print(f"  astrometric-only recovered : {astro_only:.4f}", flush=True)
    print(f"  both channels             : {both:.4f}", flush=True)
    print(f"  neither (blind spot)      : {neither:.4f}", flush=True)
    print(f"  complementary fraction    : {complementary_fraction:.4f}", flush=True)
    print(f"\n  GATE M22: {'PASS' if gate_passed else 'FAIL'} "
          f"— distinct complementary regimes "
          f"{'demonstrated' if gate_passed else 'NOT demonstrated'}", flush=True)

    # ---------------- Figures ----------------
    figdir = REPO_ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(comp) > 0:
            piv = comp.pivot_table(index="q_bin", columns="P_bin", values="frac_phot_only")
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            for ax, (col, title) in zip(
                axes.ravel(),
                [
                    ("frac_phot_only", "Photometric only"),
                    ("frac_astro_only", "Astrometric only"),
                    ("frac_both", "Both"),
                    ("frac_neither", "Neither (blind spot)"),
                ],
            ):
                pv = comp.pivot_table(index="q_bin", columns="P_bin", values=col)
                im = ax.imshow(pv.values, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
                ax.set_title(title)
                ax.set_xlabel("Period [days]")
                ax.set_ylabel("q bin")
                ax.set_xticks(range(len(pv.columns)))
                ax.set_xticklabels([f"{a}-{b}" for a, b in pv.columns], rotation=45, fontsize=7)
                ax.set_yticks(range(len(pv.index)))
                ax.set_yticklabels([f"{a}-{b}" for a, b in pv.index], fontsize=7)
                plt.colorbar(im, ax=ax)
            fig.suptitle("Four-regime photometric–astrometric complementarity (q × P)")
            fig.tight_layout()
            fig.savefig(figdir / "fig11_four_regions.png", dpi=150)
            plt.close(fig)
            print(f"\n  wrote fig11_four_regions.png", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (figure generation skipped: {exc})", flush=True)

    write_run_metadata(
        REPO_ROOT / "results" / "runs" / "complementarity",
        config=vars(args),
        metrics={
            "C_phot": float(c_phot_all),
            "C_astro": float(c_astro_all),
            "photometric_only": phot_only,
            "astrometric_only": astro_only,
            "both": both,
            "neither": neither,
            "independence_test": indep,
            "gate_m22_passed": bool(gate_passed),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
