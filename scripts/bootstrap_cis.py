"""Statistical analysis & bootstrap confidence intervals (TODO M26).

Per PRD M26:
 - Bootstrap the simulation-level quantities (C_phot, C_astro, the four
   detection regimes, and per-cell completeness) by resampling the 100k
   synthetic systems with replacement.
 - Report the physics-ablation PR-AUC / q-MAE as mean ± seed-std with a
   normal-approximation 95% confidence interval from the 3 seeds.
 - (Optional) bootstrap the classifier PR-AUC / q-MAE from persisted test
   predictions if present (see --preds-dir from train.py --save-preds).

Run via:
    PYTHONPATH=src python scripts/bootstrap_cis.py
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

try:
    from stellar_binary_selection.simulation.population import load_synthetic  # noqa: E402
except Exception:  # noqa: BLE001
    import pandas as pd

    def load_synthetic(path):  # type: ignore
        return pd.read_parquet(path)

from stellar_binary_selection.selection.astrometric import (  # noqa: E402
    astro_detection_probability,
)


def photometric_score(df: pd.DataFrame) -> np.ndarray:
    """Overluminosity above the single-star ridge (mirrors complementarity.py)."""
    parallax = pd.to_numeric(df["parallax_obs"], errors="coerce")
    dist_obs = 1000.0 / parallax.clip(lower=1e-3)
    bp_rp = df["apparent_BP"] - df["apparent_RP"]
    g_obs = pd.to_numeric(df["apparent_G"], errors="coerce")
    m_g_obs = g_obs - 5 * np.log10(dist_obs) + 5
    singles = df[~df["is_binary"]]
    s_bp = (singles["apparent_BP"] - singles["apparent_RP"]).to_numpy()
    s_par = pd.to_numeric(singles["parallax_obs"], errors="coerce").clip(lower=1e-3)
    s_mg = (
        pd.to_numeric(singles["apparent_G"], errors="coerce")
        - 5 * np.log10(1000.0 / s_par) + 5
    ).to_numpy()
    ok = np.isfinite(s_bp) & np.isfinite(s_mg)
    if ok.sum() > 100:
        coef = np.polyfit(s_bp[ok], s_mg[ok], 3)
        m_g_ridge = np.polyval(coef, bp_rp.to_numpy())
    else:
        m_g_ridge = np.full(len(df), np.nanmedian(m_g_obs))
    return (m_g_ridge - m_g_obs.to_numpy())


def bootstrap_simulation(n_boot: int = 200, rng_seed: int = 42) -> dict:
    """Resample the synthetic population and recompute C_phot / C_astro /
    the four regimes, returning 95% CIs."""
    rng = np.random.default_rng(rng_seed)
    df = load_synthetic(REPO_ROOT / "data" / "synthetic" / "synthetic_v1.parquet")

    is_bin = df["is_binary"].to_numpy(bool)
    n = len(df)

    # ---- Photometric channel (mirror complementarity.py) ----
    score = photometric_score(df)
    singles = df[~df["is_binary"]]
    thr_phot = np.quantile(score[~is_bin], 0.99)  # 1% FPR calibration
    phot_det_full = score > thr_phot

    # ---- Astrometric channel (mirror complementarity.py) ----
    m1 = df["m1_msun"].to_numpy(dtype=float)
    q = df["q"].to_numpy(dtype=float)
    P = df["log_period_days"].to_numpy(dtype=float)
    d = df["distance_pc"].to_numpy(dtype=float)
    p_astro = np.zeros(n)
    if is_bin.sum() > 0:
        p_astro[is_bin] = astro_detection_probability(
            m1[is_bin], q[is_bin], P[is_bin], d[is_bin], seed=42
        )

    # Bootstrap
    N = len(df)
    boot = {"c_phot": [], "c_astro": [], "phot_only": [], "astro_only": [], "both": [], "neither": []}
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        b = is_bin[idx]
        pd_ = phot_det_full[idx]
        ad_ = rng.random(N) < p_astro[idx]
        # All four regimes are defined over binaries, matching
        # complementarity.py. Singles must not enter the numerators.
        po = int((b & pd_ & ~ad_).sum())
        ao = int((b & ad_ & ~pd_).sum())
        bo = int((b & pd_ & ad_).sum())
        ne = int((b & ~pd_ & ~ad_).sum())
        nb = max(int(b.sum()), 1)
        boot["c_phot"].append(pd_[b].mean())
        boot["c_astro"].append(ad_[b].mean())
        boot["phot_only"].append(po / nb)
        boot["astro_only"].append(ao / nb)
        boot["both"].append(bo / nb)
        boot["neither"].append(ne / nb)

    summary = {}
    for k, arr in boot.items():
        arr = np.asarray(arr)
        summary[k] = {
            "point": float(arr.mean()),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "std": float(arr.std()),
        }
    return summary


def ablation_cis(csv_path: Path) -> pd.DataFrame:
    """Seed-level mean ± std and normal-approx 95% CI for the ablation."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        n = int(r["n_seeds"])
        se = r["pr_auc_std"] / sqrt_safe(n)
        rows.append({
            "model": r["model"],
            "pr_auc_mean": r["pr_auc"],
            "pr_auc_std": r["pr_auc_std"],
            "pr_auc_ci_lo": r["pr_auc"] - 1.96 * se,
            "pr_auc_ci_hi": r["pr_auc"] + 1.96 * se,
            "q_mae_mean": r["q_mae"],
            "q_mae_std": r["q_mae_std"],
        })
    return pd.DataFrame(rows)


def sqrt_safe(n):
    import math
    return math.sqrt(max(n, 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--outdir", default="results/tables/summary")
    args = p.parse_args()
    out = REPO_ROOT / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    print("Bootstrapping simulation-level quantities...", flush=True)
    sim = bootstrap_simulation(args.n_boot)
    sim_df = pd.DataFrame(
        [(k, v["point"], v["ci_lo"], v["ci_hi"], v["std"]) for k, v in sim.items()],
        columns=["quantity", "point", "ci_2.5%", "ci_97.5%", "boot_std"],
    )
    print(sim_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sim_df.to_csv(out / "bootstrap_simulation.csv", index=False)

    print("\nAblation seed-level CIs...", flush=True)
    abl = ablation_cis(REPO_ROOT / "results" / "tables" / "e6_ablation.csv")
    print(abl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    abl.to_csv(out / "ablation_cis.csv", index=False)

    # Combined statistical summary for the manuscript
    report = {
        "bootstrap_n": args.n_boot,
        "C_phot": sim["c_phot"],
        "C_astro": sim["c_astro"],
        "regimes": {k: sim[k] for k in ("phot_only", "astro_only", "both", "neither")},
    }
    (out / "statistical_summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}/bootstrap_simulation.csv, ablation_cis.csv, statistical_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
