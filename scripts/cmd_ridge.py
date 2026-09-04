"""M12 — classical CMD ridge baseline on real data (TODO M12, E2).

The simplest possible binary detector: a polynomial fit of M_G as a
function of (BP-RP) on the apparently-single sample, then flag sources
that sit above the ridge (brighter than expected). We compare this to
the ML / PI-NN baselines and report the ridge residual statistics.

Run via:
    PYTHONPATH=src python scripts/cmd_ridge.py
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

from stellar_binary_selection.evaluation.metrics import (  # noqa: E402
    binary_classification_metrics,
)
from stellar_binary_selection.utils import write_run_metadata  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent-derived",
        default="data/processed/gaia_dr3_2mass_derived.parquet",
    )
    p.add_argument(
        "--sample-b",
        default="data/processed/sample_b_apparently_single.parquet",
    )
    p.add_argument(
        "--sample-c",
        default="data/processed/sample_c_nss.parquet",
    )
    p.add_argument("--fpr", type=float, default=0.01)
    p.add_argument("--out", default="results/tables/cmd_ridge.json")
    p.add_argument("--figdir", default="results/figures")
    args = p.parse_args()

    # ---------------- Load ----------------
    print("Loading real Gaia+2MASS data...", flush=True)
    parent = pd.read_parquet(REPO_ROOT / args.parent_derived)
    sample_b = pd.read_parquet(REPO_ROOT / args.sample_b)
    nss_ids = pd.read_parquet(REPO_ROOT / args.sample_c)["source_id"].drop_duplicates().to_numpy()
    print(f"  parent: {len(parent):,}; Sample B: {len(sample_b):,}; NSS in parent: {len(nss_ids):,}", flush=True)

    # ---------------- Fit ridge ----------------
    print("\nFitting M_G(BP-RP) ridge on Sample B...", flush=True)
    bp = pd.to_numeric(sample_b["bp_rp"], errors="coerce").to_numpy()
    par = pd.to_numeric(sample_b["parallax"], errors="coerce").clip(lower=1e-3).to_numpy()
    g = pd.to_numeric(sample_b["phot_g_mean_mag"], errors="coerce").to_numpy()
    m_g = g - 5 * np.log10(1000.0 / par) + 5
    ok = np.isfinite(bp) & np.isfinite(m_g)
    coef = np.polyfit(bp[ok], m_g[ok], 3)
    pred = np.polyval(coef, bp)
    resid = m_g - pred
    print(
        f"  ridge M_G(BP-RP) = {coef[3]:.3f} + {coef[2]:.3f}*x + {coef[1]:.3f}*x^2 + {coef[0]:.3f}*x^3"
    )
    print(
        f"  Sample B residual scatter (std): {resid.std():.4f} mag"
    )
    print(
        f"  Sample B residual MAD×1.4826     : {1.4826*np.median(np.abs(resid-np.median(resid))):.4f} mag"
    )

    # ---------------- Apply ridge to entire parent ----------------
    print("\nApplying ridge to the full parent catalogue...", flush=True)
    par_p = pd.to_numeric(parent["parallax"], errors="coerce").clip(lower=1e-3).to_numpy()
    g_p = pd.to_numeric(parent["phot_g_mean_mag"], errors="coerce").to_numpy()
    bp_p = pd.to_numeric(parent["bp_rp"], errors="coerce").to_numpy()
    m_g_p = g_p - 5 * np.log10(1000.0 / par_p) + 5
    ridge_mg = np.polyval(coef, bp_p)
    overlum = ridge_mg - m_g_p  # positive = brighter than single-star ridge

    # Calibration threshold: the (1 - fpr) quantile of overlum on the
    # whole parent sample (mostly single stars in this realistic setting).
    thr = float(np.quantile(overlum, 1.0 - args.fpr))
    print(
        f"  threshold at FPR={args.fpr:.3f}: overluminosity > {thr:.4f} mag"
    )

    # ---------------- Score the NSS subset vs the apparently-single ----------------
    is_nss = parent["source_id"].isin(nss_ids).to_numpy()
    n_nss = int(is_nss.sum())
    n_singles_proxy = int((~is_nss).sum())
    print(f"  NSS sources in parent: {n_nss:,}; other: {n_singles_proxy:,}", flush=True)

    p_overlum_nss = overlum[is_nss]
    p_overlum_ctl = overlum[~is_nss]
    fpr_real = float((p_overlum_ctl > thr).mean())
    tpr_real = float((p_overlum_nss > thr).mean())
    print(
        f"  on real data: FPR={fpr_real:.4f}  TPR(recall)={tpr_real:.4f}  (target FPR {args.fpr:.3f})"
    )

    # Use the parent as the binary classifier test bed. Convert the
    # overluminosity (mag) to a probability-like score in [0, 1] via
    # z-scoring then logistic; brier_score_loss refuses values outside
    # [0, 1].
    y_true = is_nss.astype(int)
    if n_nss > 10 and n_singles_proxy > 10:
        # Use the Sample-B residual scatter as the natural scale.
        sigma = float(resid.std())
        z = overlum / max(sigma, 1e-3)
        y_score = 1.0 / (1.0 + np.exp(-z))
        cls = binary_classification_metrics(y_true, y_score)
    else:
        cls = {}
    print(
        f"  parent Sample C vs other: PR-AUC={cls.get('pr_auc', float('nan')):.4f}  "
        f"ROC-AUC={cls.get('roc_auc', float('nan')):.4f}"
    )

    # ---------------- Figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 9))
        sel = np.isfinite(bp_p) & np.isfinite(m_g_p) & np.isfinite(par_p)
        n_other = int((~is_nss & sel).sum())
        n_nss_vis = int((is_nss & sel).sum())
        idx_other = np.where(~is_nss & sel)[0]
        if len(idx_other) > 50_000:
            idx_other = np.random.default_rng(0).choice(idx_other, 50_000, replace=False)
        ax.scatter(
            bp_p[idx_other], m_g_p[idx_other], s=1, c="0.6", alpha=0.4,
            label=f"other ({len(idx_other):,})",
        )
        if n_nss_vis > 0:
            ax.scatter(
                bp_p[is_nss & sel], m_g_p[is_nss & sel], s=6, c="C3", alpha=0.7,
                label=f"Sample C NSS ({n_nss_vis:,})",
            )
        bx = np.linspace(0.4, 1.6, 200)
        ax.plot(bx, np.polyval(coef, bx), "k--", lw=1.5,
                label=f"ridge: M_G = poly3(BP-RP)")
        # Mark a few overluminosity contours
        for ov in (0.1, 0.3, 0.6):
            ax.plot(bx, np.polyval(coef, bx) - ov, ":", color="C2",
                    label=f"overluminosity = {ov} mag" if ov == 0.1 else None)
        ax.set_xlabel("BP - RP [mag]")
        ax.set_ylabel("M_G (distance from Sample B ridge) [mag]")
        ax.set_title("M12: Classical CMD ridge on real Gaia+2MASS")
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        out_fig = REPO_ROOT / args.figdir / "fig2_cmd_ridge.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"\nWrote {out_fig.relative_to(REPO_ROOT)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (figure skipped: {exc})", flush=True)

    # ---------------- Save ----------------
    report = {
        "ridge_coef": [float(c) for c in coef],
        "ridge_scatter_sampleB_mag": float(resid.std()),
        "ridge_MAD_sampleB_mag": float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
        "fpr_target": args.fpr,
        "threshold_overlum_mag": thr,
        "fpr_real_other": fpr_real,
        "tpr_real_nss": tpr_real,
        "pr_auc_parent": float(cls.get("pr_auc", float("nan"))),
        "roc_auc_parent": float(cls.get("roc_auc", float("nan"))),
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out.relative_to(REPO_ROOT)}", flush=True)

    write_run_metadata(
        REPO_ROOT / "results" / "runs" / "cmd_ridge",
        config=vars(args),
        metrics=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())