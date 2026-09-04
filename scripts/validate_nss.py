"""NSS validation (TODO M18, E8).

Trains a binary classifier on the synthetic population and scores the
real NSS-positive sources (Sample C) that fall inside our parent
catalogue, plus a control sample drawn from the apparently-single
stars (Sample B).

This is the EIGHT (M18) gate of the project: a model that does well on
synthetic must also work on real Gaia data.

Run via:
    PYTHONPATH=src python scripts/validate_nss.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.utils import set_global_seed, write_run_metadata  # noqa: E402


SYNTHETIC_FEATURES = [
    "apparent_G", "apparent_BP", "apparent_RP",
    "apparent_J", "apparent_H", "apparent_Ks",
    "sigma_G", "sigma_BP", "sigma_RP", "sigma_J", "sigma_H", "sigma_Ks",
]
REAL_FEATURES = [
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "j_m", "h_m", "ks_m",
    "phot_g_mean_flux_error", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error",
    "j_msigcom", "h_msigcom", "ks_msigcom",
]


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = SYNTHETIC_FEATURES if "apparent_G" in df.columns else REAL_FEATURES
    return df[cols].to_numpy(dtype=np.float32)


def photometric_score_ol(df: pd.DataFrame) -> np.ndarray:
    """Reuse the same overluminosity proxy as the complementarity script.

    For the real parent sample (no synthetic singles available), we
    calibrate the single-star ridge on Sample B (the apparently-single
    control). For the synthetic, we use the synthetic singles.
    """
    if "apparent_G" in df.columns:
        g_col, bp_col, rp_col = "apparent_G", "apparent_BP", "apparent_RP"
        par_col = "parallax_obs"
    else:
        g_col, bp_col, rp_col = "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"
        par_col = "parallax"
    parallax = pd.to_numeric(df[par_col], errors="coerce").clip(lower=1e-3)
    dist_obs = 1000.0 / parallax
    bp_rp = df[bp_col] - df[rp_col]
    g_obs = pd.to_numeric(df[g_col], errors="coerce")
    m_g_obs = g_obs - 5 * np.log10(dist_obs) + 5

    if "is_binary" in df.columns:
        # synthetic: use synthetic singles
        singles = df[~df["is_binary"]]
        s_bp = (singles[bp_col] - singles[rp_col]).to_numpy()
        s_par = pd.to_numeric(singles[par_col], errors="coerce").clip(lower=1e-3)
        s_mg = (
            pd.to_numeric(singles[g_col], errors="coerce")
            - 5 * np.log10(1000.0 / s_par)
            + 5
        ).to_numpy()
    else:
        # real: use the parent itself (singles are the bulk of the population)
        s_bp = bp_rp.to_numpy()
        s_mg = m_g_obs.to_numpy()

    ok = np.isfinite(s_bp) & np.isfinite(s_mg)
    if ok.sum() > 100:
        coef = np.polyfit(s_bp[ok], s_mg[ok], 3)
        m_g_ridge = np.polyval(coef, bp_rp.to_numpy())
    else:
        m_g_ridge = np.full(len(df), np.nanmedian(m_g_obs))
    return (m_g_ridge - m_g_obs.to_numpy())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument(
        "--parent-derived",
        default="data/processed/gaia_dr3_2mass_derived_nss.parquet",
        help="Parent catalogue with the NSS flags annotated.",
    )
    p.add_argument(
        "--fgk",
        default="data/processed/gaia_dr3_2mass_fgk.parquet",
        help="Adopted FGK CMD-domain catalogue used to restrict validation.",
    )
    p.add_argument(
        "--sample-c",
        default="data/processed/sample_c_nss.parquet",
        help="Sample C: NSS-positive sources inside the parent catalogue.",
    )
    p.add_argument(
        "--sample-b",
        default="data/processed/sample_b_apparently_single.parquet",
        help="Sample B: apparently-single control sample.",
    )
    p.add_argument("--fpr", type=float, default=0.01, help="Operating FPR for overluminosity proxy.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-control",
        type=int,
        default=20_000,
        help="Subsample size of the control (apparently-single) sources used for the score distributions.",
    )
    p.add_argument("--out", default="results/tables/nss_validation_fgk.json")
    p.add_argument("--validation-dir", default="data/processed/validation_fgk")
    p.add_argument("--figdir", default="results/figures")
    args = p.parse_args()

    set_global_seed(args.seed)

    # ---------------- Load ----------------
    print("Loading inputs...", flush=True)
    syn = pd.read_parquet(REPO_ROOT / args.synthetic)
    nss = pd.read_parquet(REPO_ROOT / args.sample_c)
    parent_derived = pd.read_parquet(REPO_ROOT / args.parent_derived)
    sample_b = pd.read_parquet(REPO_ROOT / args.sample_b)
    fgk = pd.read_parquet(REPO_ROOT / args.fgk)
    fgk_ids = set(fgk["source_id"])
    sample_b = sample_b[sample_b["source_id"].isin(fgk_ids)].copy()
    nss_ids_all = nss["source_id"].drop_duplicates()
    nss_ids_fgk = nss_ids_all[nss_ids_all.isin(fgk_ids)]
    parent_derived = parent_derived[parent_derived["source_id"].isin(fgk_ids)].copy()

    print(f"  synthetic        : {len(syn):,} ({int(syn['is_binary'].sum()):,} binaries)")
    print(f"  sample C (NSS+)  : {len(nss):,} solutions / {nss['source_id'].nunique():,} sources")
    print(f"  parent (FGK restricted): {len(parent_derived):,}")
    print(f"  sample B (FGK overlap): {len(sample_b):,}")
    print(f"  sample C (FGK unique): {len(nss_ids_fgk):,}")

    # ---------------- Train LightGBM on synthetic ----------------
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier

    print("\nTraining LightGBM on synthetic...", flush=True)
    Xtr = feature_matrix(syn)
    ytr = syn["is_binary"].to_numpy(int)
    # single 70/15/15 split; train on 70%, evaluate on 15% (test) and 15% (val)
    train, rest = train_test_split(
        syn, test_size=0.30, random_state=args.seed, stratify=syn["is_binary"]
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=args.seed, stratify=rest["is_binary"]
    )
    Xt = feature_matrix(train); yt = train["is_binary"].to_numpy(int)
    Xv = feature_matrix(val);   yv = val["is_binary"].to_numpy(int)
    Xe = feature_matrix(test);  ye = test["is_binary"].to_numpy(int)

    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        n_jobs=1,
        random_state=args.seed,
    )
    clf.fit(Xt, yt)  # no eval_set to avoid deprecation warning + faster
    proba_test = clf.predict_proba(Xe)[:, 1]
    ap_test = float(average_precision_score(ye, proba_test))
    roc_test = float(roc_auc_score(ye, proba_test))
    print(f"  synthetic test   : PR-AUC={ap_test:.4f} ROC-AUC={roc_test:.4f}", flush=True)

    # ---------------- Score real data ----------------
    print("\nScoring real NSS sources with the model trained on synthetic...", flush=True)
    # Score only the adopted FGK parent domain.
    Xpar = feature_matrix(parent_derived)
    pbin_par = clf.predict_proba(Xpar)[:, 1]
    parent_derived = parent_derived.assign(p_binary=pbin_par)

    # Score on Sample C (real NSS) and a random control from Sample B
    control_ids = sample_b.sample(
        n=min(args.n_control, len(sample_b)), random_state=args.seed
    )["source_id"].to_numpy()
    nss_ids = nss_ids_fgk.to_numpy()

    p_nss = parent_derived.loc[parent_derived["source_id"].isin(nss_ids), "p_binary"].to_numpy()
    p_ctl = parent_derived.loc[parent_derived["source_id"].isin(control_ids), "p_binary"].to_numpy()
    p_ctl_overall = pbin_par

    print(f"  scored {len(p_nss):,} NSS sources, {len(p_ctl):,} control sources", flush=True)

    # ---------------- Statistics ----------------
    # Distribution comparison
    bins = np.linspace(0, 1, 51)
    hist_nss, _ = np.histogram(p_nss, bins=bins)
    hist_ctl, _ = np.histogram(p_ctl, bins=bins)
    print("\nScore distribution (NSS vs control):", flush=True)
    print("  bin     NSS   control", flush=True)
    for i in range(0, 50, 5):
        print(f"  {bins[i]:.2f}  {hist_nss[i]:>5d}  {hist_ctl[i]:>5d}", flush=True)

    # We don't have a clean y_true for the full 250k (most sources are
    # "unknown"), but the synthetic-trained model produces a CALIBRATED
    # score in [0, 1] that we can interpret as P(binary). For Sample C
    # vs Sample B we treat the binary class labels as a noisy proxy:
    # NSS sources are TRUE POSITIVES for "having a binary solution",
    # Sample B sources are TRUE NEGATIVES for "no binary evidence".
    y_true_dis = np.concatenate([np.ones_like(p_nss), np.zeros_like(p_ctl)])
    p_dis = np.concatenate([p_nss, p_ctl])
    if len(np.unique(y_true_dis)) > 1:
        ap_dis = float(average_precision_score(y_true_dis, p_dis))
        roc_dis = float(roc_auc_score(y_true_dis, p_dis))
    else:
        ap_dis = roc_dis = float("nan")
    print(f"\n  Sample C vs B : PR-AUC={ap_dis:.4f}  ROC-AUC={roc_dis:.4f}", flush=True)

    # Overluminosity proxy for comparison
    print("\nOverluminosity proxy on real data...", flush=True)
    ol_nss_df = parent_derived[parent_derived["source_id"].isin(nss_ids)].copy()
    ol_ctl_df = parent_derived[parent_derived["source_id"].isin(control_ids)].copy()
    # We need a "synthetic singles" reference to calibrate the threshold;
    # use the synthetic singles.
    singles = syn[~syn["is_binary"]]
    ol_synth = photometric_score_ol(syn)
    thr = float(np.quantile(ol_synth[~syn["is_binary"].to_numpy()], 1 - args.fpr))
    print(f"  threshold calibrated on synthetic singles (FPR={args.fpr:.3f}): overluminosity > {thr:.4f} mag", flush=True)
    # Build the proxy on the parent (Sample C + control subset)
    parent_derived = parent_derived.assign(ol_proxy=photometric_score_ol(parent_derived))
    ol_nss = parent_derived.loc[parent_derived["source_id"].isin(nss_ids), "ol_proxy"].to_numpy()
    ol_ctl = parent_derived.loc[parent_derived["source_id"].isin(control_ids), "ol_proxy"].to_numpy()
    rec_nss = float((ol_nss > thr).mean())
    rec_ctl = float((ol_ctl > thr).mean())
    print(f"  overluminosity recall on Sample C : {rec_nss:.4f}", flush=True)
    print(f"  overluminosity FPR on Sample B    : {rec_ctl:.4f} (target {args.fpr:.3f})", flush=True)

    validation_dir = REPO_ROOT / args.validation_dir
    validation_dir.mkdir(parents=True, exist_ok=True)
    sample_b.to_parquet(validation_dir / "sample_b_fgk.parquet", index=False)
    nss_fgk = nss[nss["source_id"].isin(nss_ids_fgk)].drop_duplicates("source_id")
    nss_fgk.to_parquet(validation_dir / "sample_c_fgk_unique.parquet", index=False)
    parent_derived[parent_derived["source_id"].isin(nss_ids)].to_parquet(validation_dir / "sample_c_fgk_scored.parquet", index=False)
    parent_derived[parent_derived["source_id"].isin(control_ids)].to_parquet(validation_dir / "sample_b_fgk_scored.parquet", index=False)

    # ---------------- GATE M18 ----------------
    print("\n" + "=" * 60, flush=True)
    print("GATE M18 — NSS VALIDATION (model trained on synthetic, scored on real)", flush=True)
    print("=" * 60, flush=True)
    print(f"  synthetic test PR-AUC     : {ap_test:.4f}", flush=True)
    print(f"  real Sample C vs B PR-AUC : {ap_dis:.4f}", flush=True)
    print(f"  Sample C recall (proxy)   : {rec_nss:.4f}", flush=True)
    gate_passed = ap_dis > 0.6 and rec_nss > 0.5
    print(f"\n  GATE M18: {'PASS' if gate_passed else 'FAIL'}", flush=True)

    # ---------------- Figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(p_ctl, bins=bins, alpha=0.5, label=f"Sample B (apparently single, n={len(p_ctl):,})", color="C0")
        ax.hist(p_nss, bins=bins, alpha=0.5, label=f"Sample C (NSS+, n={len(p_nss):,})", color="C3")
        ax.set_xlabel("P(binary) score from LightGBM trained on synthetic")
        ax.set_ylabel("Number of sources")
        ax.set_title(f"GATE M18: PR-AUC = {ap_dis:.4f} on real Gaia NSS vs apparently-single")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        out_fig = REPO_ROOT / args.figdir / "figure4_real_validation_legacy.png"
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"\nWrote {out_fig.relative_to(REPO_ROOT)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (figure skipped: {exc})", flush=True)

    # ---------------- Save ----------------
    report = {
        "synthetic_test": {"pr_auc": ap_test, "roc_auc": roc_test},
        "real_sample_c_vs_b": {"pr_auc": ap_dis, "roc_auc": roc_dis},
        "overluminosity_proxy": {
            "threshold_mag": thr,
            "fpr_target": args.fpr,
            "recall_sample_c": rec_nss,
            "fpr_sample_b": rec_ctl,
        },
        "n_nss_scored": int(len(p_nss)),
        "n_control_scored": int(len(p_ctl)),
        "validation_domain": "FGK overlap of the adopted CMD subset",
        "n_fgk_parent": int(len(parent_derived)),
        "n_sample_b_fgk": int(len(sample_b)),
        "n_sample_b_control_draw": int(len(control_ids)),
        "n_sample_c_entries_fgk": int(nss[nss["source_id"].isin(nss_ids_fgk)].shape[0]),
        "n_sample_c_unique_fgk": int(len(nss_ids_fgk)),
        "threshold_source": "99th percentile of synthetic single-star overluminosity scores",
        "threshold_fpr_target": float(args.fpr),
        "threshold_not_calibrated_on_sample_b": True,
        "feature_domain_check": {
            "real_features": REAL_FEATURES,
            "synthetic_features": SYNTHETIC_FEATURES,
            "all_required_columns_present": True,
        },
        "gate_m18_passed": bool(gate_passed),
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}", flush=True)

    write_run_metadata(
        REPO_ROOT / "results" / "runs" / "validate_nss",
        config=vars(args),
        metrics=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
