"""Synthetic controlled evaluation (M17) and robustness (M23).

M17 — q-recovery and stratified PR-AUC vs q / M1 / distance.
M23 — robustness across distance, mass, metallicity, and model choice.

Run via:
    PYTHONPATH=src python scripts/eval_m17_m23.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURES = [
    "apparent_G", "apparent_BP", "apparent_RP",
    "apparent_J", "apparent_H", "apparent_Ks",
    "sigma_G", "sigma_BP", "sigma_RP", "sigma_J", "sigma_H", "sigma_Ks",
]


def load():
    df = pd.read_parquet(REPO_ROOT / "data" / "synthetic" / "synthetic_v1.parquet")
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["is_binary"].to_numpy(int)
    return df, X, y


def models():
    return {
        "RF": RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=42),
        "XGB": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             n_jobs=2, random_state=42, verbosity=0),
        "LGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               n_jobs=2, random_state=42, verbose=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=80,
                             random_state=42, early_stopping=True),
    }


def pr_auc_per_bin(mask_test, y_test, proba):
    if mask_test.sum() < 20:
        return float("nan")
    return float(average_precision_score(y_test[mask_test], proba[mask_test]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figdir", default="results/figures")
    p.add_argument("--outdir", default="results/tables/summary")
    args = p.parse_args()
    figdir = REPO_ROOT / args.figdir
    outdir = REPO_ROOT / args.outdir
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    df, X, y = load()
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
    )
    dte = df.iloc[idx_te].reset_index(drop=True)
    qte = dte["q"].to_numpy()
    m1te = dte["m1_msun"].to_numpy()
    dte_d = dte["distance_pc"].to_numpy()
    fehte = dte["feh"].to_numpy()
    bte = yte.astype(bool)

    # ===================== M17: q-recovery + stratified PR-AUC =====================
    print("\n[M17] q-recovery (MLPRegressor on binaries)...", flush=True)
    bin_tr = ytr.astype(bool)
    reg = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=80, random_state=42,
                      early_stopping=True)
    reg.fit(Xtr[bin_tr], df.iloc[idx_tr].reset_index(drop=True)["q"].to_numpy()[bin_tr])
    q_pred = reg.predict(Xte[bte])
    q_true = qte[bte]
    q_mae = float(np.mean(np.abs(q_pred - q_true)))
    q_bias = float(np.median(q_pred - q_true))
    print(f"  q-MAE={q_mae:.4f}  median-bias={q_bias:.4f}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].scatter(q_true, q_pred, s=3, alpha=0.2, color="C3")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel(r"$q_{\rm true}$"); ax[0].set_ylabel(r"$q_{\rm pred}$")
    ax[0].set_title(f"M17a: q recovery (MAE={q_mae:.3f})")
    res = q_pred - q_true
    ax[1].scatter(q_true, res, s=3, alpha=0.2, color="C1")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel(r"$q_{\rm true}$"); ax[1].set_ylabel(r"$\Delta q$")
    ax[1].set_title("M17b: q residual vs q")
    ax[2].hist(res, bins=50, color="C2")
    ax[2].set_xlabel(r"$\Delta q$"); ax[2].set_ylabel("count")
    ax[2].set_title("M17c: q residual distribution")
    fig.tight_layout(); fig.savefig(figdir / "fig17_q_recovery.png", dpi=150); plt.close(fig)

    # stratified PR-AUC (use LGBM as the reference classifier for M17/M23)
    print("[M17] training LGBM for stratified PR-AUC...", flush=True)
    clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        n_jobs=2, random_state=42, verbose=-1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    prauc = float(average_precision_score(yte, proba))
    prauc_roc = float(roc_auc_score(yte, proba))
    print(f"  LGBM PR-AUC={prauc:.4f} ROC-AUC={prauc_roc:.4f}", flush=True)

    def strat_all(col, edges):
        """PR-AUC for ALL test systems (both classes) in a property slice."""
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (col >= lo) & (col < hi)
            bins.append((f"[{lo},{hi}]", pr_auc_per_bin(mask, yte, proba)))
        return bins

    def strat_q(edges):
        """PR-AUC for binaries in a q-bin vs an equal-size random single sample
        (balanced so the metric is interpretable around 0.5)."""
        rng23 = np.random.default_rng(0)
        single_idx = np.where(~bte)[0]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            pos = np.where((qte >= lo) & (qte < hi) & bte)[0]
            if len(pos) < 20:
                bins.append((f"[{lo},{hi}]", float("nan")))
                continue
            neg = rng23.choice(single_idx, size=len(pos), replace=False)
            sel = np.concatenate([pos, neg])
            bins.append((f"[{lo},{hi}]",
                         float(average_precision_score(yte[sel], proba[sel]))))
        return bins

    q_bins = strat_q(np.linspace(0.1, 1.0, 10))
    m1_bins = strat_all(m1te, [0.6, 0.9, 1.2, 1.4])
    d_bins = strat_all(dte_d, [20, 60, 100, 150, 200])
    z_bins = strat_all(fehte, [-1.0, -0.3, 0.0, 0.5])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for axx, bins, title in [
        (ax[0], q_bins, "PR-AUC vs q"),
        (ax[1], m1_bins, "PR-AUC vs M1"),
        (ax[2], d_bins, "PR-AUC vs distance"),
    ]:
        xs = np.arange(len(bins)); vals = [v for _, v in bins]
        axx.bar(xs, vals, color="C0")
        axx.axhline(prauc, color="k", ls="--", lw=0.8, label=f"overall {prauc:.3f}")
        axx.set_xticks(xs); axx.set_xticklabels([b for b, _ in bins], rotation=45, fontsize=8)
        axx.set_ylim(0, 1); axx.set_title(f"M17: {title}"); axx.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "fig17_stratified_prauc.png", dpi=150); plt.close(fig)

    m17_tbl = pd.DataFrame({
        "slice": [f"q {b}" for b, _ in q_bins] + [f"M1 {b}" for b, _ in m1_bins]
                 + [f"d {b}" for b, _ in d_bins] + [f"feh {b}" for b, _ in z_bins],
        "pr_auc": [v for _, v in q_bins] + [v for _, v in m1_bins]
                 + [v for _, v in d_bins] + [v for _, v in z_bins],
    })
    m17_tbl.to_csv(outdir / "m17_stratified_prauc.csv", index=False)

    # ===================== M23: robustness across slices & models =====================
    print("\n[M23] robustness: per-model PR-AUC in distance/mass/metallicity slices...", flush=True)
    rows = []
    for name, m in models().items():
        m.fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        overall = float(average_precision_score(yte, pr))
        for label, bins in [("distance", [(20,60),(60,100),(100,150),(150,200)]),
                            ("mass", [(0.6,0.9),(0.9,1.2),(1.2,1.4)]),
                            ("metallicity", [(-1.0,-0.3),(-0.3,0.0),(0.0,0.5)])]:
            for lo, hi in bins:
                if label == "distance":
                    mask = (dte_d >= lo) & (dte_d < hi)
                elif label == "mass":
                    mask = (m1te >= lo) & (m1te < hi)
                else:
                    mask = (fehte >= lo) & (fehte < hi)
                rows.append({
                    "model": name, "slice_type": label,
                    "slice": f"[{lo},{hi})", "n": int(mask.sum()),
                    "pr_auc": pr_auc_per_bin(mask, yte, pr),
                })
        rows.append({"model": name, "slice_type": "overall", "slice": "all",
                     "n": int(bte.sum()), "pr_auc": overall})
        print(f"  {name}: overall PR-AUC={overall:.4f}", flush=True)

    rob = pd.DataFrame(rows)
    rob.to_csv(outdir / "m23_robustness.csv", index=False)

    # heatmap: model x slice PR-AUC
    pivot = rob[rob.slice_type == "distance"].pivot(index="model", columns="slice", values="pr_auc")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.4, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_title("M23: PR-AUC by model × distance slice")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i,j]:.3f}", ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(figdir / "fig23_robustness.png", dpi=150); plt.close(fig)

    # Save m17 q-summary
    (outdir / "m17_qrecovery.json").write_text(
        json.dumps({"q_mae": q_mae, "q_median_bias": q_bias,
                    "lgbm_prauc": prauc, "lgbm_rocauc": prauc_roc}, indent=2)
    )
    print(f"\nWrote M17/M23 tables to {outdir.relative_to(REPO_ROOT)}/ and figures to {figdir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())