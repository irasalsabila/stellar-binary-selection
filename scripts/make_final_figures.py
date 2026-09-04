"""Render the focused six-figure manuscript set into results/figures_final.

The script intentionally does not overwrite the historical figures in
``results/figures``.  All plotted quantities are derived from the repository's
stored parquet catalogues and result tables.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
                         "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300})
    return plt


def absolute_mag(df: pd.DataFrame) -> np.ndarray:
    par = pd.to_numeric(df["parallax"], errors="coerce").to_numpy()
    g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy()
    return g - 5 * np.log10(1000 / np.clip(par, 1e-3, None)) + 5


def fig1(out: Path, plt) -> dict:
    parent = pd.read_parquet(REPO_ROOT / "data/processed/gaia_dr3_2mass_derived.parquet")
    fgk = pd.read_parquet(REPO_ROOT / "data/processed/gaia_dr3_2mass_fgk.parquet")
    sample_b = pd.read_parquet(REPO_ROOT / "data/processed/sample_b_apparently_single.parquet")
    sample_c = pd.read_parquet(REPO_ROOT / "data/processed/sample_c_nss.parquet")
    nss_ids = set(sample_c["source_id"].drop_duplicates())
    x = pd.to_numeric(parent["bp_rp"], errors="coerce").to_numpy()
    y = absolute_mag(parent)
    valid = np.isfinite(x) & np.isfinite(y)
    xb, yb = x[valid], y[valid]
    x_b = pd.to_numeric(sample_b["bp_rp"], errors="coerce").to_numpy()
    y_b = absolute_mag(sample_b)
    x_f = pd.to_numeric(fgk["bp_rp"], errors="coerce").to_numpy()
    y_f = absolute_mag(fgk)
    is_nss = fgk["source_id"].isin(nss_ids).to_numpy()
    is_b = fgk["source_id"].isin(set(sample_b["source_id"])).to_numpy()
    ridge_ok = np.isfinite(x_b) & np.isfinite(y_b)
    coef = np.polyfit(x_b[ridge_ok], y_b[ridge_ok], 3)
    xx = np.linspace(0.5, 1.6, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True,
                               gridspec_kw={"width_ratios": [1.05, 1]})
    a.hexbin(xb, yb, gridsize=180, bins="log", mincnt=1, cmap="magma", linewidths=0)
    a.set_xlim(-0.2, 3.2); a.set_ylim(10.5, -2)
    a.set_title(f"(a) Parent sample ($N={len(parent):,}$)")
    a.set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$ [mag]"); a.set_ylabel(r"$M_G$ [mag]")
    a.axvspan(0.5, 1.6, 3, 8, color="white", alpha=0.08, lw=1, ec="white")
    # Keep the parent context faint, then layer the science populations.
    b.hexbin(xb, yb, gridsize=130, bins="log", mincnt=1, cmap="Greys", alpha=0.55, linewidths=0)
    b.scatter(x_f[is_b], y_f[is_b], s=3, color="#2878b5", alpha=0.35,
              label=f"Sample B in FGK ({is_b.sum():,})")
    b.scatter(x_f[is_nss], y_f[is_nss], s=10, color="#c43d4b", alpha=0.85,
              edgecolors="white", linewidths=0.2, label=f"Sample C NSS in FGK ({is_nss.sum():,})")
    b.plot(xx, np.polyval(coef, xx), "k--", lw=1.6, label="Sample B ridge")
    b.add_patch(plt.Rectangle((0.5, 3), 1.1, 5, fill=False, ec="#333333", lw=0.9, ls=":"))
    b.set_xlim(0.35, 1.75); b.set_ylim(8.5, 2.2); b.set_xlabel(r"$G_{\rm BP}-G_{\rm RP}$ [mag]")
    b.set_title(f"(b) FGK main sequence ($N={len(fgk):,}$)")
    b.legend(loc="upper right", frameon=True)
    fig.tight_layout(); fig.savefig(out / "figure1_observational_sample.png", bbox_inches="tight"); plt.close(fig)
    return {"parent": len(parent), "fgk": len(fgk), "sample_b_total": len(sample_b),
            "sample_b_fgk_overlap": int(is_b.sum()),
            "sample_c_solutions": len(sample_c), "sample_c_unique": len(nss_ids),
            "sample_c_fgk_overlap": int(is_nss.sum()),
            "ridge_coefficients": [float(v) for v in coef]}


def fig2(out: Path, plt) -> None:
    syn = pd.read_parquet(REPO_ROOT / "data/synthetic/synthetic_v1.parquet")
    singles = syn[~syn.is_binary]
    bp_s = singles.apparent_BP - singles.apparent_RP
    mg_s = singles.apparent_G - 5 * np.log10(1000 / singles.parallax_obs.clip(lower=1e-3)) + 5
    coef = np.polyfit(bp_s, mg_s, 3)
    bp = syn.apparent_BP - syn.apparent_RP
    mg = syn.apparent_G - 5 * np.log10(1000 / syn.parallax_obs.clip(lower=1e-3)) + 5
    excess = np.polyval(coef, bp) - mg
    q = syn.q.to_numpy(); binary = syn.is_binary.to_numpy()
    bins = np.linspace(0.1, 1.0, 19); centers = (bins[:-1] + bins[1:]) / 2
    med, lo, hi, p05, p95 = [], [], [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        v = excess[binary & (q >= left) & (q < right)]
        med.append(np.nanmedian(v)); lo.append(np.nanpercentile(v, 16)); hi.append(np.nanpercentile(v, 84))
        p05.append(np.nanpercentile(v, 5)); p95.append(np.nanpercentile(v, 95))
    fig, ax = plt.subplots(figsize=(7.2, 5)); ax.fill_between(centers, p05, p95, color="#9ecae1", alpha=.35, label="5--95 percentile")
    ax.fill_between(centers, lo, hi, color="#3182bd", alpha=.35, label="16--84 percentile")
    ax.plot(centers, med, "o-", color="#08519c", lw=2, ms=4, label="Median")
    ax.axhline(0, color="k", lw=.8, ls="--", label="Ridge locus")
    ax.set(xlim=(.1, 1), ylim=(-.2, .95), xlabel=r"Mass ratio $q=M_2/M_1$", ylabel=r"Overluminosity relative to fitted ridge $\Delta M_G^{\rm ridge}$ [mag]")
    ax.set_title("Photometric binary signal relative to single-star ridge"); ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(out / "figure2_photometric_physics.png", bbox_inches="tight"); plt.close(fig)


def fig3(out: Path, plt) -> None:
    d = pd.read_csv(REPO_ROOT / "results/tables/e6_ablation.csv")
    labels = ["NN-0", "+ parallax", "+ flux", "+ emulator", "+ coevality"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5)); x = np.arange(len(d))
    a.errorbar(x, d.pr_auc, yerr=d.pr_auc_std, fmt="o-", capsize=3, color="#2166ac")
    a.axhline(.5, color="0.4", ls=":"); a.set_ylabel("PR-AUC"); a.set_title("Classification"); a.set_xticks(x, labels, rotation=25, ha="right")
    b.errorbar(x, d.q_mae, yerr=d.q_mae_std, fmt="o-", capsize=3, color="#b2182b")
    b.set_ylabel(r"$q$ MAE"); b.set_title("Mass-ratio recovery"); b.set_xticks(x, labels, rotation=25, ha="right")
    fig.suptitle("RQ1: physics-informed ablation (mean +/- 1 SD; 3 seeds)"); fig.tight_layout(); fig.savefig(out / "figure3_model_ablation.png", bbox_inches="tight"); plt.close(fig)


def fig4(out: Path, plt) -> None:
    metrics_path = REPO_ROOT / "results/tables/nss_validation_fgk.json"
    m = json.loads(metrics_path.read_text()); syn = pd.read_parquet(REPO_ROOT / "data/synthetic/synthetic_v1.parquet")
    from validate_nss import photometric_score_ol
    nss_scored = pd.read_parquet(REPO_ROOT / "data/processed/validation_fgk/sample_c_fgk_scored.parquet")
    ctl_scored = pd.read_parquet(REPO_ROOT / "data/processed/validation_fgk/sample_b_fgk_scored.parquet")
    nss_scores = nss_scored["p_binary"].to_numpy(); ctl_scores = ctl_scored["p_binary"].to_numpy()
    bins = np.linspace(0, 1, 41); fig, (a, b_ax) = plt.subplots(1, 2, figsize=(10, 4.4));
    a.hist(ctl_scores, bins=bins, density=True, alpha=.55, color="#2878b5", label=f"Sample B (n={len(ctl_scores):,})")
    a.hist(nss_scores, bins=bins, density=True, alpha=.55, color="#c43d4b", label=f"Sample C (n={len(nss_scores):,})")
    a.set(xlabel="Synthetic-trained LightGBM P(binary)", ylabel="Probability density", title="Real Gaia validation"); a.legend()
    ol_syn = photometric_score_ol(syn); threshold = float(m["overluminosity_proxy"]["threshold_mag"])
    ol_nss = nss_scored["ol_proxy"].to_numpy(); ol_ctl = ctl_scored["ol_proxy"].to_numpy()
    b_ax.hist(ol_syn[~syn.is_binary], bins=60, density=True, histtype="step", lw=1.8, color="#2166ac", label="Synthetic singles")
    b_ax.hist(ol_ctl, bins=60, density=True, alpha=.45, color="#666666", label=f"Real Sample B--FGK (n={len(ol_ctl):,})")
    b_ax.axvline(threshold, color="#b2182b", ls="--", label=f"Synthetic 1% threshold = {threshold:.3f} mag")
    b_ax.set(xlabel=r"Overluminosity $\Delta M_G$ [mag]", ylabel="Probability density", title=f"Sim-to-real calibration (FGK)\nreal FPR = {m['overluminosity_proxy']['fpr_sample_b']:.1%}"); b_ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figure4_real_validation.png", bbox_inches="tight"); plt.close(fig)


def fig5(out: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    ims = []
    for ax, path, col, xlabel, title in zip(
        axes,
        ["results/tables/c_phot_vs_q_M1.csv", "results/tables/c_phot_vs_q_d.csv"],
        ["m1_msun_bin", "distance_pc_bin"],
        [r"Primary mass $M_1$ [$M_\odot$]", "Distance [pc]"],
        ["Marginalized over distance and metallicity", "Marginalized over primary mass and metallicity"],
    ):
        d = pd.read_csv(REPO_ROOT / path); d.q_bin = d.q_bin.map(ast.literal_eval); d[col] = d[col].map(ast.literal_eval)
        pv = d.pivot(index="q_bin", columns=col, values="completeness")
        lo = d.pivot(index="q_bin", columns=col, values="ci_lo"); hi = d.pivot(index="q_bin", columns=col, values="ci_hi")
        im = ax.imshow(pv, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis"); ims.append(im)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                ax.text(j, i, f"{pv.iloc[i,j]:.2f}\n[{lo.iloc[i,j]:.2f},{hi.iloc[i,j]:.2f}]", ha="center", va="center", fontsize=7, color="white" if pv.iloc[i,j] < .55 else "black")
        ax.set(xticks=range(len(pv.columns)), xticklabels=[f"{a:g}--{b:g}" for a,b in pv.columns], yticks=range(len(pv.index)), yticklabels=[f"{a:g}--{b:g}" for a,b in pv.index], xlabel=xlabel, title=title)
    axes[0].set_ylabel(r"Mass ratio $q$"); fig.colorbar(ims[0], ax=axes, label="Photometric completeness", shrink=.85)
    fig.suptitle(r"Photometric selection function $C_{\rm phot}$ (95% binomial intervals)"); fig.tight_layout(); fig.savefig(out / "figure5_photometric_selection.png", bbox_inches="tight"); plt.close(fig)


def fig6(out: Path, plt) -> None:
    d = pd.read_csv(REPO_ROOT / "results/tables/complementarity_q_P.csv"); d.q_bin=d.q_bin.map(ast.literal_eval); d.P_bin=d.P_bin.map(ast.literal_eval)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True); ims=[]
    for ax, (col, title) in zip(axes.flat, [("frac_phot_only","Photometric only"),("frac_astro_only","Astrometric only"),("frac_both","Both channels"),("frac_neither","Neither")]):
        pv=d.pivot(index="q_bin",columns="P_bin",values=col); im=ax.imshow(pv,origin="lower",aspect="auto",vmin=0,vmax=1,cmap="viridis"); ims.append(im)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)): ax.text(j,i,f"{pv.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if pv.iloc[i,j]<.55 else "black")
        ax.set_title(title); ax.set_xticks(range(len(pv.columns)),[f"{a:g}--{b:g}" for a,b in pv.columns]); ax.set_yticks(range(len(pv.index)),[f"{a:g}--{b:g}" for a,b in pv.index]); ax.set_xlabel("Period [days]"); ax.set_ylabel("Mass ratio q")
    fig.colorbar(ims[0], ax=axes, label="Fraction of binaries", shrink=.85); fig.suptitle("Photometric and astrometric complementarity"); fig.tight_layout(); fig.savefig(out / "figure6_complementarity.png", bbox_inches="tight"); plt.close(fig)


def main() -> int:
    plt = setup(); out = REPO_ROOT / "results/figures_final"; out.mkdir(parents=True, exist_ok=True)
    metadata = {"figure1": fig1(out, plt)}; fig2(out, plt); fig3(out, plt); fig4(out, plt)
    fig5(out, plt)
    fig6(out, plt); (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2)); print(f"Wrote six figures to {out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
