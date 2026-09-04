"""Generate publication figures from stored analysis products.

Run from the repository root with::

    PYTHONPATH=src python scripts/make_manuscript_figures.py

PDF and high-resolution PNG copies are written to
``results/figures_manuscript/``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8,
                         "savefig.dpi": 600, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    return plt


def save(fig, outdir: Path, stem: str, plt) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def abs_g(df: pd.DataFrame) -> np.ndarray:
    par = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
    g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    return g - 5 * np.log10(1000 / np.clip(par, 1e-6, None)) + 5


def read_bins(path: Path, columns: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path)
    for col in columns:
        d[col] = d[col].map(ast.literal_eval)
    return d


def figure1(out: Path, plt) -> None:
    parent = pd.read_parquet(ROOT / "data/processed/gaia_dr3_2mass_derived.parquet")
    fgk = pd.read_parquet(ROOT / "data/processed/gaia_dr3_2mass_fgk.parquet")
    sample_b = pd.read_parquet(ROOT / "data/processed/sample_b_apparently_single.parquet")
    sample_c = pd.read_parquet(ROOT / "data/processed/sample_c_nss.parquet")
    x = pd.to_numeric(parent.bp_rp, errors="coerce").to_numpy(float)
    y = abs_g(parent); good = np.isfinite(x) & np.isfinite(y)
    xb = pd.to_numeric(sample_b.bp_rp, errors="coerce").to_numpy(float)
    yb = abs_g(sample_b); ok = np.isfinite(xb) & np.isfinite(yb)
    coef = np.polyfit(xb[ok], yb[ok], 3)
    nss_ids = set(sample_c.source_id.drop_duplicates())
    fgk_b = fgk[fgk.source_id.isin(set(sample_b.source_id))]
    fgk_c = fgk[fgk.source_id.isin(nss_ids)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    a.hexbin(x[good], y[good], gridsize=150, bins="log", mincnt=1, cmap="viridis", linewidths=0)
    a.set(xlim=(-.2, 3.2), ylim=(10.5, -2), xlabel=r"$G_{\rm BP}-G_{\rm RP}$ [mag]", ylabel=r"$M_G$ [mag]", title="(a) Gaia--2MASS parent sample")
    xf = pd.to_numeric(fgk.bp_rp, errors="coerce").to_numpy(float); yf = abs_g(fgk); goodf = np.isfinite(xf) & np.isfinite(yf)
    b.hexbin(xf[goodf], yf[goodf], gridsize=90, bins="log", mincnt=1, cmap="Greys", alpha=.45, linewidths=0)
    b.scatter(pd.to_numeric(fgk_b.bp_rp), abs_g(fgk_b), s=4, alpha=.35, label=f"Sample B ($N={len(fgk_b):,}$)")
    b.scatter(pd.to_numeric(fgk_c.bp_rp), abs_g(fgk_c), s=9, alpha=.75, label=f"Sample C NSS ($N={len(fgk_c):,}$)")
    xx = np.linspace(.5, 1.6, 300); b.plot(xx, np.polyval(coef, xx), "--", lw=1.4, label="Single-star ridge")
    b.set(xlim=(.45, 1.65), ylim=(8.2, 2.5), xlabel=r"$G_{\rm BP}-G_{\rm RP}$ [mag]", ylabel=r"$M_G$ [mag]", title="(b) Adopted FGK domain")
    b.legend(loc="best", frameon=True); fig.tight_layout(); save(fig, out, "figure1_observational_sample", plt)


def figure2(out: Path, plt) -> None:
    syn = pd.read_parquet(ROOT / "data/synthetic/synthetic_v1.parquet")
    singles = syn[~syn.is_binary]
    bp = singles.apparent_BP - singles.apparent_RP
    mg = singles.apparent_G - 5 * np.log10(1000 / singles.parallax_obs.clip(lower=1e-6)) + 5
    coef = np.polyfit(bp.to_numpy(float), mg.to_numpy(float), 3)
    color = syn.apparent_BP - syn.apparent_RP
    mag = syn.apparent_G - 5 * np.log10(1000 / syn.parallax_obs.clip(lower=1e-6)) + 5
    excess = np.polyval(coef, color) - mag.to_numpy(float); q = syn.q.to_numpy(float); binary = syn.is_binary.to_numpy(bool)
    edges = np.linspace(.1, 1, 19); centers = (edges[:-1] + edges[1:]) / 2
    vals = {k: [] for k in ("median", "p16", "p84", "p05", "p95")}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = binary & (q >= lo) & ((q <= hi) if i == len(edges) - 2 else (q < hi)); v = excess[mask]
        vals["median"].append(np.nanmedian(v)); vals["p16"].append(np.nanpercentile(v, 16)); vals["p84"].append(np.nanpercentile(v, 84)); vals["p05"].append(np.nanpercentile(v, 5)); vals["p95"].append(np.nanpercentile(v, 95))
    fig, ax = plt.subplots(figsize=(4.25, 3.2)); ax.fill_between(centers, vals["p05"], vals["p95"], alpha=.18, label="5--95 percentile"); ax.fill_between(centers, vals["p16"], vals["p84"], alpha=.28, label="16--84 percentile"); ax.plot(centers, vals["median"], "o-", lw=1.7, ms=3.5, label="Median"); ax.axhline(0, ls="--", lw=1, label="Ridge locus")
    ax.set(xlim=(.1, 1), xlabel=r"Mass ratio $q=M_2/M_1$", ylabel=r"$\Delta M_G^{\rm ridge}$ [mag]"); ax.legend(loc="upper left"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); save(fig, out, "figure2_photometric_physics", plt)


def figure3(out: Path, plt) -> None:
    metrics = json.loads((ROOT / "results/tables/nss_validation_fgk.json").read_text())
    syn = pd.read_parquet(ROOT / "data/synthetic/synthetic_v1.parquet")
    ctl = pd.read_parquet(ROOT / "data/processed/validation_fgk/sample_b_fgk_scored.parquet")
    nss = pd.read_parquet(ROOT / "data/processed/validation_fgk/sample_c_fgk_scored.parquet")
    from validate_nss import photometric_score_ol
    bins = np.linspace(0, 1, 41); fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.15))
    a.hist(ctl.p_binary, bins=bins, density=True, alpha=.55, label=f"Sample B ($N={len(ctl):,}$)"); a.hist(nss.p_binary, bins=bins, density=True, alpha=.55, label=f"Sample C ($N={len(nss):,}$)"); a.set(xlabel="Synthetic-trained LightGBM $P$(binary)", ylabel="Probability density", title="(a) FGK real-data score distributions"); a.legend(loc="upper right"); a.text(.96, .76, f"PR-AUC = {metrics['real_sample_c_vs_b']['pr_auc']:.3f}\nROC-AUC = {metrics['real_sample_c_vs_b']['roc_auc']:.3f}", transform=a.transAxes, ha="right", va="top", bbox={"facecolor":"white", "alpha":.9})
    ol_syn = np.asarray(photometric_score_ol(syn)); singles = ~syn.is_binary.to_numpy(bool); threshold = metrics["overluminosity_proxy"]["threshold_mag"]
    b.hist(ol_syn[singles], bins=65, density=True, histtype="step", lw=1.5, label="Synthetic singles"); b.hist(ctl.ol_proxy, bins=65, density=True, alpha=.45, label=f"Real Sample B--FGK ($N={len(ctl):,}$)"); b.axvline(threshold, ls="--", lw=1.2, label=f"Synthetic 1% threshold = {threshold:.3f} mag"); b.set(xlabel=r"Overluminosity $\Delta M_G^{\rm ridge}$ [mag]", ylabel="Probability density", title="(b) Synthetic-to-real calibration"); b.legend(loc="upper right", fontsize=7); b.text(.96, .72, f"real FPR = {metrics['overluminosity_proxy']['fpr_sample_b']:.1%}", transform=b.transAxes, ha="right", va="top", bbox={"facecolor":"white", "alpha":.9}); fig.tight_layout(); save(fig, out, "figure3_real_validation", plt)


def figure4(out: Path, plt) -> None:
    dm = read_bins(ROOT / "results/appendix/appendix_c_c_phot_vs_q_M1.csv", ["q_bin", "m1_msun_bin"]); dd = read_bins(ROOT / "results/appendix/appendix_c_c_phot_vs_q_d.csv", ["q_bin", "distance_pc_bin"])
    fig, axes = plt.subplots(2, 1, figsize=(4.25, 7), sharex=True)
    for ax, d, group, title, xlabel, fmt in [(axes[0], dm, "m1_msun_bin", r"(a) $C_{\rm phot}(q)$ grouped by primary mass", r"$M_1$ [$M_\odot$]", lambda x: f"{x[0]:.1f}--{x[1]:.1f}"), (axes[1], dd, "distance_pc_bin", r"(b) $C_{\rm phot}(q)$ grouped by distance", "Distance [pc]", lambda x: f"{x[0]:g}--{x[1]:g}")]:
        for gname in list(dict.fromkeys(d[group])):
            g = d[d[group] == gname].sort_values("q_bin"); x = np.array([(u + v) / 2 for u, v in g.q_bin]); line, = ax.plot(x, g.completeness, marker="o", lw=1.7, ms=4.5, label=fmt(gname)); ax.fill_between(x, g.ci_lo, g.ci_hi, alpha=.16, color=line.get_color())
        ax.set_ylim(0, 1.02); ax.set_ylabel(r"Photometric completeness $C_{\rm phot}$"); ax.set_title(title); ax.grid(axis="y", alpha=.25); ax.legend(title=xlabel, loc="upper left")
    axes[1].set(xlabel=r"Mass ratio $q=M_2/M_1", xlim=(.1, 1)); fig.tight_layout(h_pad=1.2); save(fig, out, "figure4_photometric_selection", plt)


def figure5(out: Path, plt) -> None:
    d = read_bins(ROOT / "results/appendix/appendix_c_complementarity_q_P.csv", ["q_bin", "P_bin"]); q = sorted(set(d.q_bin), key=lambda x: x[0]); p = sorted(set(d.P_bin), key=lambda x: x[0]); qe = np.array([q[0][0]] + [x[1] for x in q]); pe = np.array([p[0][0]] + [x[1] for x in p]); stats = json.loads((ROOT / "results/appendix/appendix_c_global_statistics.json").read_text())["complementarity_metrics"]
    panels = [("frac_phot_only", "(a) Photometric only", stats["photometric_only"]), ("frac_astro_only", "(b) Astrometric only", stats["astrometric_only"]), ("frac_both", "(c) Both channels", stats["both"]), ("frac_neither", "(d) Neither channel", stats["neither"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), sharex=True, sharey=True); mesh = None
    for ax, (col, title, fraction) in zip(axes.flat, panels):
        z = np.full((len(q), len(p)), np.nan)
        for i, qb in enumerate(q):
            for j, pb in enumerate(p):
                row = d[(d.q_bin == qb) & (d.P_bin == pb)]
                if len(row): z[i, j] = row.iloc[0][col]
        mesh = ax.pcolormesh(pe, qe, z, shading="flat", vmin=0, vmax=1, cmap="viridis", rasterized=True); ax.set_xscale("log"); ax.set(xlim=(pe[0], pe[-1]), ylim=(qe[0], qe[-1]), title=title); ax.text(.96, .93, f"Fraction = {fraction:.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, bbox={"facecolor":"white", "alpha":.85, "edgecolor":".7"})
    for ax in axes[:, 0]: ax.set_ylabel(r"Mass ratio $q=M_2/M_1$");
    for ax in axes[1, :]: ax.set_xlabel("Period [days]")
    fig.colorbar(mesh, ax=axes, fraction=.03, pad=.03, label="Detection fraction"); fig.subplots_adjust(left=.1, right=.88, bottom=.12, top=.95, wspace=.18, hspace=.24); save(fig, out, "figure5_complementarity", plt)


def main() -> int:
    plt = setup(); out = ROOT / "results/figures_manuscript"; out.mkdir(parents=True, exist_ok=True)
    figure1(out, plt); figure2(out, plt); figure3(out, plt); figure4(out, plt); figure5(out, plt)
    print(f"Wrote manuscript figures to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
