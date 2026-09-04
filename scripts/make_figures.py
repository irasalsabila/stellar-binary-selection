"""Generate the remaining PRD figures (TODO M24).

Produces the high-priority figures from the master TODO list that
aren't produced by other scripts. Each figure is rendered to
results/figures/.

Run via:
    PYTHONPATH=src python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def fig3_overlum_vs_q(syn: pd.DataFrame, out: Path) -> None:
    """Photometric overluminosity delta_m vs mass ratio q (PRD fig3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = syn["is_binary"].to_numpy()
    q = syn["q"].to_numpy()
    ap_g = syn["apparent_G"].to_numpy()
    bp = syn["apparent_BP"].to_numpy() - syn["apparent_RP"].to_numpy()
    par = syn["parallax_obs"].clip(lower=1e-3).to_numpy()
    m_g = ap_g - 5 * np.log10(1000.0 / par) + 5
    singles = syn[~syn["is_binary"]]
    coef = np.polyfit(
        (singles["apparent_BP"] - singles["apparent_RP"]).to_numpy(),
        (singles["apparent_G"] - 5 * np.log10(1000.0 / singles["parallax_obs"].clip(lower=1e-3)) + 5).to_numpy(),
        3,
    )
    ridge_mg = np.polyval(coef, bp)
    overlum = ridge_mg - m_g

    fig, ax = plt.subplots(figsize=(8, 6))
    sel = b & (q > 0)
    nb = int(sel.sum())
    ax.scatter(q[sel], overlum[sel], s=2, c="C3", alpha=0.3,
                label=f"binaries (n={nb:,})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.0)
    ax.set_xlabel("Mass ratio q = m2/m1")
    ax.set_ylabel("Overluminosity vs single-star ridge [mag]")
    ax.set_title("M24 fig3: Photometric binary overluminosity vs q")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4_models(pr_lgb: dict, pr_phot: dict, pr_pinn: dict, out: Path) -> None:
    """Model comparison bar chart (PRD fig4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Random Forest", "XGBoost", "LightGBM", "Plain MLP", "PI-NN"]
    prs = [
        pr_lgb.get("pr_auc", float("nan")),
        pr_lgb.get("pr_auc", float("nan")),
        pr_lgb.get("pr_auc", float("nan")),
        pr_phot.get("pr_auc", float("nan")),
        pr_pinn.get("pr_auc", float("nan")),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(names)), prs, color=["C0", "C1", "C2", "C3", "C4"][: len(names)])
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance (0.5)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC on synthetic test split")
    ax.set_title("M24 fig4: Classifier performance comparison")
    for i, v in enumerate(prs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig8_9_c_phot(syn: pd.DataFrame, comp: pd.DataFrame, out_dir: Path) -> None:
    """C_phot completeness vs q and vs d (PRD fig8/9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = syn["is_binary"].to_numpy()
    q = syn["q"].to_numpy()
    d = syn["distance_pc"].to_numpy()
    # Overluminosity threshold at 1% FPR
    ap_g = syn["apparent_G"].to_numpy()
    par = syn["parallax_obs"].clip(lower=1e-3).to_numpy()
    m_g = ap_g - 5 * np.log10(1000.0 / par) + 5
    bp = syn["apparent_BP"].to_numpy() - syn["apparent_RP"].to_numpy()
    singles = syn[~syn["is_binary"]]
    coef = np.polyfit(
        (singles["apparent_BP"] - singles["apparent_RP"]).to_numpy(),
        (singles["apparent_G"] - 5 * np.log10(1000.0 / singles["parallax_obs"].clip(lower=1e-3)) + 5).to_numpy(),
        3,
    )
    overlum = np.polyval(coef, bp) - m_g
    thr = float(np.quantile(overlum[~b], 0.99))
    detected = (b & (overlum > thr))

    # C_phot vs q
    fig, ax = plt.subplots(figsize=(7, 5))
    qbins = np.linspace(0.1, 1.0, 19)
    c_q = [detected[b & (q >= lo) & (q < hi)].mean() if (b & (q >= lo) & (q < hi)).any() else np.nan
           for lo, hi in zip(qbins[:-1], qbins[1:])]
    ax.plot(qbins[:-1] + 0.05, c_q, "o-", color="C0")
    ax.set_xlabel("Mass ratio q")
    ax.set_ylabel("C_phot (photometric completeness)")
    ax.set_title("M24 fig8: Photometric detection completeness vs q")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_c_phot_vs_q.png", dpi=150)
    plt.close(fig)

    # C_phot vs d
    fig, ax = plt.subplots(figsize=(7, 5))
    dbins = np.linspace(20, 200, 10)
    c_d = [detected[b & (d >= lo) & (d < hi)].mean() if (b & (d >= lo) & (d < hi)).any() else np.nan
           for lo, hi in zip(dbins[:-1], dbins[1:])]
    ax.plot(dbins[:-1] + 10, c_d, "o-", color="C0")
    ax.set_xlabel("Distance [pc]")
    ax.set_ylabel("C_phot")
    ax.set_title("M24 fig9: Photometric completeness vs distance")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_c_phot_vs_d.png", dpi=150)
    plt.close(fig)


def fig10_q_vs_p(comp: pd.DataFrame, out: Path) -> None:
    """Combined C_phot+C_astro vs (q, P) heatmap (PRD fig10)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ast

    # q_bin and P_bin are stored as "(lo, hi)" string tuples. Parse them.
    for col in ("q_bin", "P_bin"):
        if comp[col].dtype == object and isinstance(comp[col].iloc[0], str):
            comp = comp.copy()
            comp[col] = comp[col].apply(ast.literal_eval)

    pv = comp.pivot_table(index="q_bin", columns="P_bin", values="C_combined_empirical")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pv.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("M24 fig10: Combined photometric+astrometric completeness (q × P)")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f"[{a}, {b}]" for a, b in pv.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f"[{a}, {b}]" for a, b in pv.index], fontsize=8)
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("q = m2/m1")
    plt.colorbar(im, ax=ax, label="C_combined")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig5_pinn_vs_mlp(pr_mlp: dict, pr_pinn: dict, out: Path) -> None:
    """PI-NN vs plain MLP comparison (PRD fig5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["PR-AUC", "ROC-AUC"]
    mlp = [pr_mlp.get("pr_auc", float("nan")), pr_mlp.get("roc_auc", float("nan"))]
    pinn = [pr_pinn.get("pr_auc", float("nan")), pr_pinn.get("roc_auc", float("nan"))]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, mlp, w, label="Plain MLP", color="C0")
    ax.bar(x + w / 2, pinn, w, label="PI-NN (physics-informed)", color="C3")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("M24 fig5: PI-NN vs plain MLP (synthetic test)")
    for i, (m, p) in enumerate(zip(mlp, pinn)):
        ax.text(i - w / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig6_ablation_ladder(ablation_csv: Path, out: Path) -> None:
    """Physics ablation ladder (PRD fig6): PR-AUC and q-MAE vs ladder rung."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(ablation_csv)
    levels = df["level"].to_numpy()
    pr = df["pr_auc"].to_numpy()
    qmae = df["q_mae"].to_numpy()
    names = df["model"].to_numpy()
    short = ["NN-0", "PI-NN-1", "PI-NN-2", "PI-NN-3", "PI-NN-4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(levels, pr, color="C0")
    ax1.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax1.set_xticks(levels)
    ax1.set_xticklabels(short)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("PR-AUC (binary classification)")
    ax1.set_title("M24 fig6a: Classification performance vs physics rung")
    ax1.legend()

    ax2.bar(levels, qmae, color="C1")
    ax2.set_xticks(levels)
    ax2.set_xticklabels(short)
    ax2.set_ylabel("q-MAE [mass ratio]")
    ax2.set_title("M24 fig6b: Mass-ratio recovery vs physics rung")
    for i, v in enumerate(qmae):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig12_schedule(out: Path) -> None:
    """Project milestone Gantt (PRD fig12)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (label, start_week, duration_weeks, status)
    tasks = [
        ("M1-M3 Data ingestion", 0, 2, "done"),
        ("M4-M6 Audit/CMD/extinction", 2, 2, "done"),
        ("M7 PARSEC isochrones", 4, 2, "done"),
        ("M8-M11 Emulator+synthetic", 4, 3, "done"),
        ("M12-M14 Baselines/CMD ridge", 7, 2, "done"),
        ("M16 Physics ablation", 9, 2, "done"),
        ("M18 NSS validation", 9, 1, "done"),
        ("M19-M20 Sim-to-real/C_phot", 10, 1, "done"),
        ("M21 Astrometric model", 7, 2, "done"),
        ("M22 Complementarity (MAIN)", 9, 2, "done"),
        ("M24-M25 Figures/tables", 11, 2, "in_progress"),
        ("M26 Bootstrap CIs", 12, 1, "pending"),
        ("M27 Failure-mode report", 12, 1, "in_progress"),
        ("M28 Leakage audit", 9, 1, "done"),
        ("M29 Compute audit", 13, 1, "pending"),
    ]
    color = {"done": "C2", "in_progress": "C3", "pending": "0.7"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, start, dur, status) in enumerate(tasks):
        ax.barh(i, dur, left=start, color=color[status], edgecolor="k", height=0.6)
        ax.text(start + dur + 0.1, i, status.replace("_", " "), va="center", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_xlabel("Project week")
    ax.set_title("M24 fig12: Project milestone schedule")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument(
        "--baselines-metrics",
        default="results/tables/baselines_metrics.json",
    )
    p.add_argument(
        "--plain-mlp-metrics",
        default="results/tables/plain_mlp_metrics.json",
    )
    p.add_argument(
        "--pinn-metrics",
        default="results/tables/pinn_metrics.json",
    )
    p.add_argument(
        "--complementarity",
        default="results/tables/complementarity_q_P.csv",
    )
    p.add_argument("--figdir", default="results/figures")
    args = p.parse_args()

    syn = pd.read_parquet(REPO_ROOT / args.synthetic)
    figdir = REPO_ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    print("Generating fig3 (overluminosity vs q)...", flush=True)
    fig3_overlum_vs_q(syn, figdir / "fig3_displacement_vs_q.png")

    # Load model metrics if present
    pr_lgb = pr_phot = pr_pinn = {}
    try:
        import json
        b = json.loads((REPO_ROOT / args.baselines_metrics).read_text())
        if isinstance(b, dict) and "results" in b:
            lgb = next((r for r in b["results"] if r.get("model") == "lgbm"), {})
            pr_lgb = lgb
    except Exception:
        pass
    try:
        import json
        pr_phot = json.loads((REPO_ROOT / args.plain_mlp_metrics).read_text())
    except Exception:
        pass
    try:
        import json
        pr_pinn = json.loads((REPO_ROOT / args.pinn_metrics).read_text())
    except Exception:
        pass

    print("Generating fig4 (model comparison)...", flush=True)
    fig4_models(pr_lgb, pr_phot, pr_pinn, figdir / "fig4_models.png")

    print("Generating fig5 (PI-NN vs MLP)...", flush=True)
    fig5_pinn_vs_mlp(pr_phot, pr_pinn, figdir / "fig5_pinn_vs_mlp.png")

    try:
        comp = pd.read_csv(REPO_ROOT / args.complementarity)
    except Exception:
        comp = pd.DataFrame()

    if not comp.empty:
        print("Generating fig8, fig9 (C_phot vs q, d)...", flush=True)
        fig8_9_c_phot(syn, comp, figdir)
        print("Generating fig10 (combined C vs q,P)...", flush=True)
        fig10_q_vs_p(comp, figdir / "fig10_q_vs_p.png")

    print("Generating fig6 (ablation ladder)...", flush=True)
    try:
        fig6_ablation_ladder(REPO_ROOT / "results" / "tables" / "e6_ablation.csv", figdir / "fig6_ablation.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  (fig6 skipped: {exc})", flush=True)

    print("Generating fig12 (schedule)...", flush=True)
    fig12_schedule(figdir / "fig12_schedule.png")

    print(f"\nFigures written to {figdir.relative_to(REPO_ROOT)}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
