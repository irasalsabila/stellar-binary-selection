"""Export machine-readable numerical material for manuscript appendices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "appendix"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    noise = json.loads((ROOT / "data/processed/noise_model.json").read_text())
    config = {
        "n_systems": 100000, "n_binaries": 50000, "n_singles": 50000,
        "seed": 42, "binary_fraction": 0.5,
        "q_intervals": [[0.1, 0.3], [0.3, 0.8], [0.8, 1.0]],
        "q_relative_weights": [1.0, 3.0, 1.0],
        "q_normalized_probabilities": [0.2 / 1.9, 1.5 / 1.9, 0.2 / 1.9],
        "primary_mass": {"range_msun": [0.6, 1.4], "distribution": "uniform"},
        "metallicity": {"range_feh": [-1.0, 0.5], "distribution": "uniform"},
        "age": {"range_gyr": [0.5, 12.0], "distribution": "uniform"},
        "distance": {"range_pc": [20.0, 200.0], "distribution": "uniform"},
        "period": {"range_days": [1.0, 3000.0], "distribution": "log_uniform"},
        "extinction": {"enabled": False},
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42},
        "noise_rng_note": "Top-level population sampling is seeded; noise injection creates independent default_rng instances and is not bitwise reproducible from the top-level seed.",
        "noise_fit_catalogue": noise["source"],
        "orbital_samplers_used": False,
        "orbital_sampler_distributions": {"eccentricity": "thermal", "inclination": "isotropic", "angles": "uniform", "phase": "uniform"},
    }
    (OUT / "appendix_a_metadata.json").write_text(json.dumps(config, indent=2))

    rows = []
    for band, vals in noise["bands"].items():
        rows.extend({"band": band, "magnitude": m, "sigma_mag": s} for m, s in zip(vals["mag"], vals["sigma_mag"]))
    pd.DataFrame(rows).to_csv(OUT / "appendix_a_photometric_noise_knots.csv", index=False)
    pd.DataFrame({"apparent_g": noise["parallax_error_vs_g"]["mag"], "sigma_parallax_mas": noise["parallax_error_vs_g"]["sigma_mas"]}).to_csv(OUT / "appendix_a_parallax_noise_knots.csv", index=False)

    model = {
        "features": ["apparent_G", "apparent_BP", "apparent_RP", "apparent_J", "apparent_H", "apparent_Ks", "sigma_G", "sigma_BP", "sigma_RP", "sigma_J", "sigma_H", "sigma_Ks"],
        "rf": {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20, "n_jobs": -1, "random_state": 42},
        "xgboost": {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "random_state": 42},
        "lightgbm": {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "random_state": 42},
        "plain_mlp": {"hidden": [128, 128, 64], "activation": "GELU", "dropout": 0.05, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 1e-5, "batch_size": 512},
        "executed_pinn_loss": "BCE + 0.5 SmoothL1(q) + 0.5 MSE(parallax)",
        "ablation_weights": {"q": 0.5, "parallax": 0.5, "photometry": 0.5, "coevality": 0.5, "bound": 0.01},
        "ablation_variants": ["NN-0", "+ parallax", "+ flux addition", "+ stellar emulator", "+ coevality/metallicity"],
        "emulator": {"backend": "EmpiricalMS", "hidden": [128, 128, 128, 64], "activation": "GELU", "outputs": ["G", "BP", "RP", "J", "H", "Ks"], "optimizer": "AdamW", "learning_rate": 5e-4, "weight_decay": 1e-5, "batch_size": 1024, "epochs": 60},
    }
    emu_metrics = json.loads((ROOT / "results/runs/train_emulator/metrics.json").read_text())
    model["emulator"]["mae_per_band"] = emu_metrics["mae_per_band"]
    (OUT / "appendix_b_model_metadata.json").write_text(json.dumps(model, indent=2))
    abl = pd.read_csv(ROOT / "results/tables/e6_ablation.csv")
    abl[["level", "model", "n_seeds", "pr_auc", "pr_auc_std", "q_mae", "q_mae_std"]].to_csv(OUT / "appendix_b_ablation_summary.csv", index=False)

    for name in ["c_phot_vs_q_M1", "c_phot_vs_q_d"]:
        pd.read_csv(ROOT / f"results/tables/{name}.csv").to_csv(OUT / f"appendix_c_{name}.csv", index=False)
    pd.read_csv(ROOT / "results/tables/complementarity_q_P.csv").to_csv(OUT / "appendix_c_complementarity_q_P.csv", index=False)
    comp_metrics = json.loads((ROOT / "results/runs/complementarity/metrics.json").read_text())
    bootstrap = ROOT / "results/tables/summary/bootstrap_simulation.csv"
    report = {"complementarity_metrics": comp_metrics, "bootstrap_intervals_available": bootstrap.exists(), "minimum_systems_per_selection_cell": 20, "excluded_selection_cells": []}
    if bootstrap.exists(): report["bootstrap_intervals"] = pd.read_csv(bootstrap).to_dict(orient="records")
    (OUT / "appendix_c_global_statistics.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote appendix data to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
