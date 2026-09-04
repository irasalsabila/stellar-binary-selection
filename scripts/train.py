"""Train baselines and PI-NN on the synthetic population (TODO M13–M16).

Run via:
    PYTHONPATH=src python scripts/train.py --model baselines
    PYTHONPATH=src python scripts/train.py --model plain_mlp
    PYTHONPATH=src python scripts/train.py --model pinn
    PYTHONPATH=src python python scripts/train.py --model ridge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.evaluation.metrics import (
    binary_classification_metrics,
    regression_metrics,
)
from stellar_binary_selection.models import (
    MainSequenceRidgeBaseline,
    PhysicsInformedNN,
    PlainMLP,
    StellarEmulator,
    make_lightgbm,
    make_random_forest,
    make_xgboost,
)
from stellar_binary_selection.utils import set_global_seed, write_run_metadata


FEATURE_COLUMNS_F1 = ["apparent_G", "apparent_BP", "apparent_RP", "parallax_obs"]  # placeholders


def load_synthetic(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split(df: pd.DataFrame, seed: int = 42):
    train, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["is_binary"])
    val, test = train_test_split(test, test_size=0.5, random_state=seed, stratify=test["is_binary"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the (n, d) feature matrix used for the empirical baselines."""
    cols = [
        "apparent_G", "apparent_BP", "apparent_RP",
        "apparent_J", "apparent_H", "apparent_Ks",
        "sigma_G", "sigma_BP", "sigma_RP", "sigma_J", "sigma_H", "sigma_Ks",
    ]
    return df[cols].to_numpy(dtype=np.float32)


def train_baseline(df: pd.DataFrame, model: str, seed: int = 42) -> dict:
    X = feature_matrix(df)
    y = df["is_binary"].to_numpy(dtype=int)
    train, val, test = split(df, seed=seed)
    Xtr = feature_matrix(train); ytr = train["is_binary"].to_numpy(int)
    Xte = feature_matrix(test); yte = test["is_binary"].to_numpy(int)

    print(f"    training {model} on {len(Xtr):,} rows...", flush=True)
    if model == "rf":
        clf = make_random_forest(seed)
    elif model == "xgb":
        clf = make_xgboost(seed)
    elif model == "lgbm":
        clf = make_lightgbm(seed)
    else:
        raise ValueError(model)
    with tqdm(total=1, desc=f"{model} fit", leave=False) as bar:
        clf.fit(Xtr, ytr)
        bar.update(1)
    proba = clf.predict_proba(Xte)[:, 1]
    metrics = binary_classification_metrics(yte, proba)
    metrics["model"] = model
    return metrics


def train_ridge(df: pd.DataFrame, seed: int = 42) -> dict:
    train, val, _test = split(df, seed=seed)
    bp_rp = (train["apparent_BP"] - train["apparent_RP"]).to_numpy()
    m_g_prelim = (train["apparent_G"] - 5 * np.log10(train["distance_pc"]) + 5).to_numpy()
    ridge = MainSequenceRidgeBaseline(polynomial_order=3)
    ridge.fit(np.column_stack([bp_rp, m_g_prelim]))
    return {"model": "ridge", "sigma_residual_mag": ridge.sigma_residual_}


def train_mlp(df: pd.DataFrame, seed: int = 42, epochs: int = 5) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(seed)
    train, val, test = split(df, seed=seed)
    Xtr = torch.from_numpy(feature_matrix(train))
    ytr = torch.from_numpy(train["is_binary"].to_numpy(dtype=np.float32))
    Xte = torch.from_numpy(feature_matrix(test))
    yte = test["is_binary"].to_numpy(int)

    model = PlainMLP(in_dim=Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=512, shuffle=True)
    for epoch in tqdm(range(epochs), desc="plain_mlp epochs", leave=False):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte).squeeze(-1)).numpy()
    metrics = binary_classification_metrics(yte, proba)
    metrics["model"] = "plain_mlp"
    return metrics


def train_pinn(df: pd.DataFrame, seed: int = 42, epochs: int = 3) -> dict:
    """Train the PI-NN on synthetic data (full pipeline)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    set_global_seed(seed)
    train, val, test = split(df, seed=seed)
    Xtr = torch.from_numpy(feature_matrix(train)).float()
    y_bin = torch.from_numpy(train["is_binary"].to_numpy(dtype=np.float32))
    y_q = torch.from_numpy(train["q"].to_numpy(dtype=np.float32))
    y_par = torch.from_numpy((1000.0 / train["distance_pc"].to_numpy(dtype=np.float32)))
    Xte = torch.from_numpy(feature_matrix(test)).float()
    yte = test["is_binary"].to_numpy(int)

    emulator = StellarEmulator()
    model = PhysicsInformedNN(in_dim=Xtr.shape[1], emulator=emulator)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = torch.nn.BCEWithLogitsLoss()
    sl1 = torch.nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(Xtr, y_bin, y_q, y_par), batch_size=512, shuffle=True)
    for epoch in tqdm(range(epochs), desc="pinn epochs", leave=False):
        model.train()
        for xb, yb, yq, yp in loader:
            opt.zero_grad()
            out = model(xb)
            l_bin = bce(torch.logit(out["p_binary"].clamp(1e-6, 1 - 1e-6)), yb)
            l_q = sl1(out["q"], yq)
            l_par = ((out["parallax_pred"] - yp) ** 2).mean()
            loss = l_bin + 0.5 * l_q + 0.5 * l_par
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(Xte)
        proba = out["p_binary"].numpy()
    metrics = binary_classification_metrics(yte, proba)
    metrics["model"] = "pinn"
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ridge", "rf", "xgb", "lgbm", "plain_mlp", "pinn", "baselines"], required=True)
    p.add_argument("--synthetic", default="data/synthetic/synthetic_v1.parquet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--metadata-dir", default="results/runs/train")
    args = p.parse_args()

    df = load_synthetic(REPO_ROOT / args.synthetic)
    print(f"Loaded {len(df):,} synthetic systems", flush=True)

    if args.model == "baselines":
        results = []
        for m in ("rf", "xgb", "lgbm"):
            print(f"Training baseline: {m}...", flush=True)
            metrics = train_baseline(df, m, seed=args.seed)
            print(f"  {m}: PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}", flush=True)
            results.append(metrics)
        summary = {"results": results}
    elif args.model == "ridge":
        summary = train_ridge(df, seed=args.seed)
    elif args.model in {"rf", "xgb", "lgbm"}:
        summary = train_baseline(df, args.model, seed=args.seed)
    elif args.model == "plain_mlp":
        summary = train_mlp(df, seed=args.seed, epochs=args.epochs)
    elif args.model == "pinn":
        summary = train_pinn(df, seed=args.seed, epochs=args.epochs)
    else:
        raise ValueError(args.model)

    write_run_metadata(
        REPO_ROOT / args.metadata_dir,
        config={"model": args.model, "seed": args.seed, "epochs": args.epochs},
        metrics=summary if isinstance(summary, dict) else {"results": summary},
    )
    out_json = REPO_ROOT / "results" / "tables" / f"{args.model}_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())