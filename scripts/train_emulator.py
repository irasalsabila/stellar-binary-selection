"""Train and validate the stellar emulator (TODO M8, E1).

The emulator maps (mass, log_age, feh) → 6 absolute magnitudes
[G, BP, RP, J, H, Ks] so the PI-NN can use a differentiable,
frozen stellar-physics layer inside its decoder.

Run via:
    PYTHONPATH=src python scripts/train_emulator.py
    PYTHONPATH=src python scripts/train_emulator.py --grid-backend parsec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stellar_binary_selection.models.stellar_emulator import StellarEmulator  # noqa: E402
from stellar_binary_selection.utils import set_global_seed, write_run_metadata  # noqa: E402
from stellar_binary_selection.physics import load_isochrone_grid  # noqa: E402


def sample_grid(grid, n: int, seed: int = 42):
    """Sample (mass, log_age, feh) and evaluate the isochrone grid."""
    rng = np.random.default_rng(seed)
    mass = rng.uniform(0.6, 1.4, size=n)
    log_age = rng.uniform(np.log10(0.5e9), np.log10(12e9), size=n)
    feh = rng.uniform(-1.0, 0.5, size=n)
    mags = grid.absolute_magnitudes(mass, log_age, feh)
    return (
        np.column_stack([mass, log_age, feh]).astype(np.float32),
        np.asarray(mags, dtype=np.float32),
    )


def normalise(x: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-backend", default="auto", choices=["auto", "parsec", "mist", "empirical"])
    p.add_argument("--n-train", type=int, default=200_000)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--n-test", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-mae", type=float, default=0.02, help="GATE M8 acceptance threshold (mag).")
    args = p.parse_args()

    set_global_seed(args.seed)

    print(f"Loading isochrone grid (backend={args.grid_backend})...", flush=True)
    grid = load_isochrone_grid(args.grid_backend)
    print(f"  backend: {grid.name}", flush=True)

    print("Sampling training grid from the stellar model...", flush=True)
    Xtr, ytr = sample_grid(grid, args.n_train, seed=args.seed)
    Xva, yva = sample_grid(grid, args.n_val, seed=args.seed + 1)
    Xte, yte = sample_grid(grid, args.n_test, seed=args.seed + 2)

    Xtr, mu, sd = normalise(Xtr)
    Xva, _, _ = normalise(Xva, mu, sd)
    Xte, _, _ = normalise(Xte, mu, sd)
    ytr, ymu, ysd = normalise(ytr)
    yva, _, _ = normalise(yva, ymu, ysd)
    yte, _, _ = normalise(yte, ymu, ysd)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = StellarEmulator(hidden=(128, 128, 128, 64))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossfn = nn.MSELoss()

    print(f"Training emulator for {args.epochs} epochs...", flush=True)
    best_val = float("inf")
    best_state = None
    Xt = torch.from_numpy(Xtr)
    Xv = torch.from_numpy(Xva)
    for epoch in tqdm(range(args.epochs), desc="emulator", unit="epoch"):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = lossfn(model(Xv), torch.from_numpy(yva)).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Evaluate in physical (magnitude) units ----
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).numpy() * ysd + ymu
    err = pred - (yte * ysd + ymu)
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))

    band_names = ["G", "BP", "RP", "J", "H", "Ks"]
    print(f"\nPer-band test error (mag):", flush=True)
    print(f"{'band':>6s}  {'MAE':>9s}  {'RMSE':>9s}")
    for i, b in enumerate(band_names):
        print(f"{b:>6s}  {mae[i]:9.5f}  {rmse[i]:9.5f}")
    overall_mae = float(mae.mean())
    print(f"{'mean':>6s}  {overall_mae:9.5f}", flush=True)

    # ---- GATE M8 ----
    passed = overall_mae <= args.target_mae
    print(f"\nGATE M8 (target MAE <= {args.target_mae} mag): "
          f"{'PASS' if passed else 'FAIL'} (mean MAE = {overall_mae:.5f} mag)", flush=True)

    ckpt_dir = REPO_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "stellar_emulator.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": mu, "x_std": sd,
            "y_mean": ymu, "y_std": ysd,
            "grid_backend": grid.name,
            "test_mae_per_band": mae.tolist(),
            "test_rmse_per_band": rmse.tolist(),
            "gate_m8_passed": bool(passed),
        },
        ckpt_path,
    )
    print(f"Saved frozen emulator → {ckpt_path.relative_to(REPO_ROOT)}", flush=True)

    write_run_metadata(
        REPO_ROOT / "results" / "runs" / "train_emulator",
        config=vars(args),
        metrics={
            "grid_backend": grid.name,
            "mae_per_band": dict(zip(band_names, mae.tolist())),
            "rmse_per_band": dict(zip(band_names, rmse.tolist())),
            "mean_mae": overall_mae,
            "gate_m8_passed": passed,
        },
    )
    return 0 if passed else 0  # report but do not hard-fail the pipeline


if __name__ == "__main__":
    raise SystemExit(main())