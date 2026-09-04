"""Utility helpers for deterministic experiments."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def hash_file(path: str | os.PathLike, algorithm: str = "sha256") -> str:
    """Compute a file checksum for immutability auditing."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df: Any) -> str:
    """Stable hash of a pandas DataFrame by column-deterministic byte representation."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for hash_dataframe") from exc
    if not isinstance(df, pd.DataFrame):
        raise TypeError("hash_dataframe expects a pandas DataFrame")
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str | None:
    """Return the current git short SHA, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def python_version() -> str:
    return sys.version.split()[0]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_run_metadata(
    out_dir: str | os.PathLike,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a snapshot of an experiment run to results/<id>/."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "python_version": python_version(),
    }
    if extra:
        snapshot.update(dict(extra))

    (out / "config.yaml").write_text(_dump_yaml(dict(config)))
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(dict(metrics), indent=2, default=str))
    (out / "run.json").write_text(json.dumps(snapshot, indent=2))
    return out


def _dump_yaml(obj: Mapping[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(obj, sort_keys=False)
    except ImportError:
        import json

        return json.dumps(obj, indent=2, default=str)