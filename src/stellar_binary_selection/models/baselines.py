"""Baseline tree models and wrappers (TODO M13)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

__all__ = ["make_random_forest", "make_xgboost", "make_lightgbm"]


def make_random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
    )


def make_xgboost(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
        tree_method="hist",
        eval_metric="logloss",
    )


def make_lightgbm(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        n_jobs=-1,
        random_state=seed,
    )