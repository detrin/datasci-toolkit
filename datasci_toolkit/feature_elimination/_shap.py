from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
import shap

_TREE_TYPES: tuple[type, ...] = ()
try:
    import lightgbm as lgb
    _TREE_TYPES += (lgb.LGBMClassifier, lgb.LGBMRegressor)
except ImportError:
    pass
try:
    import xgboost as xgb
    _TREE_TYPES += (xgb.XGBClassifier, xgb.XGBRegressor)
except ImportError:
    pass


def compute_shap_values(model: Any, X: pl.DataFrame) -> np.ndarray:
    X_np = X.to_numpy()
    if _TREE_TYPES and isinstance(model, _TREE_TYPES):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_np)
    else:
        background = shap.maskers.Independent(X_np, max_samples=100)
        explainer = shap.Explainer(model, background)
        sv = explainer(X_np).values
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def shap_importance(
    shap_values: np.ndarray,
    columns: list[str],
    method: Literal["mean", "variance_penalized"],
    variance_penalty_factor: float,
) -> pl.DataFrame:
    if method not in ("mean", "variance_penalized"):
        raise ValueError(method)
    abs_shap = np.abs(shap_values)
    means = abs_shap.mean(axis=0)
    stds = abs_shap.std(axis=0)
    if method == "variance_penalized":
        importance = means - variance_penalty_factor * stds
    else:
        importance = means
    return (
        pl.DataFrame({"feature": columns, "importance": importance, "std": stds})
        .sort("importance", descending=True)
    )
