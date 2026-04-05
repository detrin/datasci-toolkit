from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils.validation import check_is_fitted

from datasci_toolkit.feature_elimination._shap import compute_shap_values, shap_importance


def _fold_shap(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    scorer: Any,
) -> tuple[np.ndarray, float, float]:
    cloned = clone(model)
    cloned.fit(X[train_idx], y[train_idx])
    X_val_df = pl.DataFrame({c: X[val_idx, i] for i, c in enumerate(columns)})
    shap_vals = compute_shap_values(cloned, X_val_df)
    train_score = scorer(cloned, X[train_idx], y[train_idx])
    val_score = scorer(cloned, X[val_idx], y[val_idx])
    return shap_vals, train_score, val_score


class ShapImportance(BaseEstimator):
    def __init__(
        self,
        model: Any = None,
        cv: int | Any = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int | None = None,
        importance_method: Literal["mean", "variance_penalized"] = "mean",
        variance_penalty_factor: float = 0.5,
    ) -> None:
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_method = importance_method
        self.variance_penalty_factor = variance_penalty_factor

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ShapImportance:
        columns = X.columns
        X_np = X.to_numpy().astype(np.float64)
        y_np = y.to_numpy().astype(np.float64)
        scorer = check_scoring(self.model, scoring=self.scoring)
        if isinstance(self.cv, int) and self.random_state is not None:
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = check_cv(self.cv, y_np, classifier=True)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fold_shap)(self.model, X_np, y_np, columns, train_idx, val_idx, scorer)
            for train_idx, val_idx in cv.split(X_np, y_np)
        )

        all_shap = np.concatenate([r[0] for r in results], axis=0)
        train_scores = np.array([r[1] for r in results])
        val_scores = np.array([r[2] for r in results])

        self.feature_importances_ = shap_importance(
            all_shap, columns, self.importance_method, self.variance_penalty_factor,
        )
        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_ = float(train_scores.std())
        self.val_score_mean_ = float(val_scores.mean())
        self.val_score_std_ = float(val_scores.std())
        return self

    def compute(self) -> pl.DataFrame:
        check_is_fitted(self)
        return self.feature_importances_
