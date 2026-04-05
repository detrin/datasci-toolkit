from __future__ import annotations

import math
from typing import Any, Literal

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from datasci_toolkit.feature_elimination.importance import ShapImportance


class ShapRFE(BaseEstimator):
    def __init__(
        self,
        model: Any = None,
        step: int | float = 1,
        min_features_to_select: int = 1,
        cv: int | Any = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int | None = None,
        importance_method: Literal["mean", "variance_penalized"] = "mean",
        variance_penalty_factor: float = 0.5,
        columns_to_keep: list[str] | None = None,
    ) -> None:
        self.model = model
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_method = importance_method
        self.variance_penalty_factor = variance_penalty_factor
        self.columns_to_keep = columns_to_keep

    def _n_features_to_remove(self, n_current: int) -> int:
        if isinstance(self.step, float):
            n_remove = max(1, math.floor(n_current * self.step))
        else:
            n_remove = self.step
        return min(n_remove, n_current - self.min_features_to_select)

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ShapRFE:
        remaining = list(X.columns)
        keep = set(self.columns_to_keep or [])
        records: list[dict[str, Any]] = []
        round_num = 0

        while len(remaining) > self.min_features_to_select:
            round_num += 1
            imp = ShapImportance(
                model=self.model, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state,
                importance_method=self.importance_method,
                variance_penalty_factor=self.variance_penalty_factor,
            )
            imp.fit(X.select(remaining), y)

            n_remove = self._n_features_to_remove(len(remaining))
            if n_remove <= 0:
                records.append({
                    "round": round_num, "n_features": len(remaining),
                    "features": list(remaining), "eliminated": [],
                    "train_score_mean": imp.train_score_mean_, "train_score_std": imp.train_score_std_,
                    "val_score_mean": imp.val_score_mean_, "val_score_std": imp.val_score_std_,
                })
                break

            ranked = imp.feature_importances_["feature"].to_list()
            removable = [f for f in reversed(ranked) if f not in keep]
            eliminated = removable[:n_remove]

            records.append({
                "round": round_num, "n_features": len(remaining),
                "features": list(remaining), "eliminated": eliminated,
                "train_score_mean": imp.train_score_mean_, "train_score_std": imp.train_score_std_,
                "val_score_mean": imp.val_score_mean_, "val_score_std": imp.val_score_std_,
            })
            remaining = [f for f in remaining if f not in set(eliminated)]

        if not records or records[-1]["n_features"] != len(remaining):
            last_imp = ShapImportance(
                model=self.model, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state,
                importance_method=self.importance_method,
                variance_penalty_factor=self.variance_penalty_factor,
            )
            last_imp.fit(X.select(remaining), y)
            records.append({
                "round": round_num + 1, "n_features": len(remaining),
                "features": list(remaining), "eliminated": [],
                "train_score_mean": last_imp.train_score_mean_, "train_score_std": last_imp.train_score_std_,
                "val_score_mean": last_imp.val_score_mean_, "val_score_std": last_imp.val_score_std_,
            })

        self.report_df_ = pl.DataFrame(records)
        self.feature_names_ = self.get_reduced_features("best")
        return self

    def compute(self) -> pl.DataFrame:
        check_is_fitted(self)
        return self.report_df_

    def get_reduced_features(
        self,
        method: Literal["best", "best_coherent", "best_parsimonious"] = "best",
        se_threshold: float = 1.0,
    ) -> list[str]:
        if method not in ("best", "best_coherent", "best_parsimonious"):
            raise ValueError(method)
        check_is_fitted(self, ["report_df_"])
        df = self.report_df_
        best_idx = int(df["val_score_mean"].arg_max())  # type: ignore[arg-type]
        best_score = df["val_score_mean"][best_idx]
        best_std = df["val_score_std"][best_idx]
        threshold = best_score - se_threshold * best_std

        if method == "best":
            return list(df["features"][best_idx])

        within = df.filter(pl.col("val_score_mean") >= threshold)
        if method == "best_coherent":
            idx = int(within["n_features"].arg_max())  # type: ignore[arg-type]
            return list(within["features"][idx])

        idx = int(within["n_features"].arg_min())  # type: ignore[arg-type]
        return list(within["features"][idx])
