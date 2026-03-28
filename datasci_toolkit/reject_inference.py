from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def _dist_weights(distances: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w = 1.0 / (distances + eps)
    return w / w.sum(axis=1, keepdims=True)


class TargetImputer(BaseEstimator):
    def __init__(
        self,
        method: str = "weighted",
        cutoff: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.method = method
        self.cutoff = cutoff
        self.seed = seed

    def fit(
        self,
        proba: np.ndarray | pl.Series,
        weights: np.ndarray | pl.Series | None = None,
    ) -> "TargetImputer":
        p = np.asarray(proba, dtype=float)
        w = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(p))
        if self.method == "weighted":
            self.targets_: np.ndarray = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
            self.weights_: np.ndarray = np.concatenate([w * p, w * (1.0 - p)])
        elif self.method == "randomized":
            rng = np.random.default_rng(self.seed)
            self.targets_ = (p > rng.uniform(size=len(p))).astype(float)
            self.weights_ = w.copy()
        elif self.method == "cutoff":
            self.targets_ = (p > self.cutoff).astype(float)
            self.weights_ = w.copy()
        else:
            raise ValueError(f"method must be 'weighted', 'randomized', or 'cutoff', got {self.method!r}")
        return self

    def transform(self) -> pl.DataFrame:
        check_is_fitted(self)
        return pl.DataFrame({"target": self.targets_.tolist(), "weight": self.weights_.tolist()})


class RejectInferenceImputer(BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 10,
        method: str = "weighted",
        cutoff: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.method = method
        self.cutoff = cutoff
        self.seed = seed

    def fit(
        self,
        X_accept: pl.DataFrame,
        y_accept: pl.Series,
        weights_accept: pl.Series | None = None,
    ) -> "RejectInferenceImputer":
        X_np = X_accept.to_numpy().astype(float)
        self.y_accept_: np.ndarray = y_accept.cast(pl.Float64).to_numpy()
        self.w_accept_: np.ndarray = (
            weights_accept.cast(pl.Float64).to_numpy()
            if weights_accept is not None
            else np.ones(len(self.y_accept_))
        )
        self.tree_: KDTree = KDTree(X_np)
        return self

    def predict_proba(self, X_reject: pl.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        X_np = X_reject.to_numpy().astype(float)
        k = min(self.n_neighbors, len(self.y_accept_))
        distances, indices = self.tree_.query(X_np, k=k)
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        w_dist = _dist_weights(distances)
        combined = w_dist * self.w_accept_[indices]
        denom = combined.sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        return (combined * self.y_accept_[indices]).sum(axis=1) / denom

    def transform(
        self,
        X_reject: pl.DataFrame,
        weights_reject: pl.Series | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        proba = self.predict_proba(X_reject)
        w = (
            weights_reject.cast(pl.Float64).to_numpy()
            if weights_reject is not None
            else np.ones(len(proba))
        )
        return TargetImputer(method=self.method, cutoff=self.cutoff, seed=self.seed).fit(proba, w).transform()
