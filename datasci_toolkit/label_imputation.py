from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted


def _dist_weights(distances: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w = 1.0 / (distances + eps)
    return w / w.sum(axis=1, keepdims=True)


class TargetImputer(BaseEstimator):
    """Converts predicted probabilities into training rows.

    Args:
        method: ``"weighted"`` duplicates each row into ``(target=1, w=p)`` +
            ``(target=0, w=1-p)``; ``"randomized"`` draws a Bernoulli sample;
            ``"cutoff"`` applies a hard threshold.
        cutoff: Threshold used when ``method="cutoff"``.
        seed: Random seed for ``method="randomized"``.

    Attributes:
        proba_: Probability array stored after `fit`.
        weights_: Sample weight array stored after `fit`.
    """

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
        probabilities = np.asarray(proba, dtype=float)
        sample_weights = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(probabilities))
        if self.method == "weighted":
            self.targets_: np.ndarray = np.concatenate([np.ones(len(probabilities)), np.zeros(len(probabilities))])
            self.weights_: np.ndarray = np.concatenate([sample_weights * probabilities, sample_weights * (1.0 - probabilities)])
        elif self.method == "randomized":
            rng = np.random.default_rng(self.seed)
            self.targets_ = (probabilities > rng.uniform(size=len(probabilities))).astype(float)
            self.weights_ = sample_weights.copy()
        elif self.method == "cutoff":
            self.targets_ = (probabilities > self.cutoff).astype(float)
            self.weights_ = sample_weights.copy()
        else:
            raise ValueError(f"method must be 'weighted', 'randomized', or 'cutoff', got {self.method!r}")
        return self

    def transform(self) -> pl.DataFrame:
        check_is_fitted(self)
        return pl.DataFrame({"target": self.targets_.tolist(), "weight": self.weights_.tolist()})


class KNNLabelImputer(BaseEstimator):
    """KNN-based label imputation for missing-outcome records.

    Finds k nearest labeled neighbours in feature space for each unlabeled
    record. Distance-weighted average of neighbour labels gives P(event).
    ``transform()`` converts these probabilities to training rows via
    ``TargetImputer``.

    Args:
        n_neighbors: Number of nearest neighbours.
        metric: Distance metric (any sklearn-compatible string).
        method: Passed to `TargetImputer` — how probabilities become rows.
        cutoff: Threshold for ``method="cutoff"``.
        seed: Random seed for ``method="randomized"``.

    Attributes:
        nn_: Fitted `NearestNeighbors` instance.
        y_: Labeled target array.
        weights_: Sample weights for labeled records.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "minkowski",
        method: str = "weighted",
        cutoff: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.method = method
        self.cutoff = cutoff
        self.seed = seed

    def fit(
        self,
        X_labeled: pl.DataFrame,
        y_labeled: pl.Series,
        weights_labeled: pl.Series | None = None,
    ) -> "KNNLabelImputer":
        X_np = X_labeled.to_numpy().astype(float)
        self.y_labeled_: np.ndarray = y_labeled.cast(pl.Float64).to_numpy()
        self.w_labeled_: np.ndarray = (
            weights_labeled.cast(pl.Float64).to_numpy()
            if weights_labeled is not None
            else np.ones(len(self.y_labeled_))
        )
        k = min(self.n_neighbors, len(self.y_labeled_))
        self.nn_: NearestNeighbors = NearestNeighbors(n_neighbors=k, metric=self.metric).fit(X_np)
        return self

    def predict_proba(self, X_unlabeled: pl.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        X_np = X_unlabeled.to_numpy().astype(float)
        distances, indices = self.nn_.kneighbors(X_np)
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]
        w_dist = _dist_weights(distances)
        combined = w_dist * self.w_labeled_[indices]
        denom = combined.sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        return (combined * self.y_labeled_[indices]).sum(axis=1) / denom

    def transform(
        self,
        X_unlabeled: pl.DataFrame,
        weights_unlabeled: pl.Series | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        proba = self.predict_proba(X_unlabeled)
        w = (
            weights_unlabeled.cast(pl.Float64).to_numpy()
            if weights_unlabeled is not None
            else np.ones(len(proba))
        )
        return TargetImputer(method=self.method, cutoff=self.cutoff, seed=self.seed).fit(proba, w).transform()
