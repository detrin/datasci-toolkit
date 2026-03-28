from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def gini(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
    sample_weight: np.ndarray | pl.Series | None = None,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    return 2.0 * float(roc_auc_score(yt, yp, sample_weight=sw)) - 1.0


def ks(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(ks_2samp(yp[yt == 1], yp[yt == 0]).statistic)


def lift(
    y_true: np.ndarray | pl.Series,
    y_pred: np.ndarray | pl.Series,
    perc: float = 10.0,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    cutoff = float(np.percentile(yp, perc))
    return float(yt[yp <= cutoff].mean() / yt.mean())


def iv(
    y_true: np.ndarray | pl.Series,
    x: np.ndarray | pl.Series,
) -> float:
    yt = np.asarray(y_true, dtype=float)
    xv = np.asarray(x)
    n_ev = float((yt == 1).sum()) + 1.0
    n_nev = float((yt == 0).sum()) + 1.0
    result = 0.0
    for v in np.unique(xv):
        mask = xv == v
        ev = float(((yt == 1) & mask).sum()) + 1.0
        nev = float(((yt == 0) & mask).sum()) + 1.0
        woe = np.log((nev / n_nev) / (ev / n_ev))
        result += woe * (nev / n_nev - ev / n_ev)
    return result


class BootstrapGini(BaseEstimator):
    def __init__(
        self,
        n_iter: int = 100,
        ci_level: float = 90.0,
        seed: int | None = None,
    ) -> None:
        self.n_iter = n_iter
        self.ci_level = ci_level
        self.seed = seed

    def fit(
        self,
        y_true: np.ndarray | pl.Series,
        y_pred: np.ndarray | pl.Series,
        sample_weight: np.ndarray | pl.Series | None = None,
    ) -> "BootstrapGini":
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
        rng = np.random.default_rng(self.seed)
        n = len(yt)
        scores: list[float] = []
        for _ in range(self.n_iter):
            idx = rng.integers(0, n, size=n)
            w = sw[idx] if sw is not None else None
            scores.append(gini(yt[idx], yp[idx], sample_weight=w))
        alpha = (100.0 - self.ci_level) / 2.0
        self.mean_: float = float(np.mean(scores))
        self.std_: float = float(np.std(scores))
        self.ci_: tuple[float, float] = (
            float(np.percentile(scores, alpha)),
            float(np.percentile(scores, 100.0 - alpha)),
        )
        self.scores_: np.ndarray = np.array(scores)
        return self


def feature_power(
    X: pl.DataFrame,
    y: pl.Series,
    sample_weight: pl.Series | None = None,
) -> pl.DataFrame:
    yt = y.cast(pl.Float64).to_numpy()
    sw: np.ndarray | None = sample_weight.cast(pl.Float64).to_numpy() if sample_weight is not None else None
    rows = []
    for col in X.columns:
        xp = X[col].cast(pl.Float64).to_numpy()
        rows.append({
            "feature": col,
            "gini": round(gini(yt, -xp, sample_weight=sw), 6),
            "iv": round(iv(yt, xp), 6),
        })
    return pl.DataFrame(rows).sort("gini", descending=True)
