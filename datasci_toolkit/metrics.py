from __future__ import annotations

import matplotlib.pyplot as plt
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


def gini_by_period(
    y: pl.Series,
    y_pred: pl.Series,
    periods: pl.Series,
    *,
    mask: pl.Series | None = None,
    sample_weight: pl.Series | None = None,
) -> pl.DataFrame:
    yt = y.cast(pl.Float64).to_numpy()
    yp = y_pred.cast(pl.Float64).to_numpy()
    t = periods.to_numpy()
    m = mask.to_numpy().astype(bool) if mask is not None else np.ones(len(yt), dtype=bool)
    sw = sample_weight.cast(pl.Float64).to_numpy() if sample_weight is not None else None
    rows = []
    for period in np.sort(np.unique(t)):
        idx = m & (t == period)
        if idx.sum() < 2 or len(np.unique(yt[idx])) < 2:
            continue
        g = gini(yt[idx], yp[idx], sample_weight=sw[idx] if sw is not None else None)
        rows.append({"period": period, "gini": g, "count": int(idx.sum())})
    return pl.DataFrame(rows)


def lift_by_period(
    y: pl.Series,
    y_pred: pl.Series,
    periods: pl.Series,
    *,
    perc: float = 10.0,
    mask: pl.Series | None = None,
) -> pl.DataFrame:
    yt = y.cast(pl.Float64).to_numpy()
    yp = y_pred.cast(pl.Float64).to_numpy()
    t = periods.to_numpy()
    m = mask.to_numpy().astype(bool) if mask is not None else np.ones(len(yt), dtype=bool)
    rows = []
    for period in np.sort(np.unique(t)):
        idx = m & (t == period)
        if idx.sum() < 2 or float(yt[idx].mean()) == 0.0:
            continue
        rows.append({"period": period, "lift": lift(yt[idx], yp[idx], perc), "count": int(idx.sum())})
    return pl.DataFrame(rows)


def plot_metric_by_period(
    periods: list,
    metric_arrays: list[list[float]],
    counts: list[float],
    labels: list[str],
    *,
    title: str = "",
    ylabel: str = "Metric",
    y_lim: tuple[float, float] | None = None,
    size: tuple[int, int] = (10, 5),
    output_file: str | None = None,
    show: bool = True,
) -> None:
    fig, ax1 = plt.subplots(figsize=size)
    x = np.arange(len(periods))
    ax1.bar(x, counts, color="lightgray", zorder=2)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, rotation=45, ha="right")
    ax1.grid(zorder=1)
    ax2 = ax1.twinx()
    for arr, label in zip(metric_arrays, labels):
        ax2.plot(x, arr, linewidth=2.5, marker="o", markersize=4, label=label, zorder=5)
    ax2.set_ylabel(ylabel, fontsize=11)
    if y_lim is not None:
        ax2.set_ylim(*y_lim)
    ax2.legend(loc="best")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


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
