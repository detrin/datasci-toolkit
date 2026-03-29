from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

ArrayLike = np.ndarray | pl.Series


def gini(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    return 2.0 * float(roc_auc_score(y_true, y_pred, sample_weight=weights)) - 1.0


def ks(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic)


def lift(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    perc: float = 10.0,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cutoff = float(np.percentile(y_pred, perc))
    return float(y_true[y_pred <= cutoff].mean() / y_true.mean())


def iv(
    y_true: ArrayLike,
    x: ArrayLike,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    x = np.asarray(x)
    n_events = float((y_true == 1).sum()) + 1.0
    n_nonevents = float((y_true == 0).sum()) + 1.0
    result = 0.0
    for value in np.unique(x):
        mask = x == value
        bin_events = float(((y_true == 1) & mask).sum()) + 1.0
        bin_nonevents = float(((y_true == 0) & mask).sum()) + 1.0
        woe = np.log((bin_nonevents / n_nonevents) / (bin_events / n_events))
        result += woe * (bin_nonevents / n_nonevents - bin_events / n_events)
    return result


class BootstrapGini(BaseEstimator):
    """Bootstrap confidence interval for Gini.

    Args:
        n_iter: Number of bootstrap resamples.
        ci_level: Confidence level in percent (e.g. 90.0 for 90% CI).
        seed: Random seed for reproducibility.

    Attributes:
        mean_: Mean Gini across bootstrap samples.
        std_: Standard deviation of bootstrap Gini values.
        ci_: Tuple ``(lower, upper)`` confidence interval bounds.
        samples_: Array of all bootstrap Gini values.
    """

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
        y_true: ArrayLike,
        y_pred: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "BootstrapGini":
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
        rng = np.random.default_rng(self.seed)
        n = len(y_true)
        scores: list[float] = []
        for _ in range(self.n_iter):
            idx = rng.integers(0, n, size=n)
            w = weights[idx] if weights is not None else None
            scores.append(gini(y_true[idx], y_pred[idx], sample_weight=w))
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
    y_true = y.cast(pl.Float64).to_numpy()
    y_pred_arr = y_pred.cast(pl.Float64).to_numpy()
    period_arr = periods.to_numpy()
    inclusion = mask.to_numpy().astype(bool) if mask is not None else np.ones(len(y_true), dtype=bool)
    weights = sample_weight.cast(pl.Float64).to_numpy() if sample_weight is not None else None
    rows = []
    for period in np.sort(np.unique(period_arr)):
        period_mask = inclusion & (period_arr == period)
        if period_mask.sum() < 2 or len(np.unique(y_true[period_mask])) < 2:
            continue
        period_gini = gini(
            y_true[period_mask],
            y_pred_arr[period_mask],
            sample_weight=weights[period_mask] if weights is not None else None,
        )
        rows.append({"period": period, "gini": period_gini, "count": int(period_mask.sum())})
    return pl.DataFrame(rows)


def lift_by_period(
    y: pl.Series,
    y_pred: pl.Series,
    periods: pl.Series,
    *,
    perc: float = 10.0,
    mask: pl.Series | None = None,
) -> pl.DataFrame:
    y_true = y.cast(pl.Float64).to_numpy()
    y_pred_arr = y_pred.cast(pl.Float64).to_numpy()
    period_arr = periods.to_numpy()
    inclusion = mask.to_numpy().astype(bool) if mask is not None else np.ones(len(y_true), dtype=bool)
    rows = []
    for period in np.sort(np.unique(period_arr)):
        period_mask = inclusion & (period_arr == period)
        if period_mask.sum() < 2 or float(y_true[period_mask].mean()) == 0.0:
            continue
        rows.append({"period": period, "lift": lift(y_true[period_mask], y_pred_arr[period_mask], perc), "count": int(period_mask.sum())})
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
    y_true = y.cast(pl.Float64).to_numpy()
    weights: np.ndarray | None = sample_weight.cast(pl.Float64).to_numpy() if sample_weight is not None else None
    rows = []
    for col in X.columns:
        feature_arr = X[col].cast(pl.Float64).to_numpy()
        rows.append({
            "feature": col,
            "gini": round(gini(y_true, -feature_arr, sample_weight=weights), 6),
            "iv": round(iv(y_true, feature_arr), 6),
        })
    return pl.DataFrame(rows).sort("gini", descending=True)
