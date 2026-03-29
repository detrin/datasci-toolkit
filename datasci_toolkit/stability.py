import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


def _to_series(x: object, name: str) -> pl.Series:
    if isinstance(x, pl.Series):
        return x
    try:
        warnings.warn(f"{name} is not pl.Series, converting.")
        return pl.Series(x)
    except Exception:
        raise TypeError(f"{name} cannot be converted to pl.Series")


def _weighted_dist(X: pl.Series, weights: pl.Series, missing_value: float) -> pl.DataFrame:
    dist = (
        pl.DataFrame({"cat": X, "w": weights})
        .group_by("cat")
        .agg(pl.col("w").sum())
    )
    total = dist["w"].sum()
    return (
        dist
        .with_columns((pl.col("w") / total).alias("freq"))
        .with_columns(
            pl.when(pl.col("freq").is_null() | (pl.col("freq") == 0))
            .then(pl.lit(missing_value))
            .otherwise(pl.col("freq"))
            .alias("freq")
        )
        .drop("w")
    )


class PSI(BaseEstimator):
    """Population Stability Index.

    Measures distributional shift between a reference dataset and a monitoring
    dataset. Fit on the reference, call `score` on any subsequent snapshot.

    Args:
        n_quantile_bins: Number of quantile bins for numeric features.
        missing_value: Frequency floor applied to empty bins to avoid log(0).

    Attributes:
        bin_breaks_: Quantile cut points fitted on the reference (numeric only).
        ref_dist_: Reference frequency distribution as a DataFrame.
    """

    def __init__(self, n_quantile_bins: int = 10, missing_value: float = 0.0001):
        self.n_quantile_bins = n_quantile_bins
        self.missing_value = missing_value

    def _bin(self, series: pl.Series) -> pl.Series:
        return series.cast(pl.Float64).cut(self.bin_breaks_).cast(pl.String).fill_null("missing")

    def fit(self, X: object, weights: object = None) -> "PSI":
        series = _to_series(X, "X")
        weights_series = _to_series(weights, "weights") if weights is not None else pl.Series(np.ones(len(series)))
        if series.dtype.is_numeric():
            cuts = np.unique(np.percentile(series.drop_nulls().to_numpy(), np.linspace(0, 100, self.n_quantile_bins + 1)))
            self.bin_breaks_ = list(cuts[1:-1])
            series = self._bin(series)
        self.ref_dist_ = _weighted_dist(series, weights_series, self.missing_value)
        return self

    def score(self, X: object, weights: object = None) -> float:
        check_is_fitted(self)
        series = _to_series(X, "X")
        weights_series = _to_series(weights, "weights") if weights is not None else pl.Series(np.ones(len(series)))
        if hasattr(self, "bin_breaks_"):
            series = self._bin(series)
        actual_dist = _weighted_dist(series, weights_series, self.missing_value)
        df = self.ref_dist_.join(actual_dist.rename({"freq": "freq_act"}), on="cat", how="inner")
        return float(((df["freq"] - df["freq_act"]) * (df["freq"] / df["freq_act"]).log()).sum())


class ESI:
    """Event Stability Index.

    Measures rank stability of a model score across time periods. Returns two
    variants: V1 (rank-correlation based) and V2 (event-rate-ratio based).
    """

    def score(
        self,
        data: pl.DataFrame,
        var: str,
        col_target: str,
        col_base: str,
        col_month: str,
        col_weight: str | None = None,
        exclude_nan: bool = False,
        exclude_zero: bool = False,
    ) -> dict:
        base = data.filter(pl.col(col_base) == 1)
        if exclude_nan:
            base = base.filter(pl.col(var).is_not_null())

        if col_weight is None:
            monthly_stats = base.group_by([col_month, var]).agg(
                pl.col(col_target).sum(), pl.col(col_base).sum()
            )
        else:
            monthly_stats = (
                base.with_columns([
                    (pl.col(col_target) * pl.col(col_weight)).alias(col_target),
                    (pl.col(col_base) * pl.col(col_weight)).alias(col_base),
                ])
                .group_by([col_month, var])
                .agg(pl.col(col_target).sum(), pl.col(col_base).sum())
            )

        if exclude_zero:
            monthly_stats = monthly_stats.filter(pl.col(var) != 0.0)

        ranked_stats = monthly_stats.with_columns(
            (pl.col(col_target) / pl.col(col_base)).alias("bad_rate")
        ).with_columns(
            pl.col("bad_rate").rank(method="dense", descending=True).over(col_month).alias("group_rank")
        )

        rank_counts = ranked_stats.group_by([var, "group_rank"]).agg(pl.len().alias("rank_count"))

        rank_totals = rank_counts.group_by("group_rank").agg(pl.col("rank_count").sum().alias("total"))
        rank_ratios = (
            rank_counts.join(rank_totals, on="group_rank")
            .with_columns((pl.col("rank_count") / pl.col("total")).alias("ratio"))
        )
        v2 = float(
            rank_ratios.group_by("group_rank")
            .agg(pl.col("ratio").product())
            .select(pl.col("ratio").mean())
            .item()
        )

        score_dominance = rank_counts.group_by(var).agg(
            pl.col("rank_count").sum().alias("sum"),
            pl.col("rank_count").max().alias("max"),
        )
        v1 = float(
            score_dominance.with_columns((pl.col("max") / pl.col("sum")).alias("ratio"))
            .select(pl.col("ratio").mean())
            .item()
        )

        return {"v1": v1, "v2": v2}


class StabilityMonitor(BaseEstimator):
    """Monitors PSI for a set of features over time.

    Fits one `PSI` instance per feature on a reference DataFrame and exposes
    three scoring modes: against a fixed reference, consecutive period pairs,
    or arbitrary boolean masks.

    Args:
        features: Column names to monitor.
        n_quantile_bins: Quantile bins for numeric features (passed to `PSI`).
        missing_value: Frequency floor for empty bins (passed to `PSI`).
        col_weight: Optional weight column in the input DataFrame.

    Attributes:
        psis_: Dict mapping feature name to fitted `PSI` instance.
    """

    def __init__(self, features: list, n_quantile_bins: int = 10, missing_value: float = 0.0001, col_weight: str | None = None):
        self.features = features
        self.n_quantile_bins = n_quantile_bins
        self.missing_value = missing_value
        self.col_weight = col_weight

    def _get_weights(self, subset: pl.DataFrame) -> pl.Series | None:
        return subset[self.col_weight] if self.col_weight else None

    def _make_psi(self) -> PSI:
        return PSI(n_quantile_bins=self.n_quantile_bins, missing_value=self.missing_value)

    def fit(self, df: pl.DataFrame, target: object = None) -> "StabilityMonitor":
        self.estimators_ = {
            feature: self._make_psi().fit(df[feature], self._get_weights(df))
            for feature in self.features
        }
        return self

    def _score_month(self, df: pl.DataFrame, feature: str, estimator: PSI, month: object, col_month: str) -> dict:
        subset = df.filter(pl.col(col_month) == month)
        return {"feature": feature, "month": month, "psi": estimator.score(subset[feature], self._get_weights(subset))}

    def score(self, df: pl.DataFrame, col_month: str) -> pl.DataFrame:
        check_is_fitted(self)
        months = sorted(df[col_month].unique().to_list())
        return pl.DataFrame([
            self._score_month(df, feature, estimator, month, col_month)
            for feature, estimator in self.estimators_.items()
            for month in months
        ])

    def score_consecutive(self, df: pl.DataFrame, col_month: str) -> pl.DataFrame:
        check_is_fitted(self)
        months = sorted(df[col_month].unique().to_list())
        records = []
        for feature in self.features:
            for month_1, month_2 in zip(months, months[1:]):
                sub1 = df.filter(pl.col(col_month) == month_1)
                sub2 = df.filter(pl.col(col_month) == month_2)
                ref_psi = self._make_psi().fit(sub1[feature], self._get_weights(sub1))
                records.append({"feature": feature, "months": f"{month_1}:{month_2}", "psi": ref_psi.score(sub2[feature], self._get_weights(sub2))})
        return pl.DataFrame(records)

    def score_masks(self, df: pl.DataFrame, mask_dict: dict) -> pl.DataFrame:
        check_is_fitted(self)
        records = []
        for feature in self.features:
            for mn1, mn2 in itertools.combinations(mask_dict, 2):
                sub1 = df.filter(mask_dict[mn1])
                sub2 = df.filter(mask_dict[mn2])
                ref_psi = self._make_psi().fit(sub1[feature], self._get_weights(sub1))
                records.append({"feature": feature, "mask": f"{mn1}:{mn2}", "psi": ref_psi.score(sub2[feature], self._get_weights(sub2))})
        return pl.DataFrame(records).sort("psi", descending=True)


def plot_psi_comparison(months: list, psi_values: list, labels: list, title: str = "PSI", size: tuple = (12, 8), output_folder: str | None = None, show: bool = True) -> None:
    n_models = len(psi_values)
    bar_positions = np.linspace(0, len(months), len(months))
    threshold_x = np.linspace(0, len(months) + 1, len(months) + 1)
    plt.figure(figsize=size)
    plt.grid(zorder=0)
    for model_index, arr in enumerate(psi_values):
        plt.bar(bar_positions - (1 / n_models) * model_index, arr, width=1 / n_models, label=labels[model_index], zorder=3)
    plt.plot(threshold_x, [0.1] * len(threshold_x), "black")
    plt.plot(threshold_x, [0.25] * len(threshold_x), "r")
    plt.title(title, fontsize=18)
    plt.xticks(bar_positions, months, rotation=45)
    plt.xlim(0, len(months) + 0.5)
    plt.ylim(0, 0.3)
    plt.xlabel("Months", fontsize=13)
    plt.ylabel("PSI", fontsize=13)
    plt.legend()
    if output_folder:
        plt.savefig(f"{output_folder}/psi_comparison_chart.png", format="png", dpi=72, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def psi_hist(data: pl.DataFrame, scores: list, months: list, month_col: str, pivot: int = 0, score_names: list | None = None, title: str = "PSI", bins: int = 10, output_folder: str | None = None, show: bool = True) -> None:
    psi = PSI(n_quantile_bins=bins)
    results = []
    for score_col in scores:
        ref = data.filter((pl.col(month_col) == months[pivot]) & pl.col(score_col).is_not_null())[score_col]
        psi.fit(ref)
        results.append([
            psi.score(data.filter((pl.col(month_col) == month) & pl.col(score_col).is_not_null())[score_col])
            for month in months
        ])
    plot_psi_comparison(months, results, score_names or [str(i) for i in range(len(scores))], title, output_folder=output_folder, show=show)
