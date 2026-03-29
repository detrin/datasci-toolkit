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
    def __init__(self, q: int = 10, missing_value: float = 0.0001):
        self.q = q
        self.missing_value = missing_value

    def _bin(self, X: pl.Series) -> pl.Series:
        return X.cast(pl.Float64).cut(self.bin_breaks_).cast(pl.String).fill_null("missing")

    def fit(self, X: object, weights: object = None) -> "PSI":
        X = _to_series(X, "X")
        w = _to_series(weights, "weights") if weights is not None else pl.Series(np.ones(len(X)))
        if X.dtype.is_numeric():
            cuts = np.unique(np.percentile(X.drop_nulls().to_numpy(), np.linspace(0, 100, self.q + 1)))
            self.bin_breaks_ = list(cuts[1:-1])
            X = self._bin(X)
        self.ref_dist_ = _weighted_dist(X, w, self.missing_value)
        return self

    def score(self, X: object, weights: object = None) -> float:
        check_is_fitted(self)
        X = _to_series(X, "X")
        w = _to_series(weights, "weights") if weights is not None else pl.Series(np.ones(len(X)))
        if hasattr(self, "bin_breaks_"):
            X = self._bin(X)
        act = _weighted_dist(X, w, self.missing_value)
        df = self.ref_dist_.join(act.rename({"freq": "freq_act"}), on="cat", how="inner")
        return float(((df["freq"] - df["freq_act"]) * (df["freq"] / df["freq_act"]).log()).sum())


class ESI:
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
            tmp = base.group_by([col_month, var]).agg(
                pl.col(col_target).sum(), pl.col(col_base).sum()
            )
        else:
            tmp = (
                base.with_columns([
                    (pl.col(col_target) * pl.col(col_weight)).alias(col_target),
                    (pl.col(col_base) * pl.col(col_weight)).alias(col_base),
                ])
                .group_by([col_month, var])
                .agg(pl.col(col_target).sum(), pl.col(col_base).sum())
            )

        if exclude_zero:
            tmp = tmp.filter(pl.col(var) != 0.0)

        tmp = tmp.with_columns(
            (pl.col(col_target) / pl.col(col_base)).alias("bad_rate")
        ).with_columns(
            pl.col("bad_rate").rank(method="dense", descending=True).over(col_month).alias("group_rank")
        )

        tmp = tmp.group_by([var, "group_rank"]).agg(pl.len().alias("rank_count"))

        totals = tmp.group_by("group_rank").agg(pl.col("rank_count").sum().alias("total"))
        tmp2 = (
            tmp.join(totals, on="group_rank")
            .with_columns((pl.col("rank_count") / pl.col("total")).alias("ratio"))
        )
        v2 = float(
            tmp2.group_by("group_rank")
            .agg(pl.col("ratio").product())
            .select(pl.col("ratio").mean())
            .item()
        )

        agg = tmp.group_by(var).agg(
            pl.col("rank_count").sum().alias("sum"),
            pl.col("rank_count").max().alias("max"),
        )
        v1 = float(
            agg.with_columns((pl.col("max") / pl.col("sum")).alias("ratio"))
            .select(pl.col("ratio").mean())
            .item()
        )

        return {"v1": v1, "v2": v2}


class StabilityMonitor(BaseEstimator):
    def __init__(self, features: list, q: int = 10, missing_value: float = 0.0001, col_weight: str | None = None):
        self.features = features
        self.q = q
        self.missing_value = missing_value
        self.col_weight = col_weight

    def _w(self, subset: pl.DataFrame) -> pl.Series | None:
        return subset[self.col_weight] if self.col_weight else None

    def _psi(self) -> PSI:
        return PSI(q=self.q, missing_value=self.missing_value)

    def fit(self, df: pl.DataFrame, y: object = None) -> "StabilityMonitor":
        self.estimators_ = {
            feat: self._psi().fit(df[feat], self._w(df))
            for feat in self.features
        }
        return self

    def _score_month(self, df: pl.DataFrame, feat: str, est: PSI, m: object, col_month: str) -> dict:
        subset = df.filter(pl.col(col_month) == m)
        return {"feature": feat, "month": m, "psi": est.score(subset[feat], self._w(subset))}

    def score(self, df: pl.DataFrame, col_month: str) -> pl.DataFrame:
        check_is_fitted(self)
        months = sorted(df[col_month].unique().to_list())
        return pl.DataFrame([
            self._score_month(df, feat, est, m, col_month)
            for feat, est in self.estimators_.items()
            for m in months
        ])

    def score_consecutive(self, df: pl.DataFrame, col_month: str) -> pl.DataFrame:
        check_is_fitted(self)
        months = sorted(df[col_month].unique().to_list())
        records = []
        for feat in self.features:
            for m1, m2 in zip(months, months[1:]):
                sub1 = df.filter(pl.col(col_month) == m1)
                sub2 = df.filter(pl.col(col_month) == m2)
                tmp = self._psi().fit(sub1[feat], self._w(sub1))
                records.append({"feature": feat, "months": f"{m1}:{m2}", "psi": tmp.score(sub2[feat], self._w(sub2))})
        return pl.DataFrame(records)

    def score_masks(self, df: pl.DataFrame, mask_dict: dict) -> pl.DataFrame:
        check_is_fitted(self)
        records = []
        for feat in self.features:
            for mn1, mn2 in itertools.combinations(mask_dict, 2):
                sub1 = df.filter(mask_dict[mn1])
                sub2 = df.filter(mask_dict[mn2])
                tmp = self._psi().fit(sub1[feat], self._w(sub1))
                records.append({"feature": feat, "mask": f"{mn1}:{mn2}", "psi": tmp.score(sub2[feat], self._w(sub2))})
        return pl.DataFrame(records).sort("psi", descending=True)


def plot_psi_comparison(months: list, psi_values: list, labels: list, title: str = "PSI", size: tuple = (12, 8), output_folder: str | None = None, show: bool = True) -> None:
    n = len(psi_values)
    X = np.linspace(0, len(months), len(months))
    X1 = np.linspace(0, len(months) + 1, len(months) + 1)
    plt.figure(figsize=size)
    plt.grid(zorder=0)
    for i, arr in enumerate(psi_values):
        plt.bar(X - (1 / n) * i, arr, width=1 / n, label=labels[i], zorder=3)
    plt.plot(X1, [0.1] * len(X1), "black")
    plt.plot(X1, [0.25] * len(X1), "r")
    plt.title(title, fontsize=18)
    plt.xticks(X, months, rotation=45)
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
    psi = PSI(q=bins)
    results = []
    for s in scores:
        ref = data.filter((pl.col(month_col) == months[pivot]) & pl.col(s).is_not_null())[s]
        psi.fit(ref)
        results.append([
            psi.score(data.filter((pl.col(month_col) == m) & pl.col(s).is_not_null())[s])
            for m in months
        ])
    plot_psi_comparison(months, results, score_names or [str(i) for i in range(len(scores))], title, output_folder=output_folder, show=show)
