from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PoissonSmoother(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size: int = 7,
        alpha: float = 0.1,
        eps_left: int = 20,
        eps_right: int = 5,
    ) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self.eps_left = eps_left
        self.eps_right = eps_right

    def fit(self, X: pl.DataFrame | None = None, y: None = None) -> PoissonSmoother:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.fitted_ = True
        return self

    def transform(
        self,
        X: pl.DataFrame,
        entity_cols: list[str] | None = None,
        date_col: str | None = None,
        value_col: str | None = None,
        target_date: str | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        assert entity_cols is not None
        assert date_col is not None
        assert value_col is not None
        assert target_date is not None

        today = X.filter(pl.col(date_col) == target_date)
        history = X.filter(pl.col(date_col) != target_date)

        today_agg = today.group_by(entity_cols).agg(
            pl.col(value_col).sum().alias("today_count")
        )
        hist_agg = history.group_by(entity_cols).agg(
            pl.col(value_col).sum().alias("history_sum")
        )

        n_hist = self.window_size - 1
        merged = today_agg.join(hist_agg, on=entity_cols, how="inner")
        merged = merged.with_columns(
            (pl.col("history_sum") / n_hist).alias("history_mean")
        )

        k_arr = merged["today_count"].to_numpy().astype(float)
        mu_arr = merged["history_mean"].to_numpy().astype(float)
        hist_sum_arr = merged["history_sum"].to_numpy().astype(float)

        pvals = np.where(
            k_arr <= mu_arr,
            poisson.cdf(np.floor(k_arr).astype(int) + self.eps_left, mu_arr + self.eps_left),
            1 - poisson.cdf(np.ceil(k_arr).astype(int) + self.eps_right, mu_arr + self.eps_right),
        )

        w_today = 1 - np.power(pvals, self.alpha)
        w_hist = np.power(pvals, self.alpha) / n_hist
        smoothed = w_today * k_arr + w_hist * hist_sum_arr

        result = merged.with_columns(
            pl.Series("pvalue", pvals),
            pl.Series("smoothed_count", smoothed),
        ).drop("history_sum")

        return result.filter(pl.col("smoothed_count") > 0)
