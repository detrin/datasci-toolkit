from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WeightedTFIDF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        zscore_thresh: float = 2.0,
        score_threshold: float = 0.1,
        weight_col: str | None = None,
        level_col: str | None = None,
    ) -> None:
        self.zscore_thresh = zscore_thresh
        self.score_threshold = score_threshold
        self.weight_col = weight_col
        self.level_col = level_col

    def fit(
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
        y: None = None,
    ) -> WeightedTFIDF:
        if entity_col is None:
            raise ValueError("entity_col is required")
        if tag_col is None:
            raise ValueError("tag_col is required")
        if value_col is None:
            raise ValueError("value_col is required")

        self.entity_col_ = entity_col
        self.tag_col_ = tag_col
        self.value_col_ = value_col

        self.n_entities_ = X[entity_col].n_unique()

        corpus_freq = (
            X.select(entity_col, tag_col)
            .unique()
            .group_by(tag_col)
            .agg(pl.len().alias("corpus_count"))
        )
        n = self.n_entities_
        self.idf_ = corpus_freq.with_columns(
            (pl.col("corpus_count").cast(pl.Float64).map_batches(
                lambda s: pl.Series(np.abs(np.log10(n / (1 + s.to_numpy())))),
                return_dtype=pl.Float64,
            )).alias("idf")
        )
        return self

    def transform(
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        entity_col = entity_col or self.entity_col_
        tag_col = tag_col or self.tag_col_
        value_col = value_col or self.value_col_

        wc = self.weight_col
        lc = self.level_col

        if wc and wc in X.columns:
            df = X.with_columns((pl.col(wc) * pl.col(value_col)).alias("_wv"))
        else:
            df = X.with_columns(pl.col(value_col).cast(pl.Float64).alias("_wv"))

        tf_num = df.group_by([entity_col, tag_col]).agg(
            pl.col("_wv").sum().alias("_in_doc"),
        )
        if lc and lc in X.columns:
            level_agg = df.group_by([entity_col, tag_col]).agg(
                pl.col(lc).first().alias("_level"),
            )
            tf_num = tf_num.join(level_agg, on=[entity_col, tag_col])
        else:
            tf_num = tf_num.with_columns(pl.lit(1.0).alias("_level"))

        doc_len = df.group_by(entity_col).agg(pl.col("_wv").sum().alias("_doc_len"))
        tf = tf_num.join(doc_len, on=entity_col).with_columns(
            (pl.col("_in_doc") / pl.col("_doc_len")).alias("_tf")
        )

        scored = tf.join(self.idf_, on=tag_col).with_columns(
            (pl.col("_level") * pl.col("_tf") * pl.col("idf")).alias("score")
        )

        mu = scored.group_by(entity_col).agg(pl.col("score").mean().alias("_mu"))
        sigma = scored.group_by(entity_col).agg(pl.col("score").std(ddof=0).alias("_sigma"))

        scored = scored.join(mu, on=entity_col).join(sigma, on=entity_col)
        scored = scored.with_columns(
            pl.when(pl.col("_sigma") == 0)
            .then(pl.lit(3.0))
            .otherwise((pl.col("score") - pl.col("_mu")) / pl.col("_sigma"))
            .alias("_zscore")
        )

        dominant = scored.filter(pl.col("_zscore") > self.zscore_thresh).with_columns(
            pl.lit(1.0).alias("final_score")
        )

        normal = scored.filter(pl.col("_zscore").abs() <= self.zscore_thresh)
        min_max = normal.group_by(entity_col).agg(
            pl.col("score").max().alias("_smax"),
            pl.col("score").min().alias("_smin"),
        )
        normal = normal.join(min_max, on=entity_col).with_columns(
            pl.when(pl.col("_smax") == pl.col("_smin"))
            .then(pl.lit(1.0))
            .otherwise(
                (pl.col("score") - pl.col("_smin")) / (pl.col("_smax") - pl.col("_smin"))
            )
            .alias("final_score")
        )
        normal = normal.filter(pl.col("final_score") >= self.score_threshold)

        return pl.concat([
            dominant.select(entity_col, tag_col, "final_score"),
            normal.select(entity_col, tag_col, "final_score"),
        ])

    def fit_transform(
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
        y: None = None,
    ) -> pl.DataFrame:
        return self.fit(X, entity_col, tag_col, value_col).transform(X, entity_col, tag_col, value_col)
