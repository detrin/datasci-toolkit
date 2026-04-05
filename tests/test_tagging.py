from __future__ import annotations

import polars as pl
import pytest

from datasci_toolkit.tagging import WeightedTFIDF


@pytest.fixture
def corpus_df() -> pl.DataFrame:
    return pl.DataFrame({
        "entity": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
        "tag": ["x", "y", "z", "x", "y", "x", "y", "z", "w"],
        "value": [10, 5, 1, 8, 3, 12, 6, 2, 1],
    })


@pytest.fixture
def corpus_with_weights() -> pl.DataFrame:
    return pl.DataFrame({
        "entity": ["A", "A", "A", "B", "B"],
        "tag": ["x", "y", "z", "x", "y"],
        "value": [10, 5, 1, 8, 3],
        "weight": [0.9, 0.5, 0.1, 0.8, 0.6],
        "level": [1.0, 1.0, 0.5, 1.0, 1.0],
    })


class TestWeightedTFIDFInit:
    def test_default_params(self) -> None:
        t = WeightedTFIDF()
        assert t.zscore_thresh == 2.0
        assert t.score_threshold == 0.1
        assert t.weight_col is None
        assert t.level_col is None

    def test_custom_params(self) -> None:
        t = WeightedTFIDF(zscore_thresh=3.0, score_threshold=0.2, weight_col="w", level_col="l")
        assert t.zscore_thresh == 3.0
        assert t.weight_col == "w"


class TestWeightedTFIDFFit:
    def test_fit_returns_self(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF()
        result = t.fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert result is t

    def test_fit_stores_n(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert t.n_entities_ == 3

    def test_fit_stores_idf(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert "tag" in t.idf_.columns
        assert "idf" in t.idf_.columns
        assert len(t.idf_) == 4

    def test_idf_values(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        idf = t.idf_.sort("tag")
        w_idf = idf.filter(pl.col("tag") == "w")["idf"][0]
        x_idf = idf.filter(pl.col("tag") == "x")["idf"][0]
        assert w_idf > x_idf


class TestWeightedTFIDFTransform:
    def test_transform_before_fit_raises(self) -> None:
        t = WeightedTFIDF()
        with pytest.raises(Exception):
            t.transform(pl.DataFrame(), entity_col="e", tag_col="t", value_col="v")

    def test_output_columns(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert set(result.columns) == {"entity", "tag", "final_score"}

    def test_scores_between_0_and_1(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert result["final_score"].min() >= 0.0
        assert result["final_score"].max() <= 1.0

    def test_score_threshold_filters(self, corpus_df: pl.DataFrame) -> None:
        t_low = WeightedTFIDF(score_threshold=0.01).fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        t_high = WeightedTFIDF(score_threshold=0.5).fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        r_low = t_low.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        r_high = t_high.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert len(r_high) <= len(r_low)

    def test_with_weight_and_level(self, corpus_with_weights: pl.DataFrame) -> None:
        t = WeightedTFIDF(weight_col="weight", level_col="level")
        t.fit(corpus_with_weights, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_with_weights, entity_col="entity", tag_col="tag", value_col="value")
        assert len(result) > 0
        assert result["final_score"].min() >= 0.0

    def test_single_tag_entity_dominant(self) -> None:
        df = pl.DataFrame({
            "entity": ["A", "A", "B"],
            "tag": ["x", "y", "x"],
            "value": [10, 5, 10],
        })
        t = WeightedTFIDF().fit(df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(df, entity_col="entity", tag_col="tag", value_col="value")
        b_row = result.filter(pl.col("entity") == "B")
        assert len(b_row) == 1
        assert b_row["final_score"][0] == 1.0

    def test_fit_transform(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF()
        result = t.fit_transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert len(result) > 0
        assert "final_score" in result.columns
