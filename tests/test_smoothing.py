from __future__ import annotations

import polars as pl
import pytest

from datasci_toolkit.smoothing import PoissonSmoother, PredictionSmoother


@pytest.fixture
def count_df() -> pl.DataFrame:
    rows = []
    for eid in range(1, 4):
        for d in range(1, 8):
            rows.append({"entity_id": eid, "date": f"2024-01-{d:02d}", "count": eid * 10 + d})
    return pl.DataFrame(rows)


class TestPoissonSmootherInit:
    def test_default_params(self) -> None:
        ps = PoissonSmoother(window_size=7)
        assert ps.window_size == 7
        assert ps.alpha == 0.1
        assert ps.eps_left == 20
        assert ps.eps_right == 5

    def test_custom_params(self) -> None:
        ps = PoissonSmoother(window_size=10, alpha=0.2, eps_left=15, eps_right=3)
        assert ps.alpha == 0.2
        assert ps.eps_left == 15

    def test_window_size_validation(self) -> None:
        ps = PoissonSmoother(window_size=1)
        with pytest.raises(ValueError, match="window_size"):
            ps.fit()


class TestPoissonSmootherFitTransform:
    def test_fit_returns_self(self) -> None:
        ps = PoissonSmoother(window_size=7)
        result = ps.fit()
        assert result is ps

    def test_transform_before_fit_raises(self) -> None:
        ps = PoissonSmoother(window_size=7)
        with pytest.raises(Exception):
            ps.transform(pl.DataFrame(), entity_cols=["id"], date_col="d", value_col="v", target_date="x")

    def test_transform_output_columns(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        assert "entity_id" in result.columns
        assert "today_count" in result.columns
        assert "history_mean" in result.columns
        assert "pvalue" in result.columns
        assert "smoothed_count" in result.columns

    def test_transform_row_count(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        assert len(result) == 3

    def test_smoothed_between_today_and_history(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        for row in result.iter_rows(named=True):
            assert row["smoothed_count"] > 0

    def test_anomalous_count_trusts_today(self) -> None:
        rows = []
        for d in range(1, 7):
            rows.append({"eid": "A", "date": f"2024-01-{d:02d}", "count": 50})
        rows.append({"eid": "A", "date": "2024-01-07", "count": 500})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid"], date_col="date", value_col="count", target_date="2024-01-07")
        smoothed = result["smoothed_count"][0]
        assert smoothed > 300

    def test_normal_count_smooths_toward_history(self) -> None:
        rows = []
        for d in range(1, 7):
            rows.append({"eid": "A", "date": f"2024-01-{d:02d}", "count": 50})
        rows.append({"eid": "A", "date": "2024-01-07", "count": 48})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid"], date_col="date", value_col="count", target_date="2024-01-07")
        smoothed = result["smoothed_count"][0]
        assert abs(smoothed - 50) < abs(48 - 50)

    def test_multiple_entity_cols(self) -> None:
        rows = []
        for seg in ["X", "Y"]:
            for d in range(1, 8):
                rows.append({"eid": "A", "seg": seg, "date": f"2024-01-{d:02d}", "count": 50})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid", "seg"], date_col="date", value_col="count", target_date="2024-01-07")
        assert len(result) == 2
        assert "eid" in result.columns
        assert "seg" in result.columns


@pytest.fixture
def binary_pred_df() -> pl.DataFrame:
    return pl.DataFrame({
        "eid": ["A", "A", "A", "B", "B"],
        "month": [1, 2, 3, 1, 2],
        "prob_default": [0.6, 0.4, 0.5, 0.8, 0.9],
    })


@pytest.fixture
def multiclass_pred_df() -> pl.DataFrame:
    return pl.DataFrame({
        "eid": ["A", "A", "A", "B", "B"],
        "month": [1, 2, 3, 1, 2],
        "prob_cat": [0.6, 0.3, 0.4, 0.1, 0.2],
        "prob_dog": [0.2, 0.5, 0.4, 0.8, 0.7],
        "prob_bird": [0.2, 0.2, 0.2, 0.1, 0.1],
    })


class TestPredictionSmootherInit:
    def test_default_params(self) -> None:
        ps = PredictionSmoother()
        assert ps.min_observations == 1

    def test_custom_params(self) -> None:
        ps = PredictionSmoother(min_observations=3)
        assert ps.min_observations == 3


class TestPredictionSmootherBinary:
    def test_fit_returns_self(self) -> None:
        ps = PredictionSmoother()
        assert ps.fit() is ps

    def test_binary_output_columns(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        assert "eid" in result.columns
        assert "prob_default" in result.columns
        assert "observation_count" in result.columns
        assert "predicted_label" not in result.columns

    def test_binary_averages_correctly(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        a_row = result.filter(pl.col("eid") == "A")
        assert abs(a_row["prob_default"][0] - 0.5) < 1e-6

    def test_binary_observation_count(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        a_row = result.filter(pl.col("eid") == "A")
        b_row = result.filter(pl.col("eid") == "B")
        assert a_row["observation_count"][0] == 3
        assert b_row["observation_count"][0] == 2

    def test_min_observations_filter(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother(min_observations=3).fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        assert len(result) == 1
        assert result["eid"][0] == "A"


class TestPredictionSmootherMulticlass:
    def test_multiclass_output_columns(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        assert "predicted_label" in result.columns
        assert "observation_count" in result.columns
        for c in ["prob_cat", "prob_dog", "prob_bird"]:
            assert c in result.columns

    def test_multiclass_argmax(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        a_row = result.filter(pl.col("eid") == "A")
        b_row = result.filter(pl.col("eid") == "B")
        assert a_row["predicted_label"][0] == "prob_cat"
        assert b_row["predicted_label"][0] == "prob_dog"

    def test_multiclass_averages(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        a_row = result.filter(pl.col("eid") == "A")
        assert abs(a_row["prob_cat"][0] - 1.3 / 3) < 1e-6
        assert abs(a_row["prob_dog"][0] - 1.1 / 3) < 1e-6
        assert abs(a_row["prob_bird"][0] - 0.2) < 1e-6

    def test_multiple_entity_cols(self) -> None:
        df = pl.DataFrame({
            "eid": ["A", "A", "A", "A"],
            "seg": ["X", "X", "Y", "Y"],
            "month": [1, 2, 1, 2],
            "p": [0.6, 0.4, 0.8, 0.9],
        })
        ps = PredictionSmoother().fit()
        result = ps.transform(df, entity_cols=["eid", "seg"], period_col="month", prob_cols="p")
        assert len(result) == 2
