import polars as pl
from datetime import date
import pytest
from datasci_toolkit.temporal import (
    AggSpec,
    TimeSinceSpec,
    RatioSpec,
    TemporalFeatureEngineer,
    _parse_window_days,
    _agg_col_name,
    _time_since_col_name,
    _sanitize_query,
)

# ── shared fixtures ──────────────────────────────────────────────────────────
REFERENCE_DATE = "2024-01-01"

TRANSACTIONS = pl.DataFrame({
    "user_id": [1, 1, 1, 2, 2],
    "date": [
        date(2023, 12, 22),
        date(2023, 12, 12),
        date(2023, 11, 17),
        date(2023, 12, 27),
        date(2023, 11,  2),
    ],
    "amount": [100.0, 200.0, 300.0, 150.0, 250.0],
    "status": ["paid", "paid", "unpaid", "paid", "unpaid"],
})

USER_INFO = pl.DataFrame({
    "user_id": [1, 2, 3],
    "tier":    ["gold", "silver", "bronze"],
})

TABLES_SINGLE = {"transactions": TRANSACTIONS}
TABLES_MULTI  = {"transactions": TRANSACTIONS, "user_info": USER_INFO}

# ── spec dataclasses ─────────────────────────────────────────────────────────
def test_agg_spec_defaults():
    spec = AggSpec("amount", ["sum"], ["30d"], "transactions")
    assert spec.query is None

def test_time_since_spec_defaults():
    spec = TimeSinceSpec("date", "last", "days", "transactions")
    assert spec.query is None

def test_ratio_spec():
    spec = RatioSpec("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
    assert spec.numerator == "SUM_AMOUNT_30d"
    assert spec.denominator == "SUM_AMOUNT_inf"

# ── _parse_window_days ───────────────────────────────────────────────────────
def test_parse_window_days_d():
    assert _parse_window_days("30d") == 30.0

def test_parse_window_days_7d():
    assert _parse_window_days("7d") == 7.0

def test_parse_window_days_mo():
    assert abs(_parse_window_days("1mo") - 30.4375) < 0.001

def test_parse_window_days_90d():
    assert _parse_window_days("90d") == 90.0

def test_parse_window_days_inf():
    assert _parse_window_days("inf") is None

def test_parse_window_days_unknown():
    with pytest.raises(ValueError):
        _parse_window_days("2y")

# ── _sanitize_query ──────────────────────────────────────────────────────────
def test_sanitize_query_simple():
    assert _sanitize_query("status = 'paid'") == "status__paid"

def test_sanitize_query_spaces():
    assert _sanitize_query("amount > 100") == "amount__100"

# ── _agg_col_name ────────────────────────────────────────────────────────────
def test_agg_col_name_no_query():
    assert _agg_col_name("sum", "amount", "30d", None) == "SUM_AMOUNT_30d"

def test_agg_col_name_with_query():
    result = _agg_col_name("count", "amount", "30d", "status = 'paid'")
    assert result == "COUNT_AMOUNT_30d__status__paid"

def test_agg_col_name_uppercase():
    assert _agg_col_name("mean", "transaction_amount", "90d", None) == "MEAN_TRANSACTION_AMOUNT_90d"

# ── _time_since_col_name ─────────────────────────────────────────────────────
def test_time_since_col_name_last_days():
    assert _time_since_col_name("last", "date", "days") == "TIME_SINCE_LAST_DATE_days"

def test_time_since_col_name_first_months():
    assert _time_since_col_name("first", "event_date", "months") == "TIME_SINCE_FIRST_EVENT_DATE_months"

# ── _join_tables ─────────────────────────────────────────────────────────────
def test_join_tables_single():
    fe = TemporalFeatureEngineer()
    fe.entity_col_ = "user_id"
    fe.primary_ = "transactions"
    result = fe._join_tables(TABLES_SINGLE)
    assert result.shape == TRANSACTIONS.shape
    assert set(result.columns) == set(TRANSACTIONS.columns)

def test_join_tables_multi_adds_columns():
    fe = TemporalFeatureEngineer()
    fe.entity_col_ = "user_id"
    fe.primary_ = "transactions"
    result = fe._join_tables(TABLES_MULTI)
    assert "tier" in result.columns
    assert result.shape[0] == TRANSACTIONS.shape[0]

def test_join_tables_multi_null_for_missing_entity():
    fe = TemporalFeatureEngineer()
    fe.entity_col_ = "user_id"
    fe.primary_ = "transactions"
    result = fe._join_tables(TABLES_MULTI)
    assert 3 not in result["user_id"].to_list()

# ── aggregation ──────────────────────────────────────────────────────────────
def test_agg_sum_30d():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    row1 = result.filter(pl.col("user_id") == 1)["SUM_AMOUNT_30d"][0]
    row2 = result.filter(pl.col("user_id") == 2)["SUM_AMOUNT_30d"][0]
    # user 1: 10d + 20d events → 100 + 200 = 300
    assert row1 == pytest.approx(300.0)
    # user 2: 5d event only → 150
    assert row2 == pytest.approx(150.0)

def test_agg_sum_inf():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    row1 = result.filter(pl.col("user_id") == 1)["SUM_AMOUNT_inf"][0]
    row2 = result.filter(pl.col("user_id") == 2)["SUM_AMOUNT_inf"][0]
    assert row1 == pytest.approx(600.0)
    assert row2 == pytest.approx(400.0)

def test_agg_mean():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["mean"], ["inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    row1 = result.filter(pl.col("user_id") == 1)["MEAN_AMOUNT_inf"][0]
    assert row1 == pytest.approx(200.0)  # (100+200+300)/3

def test_agg_count():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["count"], ["inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    assert result.filter(pl.col("user_id") == 1)["COUNT_AMOUNT_inf"][0] == 3
    assert result.filter(pl.col("user_id") == 2)["COUNT_AMOUNT_inf"][0] == 2

def test_agg_multiple_functions_and_windows():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum", "mean"], ["30d", "inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    expected_cols = {"SUM_AMOUNT_30d", "MEAN_AMOUNT_30d", "SUM_AMOUNT_inf", "MEAN_AMOUNT_inf"}
    assert expected_cols.issubset(set(result.columns))

def test_agg_with_query():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["inf"], "transactions", query="status = 'paid'")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "SUM_AMOUNT_inf__status__paid"
    assert col in result.columns
    # user 1: paid rows are 100, 200 → sum=300
    assert result.filter(pl.col("user_id") == 1)[col][0] == pytest.approx(300.0)

def test_agg_missing_entity_is_null():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d"], "transactions")
    )
    fe.fit(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    fe.entities_ = pl.Series([1, 2, 3])
    result = fe.transform(TABLES_SINGLE)
    row3 = result.filter(pl.col("user_id") == 3)["SUM_AMOUNT_30d"][0]
    assert row3 is None

def test_agg_output_one_row_per_entity():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d", "inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    assert result["user_id"].n_unique() == result.shape[0]

# ── time-since ───────────────────────────────────────────────────────────────
def test_time_since_last_days():
    fe = (
        TemporalFeatureEngineer()
        .add_time_since("date", from_="last", unit="days", table="transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "TIME_SINCE_LAST_DATE_days"
    assert col in result.columns
    # user 1: most recent event is 10d before ref
    assert result.filter(pl.col("user_id") == 1)[col][0] == pytest.approx(10.0)
    # user 2: most recent event is 5d before ref
    assert result.filter(pl.col("user_id") == 2)[col][0] == pytest.approx(5.0)

def test_time_since_first_days():
    fe = (
        TemporalFeatureEngineer()
        .add_time_since("date", from_="first", unit="days", table="transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "TIME_SINCE_FIRST_DATE_days"
    # user 1: oldest event is 45d before ref
    assert result.filter(pl.col("user_id") == 1)[col][0] == pytest.approx(45.0)
    # user 2: oldest event is 60d before ref
    assert result.filter(pl.col("user_id") == 2)[col][0] == pytest.approx(60.0)

def test_time_since_months():
    fe = (
        TemporalFeatureEngineer()
        .add_time_since("date", from_="last", unit="months", table="transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "TIME_SINCE_LAST_DATE_months"
    # user 1: 10 days / 30.4375 ≈ 0.328 months
    val = result.filter(pl.col("user_id") == 1)[col][0]
    assert abs(val - 10 / 30.4375) < 0.01

def test_time_since_with_query():
    fe = (
        TemporalFeatureEngineer()
        .add_time_since("date", from_="last", unit="days", table="transactions",
                        query="status = 'paid'")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "TIME_SINCE_LAST_DATE_days"
    # user 1: paid events at 10d and 20d → last paid = 10d
    assert result.filter(pl.col("user_id") == 1)[col][0] == pytest.approx(10.0)

# ── ratio ────────────────────────────────────────────────────────────────────
def test_ratio_normal():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d", "inf"], "transactions")
        .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf"
    assert col in result.columns
    # user 1: 300 / 600 = 0.5
    assert result.filter(pl.col("user_id") == 1)[col][0] == pytest.approx(0.5)
    # user 2: 150 / 400 = 0.375
    assert result.filter(pl.col("user_id") == 2)[col][0] == pytest.approx(0.375)

def test_ratio_zero_denominator_is_null():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["3d", "inf"], "transactions")
        .add_ratio("SUM_AMOUNT_3d", "SUM_AMOUNT_inf")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    col = "RATIO_SUM_AMOUNT_3d__SUM_AMOUNT_inf"
    # Both users have null for 3d window → null ratio
    val1 = result.filter(pl.col("user_id") == 1)[col][0]
    assert val1 is None

def test_ratio_column_present():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d", "inf"], "transactions")
        .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    assert "RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf" in result.columns

# ── integration ──────────────────────────────────────────────────────────────
def test_fit_transform_full_pipeline():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum", "mean", "count"], ["30d", "inf"], "transactions")
        .add_time_since("date", from_="last", unit="days", table="transactions")
        .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    expected_cols = {
        "user_id",
        "SUM_AMOUNT_30d", "MEAN_AMOUNT_30d", "COUNT_AMOUNT_30d",
        "SUM_AMOUNT_inf", "MEAN_AMOUNT_inf", "COUNT_AMOUNT_inf",
        "TIME_SINCE_LAST_DATE_days",
        "RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf",
    }
    assert expected_cols.issubset(set(result.columns))
    assert result["user_id"].n_unique() == result.shape[0]

def test_transform_after_separate_fit():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d"], "transactions")
    )
    fe.fit(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    result = fe.transform(TABLES_SINGLE)
    assert "SUM_AMOUNT_30d" in result.columns

def test_multi_table_agg_on_secondary_column():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["inf"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_MULTI, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    assert "SUM_AMOUNT_inf" in result.columns
    assert result.filter(pl.col("user_id") == 1)["SUM_AMOUNT_inf"][0] == pytest.approx(600.0)

def test_output_entity_col_present():
    fe = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum"], ["30d"], "transactions")
    )
    result = fe.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )
    assert "user_id" in result.columns

def test_check_is_fitted_before_transform():
    fe = TemporalFeatureEngineer().add_aggregation("amount", ["sum"], ["30d"], "transactions")
    with pytest.raises(Exception):
        fe.transform(TABLES_SINGLE)
