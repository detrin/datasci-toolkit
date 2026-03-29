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
