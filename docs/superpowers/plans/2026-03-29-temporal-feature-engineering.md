# Temporal Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `datasci_toolkit/temporal.py` — a general-purpose temporal feature engineering module that aggregates multi-table event data into a flat feature vector per entity, with a fluent builder API, config dict API, and tutorial.

**Architecture:** Three spec dataclasses (`AggSpec`, `TimeSinceSpec`, `RatioSpec`) describe what to compute. `TemporalFeatureEngineer(BaseEstimator, TransformerMixin)` owns the spec lists, joins multi-table input internally via polars, and exposes both fluent builder methods and a `from_config` classmethod. Private helpers handle joining, windowing, and per-feature-family computation.

**Tech Stack:** Python 3.12, polars, scikit-learn (BaseEstimator, TransformerMixin, check_is_fitted), pytest

---

## Files

| Action | Path |
|---|---|
| Create | `datasci_toolkit/temporal.py` |
| Create | `tests/test_temporal.py` |
| Modify | `datasci_toolkit/__init__.py` |
| Create | `docs/tutorials/temporal.md` |

---

## Shared Test Fixtures (used across all tasks)

All test tasks import and use this fixture — copy it into `tests/test_temporal.py` in Task 1 and reuse it.

```python
import polars as pl
from datetime import date

REFERENCE_DATE = "2024-01-01"

# user 1: events at 10d, 20d, 45d before ref — amounts 100, 200, 300
# user 2: events at 5d, 60d before ref — amounts 150, 250
# user 3: no events (tests missing-entity null-fill)
TRANSACTIONS = pl.DataFrame({
    "user_id": [1, 1, 1, 2, 2],
    "date":    [
        date(2023, 12, 22),  # 10d before ref
        date(2023, 12, 12),  # 20d before ref
        date(2023, 11, 17),  # 45d before ref
        date(2023, 12, 27),  # 5d before ref
        date(2023, 11,  2),  # 60d before ref
    ],
    "amount": [100.0, 200.0, 300.0, 150.0, 250.0],
    "status": ["paid", "paid", "unpaid", "paid", "unpaid"],
})

# secondary table: tier per user (used for multi-table tests)
USER_INFO = pl.DataFrame({
    "user_id": [1, 2, 3],
    "tier":    ["gold", "silver", "bronze"],
})

TABLES_SINGLE = {"transactions": TRANSACTIONS}
TABLES_MULTI  = {"transactions": TRANSACTIONS, "user_info": USER_INFO}
```

---

## Task 1: Spec dataclasses and pure helper functions

**Files:**
- Create: `datasci_toolkit/temporal.py`
- Create: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_temporal.py`:

```python
import re
import polars as pl
from datetime import date
import pytest
from datasci_toolkit.temporal import (
    AggSpec,
    TimeSinceSpec,
    RatioSpec,
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` — file doesn't exist yet.

- [ ] **Step 3: Create `datasci_toolkit/temporal.py` with specs and helpers**

```python
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class AggSpec:
    variable: str
    functions: list[str]
    windows: list[str]
    table: str
    query: str | None = None


@dataclass
class TimeSinceSpec:
    variable: str
    from_: str
    unit: str
    table: str
    query: str | None = None


@dataclass
class RatioSpec:
    numerator: str
    denominator: str


_AGG_EXPRS: dict[str, Callable[[str], pl.Expr]] = {
    "sum":   lambda c: pl.col(c).sum(),
    "mean":  lambda c: pl.col(c).mean(),
    "min":   lambda c: pl.col(c).min(),
    "max":   lambda c: pl.col(c).max(),
    "count": lambda c: pl.col(c).len(),
    "std":   lambda c: pl.col(c).std(),
    "mode":  lambda c: pl.col(c).mode().first(),
}

_UNIT_SCALE: dict[str, float] = {
    "days":   1.0,
    "months": 30.4375,
    "hours":  1.0 / 24.0,
}


def _parse_window_days(w: str) -> float | None:
    if w == "inf":
        return None
    if w.endswith("mo"):
        return int(w[:-2]) * 30.4375
    if w.endswith("d"):
        return float(w[:-1])
    if w.endswith("h"):
        return int(w[:-1]) / 24.0
    raise ValueError(f"Unknown window format: {w}")


def _sanitize_query(q: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", q.lower().replace(" ", "_"))


def _agg_col_name(func: str, var: str, window: str, query: str | None) -> str:
    name = f"{func.upper()}_{var.upper()}_{window}"
    if query:
        name += f"__{_sanitize_query(query)}"
    return name


def _time_since_col_name(from_: str, var: str, unit: str) -> str:
    return f"TIME_SINCE_{from_.upper()}_{var.upper()}_{unit}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add datasci_toolkit/temporal.py tests/test_temporal.py && git commit -m "feat: add temporal spec dataclasses and helper functions"
```

---

## Task 2: `TemporalFeatureEngineer` skeleton and `_join_tables`

**Files:**
- Modify: `datasci_toolkit/temporal.py` (append class + method)
- Modify: `tests/test_temporal.py` (append tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_temporal.py`:

```python
from datasci_toolkit.temporal import TemporalFeatureEngineer

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
    # user 3 is in USER_INFO but not in TRANSACTIONS → no rows expected for user 3
    assert 3 not in result["user_id"].to_list()
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py::test_join_tables_single -v
```

Expected: `ImportError: cannot import name 'TemporalFeatureEngineer'`

- [ ] **Step 3: Append class skeleton to `datasci_toolkit/temporal.py`**

Append after the helper functions:

```python
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._agg_specs: list[AggSpec] = []
        self._time_since_specs: list[TimeSinceSpec] = []
        self._ratio_specs: list[RatioSpec] = []
        self._pending_meta: dict | None = None

    def _join_tables(self, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
        result = tables[self.primary_]
        for name, tbl in tables.items():
            if name == self.primary_:
                continue
            result = result.join(tbl, on=self.entity_col_, how="left")
        return result

    def fit(
        self,
        tables: dict[str, pl.DataFrame],
        entity_col: str | None = None,
        time_col: str | None = None,
        reference_date: str | None = None,
        primary: str | None = None,
    ) -> "TemporalFeatureEngineer":
        if self._pending_meta is not None:
            meta = self._pending_meta
            entity_col = meta["entity_col"]
            time_col = meta["time_col"]
            reference_date = meta["reference_date"]
            primary = meta["primary"]
        self.entity_col_ = entity_col
        self.time_col_ = time_col
        self.reference_date_ = reference_date
        self.primary_ = primary
        df = self._join_tables(tables)
        self.entities_ = df[self.entity_col_].unique()
        return self

    def transform(self, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
        check_is_fitted(self)
        df = self._join_tables(tables)
        df = df.with_columns(
            (
                pl.lit(self.reference_date_).str.to_date()
                - pl.col(self.time_col_).cast(pl.Date)
            )
            .dt.total_days()
            .alias("TIME_ORDER")
        )
        frames: list[pl.DataFrame] = []
        if self._agg_specs:
            frames.append(self._compute_agg(df))
        if self._time_since_specs:
            frames.append(self._compute_time_since(df))

        entity_frame = pl.DataFrame({self.entity_col_: self.entities_})
        if not frames:
            result = entity_frame
        else:
            result = frames[0]
            for f in frames[1:]:
                result = result.join(f, on=self.entity_col_, how="left")

        if self._ratio_specs:
            ratio_df = self._compute_ratio(result)
            result = result.join(ratio_df, on=self.entity_col_, how="left")

        return entity_frame.join(result, on=self.entity_col_, how="left")

    def fit_transform(
        self,
        tables: dict[str, pl.DataFrame],
        entity_col: str | None = None,
        time_col: str | None = None,
        reference_date: str | None = None,
        primary: str | None = None,
    ) -> pl.DataFrame:
        return self.fit(tables, entity_col, time_col, reference_date, primary).transform(tables)

    def _compute_agg(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.DataFrame({self.entity_col_: self.entities_})
        for spec in self._agg_specs:
            for window in spec.windows:
                max_days = _parse_window_days(window)
                filtered = df.filter(pl.col("TIME_ORDER") >= 0)
                if max_days is not None:
                    filtered = filtered.filter(pl.col("TIME_ORDER") <= max_days)
                if spec.query:
                    filtered = pl.SQLContext({"t": filtered}, eager=True).execute(
                        f"SELECT * FROM t WHERE {spec.query}"
                    )
                agg_exprs = [
                    _AGG_EXPRS[func](spec.variable).alias(
                        _agg_col_name(func, spec.variable, window, spec.query)
                    )
                    for func in spec.functions
                ]
                agg = filtered.group_by(self.entity_col_).agg(agg_exprs)
                result = result.join(agg, on=self.entity_col_, how="left")
        return result

    def _compute_time_since(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.DataFrame({self.entity_col_: self.entities_})
        for spec in self._time_since_specs:
            filtered = df
            if spec.query:
                filtered = pl.SQLContext({"t": filtered}, eager=True).execute(
                    f"SELECT * FROM t WHERE {spec.query}"
                )
            descending = spec.from_ == "last"
            sorted_df = filtered.sort(self.time_col_, descending=descending)
            agg = sorted_df.group_by(self.entity_col_).agg(
                pl.col(spec.variable).first().alias("_ts")
            )
            scale = _UNIT_SCALE[spec.unit]
            col_name = _time_since_col_name(spec.from_, spec.variable, spec.unit)
            agg = agg.with_columns(
                (
                    (
                        pl.lit(self.reference_date_).str.to_date()
                        - pl.col("_ts").cast(pl.Date)
                    )
                    .dt.total_days()
                    .cast(pl.Float64)
                    / scale
                ).alias(col_name)
            ).drop("_ts")
            result = result.join(agg, on=self.entity_col_, how="left")
        return result

    def _compute_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        cols: list[str] = [self.entity_col_]
        for spec in self._ratio_specs:
            col_name = f"RATIO_{spec.numerator}__{spec.denominator}"
            df = df.with_columns(
                pl.when(
                    pl.col(spec.denominator).is_null()
                    | (pl.col(spec.denominator) == 0)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(spec.numerator) / pl.col(spec.denominator))
                .alias(col_name)
            )
            cols.append(col_name)
        return df.select(cols)

    def add_aggregation(
        self,
        variable: str,
        functions: list[str],
        windows: list[str],
        table: str,
        query: str | None = None,
    ) -> "TemporalFeatureEngineer":
        self._agg_specs.append(AggSpec(variable, functions, windows, table, query))
        return self

    def add_time_since(
        self,
        variable: str,
        from_: str,
        unit: str,
        table: str,
        query: str | None = None,
    ) -> "TemporalFeatureEngineer":
        self._time_since_specs.append(TimeSinceSpec(variable, from_, unit, table, query))
        return self

    def add_ratio(self, numerator: str, denominator: str) -> "TemporalFeatureEngineer":
        self._ratio_specs.append(RatioSpec(numerator, denominator))
        return self

    @classmethod
    def from_config(cls, cfg: dict) -> "TemporalFeatureEngineer":
        fe = cls()
        fe._pending_meta = cfg["meta"]
        for spec in cfg.get("aggregations", []):
            fe.add_aggregation(
                spec["variable"], spec["functions"], spec["windows"],
                spec["table"], spec.get("query"),
            )
        for spec in cfg.get("time_since", []):
            fe.add_time_since(
                spec["variable"], spec["from"], spec["unit"],
                spec["table"], spec.get("query"),
            )
        for spec in cfg.get("ratios", []):
            fe.add_ratio(spec["numerator"], spec["denominator"])
        return fe
```

- [ ] **Step 4: Run `_join_tables` tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py::test_join_tables_single tests/test_temporal.py::test_join_tables_multi_adds_columns tests/test_temporal.py::test_join_tables_multi_null_for_missing_entity -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run full test file to confirm no regressions**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add datasci_toolkit/temporal.py tests/test_temporal.py && git commit -m "feat: add TemporalFeatureEngineer skeleton with _join_tables and all core methods"
```

---

## Task 3: Aggregation tests

**Files:**
- Modify: `tests/test_temporal.py` (append aggregation tests)

- [ ] **Step 1: Write aggregation tests**

Append to `tests/test_temporal.py`:

```python
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
    # add user 3 to entity set manually to test null fill
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
```

- [ ] **Step 2: Run aggregation tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -k "agg" -v
```

Expected: all aggregation tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add tests/test_temporal.py && git commit -m "test: add aggregation feature tests"
```

---

## Task 4: Time-since tests

**Files:**
- Modify: `tests/test_temporal.py` (append time-since tests)

- [ ] **Step 1: Write time-since tests**

Append to `tests/test_temporal.py`:

```python
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
    # user 1: 10 days ÷ 30.4375 ≈ 0.328 months
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
```

- [ ] **Step 2: Run time-since tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -k "time_since" -v
```

Expected: all PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add tests/test_temporal.py && git commit -m "test: add time-since feature tests"
```

---

## Task 5: Ratio tests

**Files:**
- Modify: `tests/test_temporal.py` (append ratio tests)

- [ ] **Step 1: Write ratio tests**

Append to `tests/test_temporal.py`:

```python
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
    # Create data where user has events in 30d but also in inf (so denom > 0),
    # but we test the zero case with a user who has no events at all in a window.
    # Use a narrow window that excludes all events for user 2
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
    # Both users have null for 3d window (no events within 3d of ref date)
    # null denominator or zero denominator → null ratio
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
```

- [ ] **Step 2: Run ratio tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -k "ratio" -v
```

Expected: all PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add tests/test_temporal.py && git commit -m "test: add ratio feature tests"
```

---

## Task 6: Integration and multi-table tests

**Files:**
- Modify: `tests/test_temporal.py` (append integration tests)

- [ ] **Step 1: Write integration tests**

Append to `tests/test_temporal.py`:

```python
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
    # Secondary table joined; primary agg still works
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
```

- [ ] **Step 2: Run integration tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -k "integration or multi_table or fit_transform or transform_after or output_entity or check_is_fitted" -v
```

Expected: all PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add tests/test_temporal.py && git commit -m "test: add integration and multi-table tests"
```

---

## Task 7: `from_config` and fluent-builder equivalence tests

**Files:**
- Modify: `tests/test_temporal.py` (append from_config tests)

- [ ] **Step 1: Write from_config tests**

Append to `tests/test_temporal.py`:

```python
# ── from_config ──────────────────────────────────────────────────────────────
CFG = {
    "meta": {
        "entity_col": "user_id",
        "time_col":   "date",
        "reference_date": REFERENCE_DATE,
        "primary":    "transactions",
    },
    "aggregations": [
        {"variable": "amount", "functions": ["sum", "mean"], "windows": ["30d", "inf"], "table": "transactions"},
    ],
    "time_since": [
        {"variable": "date", "from": "last", "unit": "days", "table": "transactions"},
    ],
    "ratios": [
        {"numerator": "SUM_AMOUNT_30d", "denominator": "SUM_AMOUNT_inf"},
    ],
}

def test_from_config_produces_correct_columns():
    fe = TemporalFeatureEngineer.from_config(CFG)
    result = fe.fit_transform(TABLES_SINGLE)
    expected = {
        "user_id", "SUM_AMOUNT_30d", "MEAN_AMOUNT_30d",
        "SUM_AMOUNT_inf", "MEAN_AMOUNT_inf",
        "TIME_SINCE_LAST_DATE_days",
        "RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf",
    }
    assert expected.issubset(set(result.columns))

def test_from_config_matches_fluent_builder():
    fe_cfg = TemporalFeatureEngineer.from_config(CFG)
    result_cfg = fe_cfg.fit_transform(TABLES_SINGLE)

    fe_fluent = (
        TemporalFeatureEngineer()
        .add_aggregation("amount", ["sum", "mean"], ["30d", "inf"], "transactions")
        .add_time_since("date", from_="last", unit="days", table="transactions")
        .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
    )
    result_fluent = fe_fluent.fit_transform(
        TABLES_SINGLE, entity_col="user_id", time_col="date",
        reference_date=REFERENCE_DATE, primary="transactions",
    )

    # Sort both by user_id for stable comparison
    result_cfg    = result_cfg.sort("user_id")
    result_fluent = result_fluent.sort("user_id")

    for col in result_cfg.columns:
        if result_cfg[col].dtype == pl.Float64:
            assert result_cfg[col].to_list() == pytest.approx(
                result_fluent[col].to_list(), nan_ok=True
            )
        else:
            assert result_cfg[col].to_list() == result_fluent[col].to_list()

def test_from_config_no_aggregations():
    cfg = {
        "meta": {
            "entity_col": "user_id", "time_col": "date",
            "reference_date": REFERENCE_DATE, "primary": "transactions",
        },
        "time_since": [
            {"variable": "date", "from": "last", "unit": "days", "table": "transactions"},
        ],
    }
    fe = TemporalFeatureEngineer.from_config(cfg)
    result = fe.fit_transform(TABLES_SINGLE)
    assert "TIME_SINCE_LAST_DATE_days" in result.columns
```

- [ ] **Step 2: Run from_config tests**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -k "from_config" -v
```

Expected: all PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/test_temporal.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add tests/test_temporal.py && git commit -m "test: add from_config and fluent builder equivalence tests"
```

---

## Task 8: Export from `__init__.py` and pre-commit validation

**Files:**
- Modify: `datasci_toolkit/__init__.py`

- [ ] **Step 1: Add export**

In `datasci_toolkit/__init__.py`, add the import line alongside the other imports:

```python
from datasci_toolkit.temporal import AggSpec, RatioSpec, TemporalFeatureEngineer, TimeSinceSpec
```

And add to `__all__`:

```python
"TemporalFeatureEngineer",
"AggSpec",
"TimeSinceSpec",
"RatioSpec",
```

The full updated `__init__.py` should look like:

```python
from datasci_toolkit.bin_editor import BinEditor
from datasci_toolkit.bin_editor_widget import BinEditorWidget
from datasci_toolkit.grouping import StabilityGrouping, WOETransformer
from datasci_toolkit.metrics import BootstrapGini, feature_power, gini, gini_by_period, iv, ks, lift, lift_by_period, plot_metric_by_period
from datasci_toolkit.model_selection import AUCStepwiseLogit
from datasci_toolkit.variable_clustering import CorrVarClus
from datasci_toolkit.label_imputation import KNNLabelImputer, TargetImputer
from datasci_toolkit.stability import ESI, PSI, StabilityMonitor, plot_psi_comparison, psi_hist
from datasci_toolkit.temporal import AggSpec, RatioSpec, TemporalFeatureEngineer, TimeSinceSpec

__all__ = [
    "PSI",
    "ESI",
    "StabilityMonitor",
    "plot_psi_comparison",
    "psi_hist",
    "WOETransformer",
    "StabilityGrouping",
    "AUCStepwiseLogit",
    "gini",
    "ks",
    "lift",
    "iv",
    "BootstrapGini",
    "feature_power",
    "TargetImputer",
    "KNNLabelImputer",
    "BinEditor",
    "BinEditorWidget",
    "CorrVarClus",
    "gini_by_period",
    "lift_by_period",
    "plot_metric_by_period",
    "TemporalFeatureEngineer",
    "AggSpec",
    "TimeSinceSpec",
    "RatioSpec",
]
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && python -c "from datasci_toolkit import TemporalFeatureEngineer, AggSpec, TimeSinceSpec, RatioSpec; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite including pre-commit checks**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && pytest tests/ -q
```

Expected: all tests pass (279+ tests).

- [ ] **Step 4: Run ruff and mypy manually to catch any issues before commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && source .venv/bin/activate && ruff check datasci_toolkit && mypy datasci_toolkit
```

Fix any issues reported. Common issues:
- Missing type annotation → add return type or parameter type
- `dict` without type params → use `dict[str, ...]`
- Unused import → remove it

- [ ] **Step 5: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add datasci_toolkit/__init__.py && git commit -m "feat: export TemporalFeatureEngineer and spec classes from package"
```

---

## Task 9: Tutorial

**Files:**
- Create: `docs/tutorials/temporal.md`

- [ ] **Step 1: Write tutorial**

Create `docs/tutorials/temporal.md` with this content:

```markdown
# Temporal Feature Engineering

Generate aggregated features from longitudinal event data — any dataset with a grouping key (entity) and timestamps. Works with a single table or multiple joined tables.

## Quick start — single table

```python
import polars as pl
from datetime import date
from datasci_toolkit import TemporalFeatureEngineer

transactions = pl.DataFrame({
    "user_id": [1, 1, 1, 2, 2],
    "date": [
        date(2023, 12, 22), date(2023, 12, 12), date(2023, 11, 17),
        date(2023, 12, 27), date(2023, 11,  2),
    ],
    "amount": [100.0, 200.0, 300.0, 150.0, 250.0],
    "status": ["paid", "paid", "unpaid", "paid", "unpaid"],
})

fe = (
    TemporalFeatureEngineer()
    .add_aggregation("amount", ["sum", "mean", "count"], windows=["30d", "90d", "inf"], table="transactions")
)
features = fe.fit_transform(
    {"transactions": transactions},
    entity_col="user_id",
    time_col="date",
    reference_date="2024-01-01",
    primary="transactions",
)
print(features)
```

Output: one row per `user_id`, columns `SUM_AMOUNT_30d`, `MEAN_AMOUNT_30d`, `COUNT_AMOUNT_30d`, `SUM_AMOUNT_90d`, etc.

## Multi-table input

Pass a dict of DataFrames. Each table must contain `entity_col`. Secondary tables are left-joined onto the primary on `entity_col`.

```python
user_info = pl.DataFrame({
    "user_id": [1, 2, 3],
    "tier": ["gold", "silver", "bronze"],
})

fe = (
    TemporalFeatureEngineer()
    .add_aggregation("amount", ["sum"], windows=["30d", "inf"], table="transactions")
)
features = fe.fit_transform(
    {"transactions": transactions, "user_info": user_info},
    entity_col="user_id",
    time_col="date",
    reference_date="2024-01-01",
    primary="transactions",
)
```

User 3 (present in `user_info` but not `transactions`) appears with null feature values — the entity set is derived from the primary table during `fit`.

## Time-since features

How many days/months since the first or last event per entity.

```python
fe = (
    TemporalFeatureEngineer()
    .add_time_since("date", from_="last", unit="days",   table="transactions")
    .add_time_since("date", from_="first", unit="months", table="transactions")
)
features = fe.fit_transform(
    {"transactions": transactions},
    entity_col="user_id",
    time_col="date",
    reference_date="2024-01-01",
    primary="transactions",
)
# Columns: TIME_SINCE_LAST_DATE_days, TIME_SINCE_FIRST_DATE_months
```

Supported units: `"days"`, `"months"`, `"hours"`.

## Query filters

Restrict rows before aggregating using a SQL WHERE clause string.

```python
fe = (
    TemporalFeatureEngineer()
    .add_aggregation("amount", ["sum", "count"], windows=["30d", "inf"],
                     table="transactions", query="status = 'paid'")
    .add_time_since("date", from_="last", unit="days",
                    table="transactions", query="status = 'paid'")
)
features = fe.fit_transform(
    {"transactions": transactions},
    entity_col="user_id",
    time_col="date",
    reference_date="2024-01-01",
    primary="transactions",
)
# Columns include: SUM_AMOUNT_30d__status__paid, TIME_SINCE_LAST_DATE_days
```

## Ratio features

Divide two already-computed aggregation columns. Reference them by their generated column name.

```python
fe = (
    TemporalFeatureEngineer()
    .add_aggregation("amount", ["sum"], windows=["30d", "inf"], table="transactions")
    .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_inf")
)
features = fe.fit_transform(
    {"transactions": transactions},
    entity_col="user_id",
    time_col="date",
    reference_date="2024-01-01",
    primary="transactions",
)
# Column: RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf
# Zero or null denominator → null (no inf values)
```

## Config dict

Equivalent to the fluent builder — useful for serialising feature specs to YAML/JSON.

```python
from datasci_toolkit import TemporalFeatureEngineer

cfg = {
    "meta": {
        "entity_col":     "user_id",
        "time_col":       "date",
        "reference_date": "2024-01-01",
        "primary":        "transactions",
    },
    "aggregations": [
        {
            "variable":  "amount",
            "functions": ["sum", "mean", "count"],
            "windows":   ["30d", "90d", "inf"],
            "table":     "transactions",
        },
        {
            "variable":  "amount",
            "functions": ["sum"],
            "windows":   ["30d"],
            "table":     "transactions",
            "query":     "status = 'paid'",
        },
    ],
    "time_since": [
        {"variable": "date", "from": "last",  "unit": "days",   "table": "transactions"},
        {"variable": "date", "from": "first", "unit": "months", "table": "transactions"},
    ],
    "ratios": [
        {"numerator": "SUM_AMOUNT_30d", "denominator": "SUM_AMOUNT_inf"},
    ],
}

fe = TemporalFeatureEngineer.from_config(cfg)
features = fe.fit_transform({"transactions": transactions})
```

## Feature naming reference

| Feature type | Column name pattern | Example |
|---|---|---|
| Aggregation | `{FUNC}_{VAR}_{WINDOW}` | `SUM_AMOUNT_30d` |
| Aggregation + query | `{FUNC}_{VAR}_{WINDOW}__{sanitised_query}` | `COUNT_AMOUNT_30d__status__paid` |
| Time-since | `TIME_SINCE_{FROM}_{VAR}_{UNIT}` | `TIME_SINCE_LAST_DATE_days` |
| Ratio | `RATIO_{NUMERATOR}__{DENOMINATOR}` | `RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_inf` |

**Supported aggregation functions:** `sum`, `mean`, `min`, `max`, `count`, `std`, `mode`

**Supported time windows:** `"7d"`, `"30d"`, `"90d"`, `"1mo"`, `"inf"` (and any `Nd` or `Nmo` pattern)

**Supported time-since units:** `"days"`, `"months"`, `"hours"`
```

- [ ] **Step 2: Verify tutorial renders correctly (no broken syntax)**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && python -c "
import polars as pl
from datetime import date
from datasci_toolkit import TemporalFeatureEngineer

transactions = pl.DataFrame({
    'user_id': [1, 1, 1, 2, 2],
    'date': [date(2023,12,22), date(2023,12,12), date(2023,11,17), date(2023,12,27), date(2023,11,2)],
    'amount': [100.0, 200.0, 300.0, 150.0, 250.0],
    'status': ['paid', 'paid', 'unpaid', 'paid', 'unpaid'],
})
fe = (
    TemporalFeatureEngineer()
    .add_aggregation('amount', ['sum', 'mean', 'count'], windows=['30d', '90d', 'inf'], table='transactions')
    .add_time_since('date', from_='last', unit='days', table='transactions')
    .add_ratio('SUM_AMOUNT_30d', 'SUM_AMOUNT_inf')
)
result = fe.fit_transform({'transactions': transactions}, entity_col='user_id', time_col='date', reference_date='2024-01-01', primary='transactions')
print(result)
print('Tutorial examples: OK')
"
```

Expected: prints a polars DataFrame and `Tutorial examples: OK`.

- [ ] **Step 3: Commit**

```bash
cd /Users/danherma/projects-personal/datasci-toolkit && git add docs/tutorials/temporal.md && git commit -m "docs: add temporal feature engineering tutorial"
```

---

## Self-review checklist

- [x] All spec dataclasses covered in Task 1
- [x] `_join_tables` single + multi table covered in Task 2
- [x] All aggregation functions (sum/mean/min/max/count/std) with window filtering covered in Task 3
- [x] Time-since with first/last, days/months, and query filter covered in Task 4
- [x] Ratio with zero-denominator null handling covered in Task 5
- [x] Full pipeline, multi-table, `check_is_fitted` covered in Task 6
- [x] `from_config` + fluent builder equivalence covered in Task 7
- [x] `__init__.py` export + ruff/mypy validation in Task 8
- [x] Tutorial covers all 6 planned sections in Task 9
- [x] No TBDs, no placeholders
- [x] Type names consistent: `TemporalFeatureEngineer`, `AggSpec`, `TimeSinceSpec`, `RatioSpec` throughout
- [x] Column name patterns consistent: `SUM_AMOUNT_30d`, `TIME_SINCE_LAST_DATE_days`, `RATIO_A__B`
