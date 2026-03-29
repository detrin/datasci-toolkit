# Temporal Feature Engineering — Design Spec

**Date:** 2026-03-29
**Module:** `datasci_toolkit/temporal.py`
**Tutorial:** `docs/tutorials/temporal.md`
**Tests:** `tests/test_temporal.py`

---

## Purpose

A general-purpose temporal feature engineering module that aggregates hierarchical, multi-table event data into a flat feature vector per entity. Inspired by FEFE but generalised beyond transaction data — any longitudinal/panel dataset with a grouping key and timestamps works.

---

## Architecture

```
datasci_toolkit/temporal.py
├── AggSpec          — dataclass: variable, functions, windows, table, query
├── TimeSinceSpec    — dataclass: variable, from_ (first/last), unit, table, query
├── RatioSpec        — dataclass: numerator, denominator (column name references)
└── TemporalFeatureEngineer(BaseEstimator, TransformerMixin)
    ├── fit(tables, entity_col, time_col, reference_date, primary)
    ├── transform(tables)
    ├── fit_transform(tables, ...)
    ├── add_aggregation(...)   → self
    ├── add_time_since(...)    → self
    ├── add_ratio(...)         → self
    └── from_config(cfg)       → cls

    private:
    ├── _join_tables(tables)
    ├── _compute_agg(df)
    ├── _compute_time_since(df)
    └── _compute_ratio(df)
```

`tables` is `dict[str, pl.DataFrame]`. The primary table contains `entity_col` and `time_col`. Secondary tables contain `entity_col` and are left-joined onto the primary.

---

## Data Flow

### fit
1. `_join_tables(tables)` — left join all secondary tables onto primary on `entity_col`
2. Compute `TIME_ORDER` = `reference_date - time_col` (polars duration)
3. Store `entities_` = unique values of `entity_col`
4. Store `entity_col_`, `time_col_`, `reference_date_`

### transform
1. `_join_tables(tables)` + compute `TIME_ORDER`
2. `_compute_agg(df)` — for each `AggSpec`:
   - Filter rows: `TIME_ORDER <= window_end` and `TIME_ORDER >= window_start`
   - Apply optional `query` filter
   - `group_by(entity_col).agg(functions)`
   - Column names: `{FUNC}_{VAR}_{WINDOW}` e.g. `SUM_AMOUNT_30d`
3. `_compute_time_since(df)` — for each `TimeSinceSpec`:
   - Sort by `time_col`, take `first` or `last` per entity
   - `delta = reference_date - event_time`, convert to `unit`
   - Column names: `TIME_SINCE_{FROM}_{VAR}_{UNIT}` e.g. `TIME_SINCE_LAST_EVENT_DATE_days`
4. `_compute_ratio(df)` — for each `RatioSpec`:
   - Reference already-computed agg columns by name
   - Divide numerator / denominator, replace inf/zero with null
   - Column names: `RATIO_{NUMERATOR}__{DENOMINATOR}` e.g. `RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_90d`
5. Join all feature frames on `entity_col`
6. Reindex to `entities_` — missing entities filled with null

### Output
`pl.DataFrame` — one row per entity, N feature columns.

---

## Configuration Interface

### Fluent builder
```python
fe = (
    TemporalFeatureEngineer()
    .add_aggregation("amount", ["sum", "mean", "max"], windows=["30d", "90d", "inf"], table="transactions")
    .add_aggregation("amount", ["count"], windows=["7d", "30d"], table="transactions", query="status == 'paid'")
    .add_time_since("event_date", from_="last", unit="days", table="transactions")
    .add_ratio("SUM_AMOUNT_30d", "SUM_AMOUNT_90d")
)
fe.fit(tables, entity_col="user_id", time_col="date", reference_date="2024-01-01", primary="transactions")
features = fe.transform(tables)
```

### Config dict
```python
cfg = {
    "meta": {
        "entity_col": "user_id",
        "time_col": "date",
        "reference_date": "2024-01-01",
        "primary": "transactions"
    },
    "aggregations": [
        {"variable": "amount", "functions": ["sum", "mean"], "windows": ["30d", "90d"], "table": "transactions"},
        {"variable": "amount", "functions": ["count"], "windows": ["7d"], "table": "transactions", "query": "status == 'paid'"}
    ],
    "time_since": [
        {"variable": "event_date", "from": "last", "unit": "days", "table": "transactions"}
    ],
    "ratios": [
        {"numerator": "SUM_AMOUNT_30d", "denominator": "SUM_AMOUNT_90d"}
    ]
}
fe = TemporalFeatureEngineer.from_config(cfg)
features = fe.fit_transform(tables)
```

---

## Spec Dataclasses

```python
@dataclass
class AggSpec:
    variable: str
    functions: list[str]       # sum, mean, min, max, count, std, mode
    windows: list[str]         # "30d", "90d", "1mo", "inf"
    table: str
    query: str | None = None

@dataclass
class TimeSinceSpec:
    variable: str
    from_: str                 # "first" or "last"
    unit: str                  # "days", "months", "hours"
    table: str
    query: str | None = None

@dataclass
class RatioSpec:
    numerator: str             # column name of already-computed agg feature
    denominator: str           # column name of already-computed agg feature
```

---

## Feature Naming Convention

| Feature type | Pattern | Example |
|---|---|---|
| Aggregation | `{FUNC}_{VAR}_{WINDOW}` | `SUM_AMOUNT_30d` |
| Aggregation + query | `{FUNC}_{VAR}_{WINDOW}__{QUERY_HASH}` | `COUNT_AMOUNT_30d__paid` |
| Time-since | `TIME_SINCE_{FROM}_{VAR}_{UNIT}` | `TIME_SINCE_LAST_EVENT_DATE_days` |
| Ratio | `RATIO_{NUMERATOR}__{DENOMINATOR}` | `RATIO_SUM_AMOUNT_30d__SUM_AMOUNT_90d` |

Query suffix is a short, sanitised string derived from the query expression (lowercase, spaces → `_`, special chars stripped).

---

## Time Window Format

Duration strings map to polars `pl.duration`:

| String | Meaning |
|---|---|
| `"7d"` | Last 7 days |
| `"30d"` | Last 30 days |
| `"1mo"` | Last 1 month |
| `"90d"` | Last 90 days |
| `"inf"` | All history |

`"inf"` means no upper bound on `TIME_ORDER`.

---

## Testing Plan (`tests/test_temporal.py`)

- Spec dataclass instantiation (valid + invalid inputs)
- `_join_tables` with primary only and primary + secondary
- Aggregation correctness: each function (sum, mean, min, max, count, std) against known values
- Time window filtering: rows outside window excluded
- Time-since: first vs last, days vs months
- Ratio: normal case, zero denominator → null, inf → null
- Full `fit_transform` round-trip with multi-table input
- `from_config` produces identical result to fluent builder
- Output shape: one row per entity, correct column names
- Missing entities (present in fit, absent in transform) → null row

---

## Tutorial Plan (`docs/tutorials/temporal.md`)

1. **Single-table example** — user transactions → aggregated features
2. **Multi-table example** — transactions + profile changes joined on `user_id`
3. **Time-since features** — days since last payment
4. **Ratio features** — paid / total transaction ratio
5. **Config dict usage** — same result via dict
6. **Feature naming reference** — column name conventions explained
