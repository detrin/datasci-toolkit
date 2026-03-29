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

**Supported time windows:** `"7d"`, `"30d"`, `"90d"`, `"1mo"`, `"inf"` (any `Nd` or `Nmo` pattern)

**Supported time-since units:** `"days"`, `"months"`, `"hours"`
