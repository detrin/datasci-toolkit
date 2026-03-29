# Grouping — WOE Binning & Stability-Constrained Binning

## Overview

```
StabilityGrouping.fit(X_train, y_train, t_train, X_val, y_val, t_val)
       │
       ├── .bin_specs_          ← export for audit / BinEditor
       └── .transform(X)        ← WOE-encoded DataFrame

WOETransformer(bin_specs=...).fit(X, y).transform(X)
       └── re-encodes any DataFrame using saved bin specs
```

## StabilityGrouping

Finds optimal bins for each feature using LightGBM, then merges bins that are unstable across time periods. Requires a train/validation split and a time column.

```python
import numpy as np
import polars as pl
from datasci_toolkit import StabilityGrouping

rng = np.random.default_rng(42)
N = 2000

f0 = rng.normal(0, 1, N)
f1 = rng.normal(0, 1, N)
event_rate = 1 / (1 + np.exp(-(f0 + 0.5 * f1)))
target = rng.binomial(1, event_rate).astype(float)
months = np.repeat(np.arange(8), N // 8)

df = pl.DataFrame({
    "f0": f0.tolist(),
    "f1": f1.tolist(),
    "target": target.tolist(),
    "month": months.tolist(),
})

train = df.filter(pl.col("month") < 5)
val   = df.filter(pl.col("month") >= 5)

sg = StabilityGrouping(stability_threshold=0.1).fit(
    X_train=train.select(["f0", "f1"]),
    y_train=train["target"],
    t_train=train["month"],
    X_val=val.select(["f0", "f1"]),
    y_val=val["target"],
    t_val=val["month"],
)

test = df.filter(pl.col("month") >= 6)
X_woe = sg.transform(test.select(["f0", "f1"]))
print(X_woe.head())
```

### Inspecting bin specs

After fitting, `bin_specs_` contains the boundary definitions — a dict that can be saved, audited, or passed to `BinEditor`.

```python
for feat, spec in sg.bin_specs_.items():
    print(feat, spec["dtype"], spec["bins"])
```

### Features that couldn't be grouped

```python
print("Ungroupable:", sg.ungroupable())
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `stability_threshold` | `0.10` | Maximum RSI allowed per bin across time periods |
| `max_bins` | `10` | Upper bound on number of bins per feature |
| `min_leaf_share` | `0.05` | Minimum fraction of records per bin |
| `min_leaf_minority` | `100` | Minimum records per bin for minority features |
| `important_minorities` | `None` | Features where `min_leaf_minority` applies |
| `must_have` | `None` | Features that are never excluded even if unstable |

## WOETransformer

Applies pre-computed bin specs and encodes features as WOE values. Use after `StabilityGrouping` or after manual editing with `BinEditor`.

```python
from datasci_toolkit import WOETransformer

# Use bin specs from StabilityGrouping
woe = WOETransformer(bin_specs=sg.bin_specs_).fit(
    train.select(["f0", "f1"]),
    train["target"],
)

train_woe = woe.transform(train.select(["f0", "f1"]))
test_woe  = woe.transform(test.select(["f0", "f1"]))
```

### Providing bin specs manually

```python
bin_specs = {
    "score": {
        "dtype": "float",
        "bins": [float("-inf"), 0.3, 0.5, 0.7, float("inf")],
    },
    "region": {
        "dtype": "category",
        "bins": {"North": 0, "South": 0, "East": 1, "West": 2},
    },
}

woe = WOETransformer(bin_specs=bin_specs).fit(X_train, y_train)
```

### Sklearn pipeline compatibility

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("woe", WOETransformer(bin_specs=sg.bin_specs_)),
    ("lr",  LogisticRegression()),
])
pipe.fit(X_train, y_train.to_numpy())
```
