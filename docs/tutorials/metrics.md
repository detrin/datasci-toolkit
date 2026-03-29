# Metrics

Binary classification metrics with Polars-native inputs and period-level breakdowns.

## Point-in-time metrics

```python
import numpy as np
import polars as pl
from datasci_toolkit.metrics import gini, ks, lift, iv, feature_power

rng = np.random.default_rng(0)
N = 1000

score = rng.uniform(0, 1, N)
y = (score + rng.normal(0, 0.25, N) > 0.5).astype(float)

y_s  = pl.Series(y.tolist())
sc_s = pl.Series(score.tolist())

print(f"Gini  = {gini(y_s, sc_s):.4f}")
print(f"KS    = {ks(y_s, sc_s):.4f}")
print(f"Lift@10% = {lift(y_s, -sc_s, perc=10.0):.4f}")
```

### Information Value

```python
binned = (sc_s > 0.5).cast(pl.Int32)
print(f"IV = {iv(y_s, binned):.4f}")
```

### Feature power — scan a whole DataFrame

Returns Gini and IV for every column, sorted descending.

```python
X = pl.DataFrame({
    "strong": (-sc_s).to_list(),
    "medium": (pl.Series(rng.normal(0, 1, N).tolist()) * 0.5).to_list(),
    "noise":  rng.normal(0, 1, N).tolist(),
})
print(feature_power(X, y_s))
```

## Bootstrap confidence interval

Estimates the sampling variability of Gini via bootstrap resampling.

```python
from datasci_toolkit.metrics import BootstrapGini

bg = BootstrapGini(n_iter=500, ci_level=95.0, seed=42).fit(y_s, sc_s)

print(f"Mean Gini = {bg.mean_:.4f}")
print(f"Std       = {bg.std_:.4f}")
print(f"95% CI    = [{bg.ci_[0]:.4f}, {bg.ci_[1]:.4f}]")
```

## Period metrics

Evaluate performance sliced by a time or cohort column.

```python
from datasci_toolkit.metrics import gini_by_period, lift_by_period

periods = pl.Series(np.repeat(np.arange(5), N // 5).tolist())

gini_df = gini_by_period(y_s, sc_s, periods)
lift_df = lift_by_period(y_s, -sc_s, periods, perc=10.0)

print(gini_df.join(lift_df.select(["period", "lift"]), on="period"))
```

Periods where all records belong to a single class are skipped automatically (they can't produce a meaningful Gini).

### With a population mask

Restrict scoring to a sub-population — e.g., only approved applications.

```python
mask = pl.Series((rng.uniform(size=N) > 0.3).tolist())
gini_approved = gini_by_period(y_s, sc_s, periods, mask=mask)
```

### Weighted metrics

```python
weights = pl.Series(rng.uniform(0.5, 1.5, N).tolist())
gini_weighted = gini_by_period(y_s, sc_s, periods, sample_weight=weights)
```

## Plotting period metrics

```python
from datasci_toolkit.metrics import plot_metric_by_period

plot_metric_by_period(
    gini_df["period"].to_list(),
    [gini_df["gini"].to_list(), lift_df["lift"].to_list()],
    gini_df["count"].to_list(),
    labels=["Gini", "Lift@10%"],
    ylabel="Score",
    title="Performance over time",
)
```

Multi-series overlay — primary y-axis is bar chart of record counts, secondary y-axis is line chart per metric series.
