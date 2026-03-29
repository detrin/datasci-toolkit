# Stability Index — PSI & ESI

Monitor whether feature distributions and event rates shift over time.

## PSI — Population Stability Index

Measures how much a feature's distribution has changed between a reference period and a monitoring period. PSI < 0.1 is stable, 0.1–0.2 is marginal, > 0.2 indicates significant drift.

```python
import numpy as np
import polars as pl
from datasci_toolkit import PSI

rng = np.random.default_rng(0)
N = 1000

# Reference distribution (training data)
X_ref = pl.Series(rng.normal(0, 1, N).tolist())

# Monitoring distribution — slight drift
X_mon = pl.Series(rng.normal(0.3, 1.1, N).tolist())

psi = PSI(q=10).fit(X_ref)
score = psi.score(X_mon)
print(f"PSI = {score:.4f}")
```

### Categorical features

PSI works on categorical series too — it uses the label frequency distribution directly rather than quantile binning.

```python
X_ref_cat = pl.Series(rng.choice(["A", "B", "C"], N, p=[0.5, 0.3, 0.2]).tolist())
X_mon_cat = pl.Series(rng.choice(["A", "B", "C"], N, p=[0.4, 0.4, 0.2]).tolist())

psi_cat = PSI().fit(X_ref_cat)
print(f"PSI (categorical) = {psi_cat.score(X_mon_cat):.4f}")
```

## StabilityMonitor — multiple features at once

Fits one PSI per feature and can score against any time slice of a DataFrame.

```python
from datasci_toolkit import StabilityMonitor

N_MONTHS = 8
months = np.repeat(np.arange(N_MONTHS), N // N_MONTHS)

f0 = rng.normal(0, 1, N)
f1 = rng.normal(0, 1, N)

# Introduce drift after month 5
f0[months >= 5] += 0.4

df = pl.DataFrame({
    "f0": f0.tolist(),
    "f1": f1.tolist(),
    "month": months.tolist(),
})

# Fit on months 0–4 as reference
monitor = StabilityMonitor(features=["f0", "f1"]).fit(
    df.filter(pl.col("month") < 5)
)

# Score each month against the reference
psi_df = monitor.score(df, col_month="month")
print(psi_df)
```

### Consecutive-period scoring

Fit on month N, score month N+1 — useful when there's no fixed reference baseline.

```python
consec_df = monitor.score_consecutive(df, col_month="month")
print(consec_df)
```

### Mask-based scoring

Compare arbitrary slices — e.g., approved vs. rejected population.

```python
masks = {
    "approved": pl.Series((rng.uniform(size=N) > 0.3).tolist()),
    "rejected": pl.Series((rng.uniform(size=N) <= 0.3).tolist()),
}
mask_df = monitor.score_masks(df, mask_dict=masks)
print(mask_df)
```

## ESI — Event Stability Index

Measures rank stability of a model score across time periods. Returns two variants: V1 (rank correlation) and V2 (event rate ratio).

```python
from datasci_toolkit import ESI

score_col = pl.Series(rng.uniform(0, 1, N).tolist())
target = (score_col + pl.Series(rng.normal(0, 0.3, N).to_numpy()) > 0.5).cast(pl.Int32)

df_esi = pl.DataFrame({
    "score": score_col.to_list(),
    "target": target.to_list(),
    "month": months.tolist(),
    "base": pl.Series(np.ones(N).tolist()),
})

esi = ESI()
result = esi.score(
    df_esi,
    var="score",
    col_target="target",
    col_base="base",
    col_month="month",
)
print(result)  # {"v1": ..., "v2": ...}
```

## Plotting

```python
from datasci_toolkit.stability import plot_psi_comparison

periods = list(range(N_MONTHS))
psi_values = psi_df["f0"].to_list()

plot_psi_comparison(
    periods,
    [psi_values],
    labels=["f0"],
    title="PSI over time",
)
```
