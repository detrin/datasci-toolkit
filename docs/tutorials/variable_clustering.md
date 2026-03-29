# Variable Clustering — CorrVarClus

Groups features by correlation structure to identify redundant predictors before fitting a logistic regression.

## Algorithm

1. Drop zero-variance columns
2. Hierarchical clustering with `metric="correlation"`, `method="average"`
3. Cut the dendrogram at `max_correlation` (or `max_clusters`)
4. Rank features within each cluster by absolute Gini (descending)

`best_features()` returns the top-ranked feature per cluster.

## Basic usage

```python
import numpy as np
import polars as pl
from datasci_toolkit import CorrVarClus

rng = np.random.default_rng(0)
N = 800

# Group 1: f0, f1, f2 — strongly correlated
f0 = rng.normal(0, 1, N)
f1 = f0 + rng.normal(0, 0.2, N)
f2 = f0 + rng.normal(0, 0.3, N)

# Group 2: f3, f4 — correlated with each other, not with group 1
f3 = rng.normal(0, 1, N)
f4 = f3 + rng.normal(0, 0.2, N)

# Noise
f5 = rng.normal(0, 1, N)

logit = 1.2 * f0 + 0.8 * f3 + 0.1 * f5
y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=N)).astype(float)

X = pl.DataFrame({
    "f0": f0.tolist(), "f1": f1.tolist(), "f2": f2.tolist(),
    "f3": f3.tolist(), "f4": f4.tolist(), "f5": f5.tolist(),
})
y_s = pl.Series(y.tolist())

cc = CorrVarClus(max_correlation=0.5).fit(X, y_s)
print(cc.best_features())  # → ['f0', 'f3', 'f5']  (one per cluster)
```

## Cluster table

```python
print(cc.cluster_table_.sort(["cluster", "rank"]))
# columns: feature, cluster, gini, rank
```

`rank=1` is the best feature in each cluster by absolute Gini.

## Dendrogram

```python
cc.plot_dendrogram()
# The dashed line shows the max_correlation cut
```

Save to file:

```python
cc.plot_dendrogram(output_file="dendrogram.png", show=False)
```

## Controlling cluster granularity

**By correlation threshold** (lower = more clusters):

```python
cc_strict = CorrVarClus(max_correlation=0.2).fit(X, y_s)
print(len(cc_strict.cluster_table_["cluster"].unique()), "clusters")
```

**By count** (hard cap):

```python
cc_3 = CorrVarClus(max_clusters=3).fit(X, y_s)
print(cc_3.best_features())
```

## Full workflow with model selection

```python
from datasci_toolkit import CorrVarClus, AUCStepwiseLogit

# Step 1: cluster to remove redundancy
cc = CorrVarClus(max_correlation=0.5).fit(X_woe, y_train)
candidates = cc.best_features()

# Step 2: stepwise selection from the reduced candidate set
model = AUCStepwiseLogit(
    max_predictors=8,
    max_correlation=0.8,
).fit(
    X_woe.select(candidates), y_train,
    X_val=X_val_woe.select(candidates), y_val=y_val,
)
print("Final predictors:", model.predictors_)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_correlation` | `0.5` | Dendrogram cut height — features correlated above this are in the same cluster |
| `max_clusters` | `None` | Hard cap on number of clusters (overrides `max_correlation`) |
| `sample` | `0` | Subsample rows before clustering for large datasets (0 = use all) |
