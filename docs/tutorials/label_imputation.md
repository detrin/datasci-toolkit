# Label Imputation

Handles the **missing label problem**: part of the population has unknown outcomes and can't be used directly for supervised training.

Common scenarios:

- Loan applicants who were rejected — their default status is never observed
- Holdout / control groups in A/B tests
- Records from before a monitoring system was deployed

## KNNLabelImputer

Finds k nearest labeled neighbours in feature space for each unlabeled record. Distance-weighted average of neighbour labels gives P(event).

```python
import numpy as np
import polars as pl
from datasci_toolkit import KNNLabelImputer

rng = np.random.default_rng(0)
N_labeled   = 500
N_unlabeled = 200

X_lab = pl.DataFrame({f"f{i}": rng.normal(0, 1, N_labeled).tolist() for i in range(4)})
y_lab = pl.Series(((X_lab["f0"] + X_lab["f1"]).to_numpy() > 0).astype(float).tolist())

# Unlabeled comes from a biased region (selection bias)
X_unl = pl.DataFrame({
    "f0": rng.normal(0.5, 1, N_unlabeled).tolist(),
    **{f"f{i}": rng.normal(0, 1, N_unlabeled).tolist() for i in range(1, 4)},
})

imputer = KNNLabelImputer(n_neighbors=10, method="weighted").fit(X_lab, y_lab)

# P(event) for each unlabeled record
proba = imputer.predict_proba(X_unl)
print(proba[:5])
```

### `transform()` — weighted duplication

Converts each unlabeled record into two rows: `(target=1, weight=p̂)` and `(target=0, weight=1−p̂)`. This preserves expected event count while keeping all records in training.

```python
imputed = imputer.transform(X_unl)
print(imputed.head(6))
# Each original row becomes two rows with complementary weights
```

### Combining labeled and imputed data

```python
# Labeled data
labeled = X_lab.with_columns([
    y_lab.alias("target"),
    pl.lit(1.0).alias("weight"),
])

# Stack labeled + imputed
training_set = pl.concat([labeled, imputed], how="diagonal")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_neighbors` | `10` | Number of nearest neighbours |
| `metric` | `"minkowski"` | Distance metric (any sklearn-compatible metric) |
| `method` | `"weighted"` | `"weighted"` (distance-weighted) or `"uniform"` |

## TargetImputer

When you already have probabilities from another source and just need to convert them to training rows.

```python
from datasci_toolkit import TargetImputer
import numpy as np

proba = np.array([0.1, 0.3, 0.55, 0.7, 0.9])
```

### Three strategies

**`weighted`** — duplicate each row: `(target=1, w=p)` + `(target=0, w=1−p)`

```python
t = TargetImputer(method="weighted").fit(proba)
print(t.transform())
```

**`randomized`** — Bernoulli draw: single row, `target ∈ {0, 1}`

```python
t = TargetImputer(method="randomized", seed=42).fit(proba)
print(t.transform())
```

**`cutoff`** — hard threshold at `cutoff` (default 0.5)

```python
t = TargetImputer(method="cutoff", cutoff=0.5).fit(proba)
print(t.transform())
```
