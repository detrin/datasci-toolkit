# Model Selection — AUCStepwiseLogit

Stepwise logistic regression that selects features by Gini improvement rather than p-values.

## Why Gini instead of p-values?

P-value stepwise selection optimises for statistical inference. Gini-based selection directly maximises predictive performance — which is what you actually care about in production models.

Additional constraints for regulated environments:

- `max_correlation` — rejects candidates correlated with already-selected features
- `enforce_coef_sign` — rejects features whose coefficient flips sign when added (scorecard monotonicity)
- `use_cv` — evaluate on cross-validation folds instead of a fixed validation set

## Basic usage

```python
import numpy as np
import polars as pl
from datasci_toolkit import AUCStepwiseLogit

rng = np.random.default_rng(0)
N = 2000

f0 = rng.normal(0, 1, N)
f1 = rng.normal(0, 1, N)
f2 = rng.normal(0, 1, N)
f0_clone = f0 + rng.normal(0, 0.1, N)   # nearly identical to f0
noise    = rng.normal(0, 1, N)

logit = 0.8 * f0 + 0.5 * f1 + 0.3 * f2
y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=N)).astype(float)

X_train = pl.DataFrame({
    "f0": f0[:1500].tolist(), "f1": f1[:1500].tolist(),
    "f2": f2[:1500].tolist(), "f0_clone": f0_clone[:1500].tolist(),
    "noise": noise[:1500].tolist(),
})
X_val = pl.DataFrame({
    "f0": f0[1500:].tolist(), "f1": f1[1500:].tolist(),
    "f2": f2[1500:].tolist(), "f0_clone": f0_clone[1500:].tolist(),
    "noise": noise[1500:].tolist(),
})
y_train = pl.Series(y[:1500].tolist())
y_val   = pl.Series(y[1500:].tolist())

model = AUCStepwiseLogit(
    selection_method="stepwise",
    min_increase=0.002,
    max_correlation=0.8,
    max_predictors=5,
).fit(X_train, y_train, X_val=X_val, y_val=y_val)

print("Selected:", model.predictors_)
print("Validation Gini:", model.score(X_val, y_val))
```

`f0_clone` will be rejected because it is correlated > 0.8 with `f0`.

## Inspecting selection progress

`progress_` is a DataFrame logging every add/remove step.

```python
# Rows where addrm == 0 = feature was added
print(model.progress_.filter(model.progress_["addrm"] == 0))
```

## Coefficients

```python
print(pl.DataFrame({
    "predictor":   model.predictors_,
    "coefficient": [round(float(c), 4) for c in model.coef_],
}))
```

## Cross-validated selection

Pass `use_cv=True` to score via k-fold CV on the training set — avoids overfitting the selection to the validation set.

```python
model_cv = AUCStepwiseLogit(
    selection_method="forward",
    min_increase=0.002,
    max_correlation=0.8,
    use_cv=True,
    cv_folds=5,
    cv_seed=42,
).fit(X_train, y_train)

print("CV-selected:", model_cv.predictors_)
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `selection_method` | `"stepwise"` | `"forward"`, `"backward"`, or `"stepwise"` |
| `min_increase` | `0.005` | Minimum Gini gain required to add a feature |
| `max_decrease` | `0.0025` | Maximum Gini drop allowed before removing a feature (stepwise) |
| `max_predictors` | `0` | Hard cap on number of features (0 = unlimited) |
| `max_correlation` | `1.0` | Reject candidates correlated above this threshold |
| `enforce_coef_sign` | `False` | Reject features that flip coefficient sign |
| `use_cv` | `False` | Score via cross-validation instead of validation set |
| `cv_folds` | `5` | Number of CV folds |
