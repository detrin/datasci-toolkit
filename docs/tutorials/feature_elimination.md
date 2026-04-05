# Feature Elimination — SHAP-based Backward Selection

Select features by iteratively removing the least important ones according to SHAP values, with cross-validated performance tracking at each step.

## Why SHAP instead of model feature importance?

Tree-based feature importances (gain, split count) measure how much the model uses a feature — not how much it contributes to predictions. SHAP values measure actual marginal contribution, giving a more reliable ranking for feature elimination.

## One-shot ranking with ShapImportance

Rank features by SHAP importance without any elimination loop.

```python
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from datasci_toolkit import ShapImportance

rng = np.random.default_rng(0)
N = 2000

f0 = rng.normal(0, 1, N)
f1 = rng.normal(0, 1, N)
f2 = rng.normal(0, 1, N)
noise1 = rng.normal(0, 1, N)
noise2 = rng.normal(0, 1, N)

logit = 0.8 * f0 + 0.5 * f1 + 0.2 * f2
y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=N)).astype(int)

X = pl.DataFrame({
    "f0": f0.tolist(), "f1": f1.tolist(), "f2": f2.tolist(),
    "noise1": noise1.tolist(), "noise2": noise2.tolist(),
})
y = pl.Series("target", y.tolist())

ranker = ShapImportance(
    model=LGBMClassifier(n_estimators=50, verbose=-1, random_state=42),
    cv=5,
    scoring="roc_auc",
    random_state=42,
)
ranker.fit(X, y)
print(ranker.feature_importances_)
```

The informative features (`f0`, `f1`, `f2`) should rank highest, with `noise1` and `noise2` at the bottom.

## Variance-penalized importance

Features with high SHAP variance across CV folds may be unreliable. The `variance_penalized` method penalises them:

`importance = mean|SHAP| - factor * std|SHAP|`

```python
ranker_penalized = ShapImportance(
    model=LGBMClassifier(n_estimators=50, verbose=-1, random_state=42),
    cv=5,
    scoring="roc_auc",
    random_state=42,
    importance_method="variance_penalized",
    variance_penalty_factor=1.0,
)
ranker_penalized.fit(X, y)
print(ranker_penalized.feature_importances_)
```

A higher `variance_penalty_factor` penalises unstable features more aggressively.

## Backward elimination with ShapRFE

Iteratively remove the least important features and track how performance changes.

```python
from datasci_toolkit import ShapRFE

rfe = ShapRFE(
    model=LGBMClassifier(n_estimators=50, verbose=-1, random_state=42),
    step=1,
    min_features_to_select=1,
    cv=5,
    scoring="roc_auc",
    random_state=42,
)
rfe.fit(X, y)
print(rfe.report_df_)
```

Each row in `report_df_` logs one elimination round: the feature set, which features were removed, and train/validation scores.

## Choosing the right feature set

Three selection strategies after elimination:

```python
best = rfe.get_reduced_features("best")
print("Best:", best)

coherent = rfe.get_reduced_features("best_coherent")
print("Most features within 1 SE of best:", coherent)

parsimonious = rfe.get_reduced_features("best_parsimonious")
print("Fewest features within 1 SE of best:", parsimonious)
```

| Method | Logic |
|---|---|
| `"best"` | Feature set with the highest validation score |
| `"best_coherent"` | Most features whose score is within `se_threshold * SE` of the best |
| `"best_parsimonious"` | Fewest features whose score is within `se_threshold * SE` of the best |

`"best_parsimonious"` is useful in regulated environments where fewer features means simpler model documentation.

## Removing multiple features per round

Use `step` as a float to remove a fraction of features each round — faster for high-dimensional datasets.

```python
rfe_fast = ShapRFE(
    model=LGBMClassifier(n_estimators=50, verbose=-1, random_state=42),
    step=0.2,
    min_features_to_select=1,
    cv=5,
    random_state=42,
)
rfe_fast.fit(X, y)
print(rfe_fast.report_df_)
```

`step=0.2` removes 20% of remaining features per round (minimum 1).

## Protecting features

Force specific features to survive all elimination rounds:

```python
rfe_keep = ShapRFE(
    model=LGBMClassifier(n_estimators=50, verbose=-1, random_state=42),
    step=1,
    min_features_to_select=1,
    cv=5,
    random_state=42,
    columns_to_keep=["f0"],
)
rfe_keep.fit(X, y)

last_features = rfe_keep.report_df_["features"].to_list()[-1]
print("Final features:", last_features)
assert "f0" in last_features
```

## XGBoost support

Works the same way — just swap the model:

```python
from xgboost import XGBClassifier

rfe_xgb = ShapRFE(
    model=XGBClassifier(n_estimators=50, verbosity=0, random_state=42),
    step=1,
    min_features_to_select=2,
    cv=5,
    random_state=42,
)
rfe_xgb.fit(X, y)
print(rfe_xgb.get_reduced_features("best"))
```

## Visualising the elimination curve

```python
from datasci_toolkit import plot_shap_elimination

fig = plot_shap_elimination(rfe.report_df_)
```

Shows train and validation scores (with +-1 std bands) as features are removed from right to left.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | — | Any sklearn-compatible classifier (LightGBM, XGBoost, etc.) |
| `step` | `1` | Features to remove per round — int for count, float for fraction |
| `min_features_to_select` | `1` | Stop when this many features remain |
| `cv` | `5` | Number of CV folds or a sklearn CV splitter |
| `scoring` | `"roc_auc"` | sklearn scoring metric |
| `importance_method` | `"mean"` | `"mean"` or `"variance_penalized"` |
| `variance_penalty_factor` | `0.5` | Penalty weight for variance method |
| `columns_to_keep` | `None` | Features protected from elimination |
| `random_state` | `None` | Reproducibility seed |
