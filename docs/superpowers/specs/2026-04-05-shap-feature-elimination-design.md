# SHAP Feature Elimination — Design Spec

## Overview

Backward feature elimination using SHAP importance with cross-validation. Extracted from ING's probatus package, rewritten for Polars + MLE style. LightGBM and XGBoost support only.

## Module Structure

```
datasci_toolkit/feature_elimination/
    __init__.py          # exports ShapImportance, ShapRFE, plot_shap_elimination
    importance.py        # ShapImportance
    elimination.py       # ShapRFE
    _shap.py             # compute_shap_values, shap_importance
    _plot.py             # plot_shap_elimination

tests/test_feature_elimination.py
```

## Classes

### `ShapImportance(BaseEstimator)` — `importance.py`

One-shot SHAP feature ranker. Fits model across CV folds, computes SHAP on validation folds, aggregates importance.

**Constructor params:**

| Param | Type | Default | Purpose |
|---|---|---|---|
| `model` | Any | required | sklearn-compatible estimator |
| `cv` | int \| BaseCrossValidator | 5 | CV strategy |
| `scoring` | str | "roc_auc" | sklearn scorer string |
| `n_jobs` | int | -1 | joblib parallelism |
| `random_state` | int \| None | None | reproducibility |
| `importance_method` | str | "mean" | "mean" or "variance_penalized" |
| `variance_penalty_factor` | float | 0.5 | penalty weight (only when method="variance_penalized") |

**Fitted attributes:**

| Attribute | Type | Content |
|---|---|---|
| `feature_importances_` | pl.DataFrame | columns: feature, importance, std — sorted desc by importance |

**Methods:**

- `fit(X: pl.DataFrame, y: pl.Series) -> self` — runs CV, computes SHAP, stores importances
- `compute() -> pl.DataFrame` — returns `feature_importances_` (requires fitted)

**Algorithm:**

1. Clone model per CV fold
2. Fit clone on train fold
3. Compute SHAP values on val fold via `_shap.compute_shap_values()`
4. Concatenate SHAP arrays across folds
5. Aggregate via `_shap.shap_importance()` using configured method
6. Store as `feature_importances_`

---

### `ShapRFE(BaseEstimator)` — `elimination.py`

Recursive backward elimination. Each round creates a `ShapImportance`, identifies lowest-importance features, removes them, logs metrics.

**Constructor params:**

| Param | Type | Default | Purpose |
|---|---|---|---|
| `model` | Any | required | sklearn-compatible estimator |
| `step` | int \| float | 1 | int=N features removed per round, float=fraction |
| `min_features_to_select` | int | 1 | stopping criterion |
| `cv` | int \| BaseCrossValidator | 5 | CV strategy |
| `scoring` | str | "roc_auc" | sklearn scorer string |
| `n_jobs` | int | -1 | joblib parallelism |
| `random_state` | int \| None | None | reproducibility |
| `importance_method` | str | "mean" | "mean" or "variance_penalized" |
| `variance_penalty_factor` | float | 0.5 | penalty weight |
| `columns_to_keep` | list[str] \| None | None | protected features |

**Fitted attributes:**

| Attribute | Type | Content |
|---|---|---|
| `report_df_` | pl.DataFrame | round, n_features, features, eliminated, train_score_mean, train_score_std, val_score_mean, val_score_std |
| `feature_names_` | list[str] | best feature set (from "best" method) |

**Methods:**

- `fit(X: pl.DataFrame, y: pl.Series) -> self` — runs elimination loop
- `compute() -> pl.DataFrame` — returns `report_df_`
- `get_reduced_features(method: str = "best", se_threshold: float = 1.0) -> list[str]` — feature selection from results

**Selection methods for `get_reduced_features`:**

| Method | Logic |
|---|---|
| `"best"` | Round with highest val_score_mean |
| `"best_coherent"` | Most features within se_threshold * SE of best val score |
| `"best_parsimonious"` | Fewest features within se_threshold * SE of best val score |

**Elimination algorithm:**

```
remaining = all features from X
WHILE len(remaining) > min_features_to_select:
    imp = ShapImportance(model, cv, scoring, ...).fit(X[remaining], y)
    
    # identify features to remove
    removable = [f for f in imp.feature_importances_ if f not in columns_to_keep]
    n_remove = step if int else floor(len(remaining) * step), min 1
    n_remove = min(n_remove, len(remaining) - min_features_to_select)
    to_remove = bottom n_remove from removable by importance
    
    # log round
    append to report_df_: round metrics from ShapImportance CV scores
    
    remaining = remaining - to_remove

feature_names_ = best feature set from report_df_
```

---

## Helper Functions

### `_shap.py`

**`compute_shap_values(model, X: pl.DataFrame) -> np.ndarray`**

- Input: fitted model + Polars DataFrame
- Converts X to numpy internally
- TreeExplainer for LightGBM/XGBoost (isinstance check)
- Generic Explainer with background sample (100 rows) for anything else
- Returns: np.ndarray shape (n_samples, n_features)

**`shap_importance(shap_values: np.ndarray, columns: list[str], method: str, variance_penalty_factor: float) -> pl.DataFrame`**

- `"mean"`: mean |SHAP| per feature
- `"variance_penalized"`: mean|SHAP| - factor * std|SHAP|
- Returns: pl.DataFrame(feature, importance, std) sorted desc by importance

### `_plot.py`

**`plot_shap_elimination(report: pl.DataFrame, show: bool = True) -> Figure`**

- Input: `report_df_` from ShapRFE
- X-axis: n_features (inverted, right to left)
- Y-axis: metric score
- Two lines: train mean, val mean
- Shaded ±1 std bands
- Returns matplotlib Figure

## Dependencies

- `shap` — needs adding to pyproject.toml
- `xgboost` — needs adding to pyproject.toml
- `lightgbm` — already in pyproject.toml
- `scikit-learn` — already present (BaseEstimator, clone, check_is_fitted, cross-validation)
- `joblib` — comes with sklearn
- `numpy`, `polars`, `matplotlib` — already present

## Package Exports

Add to `datasci_toolkit/__init__.py`:

```python
from datasci_toolkit.feature_elimination import ShapImportance, ShapRFE, plot_shap_elimination
```

Add to `__all__`:

```python
"ShapImportance",
"ShapRFE",
"plot_shap_elimination",
```

## Test Plan

`tests/test_feature_elimination.py`:

- ShapImportance: fit returns self, feature_importances_ has correct schema, sorted desc, both importance methods produce valid output, works with LightGBM, works with XGBoost
- ShapRFE: fit runs full loop, report_df_ has correct schema, features decrease each round, columns_to_keep survives all rounds, step=int vs step=float, min_features_to_select respected
- get_reduced_features: "best" returns highest-scoring set, "best_coherent" returns most features within SE, "best_parsimonious" returns fewest within SE
- plot_shap_elimination: returns Figure, no errors on valid report_df_
- Edge cases: single feature, step larger than remaining features, all features in columns_to_keep
