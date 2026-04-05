# datasci-toolkit

Polars-native Python toolkit for binary classification, scorecard development, and model validation.

## Modules

### Monitoring & Stability

| Module | Classes / Functions | Use case |
|---|---|---|
| [`stability`](api/stability.md) | `PSI`, `ESI`, `StabilityMonitor` | Detect population drift between training and production -- catch when input distributions shift before model performance degrades |
| [`metrics`](api/metrics.md) | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power` | Evaluate binary classifiers with confidence intervals -- report Gini/KS/lift by month, identify which features drive predictive power |

### Feature Engineering & Selection

| Module | Classes / Functions | Use case |
|---|---|---|
| [`grouping`](api/grouping.md) | `StabilityGrouping`, `WOETransformer` | Bin continuous features into stable WOE-encoded groups for scorecard development -- ensures bins don't drift across time periods |
| [`feature_elimination`](api/feature_elimination.md) | `ShapImportance`, `ShapRFE` | Reduce a 500-feature dataset to the 20 that matter -- backward elimination using SHAP values with cross-validation |
| [`variable_clustering`](api/variable_clustering.md) | `CorrVarClus` | Remove redundant features before modeling -- hierarchical clustering picks one representative from each correlated group |
| [`temporal`](tutorials/temporal.md) | `TemporalFeatureEngineer` | Generate time-windowed aggregations (sum/mean/max over 30d/90d/1y) from transaction histories for credit scoring or churn prediction |
| [`tagging`](api/tagging.md) | `WeightedTFIDF` | Profile entities with ranked tags -- find top product attributes from reviews, build customer interest profiles from transactions, with external quality signals |

### Model Building & Post-processing

| Module | Classes / Functions | Use case |
|---|---|---|
| [`model_selection`](api/model_selection.md) | `AUCStepwiseLogit` | Build interpretable scorecards -- stepwise logistic regression that adds features by Gini lift and enforces correlation constraints |
| [`bin_editor`](api/bin_editor.md) | `BinEditor`, `BinEditorWidget` | Manually adjust bin boundaries after auto-binning -- headless API for pipelines, interactive widget for notebooks |
| [`label_imputation`](api/label_imputation.md) | `KNNLabelImputer`, `TargetImputer` | Recover labels for rejected loan applications (reject inference) or fill missing targets in semi-supervised settings |
| [`smoothing`](api/smoothing.md) | `PoissonSmoother`, `PredictionSmoother` | Stabilize noisy count features before modeling (Poisson), or eliminate monthly prediction jitter so customers don't flip between risk tiers |

## Install

```bash
pip install datasci-toolkit
```

## Quick start

```python
import polars as pl
from lightgbm import LGBMClassifier
from datasci_toolkit import ShapRFE, StabilityGrouping, AUCStepwiseLogit

# 1. Stability-constrained binning
sg = StabilityGrouping(stability_threshold=0.1).fit(
    X_train, y_train, t_train=month_train,
    X_val=X_val, y_val=y_val, t_val=month_val,
)
X_woe = sg.transform(X_test)

# 2. SHAP-based feature elimination
rfe = ShapRFE(
    model=LGBMClassifier(n_estimators=100, verbose=-1),
    step=1, cv=5, min_features_to_select=5,
).fit(X_woe, y_train)
features = rfe.get_reduced_features("best_parsimonious")

# 3. Stepwise logistic regression
model = AUCStepwiseLogit(max_predictors=10, max_correlation=0.8).fit(
    X_woe.select(features), y_train,
    X_val=X_val_woe.select(features), y_val=y_val,
)
```

## Stack

- Python 3.12, `polars` -- no pandas
- `scikit-learn` estimator conventions (`fit` / `transform` / `score`)
- `shap` + `lightgbm` + `xgboost` for SHAP-based feature selection
- `matplotlib` for standalone plot functions
