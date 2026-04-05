# datasci-toolkit

Polars-native Python toolkit for binary classification, scorecard development, and model validation.

## Modules

| Module | Classes / Functions | Description |
|---|---|---|
| [`stability`](api/stability.md) | `PSI`, `ESI`, `StabilityMonitor` | Population and event stability indices |
| [`grouping`](api/grouping.md) | `StabilityGrouping`, `WOETransformer` | Stability-constrained optimal binning and WOE encoding |
| [`metrics`](api/metrics.md) | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power` | Binary classification metrics with period breakdowns |
| [`model_selection`](api/model_selection.md) | `AUCStepwiseLogit` | Gini-based stepwise logistic regression |
| [`feature_elimination`](api/feature_elimination.md) | `ShapImportance`, `ShapRFE` | SHAP-based backward feature elimination with CV |
| [`label_imputation`](api/label_imputation.md) | `KNNLabelImputer`, `TargetImputer` | Missing label imputation |
| [`bin_editor`](api/bin_editor.md) | `BinEditor`, `BinEditorWidget` | Headless and interactive bin boundary editor |
| [`variable_clustering`](api/variable_clustering.md) | `CorrVarClus` | Hierarchical correlation clustering |
| [`temporal`](tutorials/temporal.md) | `TemporalFeatureEngineer` | Time-based feature generation |
| [`smoothing`](api/smoothing.md) | `PoissonSmoother`, `PredictionSmoother` | Adaptive temporal smoothing |
| [`tagging`](api/tagging.md) | `WeightedTFIDF` | Weighted TF-IDF entity tagging |

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
