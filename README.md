# datasci-toolkit

My personal Python toolkit for data science — a clean rewrite of tools I use day-to-day for binary classification, scorecard development, and model validation.

Polars-native, sklearn-compatible, zero external state.

## Modules

| Module | Classes / Functions | Description |
|---|---|---|
| `stability` | `PSI`, `ESI`, `StabilityMonitor` | Population and event stability indices |
| `grouping` | `StabilityGrouping`, `WOETransformer` | Stability-constrained optimal binning and WOE encoding |
| `metrics` | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power`, `gini_by_period`, `lift_by_period`, `plot_metric_by_period` | Binary classification metrics with period breakdowns |
| `model_selection` | `AUCStepwiseLogit` | Gini-based stepwise logistic regression |
| `label_imputation` | `KNNLabelImputer`, `TargetImputer` | KNN imputation for records with missing labels |
| `bin_editor` | `BinEditor`, `BinEditorWidget` | Headless and interactive bin boundary editor |
| `variable_clustering` | `CorrVarClus` | Hierarchical correlation clustering for variable reduction |

## Quick start

```python
import polars as pl
from datasci_toolkit import StabilityGrouping, AUCStepwiseLogit, CorrVarClus

# 1. Stability-constrained binning
sg = StabilityGrouping(stability_threshold=0.1).fit(
    X_train, y_train, t_train=month_train,
    X_val=X_val, y_val=y_val, t_val=month_val,
)
X_woe = sg.transform(X_test)

# 2. Remove correlated features
cc = CorrVarClus(max_correlation=0.5).fit(X_woe, y_train)
features = cc.best_features()

# 3. Stepwise selection
model = AUCStepwiseLogit(max_predictors=10, max_correlation=0.8).fit(
    X_woe.select(features), y_train,
    X_val=X_val_woe.select(features), y_val=y_val,
)
```

## Documentation

**[detrin.github.io/datasci-toolkit](https://detrin.github.io/datasci-toolkit)**

## Stack

- Python 3.12, `polars` — no pandas
- `scikit-learn` for estimator conventions
- `matplotlib` for standalone plot functions
