# datasci-toolkit

My personal data science toolkit — a Polars-native Python library for supervised learning workflows.

## Modules

| Module | Classes / Functions | Description |
|---|---|---|
| [`stability`](api/stability.md) | `PSI`, `ESI`, `StabilityMonitor` | Population and event stability indices |
| [`grouping`](api/grouping.md) | `StabilityGrouping`, `WOETransformer` | Stability-constrained optimal binning and WOE encoding |
| [`metrics`](api/metrics.md) | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power`, `gini_by_period`, `lift_by_period` | Binary classification metrics |
| [`model_selection`](api/model_selection.md) | `AUCStepwiseLogit` | Gini-based stepwise logistic regression |
| [`label_imputation`](api/label_imputation.md) | `KNNLabelImputer`, `TargetImputer` | Missing label imputation |
| [`bin_editor`](api/bin_editor.md) | `BinEditor`, `BinEditorWidget` | Headless and interactive bin boundary editor |
| [`variable_clustering`](api/variable_clustering.md) | `CorrVarClus` | Hierarchical correlation clustering |

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

# 3. Stepwise feature selection
model = AUCStepwiseLogit(max_predictors=10, max_correlation=0.8).fit(
    X_woe.select(features), y_train,
    X_val=X_val_woe.select(features), y_val=y_val,
)
```

## Stack

- Python 3.12, `polars` — no pandas
- `scikit-learn` estimator conventions (`fit` / `transform` / `score`)
- `matplotlib` for standalone plot functions
