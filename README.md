# datasci-toolkit

Clean Python library for credit scoring and binary classification workflows.
Polars-native, sklearn-compatible, zero external state.

## Modules

| Module | Classes / Functions | Description |
|---|---|---|
| `stability` | `PSI`, `ESI`, `StabilityMonitor` | Population and event stability indices |
| `grouping` | `StabilityGrouping`, `WOETransformer` | Stability-constrained optimal binning and WOE encoding |
| `metrics` | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power`, `gini_by_period`, `lift_by_period`, `plot_metric_by_period` | Binary classification metrics with period breakdowns |
| `model_selection` | `AUCStepwiseLogit` | Gini-based stepwise logistic regression |
| `label_imputation` | `KNNLabelImputer`, `TargetImputer` | WOE-space KNN imputation for missing labels |
| `bin_editor` | `BinEditor`, `BinEditorWidget` | Headless and interactive bin boundary editor |
| `variable_clustering` | `CorrVarClus` | Hierarchical correlation clustering for variable reduction |

## Install

```bash
pip install datasci-toolkit
```

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

Interactive examples are published to GitHub Pages:
**[detrin.github.io/datasci-toolkit](https://detrin.github.io/datasci-toolkit)**

| Notebook | Topic |
|---|---|
| [01 Stability](https://detrin.github.io/datasci-toolkit/01_stability.html) | PSI drift detection, StabilityMonitor, ESI |
| [02 Grouping](https://detrin.github.io/datasci-toolkit/02_grouping.html) | StabilityGrouping, WOETransformer |
| [03 Metrics](https://detrin.github.io/datasci-toolkit/03_metrics.html) | Gini, KS, lift, IV, bootstrap CI, period breakdowns |
| [04 Model selection](https://detrin.github.io/datasci-toolkit/04_model_selection.html) | AUCStepwiseLogit, correlation filter, CV mode |
| [05 Label imputation](https://detrin.github.io/datasci-toolkit/05_label_imputation.html) | KNNLabelImputer, TargetImputer |
| [06 Bin editor](https://detrin.github.io/datasci-toolkit/06_bin_editor.html) | BinEditor headless API, BinEditorWidget |
| [07 Variable clustering](https://detrin.github.io/datasci-toolkit/07_variable_clustering.html) | CorrVarClus dendrogram, best_features |

## Stack

- Python 3.12, `polars` — no pandas
- `scikit-learn` for estimator conventions
- `matplotlib` for standalone plot functions
