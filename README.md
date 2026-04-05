# datasci-toolkit

[![CI](https://github.com/detrin/datasci-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/detrin/datasci-toolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/datasci-toolkit)](https://pypi.org/project/datasci-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/datasci-toolkit)](https://pypi.org/project/datasci-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Polars-native Python toolkit for binary classification, scorecard development, and model validation.

sklearn-compatible estimators. Zero pandas. Zero comments.

## Install

```bash
pip install datasci-toolkit
```

## Modules

| Module | Classes / Functions | Description |
|---|---|---|
| `stability` | `PSI`, `ESI`, `StabilityMonitor` | Population and event stability indices |
| `grouping` | `StabilityGrouping`, `WOETransformer` | Stability-constrained optimal binning and WOE encoding |
| `metrics` | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power` | Binary classification metrics with period breakdowns |
| `model_selection` | `AUCStepwiseLogit` | Gini-based stepwise logistic regression |
| `feature_elimination` | `ShapImportance`, `ShapRFE` | SHAP-based backward feature elimination with CV |
| `label_imputation` | `KNNLabelImputer`, `TargetImputer` | KNN imputation for records with missing labels |
| `bin_editor` | `BinEditor`, `BinEditorWidget` | Headless and interactive bin boundary editor |
| `variable_clustering` | `CorrVarClus` | Hierarchical correlation clustering for variable reduction |
| `temporal` | `TemporalFeatureEngineer` | Time-based feature generation |

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

## Documentation

**[detrin.github.io/datasci-toolkit](https://detrin.github.io/datasci-toolkit)**

| Tutorial | Topic |
|---|---|
| [Stability](https://detrin.github.io/datasci-toolkit/tutorials/stability/) | PSI drift detection, StabilityMonitor, ESI |
| [Grouping](https://detrin.github.io/datasci-toolkit/tutorials/grouping/) | StabilityGrouping, WOETransformer |
| [Metrics](https://detrin.github.io/datasci-toolkit/tutorials/metrics/) | Gini, KS, lift, IV, bootstrap CI, period breakdowns |
| [Model Selection](https://detrin.github.io/datasci-toolkit/tutorials/model_selection/) | AUCStepwiseLogit, correlation filter, CV mode |
| [Feature Elimination](https://detrin.github.io/datasci-toolkit/tutorials/feature_elimination/) | ShapImportance, ShapRFE, SHAP-based backward selection |
| [Label Imputation](https://detrin.github.io/datasci-toolkit/tutorials/label_imputation/) | KNNLabelImputer, TargetImputer |
| [Bin Editor](https://detrin.github.io/datasci-toolkit/tutorials/bin_editor/) | BinEditor headless API, BinEditorWidget |
| [Variable Clustering](https://detrin.github.io/datasci-toolkit/tutorials/variable_clustering/) | CorrVarClus dendrogram, best_features |
| [Temporal](https://detrin.github.io/datasci-toolkit/tutorials/temporal/) | TemporalFeatureEngineer, AggSpec, TimeSinceSpec |

## Stack

- Python 3.12, `polars` -- no pandas
- `scikit-learn` estimator conventions (`fit` / `transform` / `score`)
- `shap` + `lightgbm` + `xgboost` for SHAP-based feature selection
- `matplotlib` for standalone plot functions
