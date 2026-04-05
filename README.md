# datasci-toolkit

[![CI](https://github.com/detrin/datasci-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/detrin/datasci-toolkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/datasci-toolkit)](https://pypi.org/project/datasci-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/datasci-toolkit)](https://pypi.org/project/datasci-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Polars-native Python toolkit for binary classification and model validation.

sklearn-compatible estimators. Zero pandas. Zero comments.

## Install

```bash
pip install datasci-toolkit
```

## Modules

### Monitoring & Stability

| Module | Classes / Functions | Use case |
|---|---|---|
| `stability` | `PSI`, `ESI`, `StabilityMonitor` | Detect population drift between training and production data -- catch when your input distributions shift before model performance degrades |
| `metrics` | `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power` | Evaluate binary classifiers with confidence intervals -- report Gini/KS/lift by month to stakeholders, identify which features drive predictive power |

### Feature Engineering & Selection

| Module | Classes / Functions | Use case |
|---|---|---|
| `grouping` | `StabilityGrouping`, `WOETransformer` | Bin continuous features into stable WOE-encoded groups for scorecard development -- ensures bins don't drift across time periods |
| `feature_elimination` | `ShapImportance`, `ShapRFE` | Reduce a 500-feature dataset to the 20 that matter -- backward elimination using SHAP values with cross-validation, not just feature importance |
| `variable_clustering` | `CorrVarClus` | Remove redundant features before modeling -- hierarchical clustering picks one representative from each correlated group |
| `temporal` | `TemporalFeatureEngineer` | Generate time-windowed aggregations (sum/mean/max over 30d/90d/1y) from transaction histories for credit scoring or churn prediction |
| `tagging` | `WeightedTFIDF` | Profile entities with ranked tags -- e.g., find the top 5 product attributes from reviews, or build customer interest profiles from transaction categories, with external quality signals and cross-entity normalization |

### Model Building & Post-processing

| Module | Classes / Functions | Use case |
|---|---|---|
| `model_selection` | `AUCStepwiseLogit` | Build interpretable scorecards -- stepwise logistic regression that adds features by Gini lift and enforces correlation constraints |
| `bin_editor` | `BinEditor`, `BinEditorWidget` | Manually adjust bin boundaries after auto-binning -- headless API for pipelines, interactive widget for notebooks |
| `label_imputation` | `KNNLabelImputer`, `TargetImputer` | Recover labels for rejected loan applications (reject inference) or fill missing targets in semi-supervised settings |
| `smoothing` | `PoissonSmoother`, `PredictionSmoother` | Stabilize noisy count features before modeling (Poisson), or eliminate monthly prediction jitter so a customer doesn't flip between risk tiers due to noise |

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
| [Smoothing](https://detrin.github.io/datasci-toolkit/tutorials/smoothing/) | PoissonSmoother, PredictionSmoother |
| [Tagging](https://detrin.github.io/datasci-toolkit/tutorials/tagging/) | WeightedTFIDF, Z-score normalization |

## Stack

- Python 3.12, `polars` -- no pandas
- `scikit-learn` estimator conventions (`fit` / `transform` / `score`)
- `shap` + `lightgbm` + `xgboost` for SHAP-based feature selection
- `matplotlib` for standalone plot functions
