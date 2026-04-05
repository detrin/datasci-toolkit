# Changelog

## [Unreleased]

### Added
- `feature_elimination` module: SHAP-based backward feature elimination
  - `ShapImportance` — one-shot SHAP feature ranker with CV
  - `ShapRFE` — recursive backward elimination using SHAP importance
  - `plot_shap_elimination` — standalone elimination curve plot
  - Two importance methods: mean |SHAP| and variance-penalized
  - Three selection strategies: `"best"`, `"best_coherent"`, `"best_parsimonious"`
  - LightGBM and XGBoost support via TreeExplainer
  - `columns_to_keep` for protected features

### Dependencies
- Added `shap` and `xgboost`

## [0.2.0] - 2026-03-30

### Added
- `temporal` module: `TemporalFeatureEngineer` for time-based feature generation
  - Spec dataclasses: `AggSpec`, `TimeSinceSpec`, `RatioSpec`
  - `from_config` for dict-driven construction, fluent builder API

## [0.1.0] - 2026-03-29

### Added
- `stability` module: `PSI`, `ESI`, `StabilityMonitor`, `plot_psi_comparison`, `psi_hist`
- `grouping` module: `StabilityGrouping`, `WOETransformer`
- `model_selection` module: `AUCStepwiseLogit`
- `label_imputation` module: `KNNLabelImputer`, `TargetImputer`
- `bin_editor` module: `BinEditor`, `BinEditorWidget`
- `metrics` module: `gini`, `ks`, `lift`, `iv`, `BootstrapGini`, `feature_power`, `gini_by_period`, `lift_by_period`, `plot_metric_by_period`
- `variable_clustering` module: `CorrVarClus`
- MkDocs Material documentation site
- PyPI publish workflow via trusted publishing
