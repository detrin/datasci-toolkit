from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from datasci_toolkit.feature_elimination._shap import compute_shap_values, shap_importance
from datasci_toolkit.feature_elimination.elimination import ShapRFE
from datasci_toolkit.feature_elimination.importance import ShapImportance

RNG = np.random.default_rng(42)
N = 500
N_FEATURES = 5


@pytest.fixture
def binary_dataset() -> tuple[pl.DataFrame, pl.Series]:
    X_np = RNG.normal(size=(N, N_FEATURES))
    y_np = (X_np[:, 0] + 0.5 * X_np[:, 1] + RNG.normal(scale=0.3, size=N) > 0).astype(int)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    return pl.DataFrame({c: X_np[:, i] for i, c in enumerate(cols)}), pl.Series("target", y_np)


@pytest.fixture
def fitted_lgb(binary_dataset: tuple[pl.DataFrame, pl.Series]) -> lgb.LGBMClassifier:
    X, y = binary_dataset
    m = lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42)
    m.fit(X.to_numpy(), y.to_numpy())
    return m


@pytest.fixture
def fitted_xgb(binary_dataset: tuple[pl.DataFrame, pl.Series]) -> xgb.XGBClassifier:
    X, y = binary_dataset
    m = xgb.XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
    m.fit(X.to_numpy(), y.to_numpy())
    return m


class TestComputeShapValues:
    def test_lgb_returns_correct_shape(self, fitted_lgb: lgb.LGBMClassifier, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, _ = binary_dataset
        result = compute_shap_values(fitted_lgb, X)
        assert result.shape == (N, N_FEATURES)

    def test_xgb_returns_correct_shape(self, fitted_xgb: xgb.XGBClassifier, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, _ = binary_dataset
        result = compute_shap_values(fitted_xgb, X)
        assert result.shape == (N, N_FEATURES)

    def test_returns_numpy_array(self, fitted_lgb: lgb.LGBMClassifier, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, _ = binary_dataset
        result = compute_shap_values(fitted_lgb, X)
        assert isinstance(result, np.ndarray)


class TestShapImportance:
    def test_mean_method_returns_correct_schema(self) -> None:
        shap_vals = RNG.normal(size=(100, 3))
        result = shap_importance(shap_vals, ["a", "b", "c"], "mean", 0.5)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["feature", "importance", "std"]
        assert len(result) == 3

    def test_mean_method_sorted_descending(self) -> None:
        shap_vals = np.column_stack([
            RNG.normal(0, 0.1, 100),
            RNG.normal(0, 1.0, 100),
            RNG.normal(0, 5.0, 100),
        ])
        result = shap_importance(shap_vals, ["low", "mid", "high"], "mean", 0.5)
        assert result["feature"].to_list()[0] == "high"
        assert result["feature"].to_list()[-1] == "low"

    def test_variance_penalized_differs_from_mean(self) -> None:
        shap_vals = np.column_stack([
            RNG.normal(0, 0.1, 100),
            RNG.normal(0, 10.0, 100),
        ])
        mean_result = shap_importance(shap_vals, ["stable", "noisy"], "mean", 0.5)
        penalized_result = shap_importance(shap_vals, ["stable", "noisy"], "variance_penalized", 2.0)
        mean_order = mean_result["feature"].to_list()
        penalized_order = penalized_result["feature"].to_list()
        assert mean_order != penalized_order or penalized_result["importance"][1] < mean_result["importance"][1]

    def test_all_importances_finite(self) -> None:
        shap_vals = RNG.normal(size=(100, 4))
        result = shap_importance(shap_vals, ["a", "b", "c", "d"], "mean", 0.5)
        assert result["importance"].is_nan().sum() == 0
        assert result["importance"].is_infinite().sum() == 0


class TestShapImportanceEstimator:
    def test_fit_returns_self(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), cv=3, random_state=42)
        result = est.fit(X, y)
        assert result is est

    def test_feature_importances_schema(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), cv=3, random_state=42)
        est.fit(X, y)
        df = est.feature_importances_
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["feature", "importance", "std"]
        assert len(df) == N_FEATURES

    def test_feature_importances_sorted_desc(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), cv=3, random_state=42)
        est.fit(X, y)
        importances = est.feature_importances_["importance"].to_list()
        assert importances == sorted(importances, reverse=True)

    def test_compute_returns_importances(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), cv=3, random_state=42)
        est.fit(X, y)
        result = est.compute()
        assert result.equals(est.feature_importances_)

    def test_compute_before_fit_raises(self) -> None:
        from sklearn.exceptions import NotFittedError
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42))
        with pytest.raises(NotFittedError):
            est.compute()

    def test_works_with_xgboost(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=xgb.XGBClassifier(n_estimators=10, verbosity=0, random_state=42), cv=3, random_state=42)
        est.fit(X, y)
        assert len(est.feature_importances_) == N_FEATURES

    def test_variance_penalized_method(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(
            model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42),
            cv=3, random_state=42, importance_method="variance_penalized", variance_penalty_factor=1.0,
        )
        est.fit(X, y)
        assert len(est.feature_importances_) == N_FEATURES

    def test_stores_train_and_val_scores(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), cv=3, random_state=42)
        est.fit(X, y)
        assert hasattr(est, "train_score_mean_")
        assert hasattr(est, "train_score_std_")
        assert hasattr(est, "val_score_mean_")
        assert hasattr(est, "val_score_std_")
        assert 0.5 < est.val_score_mean_ < 1.0



class TestShapRFE:
    def test_fit_returns_self(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        result = rfe.fit(X, y)
        assert result is rfe

    def test_report_df_schema(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        rfe.fit(X, y)
        df = rfe.report_df_
        assert isinstance(df, pl.DataFrame)
        expected_cols = ["round", "n_features", "features", "eliminated", "train_score_mean", "train_score_std", "val_score_mean", "val_score_std"]
        assert df.columns == expected_cols

    def test_features_decrease_each_round(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        rfe.fit(X, y)
        n_features = rfe.report_df_["n_features"].to_list()
        assert n_features == sorted(n_features, reverse=True)
        assert n_features[-1] >= 2

    def test_min_features_respected(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=2, min_features_to_select=3, cv=3, random_state=42)
        rfe.fit(X, y)
        assert rfe.report_df_["n_features"].to_list()[-1] >= 3

    def test_step_float(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=0.3, min_features_to_select=1, cv=3, random_state=42)
        rfe.fit(X, y)
        assert rfe.report_df_["n_features"].to_list()[-1] >= 1

    def test_columns_to_keep_survives(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(
            model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42),
            step=1, min_features_to_select=1, cv=3, random_state=42, columns_to_keep=["f0"],
        )
        rfe.fit(X, y)
        last_features = rfe.report_df_["features"].to_list()[-1]
        assert "f0" in last_features

    def test_compute_returns_report(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        rfe.fit(X, y)
        assert rfe.compute().equals(rfe.report_df_)

    def test_feature_names_set(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        rfe.fit(X, y)
        assert isinstance(rfe.feature_names_, list)
        assert len(rfe.feature_names_) >= 2

    def test_works_with_xgboost(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=xgb.XGBClassifier(n_estimators=10, verbosity=0, random_state=42), step=1, min_features_to_select=3, cv=3, random_state=42)
        rfe.fit(X, y)
        assert len(rfe.report_df_) >= 1


class TestGetReducedFeatures:
    @pytest.fixture
    def fitted_rfe(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> ShapRFE:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=1, cv=3, random_state=42)
        rfe.fit(X, y)
        return rfe

    def test_best_returns_list(self, fitted_rfe: ShapRFE) -> None:
        result = fitted_rfe.get_reduced_features("best")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_best_coherent_has_most_features_within_se(self, fitted_rfe: ShapRFE) -> None:
        best = fitted_rfe.get_reduced_features("best")
        coherent = fitted_rfe.get_reduced_features("best_coherent")
        assert len(coherent) >= len(best)

    def test_best_parsimonious_has_fewest_features_within_se(self, fitted_rfe: ShapRFE) -> None:
        coherent = fitted_rfe.get_reduced_features("best_coherent")
        parsimonious = fitted_rfe.get_reduced_features("best_parsimonious")
        assert len(parsimonious) <= len(coherent)

    def test_all_methods_return_valid_features(self, fitted_rfe: ShapRFE) -> None:
        all_features = set(fitted_rfe.report_df_["features"][0])
        for method in ["best", "best_coherent", "best_parsimonious"]:
            result = fitted_rfe.get_reduced_features(method)
            assert set(result).issubset(all_features)
