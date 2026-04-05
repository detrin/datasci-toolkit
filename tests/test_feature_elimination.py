from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from datasci_toolkit.feature_elimination._shap import compute_shap_values, shap_importance

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
