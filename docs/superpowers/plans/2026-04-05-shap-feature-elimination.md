# SHAP Feature Elimination — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backward feature elimination using SHAP importance with CV, extracted from probatus, rewritten for Polars + MLE style.

**Architecture:** Submodule `feature_elimination/` with two public classes (`ShapImportance`, `ShapRFE`), two private helpers (`_shap`, `_plot`), and a standalone plot function. `ShapRFE` composes `ShapImportance` internally per elimination round.

**Tech Stack:** Python 3.12, Polars, scikit-learn (BaseEstimator), shap, LightGBM, XGBoost, joblib, matplotlib, numpy

---

### Task 0: Install dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add shap and xgboost to pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list:

```toml
"shap>=0.46.0",
"xgboost>=2.1.0",
```

- [ ] **Step 2: Install**

Run: `source .venv/bin/activate && uv sync`

- [ ] **Step 3: Verify**

Run: `source .venv/bin/activate && python -c "import shap, xgboost; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add shap and xgboost dependencies"
```

---

### Task 1: `_shap.py` — SHAP computation helpers

**Files:**
- Create: `datasci_toolkit/feature_elimination/__init__.py` (empty for now)
- Create: `datasci_toolkit/feature_elimination/_shap.py`
- Create: `tests/test_feature_elimination.py`

- [ ] **Step 1: Create the submodule directory with empty `__init__.py`**

Create `datasci_toolkit/feature_elimination/__init__.py` with empty content.

- [ ] **Step 2: Write failing tests for `compute_shap_values`**

Create `tests/test_feature_elimination.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py -v`
Expected: FAIL — `ImportError` because `_shap.py` doesn't exist yet.

- [ ] **Step 4: Implement `compute_shap_values`**

Create `datasci_toolkit/feature_elimination/_shap.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import shap


def _is_tree_model(model: Any) -> bool:
    tree_types = []
    try:
        import lightgbm as lgb
        tree_types.extend([lgb.LGBMClassifier, lgb.LGBMRegressor])
    except ImportError:
        pass
    try:
        import xgboost as xgb
        tree_types.extend([xgb.XGBClassifier, xgb.XGBRegressor])
    except ImportError:
        pass
    return isinstance(model, tuple(tree_types))


def compute_shap_values(model: Any, X: pl.DataFrame) -> np.ndarray:
    X_np = X.to_numpy()
    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_np)
    else:
        background = shap.maskers.Independent(X_np, max_samples=100)
        explainer = shap.Explainer(model, background)
        sv = explainer(X_np).values
    if isinstance(sv, list):
        sv = sv[1]
    return sv
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestComputeShapValues -v`
Expected: 3 PASS

- [ ] **Step 6: Write failing tests for `shap_importance`**

Append to `tests/test_feature_elimination.py`:

```python
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
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestShapImportance -v`
Expected: FAIL — `shap_importance` not yet defined.

- [ ] **Step 8: Implement `shap_importance`**

Append to `datasci_toolkit/feature_elimination/_shap.py`:

```python
def shap_importance(
    shap_values: np.ndarray,
    columns: list[str],
    method: str,
    variance_penalty_factor: float,
) -> pl.DataFrame:
    abs_shap = np.abs(shap_values)
    means = abs_shap.mean(axis=0)
    stds = abs_shap.std(axis=0)
    if method == "variance_penalized":
        importance = means - variance_penalty_factor * stds
    else:
        importance = means
    return (
        pl.DataFrame({"feature": columns, "importance": importance, "std": stds})
        .sort("importance", descending=True)
    )
```

- [ ] **Step 9: Run all `_shap` tests**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py -v`
Expected: 7 PASS

- [ ] **Step 10: Commit**

```bash
git add datasci_toolkit/feature_elimination/__init__.py datasci_toolkit/feature_elimination/_shap.py tests/test_feature_elimination.py
git commit -m "feat: add SHAP computation helpers"
```

---

### Task 2: `ShapImportance` — the ranker

**Files:**
- Create: `datasci_toolkit/feature_elimination/importance.py`
- Modify: `tests/test_feature_elimination.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_feature_elimination.py`:

```python
from datasci_toolkit.feature_elimination.importance import ShapImportance


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
        est = ShapImportance(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42))
        with pytest.raises(Exception):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestShapImportanceEstimator -v`
Expected: FAIL — `importance.py` doesn't exist.

- [ ] **Step 3: Implement `ShapImportance`**

Create `datasci_toolkit/feature_elimination/importance.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted

from datasci_toolkit.feature_elimination._shap import compute_shap_values, shap_importance


def _fold_shap(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    scorer: Any,
) -> tuple[np.ndarray, float, float]:
    cloned = clone(model)
    cloned.fit(X[train_idx], y[train_idx])
    X_val_df = pl.DataFrame({c: X[val_idx, i] for i, c in enumerate(columns)})
    shap_vals = compute_shap_values(cloned, X_val_df)
    train_score = scorer(cloned, X[train_idx], y[train_idx])
    val_score = scorer(cloned, X[val_idx], y[val_idx])
    return shap_vals, train_score, val_score


class ShapImportance(BaseEstimator):
    def __init__(
        self,
        model: Any = None,
        cv: int | Any = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int | None = None,
        importance_method: str = "mean",
        variance_penalty_factor: float = 0.5,
    ) -> None:
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_method = importance_method
        self.variance_penalty_factor = variance_penalty_factor

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ShapImportance:
        columns = X.columns
        X_np = X.to_numpy().astype(np.float64)
        y_np = y.to_numpy().astype(np.float64)
        scorer = check_scoring(self.model, scoring=self.scoring)
        cv = check_cv(self.cv, y_np, classifier=True)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fold_shap)(self.model, X_np, y_np, columns, train_idx, val_idx, scorer)
            for train_idx, val_idx in cv.split(X_np, y_np)
        )

        all_shap = np.concatenate([r[0] for r in results], axis=0)
        train_scores = np.array([r[1] for r in results])
        val_scores = np.array([r[2] for r in results])

        self.feature_importances_ = shap_importance(
            all_shap, columns, self.importance_method, self.variance_penalty_factor,
        )
        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_ = float(train_scores.std())
        self.val_score_mean_ = float(val_scores.mean())
        self.val_score_std_ = float(val_scores.std())
        return self

    def compute(self) -> pl.DataFrame:
        check_is_fitted(self)
        return self.feature_importances_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestShapImportanceEstimator -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/feature_elimination/importance.py tests/test_feature_elimination.py
git commit -m "feat: add ShapImportance estimator"
```

---

### Task 3: `ShapRFE` — the eliminator

**Files:**
- Create: `datasci_toolkit/feature_elimination/elimination.py`
- Modify: `tests/test_feature_elimination.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_feature_elimination.py`:

```python
from datasci_toolkit.feature_elimination.elimination import ShapRFE


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestShapRFE -v`
Expected: FAIL — `elimination.py` doesn't exist.

- [ ] **Step 3: Implement `ShapRFE`**

Create `datasci_toolkit/feature_elimination/elimination.py`:

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from datasci_toolkit.feature_elimination.importance import ShapImportance


class ShapRFE(BaseEstimator):
    def __init__(
        self,
        model: Any = None,
        step: int | float = 1,
        min_features_to_select: int = 1,
        cv: int | Any = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int | None = None,
        importance_method: str = "mean",
        variance_penalty_factor: float = 0.5,
        columns_to_keep: list[str] | None = None,
    ) -> None:
        self.model = model
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_method = importance_method
        self.variance_penalty_factor = variance_penalty_factor
        self.columns_to_keep = columns_to_keep

    def _n_features_to_remove(self, n_current: int) -> int:
        if isinstance(self.step, float):
            n_remove = max(1, math.floor(n_current * self.step))
        else:
            n_remove = self.step
        return min(n_remove, n_current - self.min_features_to_select)

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ShapRFE:
        remaining = list(X.columns)
        keep = set(self.columns_to_keep or [])
        records: list[dict[str, Any]] = []
        round_num = 0

        while len(remaining) > self.min_features_to_select:
            round_num += 1
            imp = ShapImportance(
                model=self.model, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state,
                importance_method=self.importance_method,
                variance_penalty_factor=self.variance_penalty_factor,
            )
            imp.fit(X.select(remaining), y)

            n_remove = self._n_features_to_remove(len(remaining))
            if n_remove <= 0:
                records.append({
                    "round": round_num, "n_features": len(remaining),
                    "features": list(remaining), "eliminated": [],
                    "train_score_mean": imp.train_score_mean_, "train_score_std": imp.train_score_std_,
                    "val_score_mean": imp.val_score_mean_, "val_score_std": imp.val_score_std_,
                })
                break

            ranked = imp.feature_importances_["feature"].to_list()
            removable = [f for f in reversed(ranked) if f not in keep]
            eliminated = removable[:n_remove]

            records.append({
                "round": round_num, "n_features": len(remaining),
                "features": list(remaining), "eliminated": eliminated,
                "train_score_mean": imp.train_score_mean_, "train_score_std": imp.train_score_std_,
                "val_score_mean": imp.val_score_mean_, "val_score_std": imp.val_score_std_,
            })
            remaining = [f for f in remaining if f not in set(eliminated)]

        if not records or records[-1]["n_features"] != len(remaining):
            last_imp = ShapImportance(
                model=self.model, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs, random_state=self.random_state,
                importance_method=self.importance_method,
                variance_penalty_factor=self.variance_penalty_factor,
            )
            last_imp.fit(X.select(remaining), y)
            records.append({
                "round": round_num + 1, "n_features": len(remaining),
                "features": list(remaining), "eliminated": [],
                "train_score_mean": last_imp.train_score_mean_, "train_score_std": last_imp.train_score_std_,
                "val_score_mean": last_imp.val_score_mean_, "val_score_std": last_imp.val_score_std_,
            })

        self.report_df_ = pl.DataFrame(records)
        self.feature_names_ = self.get_reduced_features("best")
        return self

    def compute(self) -> pl.DataFrame:
        check_is_fitted(self)
        return self.report_df_

    def get_reduced_features(self, method: str = "best", se_threshold: float = 1.0) -> list[str]:
        check_is_fitted(self, ["report_df_"])
        df = self.report_df_
        best_idx = int(df["val_score_mean"].arg_max())
        best_score = df["val_score_mean"][best_idx]
        best_std = df["val_score_std"][best_idx]
        threshold = best_score - se_threshold * best_std

        if method == "best":
            return df["features"][best_idx]

        within = df.filter(pl.col("val_score_mean") >= threshold)
        if method == "best_coherent":
            idx = int(within["n_features"].arg_max())
            return within["features"][idx]

        idx = int(within["n_features"].arg_min())
        return within["features"][idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestShapRFE -v`
Expected: 9 PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/feature_elimination/elimination.py tests/test_feature_elimination.py
git commit -m "feat: add ShapRFE backward elimination"
```

---

### Task 4: `get_reduced_features` selection methods

**Files:**
- Modify: `tests/test_feature_elimination.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_feature_elimination.py`:

```python
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
```

- [ ] **Step 2: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestGetReducedFeatures -v`
Expected: 4 PASS (implementation already in Task 3)

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_elimination.py
git commit -m "test: add get_reduced_features selection method tests"
```

---

### Task 5: `_plot.py` — standalone plot function

**Files:**
- Create: `datasci_toolkit/feature_elimination/_plot.py`
- Modify: `tests/test_feature_elimination.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_feature_elimination.py`:

```python
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from datasci_toolkit.feature_elimination._plot import plot_shap_elimination


class TestPlotShapElimination:
    def test_returns_figure(self, binary_dataset: tuple[pl.DataFrame, pl.Series]) -> None:
        X, y = binary_dataset
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=2, cv=3, random_state=42)
        rfe.fit(X, y)
        fig = plot_shap_elimination(rfe.report_df_, show=False)
        assert isinstance(fig, Figure)

    def test_no_error_on_single_round(self) -> None:
        report = pl.DataFrame({
            "round": [1],
            "n_features": [5],
            "features": [["a", "b", "c", "d", "e"]],
            "eliminated": [[]],
            "train_score_mean": [0.9],
            "train_score_std": [0.01],
            "val_score_mean": [0.85],
            "val_score_std": [0.02],
        })
        fig = plot_shap_elimination(report, show=False)
        assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestPlotShapElimination -v`
Expected: FAIL — `_plot.py` doesn't exist.

- [ ] **Step 3: Implement `plot_shap_elimination`**

Create `datasci_toolkit/feature_elimination/_plot.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def plot_shap_elimination(report: pl.DataFrame, show: bool = True) -> Figure:
    n_features = report["n_features"].to_list()
    train_mean = report["train_score_mean"].to_numpy()
    train_std = report["train_score_std"].to_numpy()
    val_mean = report["val_score_mean"].to_numpy()
    val_std = report["val_score_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_features, train_mean, label="Train")
    ax.fill_between(n_features, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(n_features, val_mean, label="Validation")
    ax.fill_between(n_features, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Score")
    ax.set_title("SHAP Backward Feature Elimination")
    ax.invert_xaxis()
    ax.legend()

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestPlotShapElimination -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/feature_elimination/_plot.py tests/test_feature_elimination.py
git commit -m "feat: add SHAP elimination plot function"
```

---

### Task 6: Package exports and integration

**Files:**
- Modify: `datasci_toolkit/feature_elimination/__init__.py`
- Modify: `datasci_toolkit/__init__.py`

- [ ] **Step 1: Write the submodule `__init__.py`**

Overwrite `datasci_toolkit/feature_elimination/__init__.py`:

```python
from datasci_toolkit.feature_elimination._plot import plot_shap_elimination
from datasci_toolkit.feature_elimination.elimination import ShapRFE
from datasci_toolkit.feature_elimination.importance import ShapImportance

__all__ = ["ShapImportance", "ShapRFE", "plot_shap_elimination"]
```

- [ ] **Step 2: Add to package `__init__.py`**

In `datasci_toolkit/__init__.py`, add import:

```python
from datasci_toolkit.feature_elimination import ShapImportance, ShapRFE, plot_shap_elimination
```

Add to `__all__`:

```python
"ShapImportance",
"ShapRFE",
"plot_shap_elimination",
```

- [ ] **Step 3: Verify top-level imports work**

Run: `source .venv/bin/activate && python -c "from datasci_toolkit import ShapImportance, ShapRFE, plot_shap_elimination; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Run full test suite**

Run: `source .venv/bin/activate && pytest tests/ -q`
Expected: All tests pass (existing + new)

- [ ] **Step 5: Run pre-commit checks**

Run: `source .venv/bin/activate && ruff check datasci_toolkit && mypy datasci_toolkit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add datasci_toolkit/feature_elimination/__init__.py datasci_toolkit/__init__.py
git commit -m "feat: export ShapImportance, ShapRFE, plot_shap_elimination from package"
```

---

### Task 7: Edge case tests

**Files:**
- Modify: `tests/test_feature_elimination.py`

- [ ] **Step 1: Write edge case tests**

Append to `tests/test_feature_elimination.py`:

```python
class TestEdgeCases:
    def test_single_feature_dataset(self) -> None:
        X = pl.DataFrame({"f0": RNG.normal(size=200).tolist()})
        y = pl.Series("target", (RNG.normal(size=200) > 0).astype(int).tolist())
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=1, min_features_to_select=1, cv=3, random_state=42)
        rfe.fit(X, y)
        assert rfe.report_df_["n_features"].to_list()[-1] == 1

    def test_step_larger_than_features(self) -> None:
        X = pl.DataFrame({"f0": RNG.normal(size=200).tolist(), "f1": RNG.normal(size=200).tolist(), "f2": RNG.normal(size=200).tolist()})
        y = pl.Series("target", (RNG.normal(size=200) > 0).astype(int).tolist())
        rfe = ShapRFE(model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42), step=10, min_features_to_select=1, cv=3, random_state=42)
        rfe.fit(X, y)
        assert rfe.report_df_["n_features"].to_list()[-1] >= 1

    def test_all_columns_to_keep(self) -> None:
        cols = [f"f{i}" for i in range(3)]
        X = pl.DataFrame({c: RNG.normal(size=200).tolist() for c in cols})
        y = pl.Series("target", (RNG.normal(size=200) > 0).astype(int).tolist())
        rfe = ShapRFE(
            model=lgb.LGBMClassifier(n_estimators=10, verbose=-1, random_state=42),
            step=1, min_features_to_select=1, cv=3, random_state=42, columns_to_keep=cols,
        )
        rfe.fit(X, y)
        last_features = rfe.report_df_["features"].to_list()[-1]
        assert set(last_features) == set(cols)
```

- [ ] **Step 2: Run edge case tests**

Run: `source .venv/bin/activate && pytest tests/test_feature_elimination.py::TestEdgeCases -v`
Expected: 3 PASS

- [ ] **Step 3: Run full suite one final time**

Run: `source .venv/bin/activate && pytest tests/ -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_feature_elimination.py
git commit -m "test: add edge case tests for feature elimination"
```
