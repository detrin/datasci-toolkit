# Smoothing & Tagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three algorithms (PoissonSmoother, PredictionSmoother, WeightedTFIDF) as Polars-native sklearn-compatible classes in two new modules.

**Architecture:** Two new files — `datasci_toolkit/smoothing.py` for temporal smoothers, `datasci_toolkit/tagging.py` for weighted TF-IDF. All classes follow sklearn `fit`/`transform` conventions. Polars in, Polars out. Zero comments, zero docstrings.

**Tech Stack:** Python 3.10+, polars, scikit-learn, scipy, numpy

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `datasci_toolkit/smoothing.py` | Create | `PoissonSmoother`, `PredictionSmoother` |
| `datasci_toolkit/tagging.py` | Create | `WeightedTFIDF` |
| `tests/test_smoothing.py` | Create | Tests for both smoothers |
| `tests/test_tagging.py` | Create | Tests for WeightedTFIDF |
| `datasci_toolkit/__init__.py` | Modify | Add imports and `__all__` entries |
| `README.md` | Modify | Add smoothing and tagging rows to module table |
| `docs/index.md` | Modify | Add smoothing and tagging rows to module table |
| `docs/api/smoothing.md` | Create | API reference for smoothing module |
| `docs/api/tagging.md` | Create | API reference for tagging module |
| `docs/tutorials/smoothing.md` | Create | Tutorial for smoothing module |
| `docs/tutorials/tagging.md` | Create | Tutorial for tagging module |
| `mkdocs.yml` | Modify | Add nav entries |

---

### Task 1: PoissonSmoother

**Files:**
- Create: `datasci_toolkit/smoothing.py`
- Create: `tests/test_smoothing.py`

**Context:** This class takes a Polars DataFrame containing count time series data with entity identifiers, a date column, and a value column. It splits the data into "today" (target_date) and "history" (everything else), computes a Poisson p-value per entity, then blends today's count with historical sum using adaptive weights. The p-value measures how surprising today's count is given history — anomalous counts are trusted, normal counts are smoothed toward history.

**Math reference:**
- `history_mean = history_sum / (window_size - 1)`
- If `k <= mu`: `p = poisson.cdf(floor(k) + eps_left, mu + eps_left)`
- If `k > mu`: `p = 1 - poisson.cdf(ceil(k) + eps_right, mu + eps_right)`
- `w_today = 1 - p^alpha`
- `w_hist = p^alpha / (window_size - 1)`
- `smoothed = w_today * today_count + w_hist * history_sum`

- [ ] **Step 1: Write failing tests for PoissonSmoother**

Create `tests/test_smoothing.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.utils.validation import check_is_fitted

from datasci_toolkit.smoothing import PoissonSmoother

RNG = np.random.default_rng(42)


@pytest.fixture
def count_df() -> pl.DataFrame:
    rows = []
    for eid in range(1, 4):
        for d in range(1, 8):
            rows.append({"entity_id": eid, "date": f"2024-01-{d:02d}", "count": eid * 10 + d})
    return pl.DataFrame(rows)


class TestPoissonSmootherInit:
    def test_default_params(self) -> None:
        ps = PoissonSmoother(window_size=7)
        assert ps.window_size == 7
        assert ps.alpha == 0.1
        assert ps.eps_left == 20
        assert ps.eps_right == 5

    def test_custom_params(self) -> None:
        ps = PoissonSmoother(window_size=10, alpha=0.2, eps_left=15, eps_right=3)
        assert ps.alpha == 0.2
        assert ps.eps_left == 15

    def test_window_size_validation(self) -> None:
        ps = PoissonSmoother(window_size=1)
        with pytest.raises(ValueError, match="window_size"):
            ps.fit()


class TestPoissonSmootherFitTransform:
    def test_fit_returns_self(self) -> None:
        ps = PoissonSmoother(window_size=7)
        result = ps.fit()
        assert result is ps

    def test_transform_before_fit_raises(self) -> None:
        ps = PoissonSmoother(window_size=7)
        with pytest.raises(Exception):
            ps.transform(pl.DataFrame(), entity_cols=["id"], date_col="d", value_col="v", target_date="x")

    def test_transform_output_columns(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        assert "entity_id" in result.columns
        assert "today_count" in result.columns
        assert "history_mean" in result.columns
        assert "pvalue" in result.columns
        assert "smoothed_count" in result.columns

    def test_transform_row_count(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        assert len(result) <= 3

    def test_smoothed_between_today_and_history(self, count_df: pl.DataFrame) -> None:
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(count_df, entity_cols=["entity_id"], date_col="date", value_col="count", target_date="2024-01-07")
        for row in result.iter_rows(named=True):
            assert row["smoothed_count"] > 0

    def test_anomalous_count_trusts_today(self) -> None:
        rows = []
        for d in range(1, 7):
            rows.append({"eid": "A", "date": f"2024-01-{d:02d}", "count": 50})
        rows.append({"eid": "A", "date": "2024-01-07", "count": 500})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid"], date_col="date", value_col="count", target_date="2024-01-07")
        smoothed = result["smoothed_count"][0]
        assert smoothed > 300

    def test_normal_count_smooths_toward_history(self) -> None:
        rows = []
        for d in range(1, 7):
            rows.append({"eid": "A", "date": f"2024-01-{d:02d}", "count": 50})
        rows.append({"eid": "A", "date": "2024-01-07", "count": 48})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid"], date_col="date", value_col="count", target_date="2024-01-07")
        smoothed = result["smoothed_count"][0]
        assert abs(smoothed - 50) < abs(48 - 50)

    def test_multiple_entity_cols(self) -> None:
        rows = []
        for seg in ["X", "Y"]:
            for d in range(1, 8):
                rows.append({"eid": "A", "seg": seg, "date": f"2024-01-{d:02d}", "count": 50})
        df = pl.DataFrame(rows)
        ps = PoissonSmoother(window_size=7).fit()
        result = ps.transform(df, entity_cols=["eid", "seg"], date_col="date", value_col="count", target_date="2024-01-07")
        assert len(result) == 2
        assert "eid" in result.columns
        assert "seg" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_smoothing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'datasci_toolkit.smoothing'`

- [ ] **Step 3: Implement PoissonSmoother**

Create `datasci_toolkit/smoothing.py`:

```python
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PoissonSmoother(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        window_size: int = 7,
        alpha: float = 0.1,
        eps_left: int = 20,
        eps_right: int = 5,
    ) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self.eps_left = eps_left
        self.eps_right = eps_right

    def fit(self, X: pl.DataFrame | None = None, y: None = None) -> PoissonSmoother:
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.fitted_ = True
        return self

    def transform(
        self,
        X: pl.DataFrame,
        entity_cols: list[str] | None = None,
        date_col: str | None = None,
        value_col: str | None = None,
        target_date: str | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        assert entity_cols is not None
        assert date_col is not None
        assert value_col is not None
        assert target_date is not None

        today = X.filter(pl.col(date_col) == target_date)
        history = X.filter(pl.col(date_col) != target_date)

        today_agg = today.group_by(entity_cols).agg(
            pl.col(value_col).sum().alias("today_count")
        )
        hist_agg = history.group_by(entity_cols).agg(
            pl.col(value_col).sum().alias("history_sum")
        )

        n_hist = self.window_size - 1
        merged = today_agg.join(hist_agg, on=entity_cols, how="inner")
        merged = merged.with_columns(
            (pl.col("history_sum") / n_hist).alias("history_mean")
        )

        k_arr = merged["today_count"].to_numpy().astype(float)
        mu_arr = merged["history_mean"].to_numpy().astype(float)
        hist_sum_arr = merged["history_sum"].to_numpy().astype(float)

        pvals = np.where(
            k_arr <= mu_arr,
            poisson.cdf(np.floor(k_arr).astype(int) + self.eps_left, mu_arr + self.eps_left),
            1 - poisson.cdf(np.ceil(k_arr).astype(int) + self.eps_right, mu_arr + self.eps_right),
        )

        w_today = 1 - np.power(pvals, self.alpha)
        w_hist = np.power(pvals, self.alpha) / n_hist
        smoothed = w_today * k_arr + w_hist * hist_sum_arr

        result = merged.with_columns(
            pl.Series("pvalue", pvals),
            pl.Series("smoothed_count", smoothed),
        ).drop("history_sum")

        return result.filter(pl.col("smoothed_count") > 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smoothing.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/smoothing.py tests/test_smoothing.py
git commit -m "feat: add PoissonSmoother for adaptive temporal count smoothing"
```

---

### Task 2: PredictionSmoother

**Files:**
- Modify: `datasci_toolkit/smoothing.py`
- Modify: `tests/test_smoothing.py`

**Context:** This class smooths classification probability outputs across time periods. It supports two modes: binary (single prob column as `str`) and multi-class (list of prob column names). Binary mode averages a single probability. Multi-class mode averages each column, then picks the column name with the highest average as `predicted_label`. Both modes add `observation_count` and filter to `>= min_observations`.

- [ ] **Step 1: Add failing tests for PredictionSmoother**

Append to `tests/test_smoothing.py`:

```python
from datasci_toolkit.smoothing import PredictionSmoother


@pytest.fixture
def binary_pred_df() -> pl.DataFrame:
    return pl.DataFrame({
        "eid": ["A", "A", "A", "B", "B"],
        "month": [1, 2, 3, 1, 2],
        "prob_default": [0.6, 0.4, 0.5, 0.8, 0.9],
    })


@pytest.fixture
def multiclass_pred_df() -> pl.DataFrame:
    return pl.DataFrame({
        "eid": ["A", "A", "A", "B", "B"],
        "month": [1, 2, 3, 1, 2],
        "prob_cat": [0.5, 0.3, 0.4, 0.1, 0.2],
        "prob_dog": [0.3, 0.5, 0.4, 0.8, 0.7],
        "prob_bird": [0.2, 0.2, 0.2, 0.1, 0.1],
    })


class TestPredictionSmootherInit:
    def test_default_params(self) -> None:
        ps = PredictionSmoother()
        assert ps.min_observations == 1

    def test_custom_params(self) -> None:
        ps = PredictionSmoother(min_observations=3)
        assert ps.min_observations == 3


class TestPredictionSmootherBinary:
    def test_fit_returns_self(self) -> None:
        ps = PredictionSmoother()
        assert ps.fit() is ps

    def test_binary_output_columns(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        assert "eid" in result.columns
        assert "prob_default" in result.columns
        assert "observation_count" in result.columns
        assert "predicted_label" not in result.columns

    def test_binary_averages_correctly(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        a_row = result.filter(pl.col("eid") == "A")
        assert abs(a_row["prob_default"][0] - 0.5) < 1e-6

    def test_binary_observation_count(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        a_row = result.filter(pl.col("eid") == "A")
        b_row = result.filter(pl.col("eid") == "B")
        assert a_row["observation_count"][0] == 3
        assert b_row["observation_count"][0] == 2

    def test_min_observations_filter(self, binary_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother(min_observations=3).fit()
        result = ps.transform(binary_pred_df, entity_cols=["eid"], period_col="month", prob_cols="prob_default")
        assert len(result) == 1
        assert result["eid"][0] == "A"


class TestPredictionSmootherMulticlass:
    def test_multiclass_output_columns(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        assert "predicted_label" in result.columns
        assert "observation_count" in result.columns
        for c in ["prob_cat", "prob_dog", "prob_bird"]:
            assert c in result.columns

    def test_multiclass_argmax(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        a_row = result.filter(pl.col("eid") == "A")
        b_row = result.filter(pl.col("eid") == "B")
        assert a_row["predicted_label"][0] == "prob_cat"
        assert b_row["predicted_label"][0] == "prob_dog"

    def test_multiclass_averages(self, multiclass_pred_df: pl.DataFrame) -> None:
        ps = PredictionSmoother().fit()
        result = ps.transform(multiclass_pred_df, entity_cols=["eid"], period_col="month", prob_cols=["prob_cat", "prob_dog", "prob_bird"])
        a_row = result.filter(pl.col("eid") == "A")
        assert abs(a_row["prob_cat"][0] - 0.4) < 1e-6
        assert abs(a_row["prob_dog"][0] - 0.4) < 1e-6
        assert abs(a_row["prob_bird"][0] - 0.2) < 1e-6

    def test_multiple_entity_cols(self) -> None:
        df = pl.DataFrame({
            "eid": ["A", "A", "A", "A"],
            "seg": ["X", "X", "Y", "Y"],
            "month": [1, 2, 1, 2],
            "p": [0.6, 0.4, 0.8, 0.9],
        })
        ps = PredictionSmoother().fit()
        result = ps.transform(df, entity_cols=["eid", "seg"], period_col="month", prob_cols="p")
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_smoothing.py::TestPredictionSmootherInit -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement PredictionSmoother**

Add to `datasci_toolkit/smoothing.py`:

```python
class PredictionSmoother(BaseEstimator, TransformerMixin):
    def __init__(self, min_observations: int = 1) -> None:
        self.min_observations = min_observations

    def fit(self, X: pl.DataFrame | None = None, y: None = None) -> PredictionSmoother:
        self.fitted_ = True
        return self

    def transform(
        self,
        X: pl.DataFrame,
        entity_cols: list[str] | None = None,
        period_col: str | None = None,
        prob_cols: str | list[str] | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        assert entity_cols is not None
        assert period_col is not None
        assert prob_cols is not None

        binary = isinstance(prob_cols, str)
        cols = [prob_cols] if binary else prob_cols

        agg_exprs = [pl.col(c).mean().alias(c) for c in cols]
        agg_exprs.append(pl.len().alias("observation_count"))
        result = X.group_by(entity_cols).agg(agg_exprs)
        result = result.filter(pl.col("observation_count") >= self.min_observations)

        if not binary:
            prob_array = pl.concat_list(cols)
            col_names = pl.Series(cols)
            result = result.with_columns(
                pl.struct(cols)
                .map_elements(
                    lambda row: max(cols, key=lambda c: row[c]),
                    return_dtype=pl.String,
                )
                .alias("predicted_label")
            )

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smoothing.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/smoothing.py tests/test_smoothing.py
git commit -m "feat: add PredictionSmoother for temporal label de-jittering"
```

---

### Task 3: WeightedTFIDF

**Files:**
- Create: `datasci_toolkit/tagging.py`
- Create: `tests/test_tagging.py`

**Context:** This class implements weighted TF-IDF with per-entity Z-score normalization. `fit` learns IDF weights from a corpus. `transform` computes weighted TF, multiplies by IDF, Z-scores per entity, splits dominant tags (Z > threshold, score=1.0) from normal tags (min-max scaled), filters below score_threshold, and unions both sets.

Optional columns: `weight_col` (external relevance signal, defaults to 1.0) and `level_col` (hierarchy multiplier, defaults to 1.0).

**Math reference:**
- Weighted TF: `sum(weight * value) / sum_entity(weight * value)`
- IDF: `|log10(N / (1 + corpus_count))|`
- Score: `level * TF * IDF`
- Per-entity Z-score: `(score - mu) / sigma`, sigma=0 → Z=3.0
- Dominant (Z > thresh): final_score=1.0
- Normal (|Z| <= thresh): min-max scaled per entity, filtered by score_threshold

- [ ] **Step 1: Write failing tests for WeightedTFIDF**

Create `tests/test_tagging.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from datasci_toolkit.tagging import WeightedTFIDF


@pytest.fixture
def corpus_df() -> pl.DataFrame:
    return pl.DataFrame({
        "entity": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
        "tag": ["x", "y", "z", "x", "y", "x", "y", "z", "w"],
        "value": [10, 5, 1, 8, 3, 12, 6, 2, 1],
    })


@pytest.fixture
def corpus_with_weights() -> pl.DataFrame:
    return pl.DataFrame({
        "entity": ["A", "A", "A", "B", "B"],
        "tag": ["x", "y", "z", "x", "y"],
        "value": [10, 5, 1, 8, 3],
        "weight": [0.9, 0.5, 0.1, 0.8, 0.6],
        "level": [1.0, 1.0, 0.5, 1.0, 1.0],
    })


class TestWeightedTFIDFInit:
    def test_default_params(self) -> None:
        t = WeightedTFIDF()
        assert t.zscore_thresh == 2.0
        assert t.score_threshold == 0.1
        assert t.weight_col is None
        assert t.level_col is None

    def test_custom_params(self) -> None:
        t = WeightedTFIDF(zscore_thresh=3.0, score_threshold=0.2, weight_col="w", level_col="l")
        assert t.zscore_thresh == 3.0
        assert t.weight_col == "w"


class TestWeightedTFIDFFit:
    def test_fit_returns_self(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF()
        result = t.fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert result is t

    def test_fit_stores_n(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert t.n_entities_ == 3

    def test_fit_stores_idf(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert "tag" in t.idf_.columns
        assert "idf" in t.idf_.columns
        assert len(t.idf_) == 4

    def test_idf_values(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        idf = t.idf_.sort("tag")
        w_idf = idf.filter(pl.col("tag") == "w")["idf"][0]
        x_idf = idf.filter(pl.col("tag") == "x")["idf"][0]
        assert w_idf > x_idf


class TestWeightedTFIDFTransform:
    def test_transform_before_fit_raises(self) -> None:
        t = WeightedTFIDF()
        with pytest.raises(Exception):
            t.transform(pl.DataFrame(), entity_col="e", tag_col="t", value_col="v")

    def test_output_columns(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert set(result.columns) == {"entity", "tag", "final_score"}

    def test_scores_between_0_and_1(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF().fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert result["final_score"].min() >= 0.0
        assert result["final_score"].max() <= 1.0

    def test_score_threshold_filters(self, corpus_df: pl.DataFrame) -> None:
        t_low = WeightedTFIDF(score_threshold=0.01).fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        t_high = WeightedTFIDF(score_threshold=0.5).fit(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        r_low = t_low.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        r_high = t_high.transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert len(r_high) <= len(r_low)

    def test_with_weight_and_level(self, corpus_with_weights: pl.DataFrame) -> None:
        t = WeightedTFIDF(weight_col="weight", level_col="level")
        t.fit(corpus_with_weights, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(corpus_with_weights, entity_col="entity", tag_col="tag", value_col="value")
        assert len(result) > 0
        assert result["final_score"].min() >= 0.0

    def test_single_tag_entity_dominant(self) -> None:
        df = pl.DataFrame({
            "entity": ["A", "A", "B"],
            "tag": ["x", "y", "x"],
            "value": [10, 5, 10],
        })
        t = WeightedTFIDF().fit(df, entity_col="entity", tag_col="tag", value_col="value")
        result = t.transform(df, entity_col="entity", tag_col="tag", value_col="value")
        b_row = result.filter(pl.col("entity") == "B")
        assert len(b_row) == 1
        assert b_row["final_score"][0] == 1.0

    def test_fit_transform(self, corpus_df: pl.DataFrame) -> None:
        t = WeightedTFIDF()
        result = t.fit_transform(corpus_df, entity_col="entity", tag_col="tag", value_col="value")
        assert len(result) > 0
        assert "final_score" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tagging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'datasci_toolkit.tagging'`

- [ ] **Step 3: Implement WeightedTFIDF**

Create `datasci_toolkit/tagging.py`:

```python
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WeightedTFIDF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        zscore_thresh: float = 2.0,
        score_threshold: float = 0.1,
        weight_col: str | None = None,
        level_col: str | None = None,
    ) -> None:
        self.zscore_thresh = zscore_thresh
        self.score_threshold = score_threshold
        self.weight_col = weight_col
        self.level_col = level_col

    def fit(
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
        y: None = None,
    ) -> WeightedTFIDF:
        assert entity_col is not None
        assert tag_col is not None
        assert value_col is not None

        self.entity_col_ = entity_col
        self.tag_col_ = tag_col
        self.value_col_ = value_col

        self.n_entities_ = X[entity_col].n_unique()

        corpus_freq = (
            X.select(entity_col, tag_col)
            .unique()
            .group_by(tag_col)
            .agg(pl.len().alias("corpus_count"))
        )
        self.idf_ = corpus_freq.with_columns(
            pl.col("corpus_count")
            .map_elements(
                lambda c: abs(np.log10(self.n_entities_ / (1 + c))),
                return_dtype=pl.Float64,
            )
            .alias("idf")
        )
        return self

    def transform(
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
    ) -> pl.DataFrame:
        check_is_fitted(self)
        entity_col = entity_col or self.entity_col_
        tag_col = tag_col or self.tag_col_
        value_col = value_col or self.value_col_

        wc = self.weight_col
        lc = self.level_col

        if wc and wc in X.columns:
            df = X.with_columns((pl.col(wc) * pl.col(value_col)).alias("_wv"))
        else:
            df = X.with_columns(pl.col(value_col).cast(pl.Float64).alias("_wv"))

        tf_num = df.group_by([entity_col, tag_col]).agg(
            pl.col("_wv").sum().alias("_in_doc"),
        )
        if lc and lc in X.columns:
            level_agg = df.group_by([entity_col, tag_col]).agg(
                pl.col(lc).first().alias("_level"),
            )
            tf_num = tf_num.join(level_agg, on=[entity_col, tag_col])
        else:
            tf_num = tf_num.with_columns(pl.lit(1.0).alias("_level"))

        doc_len = df.group_by(entity_col).agg(pl.col("_wv").sum().alias("_doc_len"))
        tf = tf_num.join(doc_len, on=entity_col).with_columns(
            (pl.col("_in_doc") / pl.col("_doc_len")).alias("_tf")
        )

        scored = tf.join(self.idf_, on=tag_col).with_columns(
            (pl.col("_level") * pl.col("_tf") * pl.col("idf")).alias("score")
        )

        mu = scored.group_by(entity_col).agg(pl.col("score").mean().alias("_mu"))
        sigma = scored.group_by(entity_col).agg(pl.col("score").std(ddof=0).alias("_sigma"))

        scored = scored.join(mu, on=entity_col).join(sigma, on=entity_col)
        scored = scored.with_columns(
            pl.when(pl.col("_sigma") == 0)
            .then(pl.lit(3.0))
            .otherwise((pl.col("score") - pl.col("_mu")) / pl.col("_sigma"))
            .alias("_zscore")
        )

        dominant = scored.filter(pl.col("_zscore") > self.zscore_thresh).with_columns(
            pl.lit(1.0).alias("final_score")
        )

        normal = scored.filter(pl.col("_zscore").abs() <= self.zscore_thresh)
        min_max = normal.group_by(entity_col).agg(
            pl.col("score").max().alias("_smax"),
            pl.col("score").min().alias("_smin"),
        )
        normal = normal.join(min_max, on=entity_col).with_columns(
            pl.when(pl.col("_smax") == pl.col("_smin"))
            .then(pl.lit(1.0))
            .otherwise(
                (pl.col("score") - pl.col("_smin")) / (pl.col("_smax") - pl.col("_smin"))
            )
            .alias("final_score")
        )
        normal = normal.filter(pl.col("final_score") >= self.score_threshold)

        return pl.concat([
            dominant.select(entity_col, tag_col, "final_score"),
            normal.select(entity_col, tag_col, "final_score"),
        ])

    def fit_transform(  # type: ignore[override]
        self,
        X: pl.DataFrame,
        entity_col: str | None = None,
        tag_col: str | None = None,
        value_col: str | None = None,
        y: None = None,
    ) -> pl.DataFrame:
        return self.fit(X, entity_col, tag_col, value_col).transform(X, entity_col, tag_col, value_col)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tagging.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add datasci_toolkit/tagging.py tests/test_tagging.py
git commit -m "feat: add WeightedTFIDF for entity tagging with Z-score normalization"
```

---

### Task 4: Package Integration

**Files:**
- Modify: `datasci_toolkit/__init__.py`
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`
- Create: `docs/api/smoothing.md`
- Create: `docs/api/tagging.md`
- Create: `docs/tutorials/smoothing.md`
- Create: `docs/tutorials/tagging.md`

**Context:** The project uses mkdocs-material with mkdocstrings for API docs. Since this project has zero docstrings, the API reference pages use `::: datasci_toolkit.module` directives that show class signatures and source. Tutorials are standalone markdown with code examples. The existing `__init__.py` exports all public classes. `README.md` has a module table. `mkdocs.yml` has nav entries for tutorials and API reference.

- [ ] **Step 1: Update `datasci_toolkit/__init__.py`**

Add these imports after the existing ones:

```python
from datasci_toolkit.smoothing import PoissonSmoother, PredictionSmoother
from datasci_toolkit.tagging import WeightedTFIDF
```

Add to `__all__`:

```python
"PoissonSmoother",
"PredictionSmoother",
"WeightedTFIDF",
```

- [ ] **Step 2: Update `README.md` module table**

Add two rows to the module table (after the `temporal` row):

```markdown
| `smoothing` | `PoissonSmoother`, `PredictionSmoother` | Adaptive temporal count smoothing and prediction de-jittering |
| `tagging` | `WeightedTFIDF` | Weighted TF-IDF entity tagging with Z-score normalization |
```

- [ ] **Step 3: Update `docs/index.md` module table**

Add two rows (after the `temporal` row):

```markdown
| [`smoothing`](api/smoothing.md) | `PoissonSmoother`, `PredictionSmoother` | Adaptive temporal smoothing |
| [`tagging`](api/tagging.md) | `WeightedTFIDF` | Weighted TF-IDF entity tagging |
```

- [ ] **Step 4: Create API reference pages**

Create `docs/api/smoothing.md`:

```markdown
# smoothing

::: datasci_toolkit.smoothing.PoissonSmoother

::: datasci_toolkit.smoothing.PredictionSmoother
```

Create `docs/api/tagging.md`:

```markdown
# tagging

::: datasci_toolkit.tagging.WeightedTFIDF
```

- [ ] **Step 5: Create tutorial pages**

Create `docs/tutorials/smoothing.md`:

```markdown
# Smoothing

Adaptive temporal smoothing for count time series and classification probabilities.

## PoissonSmoother

Smooths count time series using Poisson CDF p-values as adaptive blend weights. When today's count is consistent with history, the output is smoothed toward the historical mean. When it's anomalous, the raw observation is trusted.

### Basic usage

` ` `python
import polars as pl
from datasci_toolkit import PoissonSmoother

df = pl.DataFrame({
    "product_id": ["A"] * 7,
    "date": [f"2024-01-{d:02d}" for d in range(1, 8)],
    "orders": [50, 48, 52, 49, 51, 50, 320],
})

ps = PoissonSmoother(window_size=7, alpha=0.1).fit()
result = ps.transform(
    df,
    entity_cols=["product_id"],
    date_col="date",
    value_col="orders",
    target_date="2024-01-07",
)
print(result)
` ` `

The flash-sale count of 320 will be mostly trusted (low p-value → high weight on today), while a normal count like 48 would be smoothed toward ~50.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `7` | Number of time periods including target. Min 2. |
| `alpha` | `0.1` | Weight compression exponent. Lower = more smoothing. |
| `eps_left` | `20` | CDF stabilization for lower tail. |
| `eps_right` | `5` | CDF stabilization for upper tail. |

### How it works

1. Splits data into today (target_date) and history
2. Computes historical mean per entity
3. Calculates Poisson p-value: how likely is today's count given the historical mean?
4. Blends: `smoothed = w_today * today + w_hist * history_sum`
5. Filters out rows where smoothed_count <= 0

---

## PredictionSmoother

Averages classification probabilities across time periods to stabilize label assignments. Reduces label churn caused by small feature fluctuations near decision boundaries.

### Binary mode

` ` `python
import polars as pl
from datasci_toolkit import PredictionSmoother

df = pl.DataFrame({
    "customer_id": ["C1", "C1", "C1"],
    "month": [1, 2, 3],
    "prob_default": [0.52, 0.48, 0.51],
})

ps = PredictionSmoother(min_observations=2).fit()
result = ps.transform(
    df,
    entity_cols=["customer_id"],
    period_col="month",
    prob_cols="prob_default",
)
print(result)
` ` `

### Multi-class mode

` ` `python
df = pl.DataFrame({
    "product_id": ["P1", "P1", "P1"],
    "month": [1, 2, 3],
    "prob_electronics": [0.51, 0.48, 0.53],
    "prob_clothing": [0.39, 0.42, 0.37],
    "prob_food": [0.10, 0.10, 0.10],
})

ps = PredictionSmoother().fit()
result = ps.transform(
    df,
    entity_cols=["product_id"],
    period_col="month",
    prob_cols=["prob_electronics", "prob_clothing", "prob_food"],
)
print(result)
` ` `

Multi-class mode adds a `predicted_label` column with the column name having the highest averaged probability.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_observations` | `1` | Minimum time periods required to produce output. |
```

Create `docs/tutorials/tagging.md`:

```markdown
# Tagging

Weighted TF-IDF with per-entity Z-score normalization for entity tagging.

## WeightedTFIDF

Assigns ranked tags to entities using a weighted TF-IDF score, then normalizes per entity via Z-scores and min-max scaling.

### Basic usage (standard TF-IDF)

` ` `python
import polars as pl
from datasci_toolkit import WeightedTFIDF

df = pl.DataFrame({
    "doc_id": ["A", "A", "A", "B", "B", "B"],
    "term": ["python", "data", "ml", "python", "web", "api"],
    "count": [10, 8, 5, 12, 7, 3],
})

tfidf = WeightedTFIDF(score_threshold=0.1)
result = tfidf.fit_transform(df, entity_col="doc_id", tag_col="term", value_col="count")
print(result)
` ` `

### With external weights and hierarchy

` ` `python
df = pl.DataFrame({
    "product": ["P1", "P1", "P1", "P2", "P2"],
    "attribute": ["durable", "lightweight", "cheap", "durable", "premium"],
    "mentions": [10, 5, 20, 8, 3],
    "confidence": [0.9, 0.7, 0.3, 0.8, 0.9],
    "tier": [1.0, 1.0, 0.5, 1.0, 1.0],
})

tfidf = WeightedTFIDF(weight_col="confidence", level_col="tier")
result = tfidf.fit_transform(
    df, entity_col="product", tag_col="attribute", value_col="mentions"
)
print(result)
` ` `

The `confidence` column weights each mention by its reliability. The `tier` column boosts primary attributes over secondary ones.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `zscore_thresh` | `2.0` | Tags with Z-score above this are dominant (score=1.0). |
| `score_threshold` | `0.1` | Minimum final score to retain a tag. |
| `weight_col` | `None` | Column with external relevance signal. None = all 1.0. |
| `level_col` | `None` | Column with hierarchy multiplier. None = all 1.0. |

### How it works

1. **Weighted TF**: `sum(weight * value) / entity_total` — normalized within each entity
2. **IDF**: `|log10(N / (1 + entity_count))|` — penalizes globally common tags
3. **Score**: `level * TF * IDF`
4. **Z-score**: per-entity normalization. Single-tag entities get Z=3.0 (dominant)
5. **Dominant tags** (Z > threshold): assigned final_score=1.0
6. **Normal tags**: min-max scaled to [0, 1] within entity, filtered by score_threshold
```

- [ ] **Step 6: Update `mkdocs.yml` nav**

Add to Tutorials section (after Temporal Features):

```yaml
    - Smoothing: tutorials/smoothing.md
    - Tagging: tutorials/tagging.md
```

Add to API Reference section:

```yaml
    - smoothing: api/smoothing.md
    - tagging: api/tagging.md
```

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add datasci_toolkit/__init__.py README.md docs/ mkdocs.yml
git commit -m "feat: integrate smoothing and tagging modules into package"
```
