# Smoothing & Tagging Algorithms — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract three algorithms from a PySpark gist into Polars-native, sklearn-compatible classes across two new modules.

**Source:** `src/smoothing_tagging.md` (gitignored gist)

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Module organization | Two modules: `smoothing.py` + `tagging.py` | Temporal smoothers share a pattern; TF-IDF is a different domain |
| Poisson smoother API | Handles windowing internally | One-stop shop, matches toolkit conventions |
| Prediction smoother scope | Binary + multi-class | Binary is the project's core; multi-class as extension |
| TF-IDF required inputs | Only `value` required; `weight` and `level` default to 1.0 | Degrades to standard TF-IDF without extras |
| Sklearn pattern | All three use `fit`/`transform` | Consistency with rest of toolkit |

---

## Module 1: `datasci_toolkit/smoothing.py`

### `PoissonSmoother(BaseEstimator, TransformerMixin)`

Adaptive temporal smoothing for count time series using Poisson CDF p-values as adaptive blend weights.

**`__init__` params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `window_size` | `int` | required | Number of time periods including target. Min 2. |
| `alpha` | `float` | `0.1` | Weight compression exponent. Lower = more smoothing. |
| `eps_left` | `int` | `20` | CDF stabilization constant for lower tail |
| `eps_right` | `int` | `5` | CDF stabilization constant for upper tail |

**`fit(X=None, y=None)`** — stores config, validates `window_size >= 2`, returns self.

**`transform(X, entity_cols, date_col, value_col, target_date)`**

Input: Polars DataFrame with the full window of data (N periods per entity).

Steps:
1. Split into today (`date == target_date`) vs history (`date != target_date`)
2. Group by `entity_cols`: compute `today_count`, `history_sum`, `history_mean = history_sum / (window_size - 1)`
3. Compute Poisson p-value vectorized via numpy/scipy:
   - If `today_count <= history_mean`: `p = poisson.cdf(floor(k) + eps_left, mu + eps_left)`
   - If `today_count > history_mean`: `p = 1 - poisson.cdf(ceil(k) + eps_right, mu + eps_right)`
4. Blend: `w_today = 1 - p^alpha`, `w_hist = p^alpha / (window_size - 1)`, `smoothed = w_today * today_count + w_hist * history_sum`
5. Filter out rows where `smoothed_count <= 0`

Output: Polars DataFrame with `entity_cols` + `today_count`, `history_mean`, `pvalue`, `smoothed_count`.

**Dependencies:** `scipy.stats.poisson`, `numpy`

---

### `PredictionSmoother(BaseEstimator, TransformerMixin)`

Temporal smoothing for classification probabilities. Averages probability vectors across time periods to stabilize label assignments.

**`__init__` params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `min_observations` | `int` | `1` | Minimum time periods required to produce output |

**`fit(X=None, y=None)`** — stores config, returns self.

**`transform(X, entity_cols, period_col, prob_cols)`**

- If `prob_cols` is a `str`: binary mode. Groups by `entity_cols`, averages the single probability column, returns smoothed probability + `observation_count`.
- If `prob_cols` is a `list[str]`: multi-class mode. Averages each column, derives `predicted_label` as the column name with the highest averaged probability (argmax). Returns smoothed probabilities + `observation_count` + `predicted_label`.

Both modes filter to entities with `observation_count >= min_observations`.

Output: Polars DataFrame with `entity_cols` + smoothed probability column(s) + `observation_count` + (multi-class) `predicted_label`.

**Dependencies:** none beyond polars

---

## Module 2: `datasci_toolkit/tagging.py`

### `WeightedTFIDF(BaseEstimator, TransformerMixin)`

Weighted TF-IDF with per-entity Z-score normalization for entity tagging.

**`__init__` params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `zscore_thresh` | `float` | `2.0` | Tags above this Z-score are dominant (score=1.0) |
| `score_threshold` | `float` | `0.1` | Minimum final score to retain a tag |
| `weight_col` | `str \| None` | `None` | Column with external relevance signal. None = all 1.0 |
| `level_col` | `str \| None` | `None` | Column with hierarchy multiplier. None = all 1.0 |

**`fit(X, entity_col, tag_col, value_col)`**

Learns corpus-level IDF:
1. `N_` = count of distinct entities
2. `corpus_freq_` = count of entities per tag
3. `idf_` = `|log10(N / (1 + corpus_count))|` per tag (Polars DataFrame)

**`transform(X, entity_col, tag_col, value_col)`**

Scores entities against learned IDF:
1. Weighted TF: `weighted_val = weight * value` (or just `value` if no weight_col), normalized by entity total
2. Score: `level * TF * IDF` (or `TF * IDF` if no level_col)
3. Per-entity Z-score: population mean/std. Single-tag entities get Z=3.0 (dominant)
4. Dominant tags (Z > zscore_thresh): `final_score = 1.0`
5. Non-dominant: min-max scaled within entity to [0, 1]. If max == min, score = 1.0
6. Filter below `score_threshold`

Output: Polars DataFrame with `entity_col`, `tag_col`, `final_score`.

**Dependencies:** `numpy` for log10

---

## File Structure

| File | Contents |
|---|---|
| `datasci_toolkit/smoothing.py` | `PoissonSmoother`, `PredictionSmoother` |
| `datasci_toolkit/tagging.py` | `WeightedTFIDF` |
| `tests/test_smoothing.py` | Tests for both smoothers |
| `tests/test_tagging.py` | Tests for WeightedTFIDF |

## Package Integration

- Add imports and `__all__` entries in `datasci_toolkit/__init__.py`
- Add `scipy` to dependencies in `pyproject.toml` (already present)
- Update `README.md` module table
- Update `docs/index.md` module table
- Add tutorial docs and API reference pages
- Update `mkdocs.yml` nav

## Style

- Zero comments, zero docstrings
- DRY, minimal
- sklearn conventions: `BaseEstimator`, `fit`/`transform`, `check_is_fitted`
- Polars in, Polars out
- Vectorized operations (no row-by-row UDFs)
