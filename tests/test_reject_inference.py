import numpy as np
import polars as pl
import pytest

from datasci_toolkit.reject_inference import (
    TargetImputer,
    RejectInferenceImputer,
    _dist_weights,
)

RNG = np.random.default_rng(0)


# --- fixtures ---

@pytest.fixture
def accept_df() -> tuple[pl.DataFrame, pl.Series]:
    n = 200
    X = RNG.normal(0, 1, (n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return pl.DataFrame({f"f{i}": X[:, i].tolist() for i in range(4)}), pl.Series(y.tolist())


@pytest.fixture
def reject_df(accept_df: tuple[pl.DataFrame, pl.Series]) -> pl.DataFrame:
    X, _ = accept_df
    n = 50
    X_rej = RNG.normal(0, 1, (n, 4))
    return pl.DataFrame({f"f{i}": X_rej[:, i].tolist() for i in range(4)})


# --- _dist_weights ---

def test_dist_weights_sum_to_one() -> None:
    d = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    w = _dist_weights(d)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_dist_weights_closer_is_higher() -> None:
    d = np.array([[1.0, 10.0]])
    w = _dist_weights(d)
    assert w[0, 0] > w[0, 1]


def test_dist_weights_zero_distance_dominates() -> None:
    d = np.array([[0.0, 1.0, 2.0]])
    w = _dist_weights(d)
    assert w[0, 0] > 0.99


# --- TargetImputer ---

def test_target_imputer_weighted_doubles_rows() -> None:
    p = np.array([0.3, 0.7, 0.5])
    t = TargetImputer(method="weighted").fit(p)
    result = t.transform()
    assert len(result) == 6


def test_target_imputer_weighted_output_schema() -> None:
    p = np.array([0.4, 0.6])
    t = TargetImputer(method="weighted").fit(p)
    result = t.transform()
    assert set(result.columns) == {"target", "weight"}


def test_target_imputer_weighted_weights_sum_to_original() -> None:
    p = np.array([0.3, 0.7])
    w = np.array([2.0, 3.0])
    t = TargetImputer(method="weighted").fit(p, w)
    result = t.transform()
    for i, orig_w in enumerate(w):
        row_weights = result["weight"][i::len(p)]
        assert float(row_weights.sum()) == pytest.approx(orig_w)


def test_target_imputer_weighted_target_values() -> None:
    p = np.array([0.5])
    t = TargetImputer(method="weighted").fit(p)
    result = t.transform()
    assert set(result["target"].to_list()) == {0.0, 1.0}


def test_target_imputer_randomized_preserves_rows() -> None:
    p = np.array([0.3, 0.7, 0.5])
    t = TargetImputer(method="randomized").fit(p)
    result = t.transform()
    assert len(result) == 3


def test_target_imputer_randomized_binary_targets() -> None:
    p = RNG.uniform(size=100)
    t = TargetImputer(method="randomized").fit(p)
    result = t.transform()
    assert set(result["target"].unique().to_list()).issubset({0.0, 1.0})


def test_target_imputer_randomized_high_prob_mostly_one() -> None:
    p = np.ones(200) * 0.95
    t = TargetImputer(method="randomized", seed=0).fit(p)
    result = t.transform()
    assert result["target"].mean() > 0.8


def test_target_imputer_cutoff_above_threshold_is_one() -> None:
    p = np.array([0.4, 0.6, 0.8])
    t = TargetImputer(method="cutoff", cutoff=0.5).fit(p)
    result = t.transform()
    assert result["target"].to_list() == [0.0, 1.0, 1.0]


def test_target_imputer_cutoff_preserves_weights() -> None:
    p = np.array([0.6, 0.4])
    w = np.array([1.5, 2.5])
    t = TargetImputer(method="cutoff").fit(p, w)
    result = t.transform()
    assert result["weight"].to_list() == pytest.approx([1.5, 2.5])


def test_target_imputer_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        TargetImputer(method="invalid").fit(np.array([0.5]))


def test_target_imputer_not_fitted_raises() -> None:
    with pytest.raises(Exception):
        TargetImputer().transform()


def test_target_imputer_polars_series_input() -> None:
    p = pl.Series([0.3, 0.7])
    t = TargetImputer(method="cutoff").fit(p)
    result = t.transform()
    assert len(result) == 2


def test_target_imputer_with_polars_weights() -> None:
    p = pl.Series([0.3, 0.7])
    w = pl.Series([1.0, 2.0])
    t = TargetImputer(method="cutoff").fit(p, w)
    result = t.transform()
    assert result["weight"].to_list() == pytest.approx([1.0, 2.0])


# --- RejectInferenceImputer ---

def test_reject_inference_fit_returns_self(
    accept_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5)
    assert ri.fit(X, y) is ri


def test_reject_inference_transform_shape(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5, method="randomized").fit(X, y)
    result = ri.transform(reject_df)
    assert len(result) == len(reject_df)


def test_reject_inference_weighted_doubles_rows(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5, method="weighted").fit(X, y)
    result = ri.transform(reject_df)
    assert len(result) == 2 * len(reject_df)


def test_reject_inference_output_columns(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5).fit(X, y)
    result = ri.transform(reject_df)
    assert set(result.columns) == {"target", "weight"}


def test_reject_inference_proba_range(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5).fit(X, y)
    proba = ri.predict_proba(reject_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_reject_inference_proba_near_accept_rate(
    accept_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=20).fit(X, y)
    proba = ri.predict_proba(X)
    accept_rate = float(y.cast(pl.Float64).mean())
    assert abs(float(proba.mean()) - accept_rate) < 0.1


def test_reject_inference_high_event_region_gets_high_proba(
    accept_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=10).fit(X, y)
    X_high = X.filter(pl.col("f0") > 1.0)
    X_low = X.filter(pl.col("f0") < -1.0)
    if len(X_high) > 5 and len(X_low) > 5:
        assert ri.predict_proba(X_high).mean() > ri.predict_proba(X_low).mean()


def test_reject_inference_with_weights(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    w_acc = pl.Series(np.ones(len(X)))
    w_rej = pl.Series(np.ones(len(reject_df)) * 2.0)
    ri = RejectInferenceImputer(n_neighbors=5, method="cutoff").fit(X, y, w_acc)
    result = ri.transform(reject_df, w_rej)
    assert (result["weight"] == 2.0).all()


def test_reject_inference_not_fitted_raises(reject_df: pl.DataFrame) -> None:
    with pytest.raises(Exception):
        RejectInferenceImputer().transform(reject_df)


def test_reject_inference_n_neighbors_capped(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=10000, method="randomized").fit(X, y)
    result = ri.transform(reject_df)
    assert len(result) == len(reject_df)


def test_reject_inference_cutoff_binary_targets(
    accept_df: tuple[pl.DataFrame, pl.Series],
    reject_df: pl.DataFrame,
) -> None:
    X, y = accept_df
    ri = RejectInferenceImputer(n_neighbors=5, method="cutoff").fit(X, y)
    result = ri.transform(reject_df)
    assert set(result["target"].unique().to_list()).issubset({0.0, 1.0})
