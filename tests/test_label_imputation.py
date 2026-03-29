import numpy as np
import polars as pl
import pytest

from datasci_toolkit.label_imputation import (
    TargetImputer,
    KNNLabelImputer,
    _dist_weights,
)

RNG = np.random.default_rng(0)


# --- fixtures ---

@pytest.fixture
def labeled_df() -> tuple[pl.DataFrame, pl.Series]:
    n = 200
    X = RNG.normal(0, 1, (n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return pl.DataFrame({f"f{i}": X[:, i].tolist() for i in range(4)}), pl.Series(y.tolist())


@pytest.fixture
def unlabeled_df() -> pl.DataFrame:
    n = 50
    X = RNG.normal(0, 1, (n, 4))
    return pl.DataFrame({f"f{i}": X[:, i].tolist() for i in range(4)})


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


# --- KNNLabelImputer ---

def test_knn_label_imputer_fit_returns_self(
    labeled_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5)
    assert ki.fit(X, y) is ki


def test_knn_label_imputer_transform_shape_randomized(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5, method="randomized").fit(X, y)
    result = ki.transform(unlabeled_df)
    assert len(result) == len(unlabeled_df)


def test_knn_label_imputer_weighted_doubles_rows(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5, method="weighted").fit(X, y)
    result = ki.transform(unlabeled_df)
    assert len(result) == 2 * len(unlabeled_df)


def test_knn_label_imputer_output_columns(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5).fit(X, y)
    result = ki.transform(unlabeled_df)
    assert set(result.columns) == {"target", "weight"}


def test_knn_label_imputer_proba_range(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5).fit(X, y)
    proba = ki.predict_proba(unlabeled_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_knn_label_imputer_proba_near_labeled_rate(
    labeled_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=20).fit(X, y)
    proba = ki.predict_proba(X)
    labeled_rate = float(y.cast(pl.Float64).mean())
    assert abs(float(proba.mean()) - labeled_rate) < 0.1


def test_knn_label_imputer_high_event_region_gets_high_proba(
    labeled_df: tuple[pl.DataFrame, pl.Series],
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=10).fit(X, y)
    X_high = X.filter(pl.col("f0") > 1.0)
    X_low = X.filter(pl.col("f0") < -1.0)
    if len(X_high) > 5 and len(X_low) > 5:
        assert ki.predict_proba(X_high).mean() > ki.predict_proba(X_low).mean()


def test_knn_label_imputer_with_weights(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    w_lab = pl.Series(np.ones(len(X)))
    w_unl = pl.Series(np.ones(len(unlabeled_df)) * 2.0)
    ki = KNNLabelImputer(n_neighbors=5, method="cutoff").fit(X, y, w_lab)
    result = ki.transform(unlabeled_df, w_unl)
    assert (result["weight"] == 2.0).all()


def test_knn_label_imputer_not_fitted_raises(unlabeled_df: pl.DataFrame) -> None:
    with pytest.raises(Exception):
        KNNLabelImputer().transform(unlabeled_df)


def test_knn_label_imputer_n_neighbors_capped(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=10000, method="randomized").fit(X, y)
    result = ki.transform(unlabeled_df)
    assert len(result) == len(unlabeled_df)


def test_knn_label_imputer_cutoff_binary_targets(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5, method="cutoff").fit(X, y)
    result = ki.transform(unlabeled_df)
    assert set(result["target"].unique().to_list()).issubset({0.0, 1.0})


def test_knn_label_imputer_manhattan_metric(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5, metric="manhattan").fit(X, y)
    proba = ki.predict_proba(unlabeled_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_knn_label_imputer_cosine_metric(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    ki = KNNLabelImputer(n_neighbors=5, metric="cosine").fit(X, y)
    proba = ki.predict_proba(unlabeled_df)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_knn_label_imputer_different_metrics_differ(
    labeled_df: tuple[pl.DataFrame, pl.Series],
    unlabeled_df: pl.DataFrame,
) -> None:
    X, y = labeled_df
    p_l2 = KNNLabelImputer(n_neighbors=5, metric="minkowski").fit(X, y).predict_proba(unlabeled_df)
    p_cos = KNNLabelImputer(n_neighbors=5, metric="cosine").fit(X, y).predict_proba(unlabeled_df)
    assert not np.allclose(p_l2, p_cos)
