import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification

from datasci_toolkit.model_selection import (
    AUCStepwiseLogit,
    _auc,
    _corr_matrix,
    _cv_auc,
    _max_abs_corr,
    _same_sign,
)

RNG = np.random.default_rng(0)


# --- fixtures ---

@pytest.fixture
def binary_df() -> tuple[pl.DataFrame, pl.Series]:
    X_np, y_np = make_classification(n_samples=400, n_features=6, n_informative=3, n_redundant=1, random_state=0)
    cols = [f"f{i}" for i in range(X_np.shape[1])]
    return pl.DataFrame(dict(zip(cols, X_np.T))), pl.Series(y_np.astype(float))


@pytest.fixture
def split_df(binary_df: tuple[pl.DataFrame, pl.Series]) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]:
    X, y = binary_df
    half = len(X) // 2
    return X[:half], y[:half], X[half:], y[half:]


# --- _same_sign ---

def test_same_sign_all_positive() -> None:
    assert _same_sign(np.array([0.1, 0.5, 0.9]))


def test_same_sign_all_negative() -> None:
    assert _same_sign(np.array([-0.1, -0.5, -0.9]))


def test_same_sign_mixed_false() -> None:
    assert not _same_sign(np.array([0.5, -0.1]))


def test_same_sign_single() -> None:
    assert _same_sign(np.array([1.0]))


# --- _max_abs_corr ---

def test_max_abs_corr_ignores_diagonal() -> None:
    corr = np.eye(3)
    assert _max_abs_corr(corr, [0, 1, 2]) == 0.0


def test_max_abs_corr_single_index() -> None:
    corr = np.eye(3)
    assert _max_abs_corr(corr, [1]) == 0.0


def test_max_abs_corr_detects_correlation() -> None:
    corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    assert _max_abs_corr(corr, [0, 1]) == pytest.approx(0.9)


# --- _corr_matrix ---

def test_corr_matrix_shape() -> None:
    X = RNG.normal(0, 1, (100, 5))
    c = _corr_matrix(X, 1000)
    assert c.shape == (5, 5)


def test_corr_matrix_diagonal_ones() -> None:
    X = RNG.normal(0, 1, (100, 4))
    c = _corr_matrix(X, 1000)
    assert np.allclose(np.diag(c), 1.0)


def test_corr_matrix_samples_when_large() -> None:
    X = RNG.normal(0, 1, (1000, 3))
    c = _corr_matrix(X, 50)
    assert c.shape == (3, 3)


# --- _auc ---

def test_auc_perfect_prediction() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert _auc(y, p) > 0.99


def test_auc_returns_float() -> None:
    y = np.array([0.0, 1.0])
    p = np.array([0.3, 0.7])
    assert isinstance(_auc(y, p), float)


def test_auc_with_weights() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.ones(4)
    assert _auc(y, p, w) > 0.99


# --- _cv_auc ---

def test_cv_auc_returns_float(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    result = _cv_auc(X.to_numpy(), y.to_numpy(), None, "l2", 1000.0, 3, 42, True)
    assert isinstance(result, float)


def test_cv_auc_range(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    result = _cv_auc(X.to_numpy(), y.to_numpy(), None, "l2", 1000.0, 3, 42, True)
    assert 0.0 <= result <= 1.0


# --- AUCStepwiseLogit ---

def test_fit_returns_self(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=2, min_increase=0.001)
    assert m.fit(X, y) is m


def test_forward_selects_predictors(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=5, min_increase=0.001)
    m.fit(X, y)
    assert len(m.predictors_) >= 1


def test_backward_from_initial(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(
        initial_predictors=list(X.columns),
        selection_method="backward",
        max_iter=5,
        max_decrease=0.001,
    )
    m.fit(X, y)
    assert len(m.predictors_) <= len(X.columns)


def test_stepwise_converges(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="stepwise", max_iter=10, min_increase=0.001)
    m.fit(X, y)
    assert m.predictors_ is not None


def test_fit_with_validation_split(split_df: tuple[pl.DataFrame, pl.Series, pl.DataFrame, pl.Series]) -> None:
    X_tr, y_tr, X_va, y_va = split_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001)
    m.fit(X_tr, y_tr, X_va, y_va)
    assert len(m.predictors_) >= 1


def test_predict_shape(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (len(X),)


def test_predict_range(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    preds = m.predict(X)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_score_returns_float(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    assert isinstance(m.score(X, y), float)


def test_score_with_weights(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    w = pl.Series(np.ones(len(y)))
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    result = m.score(X, y, w)
    assert 0.0 <= result <= 1.0


def test_not_fitted_predict_raises(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, _ = binary_df
    with pytest.raises(Exception):
        AUCStepwiseLogit().predict(X)


def test_not_fitted_score_raises(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    with pytest.raises(Exception):
        AUCStepwiseLogit().score(X, y)


def test_max_predictors_limit(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=20, min_increase=0.0, max_predictors=2).fit(X, y)
    assert len(m.predictors_) <= 2


def test_initial_predictors_used(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    init = [X.columns[0]]
    m = AUCStepwiseLogit(initial_predictors=init, selection_method="forward", max_iter=2, min_increase=0.001).fit(X, y)
    assert X.columns[0] in m.predictors_


def test_all_predictors_subset(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    subset = list(X.columns[:3])
    m = AUCStepwiseLogit(all_predictors=subset, selection_method="forward", max_iter=5, min_increase=0.001).fit(X, y)
    assert all(p in subset for p in m.predictors_)


def test_progress_is_dataframe(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    assert isinstance(m.progress_, pl.DataFrame)


def test_progress_has_expected_columns(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001).fit(X, y)
    assert {"iteration", "addrm", "auc", "delta", "predictors"}.issubset(set(m.progress_.columns))


def test_coef_shape_matches_predictors(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=5, min_increase=0.001).fit(X, y)
    assert len(m.coef_) == len(m.predictors_)


def test_cv_mode(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=2, min_increase=0.001, use_cv=True, cv_folds=3).fit(X, y)
    assert len(m.predictors_) >= 1


def test_l1_penalty(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=3, min_increase=0.001, penalty="l1").fit(X, y)
    assert len(m.predictors_) >= 1


def test_high_min_increase_empty_model(binary_df: tuple[pl.DataFrame, pl.Series]) -> None:
    X, y = binary_df
    m = AUCStepwiseLogit(selection_method="forward", max_iter=5, min_increase=1.0).fit(X, y)
    assert m.predictors_ == []
