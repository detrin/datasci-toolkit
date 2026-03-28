import numpy as np
import polars as pl
import pytest

from datasci_toolkit.metrics import gini, ks, lift, iv, BootstrapGini, feature_power

RNG = np.random.default_rng(42)

N = 400
_SCORE = RNG.uniform(size=N)
_Y = (_SCORE + RNG.normal(0, 0.3, N) > 0.5).astype(float)
_W = RNG.uniform(0.5, 2.0, size=N)


# --- gini ---

def test_gini_in_range() -> None:
    g = gini(_Y, _SCORE)
    assert -1.0 <= g <= 1.0


def test_gini_perfect_score_is_one() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert gini(y, y) == pytest.approx(1.0)


def test_gini_reversed_is_minus_one() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert gini(y, 1 - y) == pytest.approx(-1.0)


def test_gini_weighted_differs_from_unweighted() -> None:
    g_uw = gini(_Y, _SCORE)
    g_w = gini(_Y, _SCORE, sample_weight=_W)
    assert g_uw != pytest.approx(g_w, abs=1e-3)


def test_gini_accepts_polars_series() -> None:
    g = gini(pl.Series(_Y.tolist()), pl.Series(_SCORE.tolist()))
    assert isinstance(g, float)


# --- ks ---

def test_ks_in_range() -> None:
    k = ks(_Y, _SCORE)
    assert 0.0 <= k <= 1.0


def test_ks_perfect_separation() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks(y, s) == pytest.approx(1.0)


def test_ks_accepts_polars_series() -> None:
    k = ks(pl.Series(_Y.tolist()), pl.Series(_SCORE.tolist()))
    assert isinstance(k, float)


# --- lift ---

def test_lift_positive() -> None:
    l = lift(_Y, -_SCORE)
    assert l > 0.0


def test_lift_good_predictor_above_one() -> None:
    l = lift(_Y, -_SCORE, perc=10.0)
    assert l > 1.0


def test_lift_accepts_polars_series() -> None:
    l = lift(pl.Series(_Y.tolist()), pl.Series(_SCORE.tolist()))
    assert isinstance(l, float)


def test_lift_perc_100_approx_one() -> None:
    l = lift(_Y, _SCORE, perc=100.0)
    assert l == pytest.approx(1.0, abs=0.1)


# --- iv ---

def test_iv_nonneg() -> None:
    x_bin = (_SCORE > 0.5).astype(int)
    assert iv(_Y, x_bin) >= 0.0


def test_iv_better_predictor_higher_iv() -> None:
    x_good = (_SCORE > 0.5).astype(int)
    x_random = RNG.integers(0, 2, N)
    assert iv(_Y, x_good) > iv(_Y, x_random)


def test_iv_accepts_polars_series() -> None:
    x_bin = (_SCORE > 0.5).astype(int)
    result = iv(pl.Series(_Y.tolist()), pl.Series(x_bin.tolist()))
    assert isinstance(result, float)


# --- BootstrapGini ---

def test_bootstrap_gini_fit_returns_self() -> None:
    bg = BootstrapGini(n_iter=20, seed=0)
    assert bg.fit(_Y, _SCORE) is bg


def test_bootstrap_gini_mean_close_to_point_gini() -> None:
    bg = BootstrapGini(n_iter=200, seed=0).fit(_Y, _SCORE)
    point = gini(_Y, _SCORE)
    assert abs(bg.mean_ - point) < 0.05


def test_bootstrap_gini_ci_ordered() -> None:
    bg = BootstrapGini(n_iter=100, seed=0).fit(_Y, _SCORE)
    assert bg.ci_[0] <= bg.mean_ <= bg.ci_[1]


def test_bootstrap_gini_std_positive() -> None:
    bg = BootstrapGini(n_iter=100, seed=0).fit(_Y, _SCORE)
    assert bg.std_ > 0.0


def test_bootstrap_gini_scores_length() -> None:
    n_iter = 50
    bg = BootstrapGini(n_iter=n_iter, seed=0).fit(_Y, _SCORE)
    assert len(bg.scores_) == n_iter


def test_bootstrap_gini_ci_level_wider() -> None:
    bg90 = BootstrapGini(n_iter=200, ci_level=90.0, seed=0).fit(_Y, _SCORE)
    bg50 = BootstrapGini(n_iter=200, ci_level=50.0, seed=0).fit(_Y, _SCORE)
    assert (bg90.ci_[1] - bg90.ci_[0]) > (bg50.ci_[1] - bg50.ci_[0])


def test_bootstrap_gini_weighted() -> None:
    bg = BootstrapGini(n_iter=50, seed=0).fit(_Y, _SCORE, sample_weight=_W)
    assert isinstance(bg.mean_, float)


def test_bootstrap_gini_accepts_polars() -> None:
    bg = BootstrapGini(n_iter=20, seed=0).fit(
        pl.Series(_Y.tolist()), pl.Series(_SCORE.tolist())
    )
    assert isinstance(bg.mean_, float)


# --- feature_power ---

def test_feature_power_returns_dataframe() -> None:
    X = pl.DataFrame({"a": (-_SCORE).tolist(), "b": RNG.normal(size=N).tolist()})
    y = pl.Series(_Y.tolist())
    result = feature_power(X, y)
    assert isinstance(result, pl.DataFrame)


def test_feature_power_columns() -> None:
    X = pl.DataFrame({"a": (-_SCORE).tolist()})
    y = pl.Series(_Y.tolist())
    result = feature_power(X, y)
    assert set(result.columns) == {"feature", "gini", "iv"}


def test_feature_power_sorted_by_gini() -> None:
    X = pl.DataFrame({"good": (-_SCORE).tolist(), "random": RNG.normal(size=N).tolist()})
    y = pl.Series(_Y.tolist())
    result = feature_power(X, y)
    ginis = result["gini"].to_list()
    assert ginis == sorted(ginis, reverse=True)


def test_feature_power_good_predictor_first() -> None:
    X = pl.DataFrame({"good": (-_SCORE).tolist(), "random": RNG.normal(size=N).tolist()})
    y = pl.Series(_Y.tolist())
    result = feature_power(X, y)
    assert result["feature"][0] == "good"


def test_feature_power_weighted() -> None:
    X = pl.DataFrame({"a": (-_SCORE).tolist()})
    y = pl.Series(_Y.tolist())
    w = pl.Series(_W.tolist())
    result = feature_power(X, y, sample_weight=w)
    assert len(result) == 1
