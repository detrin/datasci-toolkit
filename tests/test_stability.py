import numpy as np
import polars as pl
import pytest

from datasci_toolkit.stability import (
    ESI,
    PSI,
    StabilityMonitor,
    _to_series,
    _weighted_dist,
)

RNG = np.random.default_rng(42)


# --- fixtures ---

@pytest.fixture
def normal_series() -> pl.Series:
    return pl.Series(RNG.normal(0, 1, 1000).tolist())


@pytest.fixture
def shifted_series() -> pl.Series:
    return pl.Series(RNG.normal(3, 1, 1000).tolist())


@pytest.fixture
def cat_series() -> pl.Series:
    return pl.Series(["a"] * 500 + ["b"] * 300 + ["c"] * 200)


@pytest.fixture
def esi_stable() -> pl.DataFrame:
    rows = []
    for m in range(1, 5):
        for cat, bad_rate in [("A", 0.05), ("B", 0.10), ("C", 0.20)]:
            n_bad = int(bad_rate * 100)
            rows += [{"month": m, "var": cat, "target": 1, "base": 1}] * n_bad
            rows += [{"month": m, "var": cat, "target": 0, "base": 1}] * (100 - n_bad)
    return pl.DataFrame(rows)


@pytest.fixture
def esi_unstable() -> pl.DataFrame:
    rows = []
    for m in range(1, 5):
        cats = [("A", 0.20), ("B", 0.05)] if m % 2 == 0 else [("A", 0.05), ("B", 0.20)]
        for cat, bad_rate in cats:
            n_bad = int(bad_rate * 100)
            rows += [{"month": m, "var": cat, "target": 1, "base": 1}] * n_bad
            rows += [{"month": m, "var": cat, "target": 0, "base": 1}] * (100 - n_bad)
    return pl.DataFrame(rows)


@pytest.fixture
def monitor_df() -> pl.DataFrame:
    n = 500
    return pl.DataFrame({
        "feat_num": RNG.normal(0, 1, n).tolist(),
        "feat_cat": RNG.choice(["a", "b", "c"], n).tolist(),
        "month": [m for m in range(1, 6) for _ in range(100)],
        "weight": np.ones(n).tolist(),
    })


# --- _to_series ---

def test_to_series_passthrough() -> None:
    s = pl.Series([1, 2, 3])
    assert _to_series(s, "x") is s


def test_to_series_converts_list() -> None:
    with pytest.warns(UserWarning):
        s = _to_series([1, 2, 3], "x")
    assert isinstance(s, pl.Series)
    assert len(s) == 3


def test_to_series_raises_on_invalid() -> None:
    with pytest.raises(TypeError):
        _to_series(object(), "x")


# --- _weighted_dist ---

def test_weighted_dist_sums_to_one() -> None:
    X = pl.Series(["a", "b", "a", "c"])
    w = pl.Series([1.0, 1.0, 1.0, 1.0])
    dist = _weighted_dist(X, w, 0.0001)
    assert abs(dist["freq"].sum() - 1.0) < 1e-6


def test_weighted_dist_zero_replaced_with_missing_value() -> None:
    X = pl.Series(["a", "b"])
    w = pl.Series([1.0, 0.0])
    dist = _weighted_dist(X, w, 0.0001)
    assert (dist.filter(pl.col("cat") == "b")["freq"] == 0.0001).all()


def test_weighted_dist_proportional_to_weights() -> None:
    X = pl.Series(["a", "b"])
    w = pl.Series([3.0, 1.0])
    dist = _weighted_dist(X, w, 0.0001)
    freq_a = dist.filter(pl.col("cat") == "a")["freq"][0]
    freq_b = dist.filter(pl.col("cat") == "b")["freq"][0]
    assert abs(freq_a / freq_b - 3.0) < 1e-6


# --- PSI ---

def test_psi_identical_distributions_near_zero(normal_series: pl.Series) -> None:
    psi = PSI().fit(normal_series)
    assert abs(psi.score(normal_series)) < 0.05


def test_psi_shifted_distribution_positive(normal_series: pl.Series, shifted_series: pl.Series) -> None:
    psi = PSI().fit(normal_series)
    assert psi.score(shifted_series) > 0.1


def test_psi_categorical_identical_near_zero(cat_series: pl.Series) -> None:
    psi = PSI().fit(cat_series)
    assert abs(psi.score(cat_series)) < 0.01


def test_psi_categorical_different_positive(cat_series: pl.Series) -> None:
    shifted = pl.Series(["a"] * 100 + ["b"] * 100 + ["c"] * 800)
    psi = PSI().fit(cat_series)
    assert psi.score(shifted) > 0.1


def test_psi_stores_bin_breaks_for_numeric(normal_series: pl.Series) -> None:
    psi = PSI(n_quantile_bins=5).fit(normal_series)
    assert hasattr(psi, "bin_breaks_")
    assert len(psi.bin_breaks_) == 4


def test_psi_no_bin_breaks_for_categorical(cat_series: pl.Series) -> None:
    psi = PSI().fit(cat_series)
    assert not hasattr(psi, "bin_breaks_")


def test_psi_with_weights(normal_series: pl.Series) -> None:
    w = pl.Series(np.ones(len(normal_series)).tolist())
    psi = PSI().fit(normal_series, w)
    assert abs(psi.score(normal_series, w)) < 0.05


def test_psi_not_fitted_raises() -> None:
    with pytest.raises(Exception):
        PSI().score(pl.Series([1.0, 2.0, 3.0]))


def test_psi_score_returns_float(normal_series: pl.Series) -> None:
    psi = PSI().fit(normal_series)
    result = psi.score(normal_series)
    assert isinstance(result, float)


# --- ESI ---

def test_esi_returns_v1_v2(esi_stable: pl.DataFrame) -> None:
    result = ESI().score(esi_stable, "var", "target", "base", "month")
    assert set(result.keys()) == {"v1", "v2"}


def test_esi_stable_ranking_near_one(esi_stable: pl.DataFrame) -> None:
    result = ESI().score(esi_stable, "var", "target", "base", "month")
    assert result["v1"] > 0.95
    assert result["v2"] > 0.95


def test_esi_unstable_ranking_lower(esi_unstable: pl.DataFrame) -> None:
    result = ESI().score(esi_unstable, "var", "target", "base", "month")
    assert result["v1"] < 0.6
    assert result["v2"] < 0.3


def test_esi_exclude_zero() -> None:
    rows = []
    for m in range(1, 5):
        zero_rate = 0.05 if m % 2 == 0 else 0.25
        for woe, bad_rate in [(0.0, zero_rate), (0.5, 0.10), (1.0, 0.20)]:
            n_bad = int(bad_rate * 100)
            rows += [{"month": m, "var": woe, "target": 1, "base": 1}] * n_bad
            rows += [{"month": m, "var": woe, "target": 0, "base": 1}] * (100 - n_bad)
    data = pl.DataFrame(rows)
    result_with = ESI().score(data, "var", "target", "base", "month", exclude_zero=False)
    result_without = ESI().score(data, "var", "target", "base", "month", exclude_zero=True)
    assert result_with["v1"] < result_without["v1"]


def test_esi_with_weights(esi_stable: pl.DataFrame) -> None:
    data = esi_stable.with_columns(pl.lit(1.0).alias("w"))
    result = ESI().score(data, "var", "target", "base", "month", col_weight="w")
    assert 0.0 <= result["v1"] <= 1.0
    assert 0.0 <= result["v2"] <= 1.0


# --- StabilityMonitor ---

def test_monitor_fit_creates_estimators(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num", "feat_cat"]).fit(monitor_df)
    assert set(m.estimators_.keys()) == {"feat_num", "feat_cat"}


def test_monitor_estimators_are_fitted_psi(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num"]).fit(monitor_df)
    assert isinstance(m.estimators_["feat_num"], PSI)
    assert hasattr(m.estimators_["feat_num"], "ref_dist_")


def test_monitor_score_shape(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num", "feat_cat"]).fit(monitor_df)
    result = m.score(monitor_df, "month")
    assert result.shape == (10, 3)
    assert set(result.columns) == {"feature", "month", "psi"}


def test_monitor_score_psi_values_non_negative(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num"]).fit(monitor_df)
    result = m.score(monitor_df, "month")
    assert (result["psi"] >= 0).all()


def test_monitor_score_consecutive_shape(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num", "feat_cat"]).fit(monitor_df)
    result = m.score_consecutive(monitor_df, "month")
    assert result.shape == (8, 3)  # 2 features × 4 consecutive pairs
    assert set(result.columns) == {"feature", "months", "psi"}


def test_monitor_score_masks(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num"]).fit(monitor_df)
    masks = {"early": pl.col("month") <= 2, "late": pl.col("month") > 3}
    result = m.score_masks(monitor_df, masks)
    assert result.shape == (1, 3)
    assert result["psi"][0] >= 0


def test_monitor_score_masks_sorted_descending(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num", "feat_cat"]).fit(monitor_df)
    masks = {"early": pl.col("month") <= 2, "late": pl.col("month") > 3}
    result = m.score_masks(monitor_df, masks)
    assert result["psi"].is_sorted(descending=True)


def test_monitor_with_weights(monitor_df: pl.DataFrame) -> None:
    m = StabilityMonitor(["feat_num"], col_weight="weight").fit(monitor_df)
    result = m.score(monitor_df, "month")
    assert result.shape[0] == 5


def test_monitor_not_fitted_raises(monitor_df: pl.DataFrame) -> None:
    with pytest.raises(Exception):
        StabilityMonitor(["feat_num"]).score(monitor_df, "month")
