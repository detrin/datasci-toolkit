import numpy as np
import polars as pl
import pytest

from sklearn.metrics import roc_auc_score

from datasci_toolkit.grouping import (
    WOETransformer,
    StabilityGrouping,
    _woe,
    _rsi,
    _encode_cats,
    _num_bin_spec,
    _cat_bin_spec,
)

RNG = np.random.default_rng(0)


# --- fixtures ---

@pytest.fixture
def binary_data() -> dict[str, np.ndarray]:
    n = 400
    x = RNG.normal(0, 1, n)
    y = (x > 0).astype(float) + RNG.normal(0, 0.1, n)
    y = np.clip(y, 0, 1).round()
    w = np.ones(n)
    return {"x": x, "y": y, "w": w}


@pytest.fixture
def temporal_df() -> pl.DataFrame:
    n_months = 4
    n_per = 200
    rows = []
    for m in range(1, n_months + 1):
        x = RNG.normal(m * 0.1, 1.0, n_per)
        y = (x > 0).astype(float)
        for xi, yi in zip(x, y):
            rows.append({"feat": xi, "target": yi, "month": m, "weight": 1.0})
    return pl.DataFrame(rows)


@pytest.fixture
def cat_temporal_df() -> pl.DataFrame:
    n_months = 4
    n_per = 100
    rows = []
    for m in range(1, n_months + 1):
        cats = RNG.choice(["A", "B", "C"], n_per).tolist()
        for c in cats:
            y = 1.0 if c == "A" else (0.5 if c == "B" else 0.1)
            y = float(RNG.binomial(1, y))
            rows.append({"feat": c, "target": y, "month": m})
    return pl.DataFrame(rows)


@pytest.fixture
def woe_num_spec() -> dict:
    return {
        "dtype": "float",
        "bins": [-np.inf, 0.0, np.inf],
        "woes": [-0.5, 0.5],
        "nan_woe": 0.0,
    }


@pytest.fixture
def woe_cat_spec() -> dict:
    return {
        "dtype": "category",
        "bins": {"A": 0, "B": 1},
        "woes": [-0.3, 0.3],
        "unknown_woe": 0.0,
    }


# --- _woe ---

def test_woe_high_bad_rate_positive() -> None:
    result = _woe(80.0, 20.0, 100.0, 100.0)
    assert result > 0.0


def test_woe_low_bad_rate_negative() -> None:
    result = _woe(20.0, 80.0, 100.0, 100.0)
    assert result < 0.0


def test_woe_equal_rates_near_zero() -> None:
    result = _woe(50.0, 50.0, 100.0, 100.0)
    assert abs(result) < 0.1


def test_woe_zero_bads_uses_smooth() -> None:
    result = _woe(0.0, 100.0, 100.0, 100.0)
    assert isinstance(result, float)
    assert result < 0.0


def test_woe_symmetry() -> None:
    pos = _woe(80.0, 20.0, 100.0, 100.0)
    neg = _woe(20.0, 80.0, 100.0, 100.0)
    assert abs(pos + neg) < 0.05


# --- roc_auc_score (replaces _gini) ---

def test_auc_perfect_prediction_near_one() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc_score(y, p) > 0.99


def test_auc_random_prediction_near_half() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000).astype(float)
    p = rng.random(1000)
    assert abs(roc_auc_score(y, p) - 0.5) < 0.05


def test_auc_with_weights() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert roc_auc_score(y, p, sample_weight=w) > 0.99


# --- _rsi ---

def test_rsi_stable_returns_one() -> None:
    scores = np.array([0.3, 0.7, 0.3, 0.7, 0.3, 0.7])
    rates = np.array([0.1, 0.4, 0.1, 0.4, 0.1, 0.4])
    months = np.array([1, 1, 2, 2, 3, 3])
    assert _rsi(scores, rates, months, threshold=0.1) == 1.0


def test_rsi_unstable_below_one() -> None:
    scores = np.array([0.3, 0.7, 0.3, 0.7])
    rates = np.array([0.1, 0.9, 0.9, 0.1])
    months = np.array([1, 1, 2, 2])
    assert _rsi(scores, rates, months, threshold=0.05) < 1.0


def test_rsi_threshold_corrects_small_flip() -> None:
    # 3 bins: anchor at 0.80, two middle bins flip ranks between months
    # month 1: 0.1→0.10, 0.5→0.20, 0.9→0.80
    # month 2: 0.1→0.21, 0.5→0.20, 0.9→0.80  (0.1 and 0.5 swap, but leap=0.11 vs span=0.70)
    # all leaps / span < 0.5 → all bins treated as stable → RSI = 1.0
    scores = np.array([0.1, 0.5, 0.9, 0.1, 0.5, 0.9])
    rates = np.array([0.10, 0.20, 0.80, 0.21, 0.20, 0.80])
    months = np.array([1, 1, 1, 2, 2, 2])
    assert _rsi(scores, rates, months, threshold=0.5) == 1.0


def test_rsi_returns_float() -> None:
    scores = np.array([0.3, 0.7])
    rates = np.array([0.1, 0.4])
    months = np.array([1, 1])
    assert isinstance(_rsi(scores, rates, months, threshold=0.1), float)


def test_rsi_zero_span_returns_one() -> None:
    scores = np.array([0.5, 0.5, 0.5])
    rates = np.array([0.2, 0.2, 0.2])
    months = np.array([1, 2, 3])
    assert _rsi(scores, rates, months, threshold=0.1) == 1.0


# --- _encode_cats ---

def test_encode_cats_produces_numeric() -> None:
    x = np.array(["A", "B", "A", "C"])
    encoded, mapping = _encode_cats(x)
    assert encoded.dtype == float
    assert set(mapping.keys()) == {"A", "B", "C"}


def test_encode_cats_null_becomes_nan() -> None:
    x = np.array(["A", None, "B"])
    encoded, _ = _encode_cats(x)
    assert np.isnan(encoded[1])


def test_encode_cats_uses_provided_mapping() -> None:
    x = np.array(["A", "B"])
    _, mapping = _encode_cats(x)
    x2 = np.array(["B", "A", "Z"])
    enc2, _ = _encode_cats(x2, mapping)
    assert np.isnan(enc2[2])
    assert enc2[0] == mapping["B"]


# --- WOETransformer ---

def test_woe_transformer_numeric(woe_num_spec: dict) -> None:
    X = pl.DataFrame({"feat": [-1.0, 1.0, None]})
    t = WOETransformer(bins_data={"feat": woe_num_spec}).fit(X)
    result = t.transform(X)
    assert result["feat"][0] == pytest.approx(-0.5)
    assert result["feat"][1] == pytest.approx(0.5)
    assert result["feat"][2] == pytest.approx(0.0)


def test_woe_transformer_categorical(woe_cat_spec: dict) -> None:
    X = pl.DataFrame({"feat": ["A", "B", "Z"]})
    t = WOETransformer(bins_data={"feat": woe_cat_spec}).fit(X)
    result = t.transform(X)
    assert result["feat"][0] == pytest.approx(-0.3)
    assert result["feat"][1] == pytest.approx(0.3)
    assert result["feat"][2] == pytest.approx(0.0)


def test_woe_transformer_missing_column_skipped(woe_num_spec: dict) -> None:
    X = pl.DataFrame({"other": [1.0, 2.0]})
    t = WOETransformer(bins_data={"feat": woe_num_spec}).fit(X)
    result = t.transform(X)
    assert "feat" not in result.columns


def test_woe_transformer_not_fitted_raises() -> None:
    with pytest.raises(Exception):
        WOETransformer(bins_data={}).transform(pl.DataFrame({"x": [1.0]}))


def test_woe_transformer_no_bins_data_raises() -> None:
    with pytest.raises(ValueError):
        WOETransformer().fit(pl.DataFrame({"x": [1.0]}))


def test_woe_transformer_output_shape(woe_num_spec: dict, woe_cat_spec: dict) -> None:
    X = pl.DataFrame({"num": [0.5, -0.5], "cat": ["A", "B"]})
    t = WOETransformer(bins_data={"num": woe_num_spec, "cat": woe_cat_spec}).fit(X)
    result = t.transform(X)
    assert result.shape == (2, 2)


# --- StabilityGrouping ---

def test_stability_grouping_fit_returns_self(temporal_df: pl.DataFrame) -> None:
    n = len(temporal_df)
    half = n // 2
    train = temporal_df[:half]
    val = temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    result = sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    assert result is sg


def test_stability_grouping_bins_data_populated(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    assert "feat" in sg.bins_data_ or "feat" in sg.excluded_


def test_stability_grouping_transform_output_type(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    if sg.bins_data_:
        result = sg.transform(train.select("feat"))
        assert isinstance(result, pl.DataFrame)


def test_stability_grouping_transform_row_count(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    if sg.bins_data_:
        result = sg.transform(train.select("feat"))
        assert len(result) == len(train)


def test_stability_grouping_not_fitted_raises() -> None:
    with pytest.raises(Exception):
        StabilityGrouping().ungroupable()


def test_stability_grouping_ungroupable_returns_list(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    assert isinstance(sg.ungroupable(), list)


def test_stability_grouping_categorical(cat_temporal_df: pl.DataFrame) -> None:
    half = len(cat_temporal_df) // 2
    train, val = cat_temporal_df[:half], cat_temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    assert "feat" in sg.bins_data_ or "feat" in sg.excluded_


def test_stability_grouping_must_have_not_excluded(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=3, must_have=["feat"])
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    assert "feat" not in sg.excluded_


def test_stability_grouping_with_weights(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    w_tr = pl.Series(np.ones(len(train)))
    w_va = pl.Series(np.ones(len(val)))
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
        weights_train=w_tr, weights_val=w_va,
    )
    assert isinstance(sg.ungroupable(), list)


def test_stability_grouping_woe_values_finite(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    if sg.bins_data_:
        result = sg.transform(train.select("feat"))
        assert result["feat"].is_finite().all()
