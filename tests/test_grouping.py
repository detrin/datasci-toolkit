import numpy as np
import polars as pl
import pytest

from sklearn.metrics import roc_auc_score

from datasci_toolkit.grouping import (
    WOETransformer,
    StabilityGrouping,
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
def lgbm_num_bst(binary_data: dict[str, np.ndarray]) -> object:
    import lightgbm as lgb
    x, y, w = binary_data["x"], binary_data["y"], binary_data["w"]
    from datasci_toolkit.grouping import _LGBM_PARAMS, _train_lgbm
    params = {**_LGBM_PARAMS, "num_leaves": 3}
    return _train_lgbm(params, x, y, w, x, y, w, False)


@pytest.fixture
def lgbm_cat_bst(binary_data: dict[str, np.ndarray]) -> tuple[object, np.ndarray, np.ndarray]:
    from datasci_toolkit.grouping import _LGBM_PARAMS, _encode_cats, _train_lgbm
    cats = np.array(["A", "B", "C"] * (400 // 3) + ["A"])
    y = np.array([1.0 if c == "A" else 0.0 for c in cats])
    w = np.ones(len(cats))
    train_enc = _encode_cats(cats)
    enc, mapping = train_enc.values, train_enc.category_map
    params = {**_LGBM_PARAMS, "num_leaves": 2}
    bst = _train_lgbm(params, enc, y, w, enc, y, w, True)
    return bst, enc, cats


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
    result = _encode_cats(x)
    assert result.values.dtype == float
    assert set(result.category_map.keys()) == {"A", "B", "C"}


def test_encode_cats_null_becomes_nan() -> None:
    x = np.array(["A", None, "B"])
    assert np.isnan(_encode_cats(x).values[1])


def test_encode_cats_uses_provided_mapping() -> None:
    x = np.array(["A", "B"])
    mapping = _encode_cats(x).category_map
    x2 = np.array(["B", "A", "Z"])
    enc2 = _encode_cats(x2, mapping).values
    assert np.isnan(enc2[2])
    assert enc2[0] == mapping["B"]


# --- _num_bin_spec ---

def test_num_bin_spec_returns_float_dtype(lgbm_num_bst: object, binary_data: dict[str, np.ndarray]) -> None:
    spec = _num_bin_spec(lgbm_num_bst, binary_data["x"])
    assert spec["dtype"] == "float"


def test_num_bin_spec_bins_are_sorted(lgbm_num_bst: object, binary_data: dict[str, np.ndarray]) -> None:
    spec = _num_bin_spec(lgbm_num_bst, binary_data["x"])
    bins = spec["bins"]
    assert bins == sorted(bins)


def test_num_bin_spec_bins_bounded(lgbm_num_bst: object, binary_data: dict[str, np.ndarray]) -> None:
    spec = _num_bin_spec(lgbm_num_bst, binary_data["x"])
    assert spec["bins"][0] == -np.inf
    assert spec["bins"][-1] == np.inf


# --- _cat_bin_spec ---

def test_cat_bin_spec_returns_category_dtype(lgbm_cat_bst: tuple[object, np.ndarray, np.ndarray]) -> None:
    bst, enc, orig = lgbm_cat_bst
    spec = _cat_bin_spec(bst, enc, orig)
    assert spec["dtype"] == "category"


def test_cat_bin_spec_bins_is_dict(lgbm_cat_bst: tuple[object, np.ndarray, np.ndarray]) -> None:
    bst, enc, orig = lgbm_cat_bst
    spec = _cat_bin_spec(bst, enc, orig)
    assert isinstance(spec["bins"], dict)


def test_cat_bin_spec_all_cats_mapped(lgbm_cat_bst: tuple[object, np.ndarray, np.ndarray]) -> None:
    bst, enc, orig = lgbm_cat_bst
    spec = _cat_bin_spec(bst, enc, orig)
    assert set(spec["bins"].keys()) == {"A", "B", "C"}


# --- WOETransformer ---

def test_woe_transformer_numeric_fit_transform() -> None:
    X = pl.DataFrame({"feat": [-1.0, 0.5, 1.5, -0.5]})
    y = pl.Series([0.0, 1.0, 1.0, 0.0])
    spec = {"feat": {"dtype": "float", "bins": [-np.inf, 0.0, np.inf]}}
    t = WOETransformer(bin_specs=spec).fit(X, y)
    result = t.transform(X)
    assert result.shape == (4, 1)
    assert result["feat"].dtype == pl.Float64


def test_woe_transformer_categorical_fit_transform() -> None:
    X = pl.DataFrame({"feat": ["A", "B", "A", "B"]})
    y = pl.Series([1.0, 0.0, 1.0, 0.0])
    spec = {"feat": {"dtype": "category", "bins": {"A": 0, "B": 1}}}
    t = WOETransformer(bin_specs=spec).fit(X, y)
    result = t.transform(X)
    assert result.shape == (4, 1)


def test_woe_transformer_missing_column_skipped() -> None:
    X = pl.DataFrame({"other": [1.0, 2.0]})
    y = pl.Series([0.0, 1.0])
    spec = {"feat": {"dtype": "float", "bins": [-np.inf, 0.0, np.inf]}}
    t = WOETransformer(bin_specs=spec).fit(X, y)
    result = t.transform(X)
    assert "feat" not in result.columns


def test_woe_transformer_not_fitted_raises() -> None:
    with pytest.raises(Exception):
        WOETransformer(bin_specs={}).transform(pl.DataFrame({"x": [1.0]}))


def test_woe_transformer_no_bin_specs_raises() -> None:
    with pytest.raises(ValueError):
        WOETransformer().fit(pl.DataFrame({"x": [1.0]}), pl.Series([0.0]))


def test_woe_transformer_output_shape() -> None:
    X = pl.DataFrame({"a": [0.5, -0.5], "b": [1.5, -1.5]})
    y = pl.Series([1.0, 0.0])
    spec = {
        "a": {"dtype": "float", "bins": [-np.inf, 0.0, np.inf]},
        "b": {"dtype": "float", "bins": [-np.inf, 0.0, np.inf]},
    }
    t = WOETransformer(bin_specs=spec).fit(X, y)
    result = t.transform(X)
    assert result.shape == (2, 2)


def test_woe_transformer_with_weights() -> None:
    X = pl.DataFrame({"feat": [-1.0, 0.5, 1.5, -0.5]})
    y = pl.Series([0.0, 1.0, 1.0, 0.0])
    w = pl.Series([1.0, 1.0, 1.0, 1.0])
    spec = {"feat": {"dtype": "float", "bins": [-np.inf, 0.0, np.inf]}}
    t = WOETransformer(bin_specs=spec).fit(X, y, w)
    result = t.transform(X)
    assert result.shape == (4, 1)


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
    assert "feat" in sg.bin_specs_ or "feat" in sg.excluded_


def test_stability_grouping_transform_output_type(temporal_df: pl.DataFrame) -> None:
    half = len(temporal_df) // 2
    train, val = temporal_df[:half], temporal_df[half:]
    sg = StabilityGrouping(max_bins=4)
    sg.fit(
        train.select("feat"), train["target"], train["month"],
        val.select("feat"), val["target"], val["month"],
    )
    if sg.bin_specs_:
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
    if sg.bin_specs_:
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
    assert "feat" in sg.bin_specs_ or "feat" in sg.excluded_


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
    if sg.bin_specs_:
        result = sg.transform(train.select("feat"))
        assert result["feat"].is_finite().all()
