import numpy as np
import polars as pl
import pytest

from datasci_toolkit.bin_editor import BinEditor, _bin_stats, _num_assign, _cat_assign

RNG = np.random.default_rng(0)

N = 500
_X_NP = RNG.normal(0, 1, N)
_Y_NP = (_X_NP > 0).astype(float)
_CATS = np.array(["A", "B", "C"] * (N // 3) + ["A", "B"])
_Y_CAT = np.array([1.0 if c == "A" else 0.0 for c in _CATS])

_BIN_SPECS: dict = {
    "num": {"dtype": "float", "bins": [-np.inf, -0.5, 0.5, np.inf]},
    "cat": {"dtype": "category", "bins": {"A": 0, "B": 1, "C": 2}},
}


@pytest.fixture
def editor() -> BinEditor:
    X = pl.DataFrame({"num": _X_NP.tolist(), "cat": _CATS.tolist()})
    y = pl.Series(_Y_NP.tolist())
    return BinEditor(_BIN_SPECS, X, y)


_N_MONTHS = 4
_T_NP = np.repeat(np.arange(_N_MONTHS), N // _N_MONTHS + 1)[:N]


@pytest.fixture
def editor_temporal() -> BinEditor:
    X = pl.DataFrame({"num": _X_NP.tolist(), "cat": _CATS.tolist()})
    y = pl.Series(_Y_NP.tolist())
    t = pl.Series(_T_NP.tolist())
    return BinEditor(_BIN_SPECS, X, y, t=t)


# --- _bin_stats ---

def test_bin_stats_counts_sum_to_total() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    w = np.ones(4)
    a = np.array([0, 0, 1, 2])
    s = _bin_stats(y, w, a, 3)
    assert s["counts"][:3].sum() == pytest.approx(4.0)


def test_bin_stats_event_rates_in_range() -> None:
    y = RNG.integers(0, 2, 100).astype(float)
    w = np.ones(100)
    a = RNG.integers(0, 4, 100)
    s = _bin_stats(y, w, a, 3)
    rates = s["event_rates"][:3]
    assert np.all((rates >= 0) | np.isnan(rates))
    assert np.all((rates <= 1) | np.isnan(rates))


def test_bin_stats_woe_finite() -> None:
    y = RNG.integers(0, 2, 200).astype(float)
    w = np.ones(200)
    a = RNG.integers(0, 5, 200)
    s = _bin_stats(y, w, a, 4)
    assert np.all(np.isfinite(s["woe"]))


def test_bin_stats_iv_nonneg() -> None:
    y = RNG.integers(0, 2, 200).astype(float)
    w = np.ones(200)
    a = RNG.integers(0, 4, 200)
    s = _bin_stats(y, w, a, 3)
    assert s["iv"] >= 0.0


# --- _num_assign ---

def test_num_assign_no_splits_all_zero() -> None:
    x = np.array([1.0, -1.0, 0.5])
    a = _num_assign(x, [])
    assert (a == 0).all()


def test_num_assign_nan_goes_to_nan_bin() -> None:
    x = np.array([1.0, np.nan, -1.0])
    a = _num_assign(x, [0.0])
    assert a[1] == 2


def test_num_assign_correct_bins() -> None:
    x = np.array([-1.0, 0.5, 2.0])
    a = _num_assign(x, [0.0, 1.0])
    assert a[0] == 0
    assert a[1] == 1
    assert a[2] == 2


# --- _cat_assign ---

def test_cat_assign_known_cats() -> None:
    x = np.array(["A", "B", "C"])
    a = _cat_assign(x, {"A": 0, "B": 1, "C": 1})
    assert a[0] == 0
    assert a[1] == 1
    assert a[2] == 1


def test_cat_assign_unknown_goes_to_last_bin() -> None:
    x = np.array(["A", "Z"])
    a = _cat_assign(x, {"A": 0})
    assert a[1] == 1


# --- BinEditor ---

def test_editor_features(editor: BinEditor) -> None:
    assert set(editor.features()) == {"num", "cat"}


def test_editor_state_num_keys(editor: BinEditor) -> None:
    s = editor.state("num")
    assert {"feature", "dtype", "splits", "bins", "counts", "event_rates", "woe", "iv", "n_bins"} <= s.keys()


def test_editor_state_cat_keys(editor: BinEditor) -> None:
    s = editor.state("cat")
    assert {"feature", "dtype", "groups", "bins", "counts", "event_rates", "woe", "iv", "n_bins"} <= s.keys()


def test_editor_state_num_dtype(editor: BinEditor) -> None:
    assert editor.state("num")["dtype"] == "float"


def test_editor_state_cat_dtype(editor: BinEditor) -> None:
    assert editor.state("cat")["dtype"] == "category"


def test_editor_state_counts_sum(editor: BinEditor) -> None:
    s = editor.state("num")
    data_counts = sum(s["counts"][:-1])
    assert data_counts == pytest.approx(N, abs=1)


def test_editor_state_event_rates_range(editor: BinEditor) -> None:
    s = editor.state("num")
    for er in s["event_rates"][:-1]:
        if er is not None:
            assert 0.0 <= er <= 1.0


def test_editor_state_iv_nonneg(editor: BinEditor) -> None:
    assert editor.state("num")["iv"] >= 0.0


# --- split ---

def test_split_increases_n_bins(editor: BinEditor) -> None:
    before = editor.state("num")["n_bins"]
    editor.split("num", 0.0)
    after = editor.state("num")["n_bins"]
    assert after == before + 1


def test_split_sorted(editor: BinEditor) -> None:
    editor.split("num", 1.0)
    editor.split("num", -1.0)
    splits = editor.state("num")["splits"]
    assert splits == sorted(splits)


def test_split_idempotent(editor: BinEditor) -> None:
    editor.split("num", 0.5)
    n1 = editor.state("num")["n_bins"]
    editor.split("num", 0.5)
    n2 = editor.state("num")["n_bins"]
    assert n1 == n2


def test_split_returns_state(editor: BinEditor) -> None:
    s = editor.split("num", 0.1)
    assert s["dtype"] == "float"


# --- merge ---

def test_merge_num_decreases_n_bins(editor: BinEditor) -> None:
    before = editor.state("num")["n_bins"]
    editor.merge("num", 0)
    after = editor.state("num")["n_bins"]
    assert after == before - 1


def test_merge_cat_decreases_n_bins(editor: BinEditor) -> None:
    before = editor.state("cat")["n_bins"]
    editor.merge("cat", 0)
    after = editor.state("cat")["n_bins"]
    assert after == before - 1


def test_merge_out_of_range_noop(editor: BinEditor) -> None:
    n_before = editor.state("num")["n_bins"]
    editor.merge("num", 999)
    assert editor.state("num")["n_bins"] == n_before


def test_merge_cat_out_of_range_noop(editor: BinEditor) -> None:
    n_before = editor.state("cat")["n_bins"]
    editor.merge("cat", 999)
    assert editor.state("cat")["n_bins"] == n_before


def test_merge_cat_groups_renumbered(editor: BinEditor) -> None:
    editor.merge("cat", 0)
    groups = editor.state("cat")["groups"]
    assert set(groups.keys()) == {0, 1}


# --- move_boundary ---

def test_move_boundary_changes_split(editor: BinEditor) -> None:
    editor.move_boundary("num", 0, -0.3)
    splits = editor.state("num")["splits"]
    assert -0.3 in splits


def test_move_boundary_out_of_range_noop(editor: BinEditor) -> None:
    splits_before = editor.state("num")["splits"]
    editor.move_boundary("num", 999, 1.0)
    assert editor.state("num")["splits"] == splits_before


# --- undo ---

def test_undo_reverts_split(editor: BinEditor) -> None:
    splits_before = editor.state("num")["splits"]
    editor.split("num", 0.0)
    editor.undo("num")
    assert editor.state("num")["splits"] == splits_before


def test_undo_empty_noop(editor: BinEditor) -> None:
    s_before = editor.state("num")
    editor.undo("num")
    assert editor.state("num")["splits"] == s_before["splits"]


def test_undo_reverts_cat_merge(editor: BinEditor) -> None:
    n_before = editor.state("cat")["n_bins"]
    editor.merge("cat", 0)
    editor.undo("cat")
    assert editor.state("cat")["n_bins"] == n_before


# --- history ---

def test_history_grows_with_ops(editor: BinEditor) -> None:
    editor.split("num", 0.1)
    editor.split("num", 0.2)
    editor.merge("num", 0)
    assert len(editor.history("num")) == 3


def test_history_shrinks_on_undo(editor: BinEditor) -> None:
    editor.split("num", 0.1)
    editor.split("num", 0.2)
    editor.undo("num")
    assert len(editor.history("num")) == 1


def test_history_cleared_on_reset(editor: BinEditor) -> None:
    editor.split("num", 0.1)
    editor.reset("num")
    assert editor.history("num") == []


# --- reset ---

def test_reset_restores_original_splits(editor: BinEditor) -> None:
    original_splits = editor.state("num")["splits"]
    editor.split("num", 0.0)
    editor.merge("num", 0)
    editor.reset("num")
    assert editor.state("num")["splits"] == original_splits


def test_reset_cat_restores_original(editor: BinEditor) -> None:
    original_bins = editor.state("cat")["n_bins"]
    editor.merge("cat", 0)
    editor.reset("cat")
    assert editor.state("cat")["n_bins"] == original_bins


# --- suggest_splits ---

def test_suggest_splits_num_returns_floats(editor: BinEditor) -> None:
    suggestions = editor.suggest_splits("num", n=3)
    assert isinstance(suggestions, list)
    assert all(isinstance(s, float) for s in suggestions)


def test_suggest_splits_num_count(editor: BinEditor) -> None:
    suggestions = editor.suggest_splits("num", n=4)
    assert len(suggestions) <= 4


def test_suggest_splits_num_positive_iv_gain(editor: BinEditor) -> None:
    editor_no_splits = BinEditor(
        {"num": {"dtype": "float", "bins": [-np.inf, np.inf]}},
        pl.DataFrame({"num": _X_NP.tolist()}),
        pl.Series(_Y_NP.tolist()),
    )
    base_iv = editor_no_splits.state("num")["iv"]
    suggestions = editor_no_splits.suggest_splits("num", n=1)
    if suggestions:
        after_iv = editor_no_splits.split("num", suggestions[0])["iv"]
        assert after_iv >= base_iv


def test_suggest_splits_cat_returns_pairs(editor: BinEditor) -> None:
    suggestions = editor.suggest_splits("cat", n=3)
    assert isinstance(suggestions, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in suggestions)


def test_suggest_splits_cat_count(editor: BinEditor) -> None:
    suggestions = editor.suggest_splits("cat", n=2)
    assert len(suggestions) <= 2


# --- accept ---

def test_accept_returns_all_features(editor: BinEditor) -> None:
    result = editor.accept()
    assert set(result.keys()) == {"num", "cat"}


def test_accept_feature_num_format(editor: BinEditor) -> None:
    spec = editor.accept_feature("num")
    assert spec["dtype"] == "float"
    assert spec["bins"][0] == -np.inf
    assert spec["bins"][-1] == np.inf


def test_accept_feature_cat_format(editor: BinEditor) -> None:
    spec = editor.accept_feature("cat")
    assert spec["dtype"] == "category"
    assert isinstance(spec["bins"], dict)


def test_accept_after_split_has_extra_boundary(editor: BinEditor) -> None:
    editor.split("num", 0.0)
    spec = editor.accept_feature("num")
    original_spec = _BIN_SPECS["num"]
    assert len(spec["bins"]) == len(original_spec["bins"]) + 1


def test_accept_after_merge_has_fewer_categories(editor: BinEditor) -> None:
    editor.merge("cat", 0)
    spec = editor.accept_feature("cat")
    n_groups_after = max(spec["bins"].values()) + 1
    assert n_groups_after == 2


def test_accept_compatible_with_woe_transformer(editor: BinEditor) -> None:
    from datasci_toolkit.grouping import WOETransformer
    bin_specs = editor.accept()
    X = pl.DataFrame({"num": _X_NP.tolist(), "cat": _CATS.tolist()})
    y = pl.Series(_Y_NP.tolist())
    t = WOETransformer(bin_specs=bin_specs).fit(X, y)
    result = t.transform(X)
    assert result.shape[0] == N


# --- temporal ---

def test_temporal_state_has_temporal_key(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    assert "temporal" in s


def test_no_temporal_key_without_t(editor: BinEditor) -> None:
    s = editor.state("num")
    assert "temporal" not in s


def test_temporal_months_length(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    assert len(s["temporal"]["months"]) == _N_MONTHS


def test_temporal_event_rates_structure(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    n = s["n_bins"]
    er = s["temporal"]["event_rates"]
    assert len(er) == n
    assert all(len(row) == _N_MONTHS for row in er)


def test_temporal_pop_shares_structure(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    n = s["n_bins"]
    ps = s["temporal"]["pop_shares"]
    assert len(ps) == n
    assert all(len(row) == _N_MONTHS for row in ps)


def test_temporal_pop_shares_sum_to_one(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    ps = s["temporal"]["pop_shares"]
    n_months = _N_MONTHS
    for m_idx in range(n_months):
        total = sum(ps[b][m_idx] for b in range(s["n_bins"]))
        assert total == pytest.approx(1.0, abs=1e-4)


def test_temporal_rsi_in_range(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    rsi = s["temporal"]["rsi"]
    assert 0.0 <= rsi <= 1.0


def test_temporal_rsi_float(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("num")
    assert isinstance(s["temporal"]["rsi"], float)


def test_temporal_cat_feature(editor_temporal: BinEditor) -> None:
    s = editor_temporal.state("cat")
    assert "temporal" in s
    assert len(s["temporal"]["event_rates"]) == s["n_bins"]


def test_temporal_persists_after_split(editor_temporal: BinEditor) -> None:
    editor_temporal.split("num", 0.0)
    s = editor_temporal.state("num")
    assert "temporal" in s
    assert s["n_bins"] > 0
