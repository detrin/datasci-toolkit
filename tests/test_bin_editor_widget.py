import base64
import struct

import numpy as np
import polars as pl
import pytest

from datasci_toolkit.bin_editor import BinEditor
from datasci_toolkit.bin_editor_widget import BinEditorWidget

RNG = np.random.default_rng(0)

N = 500
_X_NP = RNG.normal(0, 1, N)
_Y_NP = (_X_NP > 0).astype(float)
_CATS = np.array(["A", "B", "C"] * (N // 3) + ["A", "B"])
_Y_CAT = np.array([1.0 if c == "A" else 0.0 for c in _CATS])
_N_MONTHS = 4
_T_NP = np.repeat(np.arange(_N_MONTHS), N // _N_MONTHS + 1)[:N]

_BIN_SPECS: dict = {
    "num": {"dtype": "float", "bins": [-np.inf, -0.5, 0.5, np.inf]},
    "cat": {"dtype": "category", "bins": {"A": 0, "B": 1, "C": 2}},
}


def _make_editor(temporal: bool = False) -> BinEditor:
    X = pl.DataFrame({"num": _X_NP.tolist(), "cat": _CATS.tolist()})
    y = pl.Series(_Y_NP.tolist())
    t = pl.Series(_T_NP.tolist()) if temporal else None
    return BinEditor(_BIN_SPECS, X, y, t=t)


def _send(widget: BinEditorWidget, content: dict) -> None:
    widget._handle_msg(None, content, None)


def _is_valid_png(b64: str) -> bool:
    if not b64.startswith("data:image/png;base64,"):
        return False
    raw = base64.b64decode(b64.split(",", 1)[1])
    return raw[:8] == b"\x89PNG\r\n\x1a\n"


# --- initialisation ---

def test_widget_features_match_editor() -> None:
    w = BinEditorWidget(_make_editor())
    assert set(w.features) == {"num", "cat"}


def test_widget_initial_feat_is_first() -> None:
    ed = _make_editor()
    w = BinEditorWidget(ed)
    assert w.current_feat == ed.features()[0]


def test_widget_initial_chart_is_valid_png() -> None:
    w = BinEditorWidget(_make_editor())
    assert _is_valid_png(w.chart_png)


def test_widget_initial_stability_empty_without_temporal() -> None:
    w = BinEditorWidget(_make_editor(temporal=False))
    assert w.stability_png == ""


def test_widget_initial_stability_valid_png_with_temporal() -> None:
    w = BinEditorWidget(_make_editor(temporal=True))
    assert _is_valid_png(w.stability_png)


def test_widget_initial_suggestions_empty() -> None:
    w = BinEditorWidget(_make_editor())
    assert w.suggestions == []


def test_widget_initial_message_empty() -> None:
    w = BinEditorWidget(_make_editor())
    assert w.message == ""


# --- set_feature ---

def test_set_feature_updates_current_feat() -> None:
    w = BinEditorWidget(_make_editor())
    other = [f for f in w.features if f != w.current_feat][0]
    _send(w, {"action": "set_feature", "feature": other})
    assert w.current_feat == other


def test_set_feature_refreshes_chart() -> None:
    w = BinEditorWidget(_make_editor())
    old_png = w.chart_png
    other = [f for f in w.features if f != w.current_feat][0]
    _send(w, {"action": "set_feature", "feature": other})
    assert _is_valid_png(w.chart_png)
    assert w.chart_png != old_png


def test_set_feature_clears_suggestions() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "suggest"})
    assert len(w.suggestions) > 0
    _send(w, {"action": "set_feature", "feature": w.current_feat})
    assert w.suggestions == []


# --- split ---

def test_split_adds_merge_label() -> None:
    w = BinEditorWidget(_make_editor())
    w.current_feat = "num"
    _send(w, {"action": "set_feature", "feature": "num"})
    before = len(w.merge_labels)
    _send(w, {"action": "split", "value": 1.0})
    assert len(w.merge_labels) == before + 1


def test_split_updates_chart_png() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    old = w.chart_png
    _send(w, {"action": "split", "value": 1.0})
    assert _is_valid_png(w.chart_png)
    assert w.chart_png != old


def test_split_clears_suggestions() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    _send(w, {"action": "suggest"})
    _send(w, {"action": "split", "value": 1.0})
    assert w.suggestions == []


# --- merge ---

def test_merge_removes_merge_label() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    before = len(w.merge_labels)
    _send(w, {"action": "merge", "bin_idx": 0})
    assert len(w.merge_labels) == before - 1


def test_merge_updates_chart() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    old = w.chart_png
    _send(w, {"action": "merge", "bin_idx": 0})
    assert _is_valid_png(w.chart_png)
    assert w.chart_png != old


def test_merge_cat_reduces_groups() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "cat"})
    before = len(w.merge_labels)
    _send(w, {"action": "merge", "bin_idx": 0})
    assert len(w.merge_labels) == before - 1


# --- undo ---

def test_undo_reverts_split() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    before_labels = list(w.merge_labels)
    _send(w, {"action": "split", "value": 1.0})
    _send(w, {"action": "undo"})
    assert w.merge_labels == before_labels


def test_undo_with_no_history_is_noop() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    labels_before = list(w.merge_labels)
    _send(w, {"action": "undo"})
    assert w.merge_labels == labels_before


# --- reset ---

def test_reset_returns_to_initial_state() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    initial_labels = list(w.merge_labels)
    _send(w, {"action": "split", "value": 1.0})
    _send(w, {"action": "split", "value": 2.0})
    _send(w, {"action": "reset"})
    assert w.merge_labels == initial_labels


# --- suggest ---

def test_suggest_num_populates_float_suggestions() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    _send(w, {"action": "suggest"})
    assert len(w.suggestions) > 0
    assert all(isinstance(s, float) for s in w.suggestions)


def test_suggest_cat_populates_pair_suggestions() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "cat"})
    _send(w, {"action": "suggest"})
    assert len(w.suggestions) > 0
    assert all(isinstance(s, list) and len(s) == 2 for s in w.suggestions)


# --- accept ---

def test_accept_sets_result() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "accept"})
    assert hasattr(w, "result_")
    assert set(w.result_.keys()) == set(w.features)


def test_accept_sets_message() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "accept"})
    assert "Accepted" in w.message


# --- merge_labels content ---

def test_merge_labels_float_are_split_values() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    ed_state = _make_editor().state("num")
    assert len(w.merge_labels) == len(ed_state.splits)


def test_merge_labels_cat_contain_category_names() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "cat"})
    combined = " ".join(w.merge_labels)
    assert any(c in combined for c in ["A", "B", "C"])


# --- dtype trait ---

def test_feat_dtype_float() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "num"})
    assert w.feat_dtype == "float"


def test_feat_dtype_category() -> None:
    w = BinEditorWidget(_make_editor())
    _send(w, {"action": "set_feature", "feature": "cat"})
    assert w.feat_dtype == "category"
