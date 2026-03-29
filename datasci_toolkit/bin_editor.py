from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from datasci_toolkit.grouping import _rsi

_SMOOTH = 0.5


class FeatureDtype(str, Enum):
    NUMERIC = "float"
    CATEGORICAL = "category"


@dataclass(frozen=True)
class BinStats:
    counts: np.ndarray
    event_rates: np.ndarray
    woe: np.ndarray
    iv: float


@dataclass(frozen=True)
class TemporalStats:
    months: list[Any]
    rsi: float
    event_rates: list[list[float | None]]
    pop_shares: list[list[float]]


@dataclass
class FeatureState:
    feature: str
    dtype: FeatureDtype
    n_bins: int
    bins: list[str] | dict[str, int]
    counts: list[float]
    event_rates: list[float | None]
    woe: list[float]
    iv: float
    splits: list[float] | None = None
    groups: dict[int, list[str]] | None = None
    temporal: TemporalStats | None = None


def _bin_stats(y: np.ndarray, w: np.ndarray, assignments: np.ndarray, n_bins: int) -> BinStats:
    total_ev = float((y * w).sum())
    total_nev = float(((1.0 - y) * w).sum())
    yw = y * w
    counts = np.bincount(assignments, weights=w, minlength=n_bins + 1).astype(float)
    events = np.bincount(assignments, weights=yw, minlength=n_bins + 1).astype(float)
    nonevents = counts - events
    event_rates = np.where(counts > 0, events / counts, np.nan)

    events_per_bin, nonevents_per_bin = events[:n_bins], nonevents[:n_bins]
    event_dist = (events_per_bin + _SMOOTH) / (total_ev + _SMOOTH * n_bins)
    nonevent_dist = (nonevents_per_bin + _SMOOTH) / (total_nev + _SMOOTH * n_bins)
    woe_per_bin = np.log(event_dist / nonevent_dist)
    iv = float(((event_dist - nonevent_dist) * woe_per_bin).sum())

    nan_event_dist = (events[n_bins] + _SMOOTH) / (total_ev + _SMOOTH)
    nan_nonevent_dist = (nonevents[n_bins] + _SMOOTH) / (total_nev + _SMOOTH)
    woe_nan = float(np.log(nan_event_dist / nan_nonevent_dist))

    return BinStats(counts=counts, event_rates=event_rates, woe=np.append(woe_per_bin, woe_nan), iv=iv)


def _temporal_stats(
    y: np.ndarray,
    w: np.ndarray,
    assignments: np.ndarray,
    n_bins: int,
    t: np.ndarray,
    threshold: float,
) -> TemporalStats:
    months = np.sort(np.unique(t))
    er_by_bin: list[list[float | None]] = [[] for _ in range(n_bins)]
    ps_by_bin: list[list[float]] = [[] for _ in range(n_bins)]

    for m in months:
        mask = t == m
        stats = _bin_stats(y[mask], w[mask], assignments[mask], n_bins)
        total = float(stats.counts[:n_bins].sum()) or 1.0
        for i in range(n_bins):
            er = stats.event_rates[i]
            er_by_bin[i].append(None if np.isnan(er) else round(float(er), 6))
            ps_by_bin[i].append(round(float(stats.counts[i] / total), 6))

    scores_arr: list[float] = []
    rates_arr: list[float] = []
    months_arr: list[Any] = []
    for m_idx, m in enumerate(months):
        for bin_i in range(n_bins):
            er = er_by_bin[bin_i][m_idx]
            if er is not None:
                scores_arr.append(float(bin_i))
                rates_arr.append(er)
                months_arr.append(m)

    rsi = _rsi(np.array(scores_arr), np.array(rates_arr), np.array(months_arr), threshold) if len(scores_arr) > 1 else 1.0

    return TemporalStats(
        months=months.tolist(),
        rsi=round(rsi, 4),
        event_rates=er_by_bin,
        pop_shares=ps_by_bin,
    )


def _num_assign(x: np.ndarray, splits: list[float]) -> np.ndarray:
    nan_mask = np.isnan(x)
    a = np.digitize(x, splits)
    a[nan_mask] = len(splits) + 1
    return a


def _cat_assign(x: np.ndarray, cat_bins: dict[str, int]) -> np.ndarray:
    n = max(cat_bins.values()) + 1 if cat_bins else 0
    a = np.full(len(x), n, dtype=np.intp)
    for cat, grp in cat_bins.items():
        a[x == cat] = grp
    return a


def _num_labels(splits: list[float]) -> list[str]:
    if not splits:
        return ["-inf to inf", "NaN"]
    split_strs = [f"{v:.4g}" for v in splits]
    return [f"-inf to {split_strs[0]}"] + [f"{split_strs[i]} to {split_strs[i+1]}" for i in range(len(split_strs) - 1)] + [f"{split_strs[-1]} to inf", "NaN"]


def _num_state(feat: str, splits: list[float], x: np.ndarray, y: np.ndarray, w: np.ndarray) -> FeatureState:
    n_bins = len(splits) + 1
    stats = _bin_stats(y, w, _num_assign(x, splits), n_bins)
    return FeatureState(
        feature=feat,
        dtype=FeatureDtype.NUMERIC,
        n_bins=n_bins,
        splits=list(splits),
        bins=_num_labels(splits),
        counts=stats.counts.tolist(),
        event_rates=[None if np.isnan(v) else round(float(v), 6) for v in stats.event_rates],
        woe=[round(float(v), 6) for v in stats.woe],
        iv=round(stats.iv, 6),
    )


def _cat_state(feat: str, cat_bins: dict[str, int], x: np.ndarray, y: np.ndarray, w: np.ndarray) -> FeatureState:
    n_groups = max(cat_bins.values()) + 1 if cat_bins else 0
    stats = _bin_stats(y, w, _cat_assign(x, cat_bins), n_groups)
    groups: dict[int, list[str]] = {}
    for cat, grp in cat_bins.items():
        groups.setdefault(grp, []).append(str(cat))
    return FeatureState(
        feature=feat,
        dtype=FeatureDtype.CATEGORICAL,
        n_bins=n_groups,
        groups={k: sorted(v) for k, v in groups.items()},
        bins=dict(cat_bins),
        counts=stats.counts.tolist(),
        event_rates=[None if np.isnan(v) else round(float(v), 6) for v in stats.event_rates],
        woe=[round(float(v), 6) for v in stats.woe],
        iv=round(stats.iv, 6),
    )


class BinEditor:
    """Headless state machine for editing bin boundaries.

    Works identically in plain Python scripts, notebooks, and agents. All
    edits are logged per feature with undo support. Call `accept()` to export
    the final bin specs dict for use with `WOETransformer`.

    Args:
        bin_specs: Initial bin specifications — a dict produced by
            `StabilityGrouping.bin_specs_` or built manually.
        X: Feature DataFrame matching the features in ``bin_specs``.
        y: Binary target series (0/1 or float).
        t: Optional time series for temporal stability metrics.
        weights: Optional sample weight series.
        stability_threshold: RSI threshold used to flag unstable bins in the
            state dict (does not block edits).

    Note:
        All state is accessible via `state(feat)`, which returns a `FeatureState`
        dataclass with attributes ``bins``, ``n_bins``, ``counts``, ``event_rates``,
        ``woe``, ``iv``, ``dtype``, ``groups``, and ``temporal``.
    """

    def __init__(
        self,
        bin_specs: dict[str, dict[str, Any]],
        X: pl.DataFrame,
        y: pl.Series,
        t: pl.Series | None = None,
        weights: pl.Series | None = None,
        stability_threshold: float = 0.1,
    ) -> None:
        self._y = y.cast(pl.Float64).to_numpy()
        self._w = weights.cast(pl.Float64).to_numpy() if weights is not None else np.ones(len(self._y))
        self._t: np.ndarray | None = t.to_numpy() if t is not None else None
        self._threshold = stability_threshold
        self._x: dict[str, np.ndarray] = {}
        self._splits: dict[str, list[float]] = {}
        self._cat_bins: dict[str, dict[str, int]] = {}
        self._history: dict[str, list[tuple[str, Any]]] = {}
        self._orig: dict[str, dict[str, Any]] = {}

        for feat, spec in bin_specs.items():
            if feat not in X.columns:
                continue
            self._orig[feat] = spec
            self._history[feat] = []
            if spec["dtype"] == FeatureDtype.NUMERIC:
                self._x[feat] = X[feat].cast(pl.Float64).to_numpy()
                self._splits[feat] = [float(s) for s in spec["bins"][1:-1] if np.isfinite(s)]
            else:
                self._x[feat] = X[feat].cast(pl.Utf8).to_numpy().astype(str)
                self._cat_bins[feat] = {str(k): int(v) for k, v in spec["bins"].items()}

    def features(self) -> list[str]:
        return list(self._splits.keys()) + list(self._cat_bins.keys())

    def _base_state(self, feat: str) -> FeatureState:
        if feat in self._splits:
            return _num_state(feat, self._splits[feat], self._x[feat], self._y, self._w)
        return _cat_state(feat, self._cat_bins[feat], self._x[feat], self._y, self._w)

    def _assignments(self, feat: str) -> np.ndarray:
        if feat in self._splits:
            return _num_assign(self._x[feat], self._splits[feat])
        return _cat_assign(self._x[feat], self._cat_bins[feat])

    def state(self, feat: str) -> FeatureState:
        s = self._base_state(feat)
        if self._t is not None:
            s.temporal = _temporal_stats(
                self._y, self._w, self._assignments(feat), s.n_bins, self._t, self._threshold
            )
        return s

    def _push(self, feat: str) -> None:
        if feat in self._splits:
            self._history[feat].append(("splits", list(self._splits[feat])))
        else:
            self._history[feat].append(("cat", copy.deepcopy(self._cat_bins[feat])))

    def split(self, feat: str, value: float) -> FeatureState:
        if value in self._splits[feat]:
            return self.state(feat)
        self._push(feat)
        self._splits[feat] = sorted(self._splits[feat] + [value])
        return self.state(feat)

    def merge(self, feat: str, bin_idx: int) -> FeatureState:
        if feat in self._splits:
            splits = self._splits[feat]
            if bin_idx >= len(splits):
                return self.state(feat)
            self._push(feat)
            self._splits[feat] = [s for i, s in enumerate(splits) if i != bin_idx]
        else:
            cat_bins = self._cat_bins[feat]
            n_groups = max(cat_bins.values()) + 1 if cat_bins else 0
            if bin_idx >= n_groups - 1:
                return self.state(feat)
            self._push(feat)
            self._cat_bins[feat] = {
                cat: (bin_idx if grp == bin_idx + 1 else (grp - 1 if grp > bin_idx + 1 else grp))
                for cat, grp in cat_bins.items()
            }
        return self.state(feat)

    def move_boundary(self, feat: str, bin_idx: int, new_value: float) -> FeatureState:
        splits = self._splits[feat]
        if bin_idx >= len(splits):
            return self.state(feat)
        self._push(feat)
        new = list(splits)
        new[bin_idx] = new_value
        self._splits[feat] = sorted(set(new))
        return self.state(feat)

    def reset(self, feat: str) -> FeatureState:
        self._history[feat] = []
        spec = self._orig[feat]
        if spec["dtype"] == FeatureDtype.NUMERIC:
            self._splits[feat] = [float(s) for s in spec["bins"][1:-1] if np.isfinite(s)]
        else:
            self._cat_bins[feat] = {str(k): int(v) for k, v in spec["bins"].items()}
        return self.state(feat)

    def undo(self, feat: str) -> FeatureState:
        if not self._history[feat]:
            return self.state(feat)
        kind, prev = self._history[feat].pop()
        if kind == "splits":
            self._splits[feat] = prev
        else:
            self._cat_bins[feat] = prev
        return self.state(feat)

    def history(self, feat: str) -> list[dict[str, Any]]:
        return [{"type": k, "value": v} for k, v in self._history[feat]]

    def _suggest_num(self, feat: str, n: int) -> list[float]:
        x = self._x[feat]
        x_valid = x[~np.isnan(x)]
        if len(x_valid) == 0:
            return []
        current = self._splits[feat]
        span = float(x_valid.max() - x_valid.min())
        min_gap = span * 0.01
        candidates = [
            float(c) for c in np.unique(np.percentile(x_valid, np.linspace(5, 95, 40)))
            if all(abs(c - s) > min_gap for s in current)
        ]
        base_iv = self._base_state(feat).iv
        pairs: list[tuple[float, float]] = sorted(
            [
                (
                    _bin_stats(self._y, self._w, _num_assign(x, sorted(current + [c])), len(current) + 2).iv - base_iv,
                    float(c),
                )
                for c in candidates
            ],
            reverse=True,
        )
        return [v for _, v in pairs[:n]]

    def _suggest_cat(self, feat: str, n: int) -> list[tuple[int, int]]:
        cat_bins = self._cat_bins[feat]
        n_groups = max(cat_bins.values()) + 1 if cat_bins else 0
        if n_groups <= 1:
            return []
        x = self._x[feat]
        base_iv = self._base_state(feat).iv
        pairs: list[tuple[float, tuple[int, int]]] = sorted(
            [
                (
                    base_iv - _bin_stats(
                        self._y, self._w,
                        _cat_assign(x, {
                            cat: (bin_idx if grp == bin_idx + 1 else (grp - 1 if grp > bin_idx + 1 else grp))
                            for cat, grp in cat_bins.items()
                        }),
                        n_groups - 1,
                    ).iv,
                    (bin_idx, bin_idx + 1),
                )
                for bin_idx in range(n_groups - 1)
            ]
        )
        return [pair for _, pair in pairs[:n]]

    def suggest_splits(self, feat: str, n: int = 5) -> list:  # type: ignore[type-arg]
        if feat in self._splits:
            return self._suggest_num(feat, n)
        return self._suggest_cat(feat, n)

    def accept(self) -> dict[str, dict[str, Any]]:
        return {feat: self.accept_feature(feat) for feat in self.features()}

    def accept_feature(self, feat: str) -> dict[str, Any]:
        if feat in self._splits:
            return {"dtype": "float", "bins": [-np.inf] + self._splits[feat] + [np.inf]}
        return {"dtype": "category", "bins": dict(self._cat_bins[feat])}
