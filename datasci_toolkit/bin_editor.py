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


def _bin_stats(target: np.ndarray, weights: np.ndarray, assignments: np.ndarray, n_bins: int) -> BinStats:
    total_ev = float((target * weights).sum())
    total_nev = float(((1.0 - target) * weights).sum())
    yw = target * weights
    counts = np.bincount(assignments, weights=weights, minlength=n_bins + 1).astype(float)
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
    target: np.ndarray,
    weights: np.ndarray,
    assignments: np.ndarray,
    n_bins: int,
    time_periods: np.ndarray,
    threshold: float,
) -> TemporalStats:
    months = np.sort(np.unique(time_periods))
    event_rate_by_bin: list[list[float | None]] = [[] for _ in range(n_bins)]
    pop_share_by_bin: list[list[float]] = [[] for _ in range(n_bins)]

    for month in months:
        mask = time_periods == month
        stats = _bin_stats(target[mask], weights[mask], assignments[mask], n_bins)
        total = float(stats.counts[:n_bins].sum()) or 1.0
        for bin_index in range(n_bins):
            event_rate = stats.event_rates[bin_index]
            event_rate_by_bin[bin_index].append(None if np.isnan(event_rate) else round(float(event_rate), 6))
            pop_share_by_bin[bin_index].append(round(float(stats.counts[bin_index] / total), 6))

    scores_array: list[float] = []
    rates_array: list[float] = []
    months_array: list[Any] = []
    for month_index, month in enumerate(months):
        for bin_index in range(n_bins):
            event_rate = event_rate_by_bin[bin_index][month_index]
            if event_rate is not None:
                scores_array.append(float(bin_index))
                rates_array.append(event_rate)
                months_array.append(month)

    rsi = _rsi(np.array(scores_array), np.array(rates_array), np.array(months_array), threshold) if len(scores_array) > 1 else 1.0

    return TemporalStats(
        months=months.tolist(),
        rsi=round(rsi, 4),
        event_rates=event_rate_by_bin,
        pop_shares=pop_share_by_bin,
    )


def _num_assign(values: np.ndarray, splits: list[float]) -> np.ndarray:
    missing_mask = np.isnan(values)
    assignments = np.digitize(values, splits)
    assignments[missing_mask] = len(splits) + 1
    return assignments


def _cat_assign(values: np.ndarray, category_bins: dict[str, int]) -> np.ndarray:
    n_groups = max(category_bins.values()) + 1 if category_bins else 0
    assignments = np.full(len(values), n_groups, dtype=np.intp)
    for category, group in category_bins.items():
        assignments[values == category] = group
    return assignments


def _num_labels(splits: list[float]) -> list[str]:
    if not splits:
        return ["-inf to inf", "NaN"]
    split_strs = [f"{v:.4g}" for v in splits]
    return [f"-inf to {split_strs[0]}"] + [f"{split_strs[i]} to {split_strs[i+1]}" for i in range(len(split_strs) - 1)] + [f"{split_strs[-1]} to inf", "NaN"]


def _num_state(feat: str, splits: list[float], values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> FeatureState:
    n_bins = len(splits) + 1
    stats = _bin_stats(target, weights, _num_assign(values, splits), n_bins)
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


def _cat_state(feat: str, category_bins: dict[str, int], values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> FeatureState:
    n_groups = max(category_bins.values()) + 1 if category_bins else 0
    stats = _bin_stats(target, weights, _cat_assign(values, category_bins), n_groups)
    groups: dict[int, list[str]] = {}
    for cat, grp in category_bins.items():
        groups.setdefault(grp, []).append(str(cat))
    return FeatureState(
        feature=feat,
        dtype=FeatureDtype.CATEGORICAL,
        n_bins=n_groups,
        groups={k: sorted(v) for k, v in groups.items()},
        bins=dict(category_bins),
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
        features: Feature DataFrame matching the features in ``bin_specs``.
        target: Binary target series (0/1 or float).
        time_periods: Optional time series for temporal stability metrics.
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
        features: pl.DataFrame,
        target: pl.Series,
        time_periods: pl.Series | None = None,
        weights: pl.Series | None = None,
        stability_threshold: float = 0.1,
    ) -> None:
        self._targets = target.cast(pl.Float64).to_numpy()
        self._weights = weights.cast(pl.Float64).to_numpy() if weights is not None else np.ones(len(self._targets))
        self._time: np.ndarray | None = time_periods.to_numpy() if time_periods is not None else None
        self._threshold = stability_threshold
        self._x: dict[str, np.ndarray] = {}
        self._splits: dict[str, list[float]] = {}
        self._cat_bins: dict[str, dict[str, int]] = {}
        self._history: dict[str, list[tuple[str, Any]]] = {}
        self._orig: dict[str, dict[str, Any]] = {}

        for feat, spec in bin_specs.items():
            if feat not in features.columns:
                continue
            self._orig[feat] = spec
            self._history[feat] = []
            if spec["dtype"] == FeatureDtype.NUMERIC:
                self._x[feat] = features[feat].cast(pl.Float64).to_numpy()
                self._splits[feat] = [float(s) for s in spec["bins"][1:-1] if np.isfinite(s)]
            else:
                self._x[feat] = features[feat].cast(pl.Utf8).to_numpy().astype(str)
                self._cat_bins[feat] = {str(k): int(v) for k, v in spec["bins"].items()}

    def features(self) -> list[str]:
        return list(self._splits.keys()) + list(self._cat_bins.keys())

    def _base_state(self, feat: str) -> FeatureState:
        if feat in self._splits:
            return _num_state(feat, self._splits[feat], self._x[feat], self._targets, self._weights)
        return _cat_state(feat, self._cat_bins[feat], self._x[feat], self._targets, self._weights)

    def _assignments(self, feat: str) -> np.ndarray:
        if feat in self._splits:
            return _num_assign(self._x[feat], self._splits[feat])
        return _cat_assign(self._x[feat], self._cat_bins[feat])

    def state(self, feat: str) -> FeatureState:
        s = self._base_state(feat)
        if self._time is not None:
            s.temporal = _temporal_stats(
                self._targets, self._weights, self._assignments(feat), s.n_bins, self._time, self._threshold
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

    def _suggest_num(self, feat: str, n_suggestions: int) -> list[float]:
        values = self._x[feat]
        x_valid = values[~np.isnan(values)]
        if len(x_valid) == 0:
            return []
        current = self._splits[feat]
        span = float(x_valid.max() - x_valid.min())
        min_gap = span * 0.01
        candidates = [
            float(candidate) for candidate in np.unique(np.percentile(x_valid, np.linspace(5, 95, 40)))
            if all(abs(candidate - split) > min_gap for split in current)
        ]
        base_information_value = self._base_state(feat).iv
        pairs: list[tuple[float, float]] = sorted(
            [
                (
                    _bin_stats(self._targets, self._weights, _num_assign(values, sorted(current + [candidate])), len(current) + 2).iv - base_information_value,
                    float(candidate),
                )
                for candidate in candidates
            ],
            reverse=True,
        )
        return [v for _, v in pairs[:n_suggestions]]

    def _suggest_cat(self, feat: str, n_suggestions: int) -> list[tuple[int, int]]:
        category_bins = self._cat_bins[feat]
        n_groups = max(category_bins.values()) + 1 if category_bins else 0
        if n_groups <= 1:
            return []
        values = self._x[feat]
        base_information_value = self._base_state(feat).iv
        pairs: list[tuple[float, tuple[int, int]]] = sorted(
            [
                (
                    base_information_value - _bin_stats(
                        self._targets, self._weights,
                        _cat_assign(values, {
                            category: (bin_idx if group == bin_idx + 1 else (group - 1 if group > bin_idx + 1 else group))
                            for category, group in category_bins.items()
                        }),
                        n_groups - 1,
                    ).iv,
                    (bin_idx, bin_idx + 1),
                )
                for bin_idx in range(n_groups - 1)
            ]
        )
        return [pair for _, pair in pairs[:n_suggestions]]

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
