from __future__ import annotations

import copy
from typing import Any

import numpy as np
import polars as pl

_SMOOTH = 0.5


def _bin_stats(y: np.ndarray, w: np.ndarray, assignments: np.ndarray, n_bins: int) -> dict[str, Any]:
    total_ev = float((y * w).sum())
    total_nev = float(((1.0 - y) * w).sum())
    yw = y * w
    counts = np.bincount(assignments, weights=w, minlength=n_bins + 1).astype(float)
    events = np.bincount(assignments, weights=yw, minlength=n_bins + 1).astype(float)
    nonevents = counts - events
    event_rates = np.where(counts > 0, events / counts, np.nan)

    ev_d, nev_d = events[:n_bins], nonevents[:n_bins]
    dist_ev = (ev_d + _SMOOTH) / (total_ev + _SMOOTH * n_bins)
    dist_nev = (nev_d + _SMOOTH) / (total_nev + _SMOOTH * n_bins)
    woe_d = np.log(dist_ev / dist_nev)
    iv = float(((dist_ev - dist_nev) * woe_d).sum())

    nan_de = (events[n_bins] + _SMOOTH) / (total_ev + _SMOOTH)
    nan_dn = (nonevents[n_bins] + _SMOOTH) / (total_nev + _SMOOTH)
    woe_nan = float(np.log(nan_de / nan_dn))

    return {"counts": counts, "event_rates": event_rates, "woe": np.append(woe_d, woe_nan), "iv": iv}


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
    s = [f"{v:.4g}" for v in splits]
    return [f"-inf to {s[0]}"] + [f"{s[i]} to {s[i+1]}" for i in range(len(s) - 1)] + [f"{s[-1]} to inf", "NaN"]


def _num_state(feat: str, splits: list[float], x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    n_bins = len(splits) + 1
    s = _bin_stats(y, w, _num_assign(x, splits), n_bins)
    return {
        "feature": feat,
        "dtype": "float",
        "n_bins": n_bins,
        "splits": list(splits),
        "bins": _num_labels(splits),
        "counts": s["counts"].tolist(),
        "event_rates": [None if np.isnan(v) else round(float(v), 6) for v in s["event_rates"]],
        "woe": [round(float(v), 6) for v in s["woe"]],
        "iv": round(s["iv"], 6),
    }


def _cat_state(feat: str, cat_bins: dict[str, int], x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    n_groups = max(cat_bins.values()) + 1 if cat_bins else 0
    s = _bin_stats(y, w, _cat_assign(x, cat_bins), n_groups)
    groups: dict[int, list[str]] = {}
    for cat, grp in cat_bins.items():
        groups.setdefault(grp, []).append(str(cat))
    return {
        "feature": feat,
        "dtype": "category",
        "n_bins": n_groups,
        "groups": {k: sorted(v) for k, v in groups.items()},
        "bins": dict(cat_bins),
        "counts": s["counts"].tolist(),
        "event_rates": [None if np.isnan(v) else round(float(v), 6) for v in s["event_rates"]],
        "woe": [round(float(v), 6) for v in s["woe"]],
        "iv": round(s["iv"], 6),
    }


class BinEditor:
    def __init__(
        self,
        bin_specs: dict[str, dict[str, Any]],
        X: pl.DataFrame,
        y: pl.Series,
        weights: pl.Series | None = None,
    ) -> None:
        self._y = y.cast(pl.Float64).to_numpy()
        self._w = weights.cast(pl.Float64).to_numpy() if weights is not None else np.ones(len(self._y))
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
            if spec["dtype"] == "float":
                self._x[feat] = X[feat].cast(pl.Float64).to_numpy()
                self._splits[feat] = [float(s) for s in spec["bins"][1:-1] if np.isfinite(s)]
            else:
                self._x[feat] = X[feat].cast(pl.Utf8).to_numpy().astype(str)
                self._cat_bins[feat] = {str(k): int(v) for k, v in spec["bins"].items()}

    def features(self) -> list[str]:
        return list(self._splits.keys()) + list(self._cat_bins.keys())

    def state(self, feat: str) -> dict[str, Any]:
        if feat in self._splits:
            return _num_state(feat, self._splits[feat], self._x[feat], self._y, self._w)
        return _cat_state(feat, self._cat_bins[feat], self._x[feat], self._y, self._w)

    def _push(self, feat: str) -> None:
        if feat in self._splits:
            self._history[feat].append(("splits", list(self._splits[feat])))
        else:
            self._history[feat].append(("cat", copy.deepcopy(self._cat_bins[feat])))

    def split(self, feat: str, value: float) -> dict[str, Any]:
        if value in self._splits[feat]:
            return self.state(feat)
        self._push(feat)
        self._splits[feat] = sorted(self._splits[feat] + [value])
        return self.state(feat)

    def merge(self, feat: str, bin_idx: int) -> dict[str, Any]:
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

    def move_boundary(self, feat: str, bin_idx: int, new_value: float) -> dict[str, Any]:
        splits = self._splits[feat]
        if bin_idx >= len(splits):
            return self.state(feat)
        self._push(feat)
        new = list(splits)
        new[bin_idx] = new_value
        self._splits[feat] = sorted(set(new))
        return self.state(feat)

    def reset(self, feat: str) -> dict[str, Any]:
        self._history[feat] = []
        spec = self._orig[feat]
        if spec["dtype"] == "float":
            self._splits[feat] = [float(s) for s in spec["bins"][1:-1] if np.isfinite(s)]
        else:
            self._cat_bins[feat] = {str(k): int(v) for k, v in spec["bins"].items()}
        return self.state(feat)

    def undo(self, feat: str) -> dict[str, Any]:
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

    def suggest_splits(self, feat: str, n: int = 5) -> list:  # type: ignore[type-arg]
        if feat in self._splits:
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
            base_iv = self.state(feat)["iv"]
            pairs_num: list[tuple[float, float]] = sorted(
                [
                    (
                        _bin_stats(self._y, self._w, _num_assign(x, sorted(current + [c])), len(current) + 2)["iv"] - base_iv,
                        float(c),
                    )
                    for c in candidates
                ],
                reverse=True,
            )
            return [v for _, v in pairs_num[:n]]
        else:
            cat_bins = self._cat_bins[feat]
            n_groups = max(cat_bins.values()) + 1 if cat_bins else 0
            if n_groups <= 1:
                return []
            x = self._x[feat]
            base_iv = self.state(feat)["iv"]
            pairs_cat: list[tuple[float, tuple[int, int]]] = sorted(
                [
                    (
                        base_iv - _bin_stats(
                            self._y, self._w,
                            _cat_assign(x, {
                                cat: (bin_idx if grp == bin_idx + 1 else (grp - 1 if grp > bin_idx + 1 else grp))
                                for cat, grp in cat_bins.items()
                            }),
                            n_groups - 1,
                        )["iv"],
                        (bin_idx, bin_idx + 1),
                    )
                    for bin_idx in range(n_groups - 1)
                ]
            )
            return [pair for _, pair in pairs_cat[:n]]

    def accept(self) -> dict[str, dict[str, Any]]:
        return {feat: self.accept_feature(feat) for feat in self.features()}

    def accept_feature(self, feat: str) -> dict[str, Any]:
        if feat in self._splits:
            return {"dtype": "float", "bins": [-np.inf] + self._splits[feat] + [np.inf]}
        return {"dtype": "category", "bins": dict(self._cat_bins[feat])}
