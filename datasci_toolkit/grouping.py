from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

try:
    import lightgbm as lgb

    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

_LGBM_PARAMS: dict[str, Any] = {
    "num_iterations": 1,
    "objective": "binary",
    "num_class": 1,
    "metric": "auc",
    "learning_rate": 1.0,
    "boosting_type": "gbdt",
    "max_depth": -1,
    "num_leaves": 2,
    "seed": 42,
    "verbosity": -1,
    "device_type": "cpu",
    "n_jobs": -1,
}


def _woe(bads: float, goods: float, total_bads: float, total_goods: float, smooth: float = 0.001) -> float:
    return float(np.log(((bads + smooth) / (total_bads + smooth)) / ((goods + smooth) / (total_goods + smooth))))



def _rsi(scores: np.ndarray, event_rates: np.ndarray, months: np.ndarray, threshold: float) -> float:
    span = float(event_rates.max() - event_rates.min())
    if span == 0.0:
        return 1.0

    rank_arr = np.zeros(len(scores), dtype=int)
    for m in np.unique(months):
        idx = np.where(months == m)[0]
        order = np.argsort(-event_rates[idx])
        for rank_pos, pos in enumerate(order):
            rank_arr[idx[pos]] = rank_pos

    rsi_total = 0.0
    unique_scores = np.unique(scores)
    for s in unique_scores:
        mask = scores == s
        order = np.argsort(months[mask])
        s_ranks = rank_arr[mask][order]
        s_rates = event_rates[mask][order]

        mode_rank = int(np.bincount(s_ranks).argmax())
        max_leap = 0.0
        for i in range(len(s_ranks) - 1):
            if s_ranks[i] != s_ranks[i + 1]:
                max_leap = max(max_leap, abs(float(s_rates[i]) - float(s_rates[i + 1])))

        rsi_total += 1.0 if max_leap / span <= threshold else float(np.mean(s_ranks == mode_rank))

    return rsi_total / len(unique_scores)


def _encode_cats(x: np.ndarray, mapping: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    strs = np.array([str(v) if v is not None else "__null__" for v in x])
    if mapping is None:
        known = [s for s in np.unique(strs) if s != "__null__"]
        mapping = {c: i for i, c in enumerate(known)}
    encoded = np.array(
        [float(mapping[s]) if s in mapping else np.nan for s in strs],
        dtype=float,
    )
    return encoded, mapping


def _train_lgbm(
    params: dict[str, Any],
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    w_va: np.ndarray,
    categorical: bool,
) -> Any:
    if not _LGB_AVAILABLE:
        raise ImportError("lightgbm is required for StabilityGrouping")
    cat_feature = [0] if categorical else []
    train_ds = lgb.Dataset(
        x_tr.reshape(-1, 1),
        label=y_tr,
        weight=w_tr,
        categorical_feature=cat_feature,
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(x_va.reshape(-1, 1), label=y_va, weight=w_va, reference=train_ds)
    return lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, valid_ds],
        callbacks=[lgb.log_evaluation(0)],
    )


def _num_bin_spec(
    bst: Any,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    nan_woe: float,
    smooth: float,
) -> dict[str, Any]:
    non_null = ~np.isnan(x)
    x_obs = x[non_null]
    p = bst.predict(x_obs.reshape(-1, 1), num_iteration=1)
    y_obs, w_obs = y[non_null], w[non_null]
    total_bads = float((y_obs * w_obs).sum())
    total_goods = float(((1.0 - y_obs) * w_obs).sum())

    buckets = []
    for score in sorted(np.unique(p)):
        mask = p == score
        bads = float((y_obs[mask] * w_obs[mask]).sum())
        goods = float(((1.0 - y_obs[mask]) * w_obs[mask]).sum())
        buckets.append({
            "min": float(x_obs[mask].min()),
            "max": float(x_obs[mask].max()),
            "woe": _woe(bads, goods, total_bads, total_goods, smooth),
        })

    buckets.sort(key=lambda b: b["max"])
    bins: list[float] = (
        [-np.inf]
        + [(b["max"] + nb["min"]) / 2.0 for b, nb in zip(buckets[:-1], buckets[1:])]
        + [np.inf]
    )
    return {
        "dtype": "float",
        "bins": bins,
        "woes": [b["woe"] for b in buckets],
        "nan_woe": nan_woe,
    }


def _cat_bin_spec(
    bst: Any,
    x_enc: np.ndarray,
    x_orig: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    smooth: float,
) -> dict[str, Any]:
    non_null = ~np.isnan(x_enc)
    p = bst.predict(x_enc[non_null].reshape(-1, 1), num_iteration=1)
    y_obs, w_obs = y[non_null], w[non_null]
    x_obs = x_orig[non_null]
    total_bads = float((y_obs * w_obs).sum())
    total_goods = float(((1.0 - y_obs) * w_obs).sum())

    woes: list[float] = []
    bins: dict[str, int] = {}
    for bucket_idx, score in enumerate(sorted(np.unique(p))):
        mask = p == score
        bads = float((y_obs[mask] * w_obs[mask]).sum())
        goods = float(((1.0 - y_obs[mask]) * w_obs[mask]).sum())
        woes.append(_woe(bads, goods, total_bads, total_goods, smooth))
        for cat in np.unique(x_obs[mask]):
            bins[str(cat)] = bucket_idx

    return {"dtype": "category", "bins": bins, "woes": woes, "unknown_woe": 0.0}


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins_data: dict[str, dict[str, Any]] | None = None) -> None:
        self.bins_data = bins_data

    def fit(self, X: pl.DataFrame, y: object = None) -> "WOETransformer":
        if self.bins_data is None:
            raise ValueError("bins_data must be provided to WOETransformer")
        self.bins_data_: dict[str, dict[str, Any]] = self.bins_data
        self.feature_names_in_: list[str] = list(self.bins_data_.keys())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        check_is_fitted(self)
        cols: dict[str, list[float]] = {}
        for feat, spec in self.bins_data_.items():
            if feat not in X.columns:
                continue
            s = X[feat]
            if spec["dtype"] == "float":
                x_np = s.cast(pl.Float64).to_numpy()
                boundaries = np.array(spec["bins"][1:-1], dtype=float)
                woe_arr = np.array(spec["woes"], dtype=float)
                nan_woe: float = spec["nan_woe"]
                idx = np.clip(np.digitize(x_np, boundaries), 0, len(woe_arr) - 1)
                result = np.where(np.isnan(x_np), nan_woe, woe_arr[idx])
                cols[feat] = result.tolist()
            else:
                cat_bins: dict[str, int] = spec["bins"]
                woe_list: list[float] = spec["woes"]
                unknown_woe: float = spec["unknown_woe"]
                cols[feat] = [
                    woe_list[cat_bins[str(v)]] if str(v) in cat_bins else unknown_woe
                    for v in s.to_list()
                ]
        return pl.DataFrame(cols)


class StabilityGrouping(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_bins: int = 10,
        stability_threshold: float = 0.10,
        min_leaf_share: float = 0.05,
        min_leaf_minority: int = 100,
        important_minorities: list[str] | None = None,
        must_have: list[str] | None = None,
        woe_smooth: float = 0.001,
    ) -> None:
        self.max_bins = max_bins
        self.stability_threshold = stability_threshold
        self.min_leaf_share = min_leaf_share
        self.min_leaf_minority = min_leaf_minority
        self.important_minorities = important_minorities
        self.must_have = must_have
        self.woe_smooth = woe_smooth

    def _is_categorical(self, s: pl.Series) -> bool:
        return s.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)

    def _min_leaf(self, n: int, minority: bool) -> int:
        if minority:
            return self.min_leaf_minority
        return max(1, int(np.ceil(self.min_leaf_share * n)))

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        t_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
        t_val: pl.Series,
        weights_train: pl.Series | None = None,
        weights_val: pl.Series | None = None,
    ) -> "StabilityGrouping":
        minorities = set(self.important_minorities or [])
        must = set(self.must_have or [])
        y_tr = y_train.cast(pl.Float64).to_numpy()
        y_va = y_val.cast(pl.Float64).to_numpy()
        t_va = t_val.to_numpy()
        w_tr = weights_train.cast(pl.Float64).to_numpy() if weights_train is not None else np.ones(len(y_tr))
        w_va = weights_val.cast(pl.Float64).to_numpy() if weights_val is not None else np.ones(len(y_va))

        self.bins_data_: dict[str, dict[str, Any]] = {}
        self.excluded_: list[str] = []

        for feat in X_train.columns:
            is_cat = self._is_categorical(X_train[feat])
            is_minority = feat in minorities
            is_must = feat in must
            min_leaf = self._min_leaf(len(y_tr), is_minority)

            if is_cat:
                x_tr_enc, cat_map = _encode_cats(X_train[feat].to_numpy())
                x_va_enc, _ = _encode_cats(X_val[feat].to_numpy(), cat_map)
                x_tr_orig = X_train[feat].to_numpy()
            else:
                x_tr_enc = X_train[feat].cast(pl.Float64).to_numpy()
                x_va_enc = X_val[feat].cast(pl.Float64).to_numpy()
                x_tr_orig = x_tr_enc

            n_bins, exclude = self._auto_group(
                x_tr_enc, y_tr, w_tr, x_va_enc, y_va, w_va, t_va,
                is_cat, is_minority, is_must, min_leaf,
            )

            if exclude:
                self.excluded_.append(feat)
            else:
                params = {**_LGBM_PARAMS, "num_leaves": n_bins, "min_data_in_leaf": min_leaf}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bst = _train_lgbm(params, x_tr_enc, y_tr, w_tr, x_va_enc, y_va, w_va, is_cat)

                if is_cat:
                    spec = _cat_bin_spec(bst, x_tr_enc, x_tr_orig, y_tr, w_tr, self.woe_smooth)
                else:
                    null_mask = np.isnan(x_tr_enc)
                    if null_mask.any():
                        bads_null = float((y_tr[null_mask] * w_tr[null_mask]).sum())
                        goods_null = float(((1.0 - y_tr[null_mask]) * w_tr[null_mask]).sum())
                        total_bads = float((y_tr * w_tr).sum())
                        total_goods = float(((1.0 - y_tr) * w_tr).sum())
                        nan_woe = _woe(bads_null, goods_null, total_bads, total_goods, self.woe_smooth)
                    else:
                        nan_woe = 0.0
                    spec = _num_bin_spec(bst, x_tr_enc, y_tr, w_tr, nan_woe, self.woe_smooth)

                self.bins_data_[feat] = spec

        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        check_is_fitted(self)
        return WOETransformer(bins_data=self.bins_data_).fit(X).transform(X)

    def ungroupable(self) -> list[str]:
        check_is_fitted(self)
        return self.excluded_

    def _auto_group(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        w_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        w_va: np.ndarray,
        t_va: np.ndarray,
        is_cat: bool,
        is_minority: bool,
        is_must: bool,
        min_leaf: int,
    ) -> tuple[int, bool]:
        rsi_values: list[float] = []
        gini_values: list[float] = []
        unique_months = np.unique(t_va)

        for n_leaves in range(2, self.max_bins + 1):
            params = {**_LGBM_PARAMS, "num_leaves": n_leaves, "min_data_in_leaf": min_leaf}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    bst = _train_lgbm(params, x_tr, y_tr, w_tr, x_va, y_va, w_va, is_cat)
                except Exception:
                    rsi_values.append(0.0)
                    gini_values.append(0.0)
                    continue

            p_va = bst.predict(x_va.reshape(-1, 1), num_iteration=1)

            month_ginis: list[float] = []
            for m in unique_months:
                mask = t_va == m
                if len(np.unique(y_va[mask])) < 2:
                    continue
                try:
                    month_ginis.append(float(roc_auc_score(y_va[mask], p_va[mask], sample_weight=w_va[mask])))
                except Exception:
                    pass

            avg_gini = float(np.mean(month_ginis)) if month_ginis else 0.0

            scores_all: list[float] = []
            rates_all: list[float] = []
            months_all: list[Any] = []

            for m in unique_months:
                mask = t_va == m
                p_m, y_m, w_m = p_va[mask], y_va[mask], w_va[mask]
                for s in np.unique(p_m):
                    smask = p_m == s
                    w_sum = float(w_m[smask].sum())
                    if w_sum == 0.0:
                        continue
                    scores_all.append(float(s))
                    rates_all.append(float((y_m[smask] * w_m[smask]).sum() / w_sum))
                    months_all.append(m)

            rsi = _rsi(
                np.array(scores_all),
                np.array(rates_all),
                np.array(months_all),
                self.stability_threshold,
            )
            rsi_values.append(rsi)
            gini_values.append(avg_gini)

        rsi_arr = np.array(rsi_values)
        for i in range(1, len(rsi_arr)):
            if rsi_arr[i] == 1.0 and rsi_arr[i - 1] < 1.0:
                rsi_arr[i] = rsi_arr[i - 1]

        max_rsi = float(rsi_arr.max())
        if (not is_minority) and (not is_must) and max_rsi < 1.0:
            return -1, True

        gini_arr = np.array(gini_values)
        stable_mask = rsi_arr == max_rsi
        best_gini = float(gini_arr[stable_mask].max())
        best_mask = stable_mask & (gini_arr == best_gini)
        n_bins = int(np.arange(2, 2 + len(rsi_arr))[best_mask].min())

        return n_bins, False
