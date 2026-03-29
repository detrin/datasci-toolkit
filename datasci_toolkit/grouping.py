from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from optbinning import OptimalBinning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class EncodedFeature:
    values: np.ndarray
    category_map: dict[str, int]


@dataclass(frozen=True)
class FeatureArrays:
    x_train_encoded: np.ndarray
    x_val_encoded: np.ndarray
    x_train_original: np.ndarray
    is_categorical: bool


@dataclass(frozen=True)
class GroupingResult:
    n_bins: int
    exclude: bool

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


def _rsi(scores: np.ndarray, event_rates: np.ndarray, months: np.ndarray, threshold: float) -> float:
    span = float(event_rates.max() - event_rates.min())
    if span == 0.0:
        return 1.0

    bin_ranks = np.zeros(len(scores), dtype=int)
    for month in np.unique(months):
        month_indices = np.where(months == month)[0]
        sort_order = np.argsort(-event_rates[month_indices])
        for rank_position, original_index in enumerate(sort_order):
            bin_ranks[month_indices[original_index]] = rank_position

    stability_sum = 0.0
    for bin_score in np.unique(scores):
        score_mask = scores == bin_score
        time_order = np.argsort(months[score_mask])
        sorted_ranks = bin_ranks[score_mask][time_order]
        sorted_rates = event_rates[score_mask][time_order]

        dominant_rank = int(np.bincount(sorted_ranks).argmax())
        max_rate_jump = 0.0
        for i in range(len(sorted_ranks) - 1):
            if sorted_ranks[i] != sorted_ranks[i + 1]:
                max_rate_jump = max(max_rate_jump, abs(float(sorted_rates[i]) - float(sorted_rates[i + 1])))

        stability_sum += 1.0 if max_rate_jump / span <= threshold else float(np.mean(sorted_ranks == dominant_rank))

    return stability_sum / len(np.unique(scores))


def _encode_cats(values: np.ndarray, mapping: dict[str, int] | None = None) -> EncodedFeature:
    string_values = np.array([str(v) if v is not None else "__null__" for v in values])
    if mapping is None:
        known_categories = [sv for sv in np.unique(string_values) if sv != "__null__"]
        mapping = {category: index for index, category in enumerate(known_categories)}
    encoded = np.array(
        [float(mapping[sv]) if sv in mapping else np.nan for sv in string_values],
        dtype=float,
    )
    return EncodedFeature(values=encoded, category_map=mapping)


def _train_lgbm(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weights_val: np.ndarray,
    categorical: bool,
) -> Any:
    if not _LGB_AVAILABLE:
        raise ImportError("lightgbm is required for StabilityGrouping")
    categorical_features = [0] if categorical else []
    train_ds = lgb.Dataset(
        x_train.reshape(-1, 1),
        label=y_train,
        weight=weights_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(x_val.reshape(-1, 1), label=y_val, weight=weights_val, reference=train_ds)
    return lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, valid_ds],
        callbacks=[lgb.log_evaluation(0)],
    )


def _num_bin_spec(booster: Any, feature_values: np.ndarray) -> dict[str, Any]:
    valid_mask = ~np.isnan(feature_values)
    observed_values = feature_values[valid_mask]
    leaf_predictions = booster.predict(observed_values.reshape(-1, 1), num_iteration=1)

    score_buckets = sorted(
        [{"min": float(observed_values[leaf_predictions == s].min()), "max": float(observed_values[leaf_predictions == s].max())} for s in np.unique(leaf_predictions)],
        key=lambda b: b["max"],
    )
    bins: list[float] = (
        [-np.inf]
        + [(bucket["max"] + next_bucket["min"]) / 2.0 for bucket, next_bucket in zip(score_buckets[:-1], score_buckets[1:])]
        + [np.inf]
    )
    return {"dtype": "float", "bins": bins}


def _cat_bin_spec(booster: Any, encoded_values: np.ndarray, original_values: np.ndarray) -> dict[str, Any]:
    valid_mask = ~np.isnan(encoded_values)
    leaf_predictions = booster.predict(encoded_values[valid_mask].reshape(-1, 1), num_iteration=1)
    observed_values = original_values[valid_mask]

    bins: dict[str, int] = {}
    for bucket_idx, score in enumerate(sorted(np.unique(leaf_predictions))):
        for cat in np.unique(observed_values[leaf_predictions == score]):
            bins[str(cat)] = bucket_idx

    return {"dtype": "category", "bins": bins}


def _select_best_bins(
    rsi_arr: np.ndarray,
    gini_arr: np.ndarray,
    is_minority: bool,
    is_must: bool,
) -> GroupingResult:
    rsi_arr = rsi_arr.copy()
    for i in range(1, len(rsi_arr)):
        if rsi_arr[i] == 1.0 and rsi_arr[i - 1] < 1.0:
            rsi_arr[i] = rsi_arr[i - 1]

    max_rsi = float(rsi_arr.max())
    if (not is_minority) and (not is_must) and max_rsi < 1.0:
        return GroupingResult(n_bins=-1, exclude=True)

    stable_mask = rsi_arr == max_rsi
    best_gini = float(gini_arr[stable_mask].max())
    best_mask = stable_mask & (gini_arr == best_gini)
    n_bins = int(np.arange(2, 2 + len(rsi_arr))[best_mask].min())
    return GroupingResult(n_bins=n_bins, exclude=False)


def _monthly_gini(
    predictions_val: np.ndarray,
    y_val: np.ndarray,
    weights_val: np.ndarray,
    time_val: np.ndarray,
) -> float:
    scores: list[float] = []
    for month in np.unique(time_val):
        mask = time_val == month
        if len(np.unique(y_val[mask])) < 2:
            continue
        try:
            scores.append(float(roc_auc_score(y_val[mask], predictions_val[mask], sample_weight=weights_val[mask])))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else 0.0


def _bins_rsi(
    predictions_val: np.ndarray,
    y_val: np.ndarray,
    weights_val: np.ndarray,
    time_val: np.ndarray,
    threshold: float,
) -> float:
    all_bin_scores: list[float] = []
    all_event_rates: list[float] = []
    observation_months: list[Any] = []
    for month in np.unique(time_val):
        month_mask = time_val == month
        month_predictions, month_targets, month_weights = predictions_val[month_mask], y_val[month_mask], weights_val[month_mask]
        for score in np.unique(month_predictions):
            score_mask = month_predictions == score
            total_weight = float(month_weights[score_mask].sum())
            if total_weight == 0.0:
                continue
            all_bin_scores.append(float(score))
            all_event_rates.append(float((month_targets[score_mask] * month_weights[score_mask]).sum() / total_weight))
            observation_months.append(month)
    return _rsi(
        np.array(all_bin_scores),
        np.array(all_event_rates),
        np.array(observation_months),
        threshold,
    )


class WOETransformer(BaseEstimator, TransformerMixin):
    """Encodes features as Weight of Evidence values using pre-computed bin specs.

    Sklearn-compatible transformer (works in `Pipeline`, `GridSearchCV`).
    Bin specs must be provided at construction — use `StabilityGrouping` or
    `BinEditor.accept()` to produce them.

    Args:
        bin_specs: Mapping of feature name to spec dict with keys `dtype`
            (`"float"` or `"category"`) and `bins` (list of cut points or
            `{category: group_index}` dict).

    Attributes:
        binners_: Dict of fitted `OptimalBinning` instances keyed by feature.
        feature_names_in_: List of feature names seen during `fit`.
    """

    def __init__(self, bin_specs: dict[str, dict[str, Any]] | None = None) -> None:
        self.bin_specs = bin_specs

    def fit(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series | None = None) -> "WOETransformer":
        if self.bin_specs is None:
            raise ValueError("bin_specs must be provided to WOETransformer")
        y_np = y.cast(pl.Float64).to_numpy()
        w_np = weights.cast(pl.Float64).to_numpy() if weights is not None else None
        self.binners_: dict[str, OptimalBinning] = {}
        for feat, bin_spec in self.bin_specs.items():
            if feat not in X.columns:
                continue
            if bin_spec["dtype"] == "float":
                splits = [s for s in bin_spec["bins"][1:-1] if np.isfinite(s)]
                x_np = X[feat].cast(pl.Float64).to_numpy()
                binner = OptimalBinning(
                    name=feat,
                    dtype="numerical",
                    user_splits=splits if splits else None,
                )
            else:
                category_bins: dict[str, int] = bin_spec["bins"]
                groups: dict[int, list[str]] = {}
                for cat, idx in category_bins.items():
                    groups.setdefault(idx, []).append(cat)
                user_splits = [groups[i] for i in sorted(groups)]
                x_np = X[feat].cast(pl.Utf8).to_numpy().astype(str)
                binner = OptimalBinning(
                    name=feat,
                    dtype="categorical",
                    user_splits=user_splits,
                )
            binner.fit(x_np, y_np, sample_weight=w_np)
            self.binners_[feat] = binner
        self.feature_names_in_: list[str] = list(self.bin_specs.keys())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        check_is_fitted(self)
        assert self.bin_specs is not None
        result_cols: dict[str, list[float]] = {}
        for feat, binner in self.binners_.items():
            if feat not in X.columns:
                continue
            bin_spec = self.bin_specs[feat]
            if bin_spec["dtype"] == "float":
                x_np = X[feat].cast(pl.Float64).to_numpy()
            else:
                x_np = X[feat].cast(pl.Utf8).to_numpy().astype(str)
            result_cols[feat] = binner.transform(x_np, metric="woe").tolist()
        return pl.DataFrame(result_cols)


class StabilityGrouping(BaseEstimator, TransformerMixin):
    """Stability-constrained optimal binning with WOE encoding.

    Finds optimal bins for each feature using LightGBM, then merges bins whose
    event rate shifts significantly across time periods (measured by RSI).
    Requires both a train and validation split plus a time column.

    Args:
        max_bins: Upper bound on number of bins per feature.
        stability_threshold: Maximum RSI allowed per bin across time periods.
            Bins exceeding this are merged with a neighbour.
        min_leaf_share: Minimum fraction of total records per bin leaf.
        min_leaf_minority: Minimum records per bin for minority features.
        important_minorities: Features where `min_leaf_minority` applies.
        must_have: Features that are never excluded even if unstable.

    Attributes:
        bin_specs_: Dict of bin definitions produced after fitting.
        transformer_: Fitted `WOETransformer` instance.
        excluded_: Features that could not be grouped.
    """

    def __init__(
        self,
        max_bins: int = 10,
        stability_threshold: float = 0.10,
        min_leaf_share: float = 0.05,
        min_leaf_minority: int = 100,
        important_minorities: list[str] | None = None,
        must_have: list[str] | None = None,
    ) -> None:
        self.max_bins = max_bins
        self.stability_threshold = stability_threshold
        self.min_leaf_share = min_leaf_share
        self.min_leaf_minority = min_leaf_minority
        self.important_minorities = important_minorities
        self.must_have = must_have

    def _is_categorical(self, s: pl.Series) -> bool:
        return s.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum)

    def _min_leaf(self, n: int, minority: bool) -> int:
        if minority:
            return self.min_leaf_minority
        return max(1, int(np.ceil(self.min_leaf_share * n)))

    def _prepare_feature_data(self, feat: str, X_train: pl.DataFrame, X_val: pl.DataFrame) -> FeatureArrays:
        if self._is_categorical(X_train[feat]):
            train_enc = _encode_cats(X_train[feat].to_numpy())
            val_enc = _encode_cats(X_val[feat].to_numpy(), train_enc.category_map)
            return FeatureArrays(
                x_train_encoded=train_enc.values,
                x_val_encoded=val_enc.values,
                x_train_original=X_train[feat].to_numpy(),
                is_categorical=True,
            )
        x_train = X_train[feat].cast(pl.Float64).to_numpy()
        return FeatureArrays(
            x_train_encoded=x_train,
            x_val_encoded=X_val[feat].cast(pl.Float64).to_numpy(),
            x_train_original=x_train,
            is_categorical=False,
        )

    def _fit_feature(
        self,
        feat: str,
        arrays: FeatureArrays,
        y_train: np.ndarray,
        weights_train: np.ndarray,
        y_val: np.ndarray,
        weights_val: np.ndarray,
        time_val: np.ndarray,
        is_minority: bool,
        is_must: bool,
    ) -> dict[str, Any] | None:
        min_leaf = self._min_leaf(len(y_train), is_minority)
        grouping_result = self._auto_group(
            arrays.x_train_encoded, y_train, weights_train, arrays.x_val_encoded, y_val, weights_val, time_val,
            arrays.is_categorical, is_minority, is_must, min_leaf,
        )
        if grouping_result.exclude:
            return None
        model_params = {**_LGBM_PARAMS, "num_leaves": grouping_result.n_bins, "min_data_in_leaf": min_leaf}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            booster = _train_lgbm(model_params, arrays.x_train_encoded, y_train, weights_train, arrays.x_val_encoded, y_val, weights_val, arrays.is_categorical)
        if arrays.is_categorical:
            return _cat_bin_spec(booster, arrays.x_train_encoded, arrays.x_train_original)
        return _num_bin_spec(booster, arrays.x_train_encoded)

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
        y_train_np = y_train.cast(pl.Float64).to_numpy()
        y_val_np = y_val.cast(pl.Float64).to_numpy()
        time_val_np = t_val.to_numpy()
        weights_train_np = weights_train.cast(pl.Float64).to_numpy() if weights_train is not None else np.ones(len(y_train_np))
        weights_val_np = weights_val.cast(pl.Float64).to_numpy() if weights_val is not None else np.ones(len(y_val_np))

        bin_specs: dict[str, dict[str, Any]] = {}
        self.excluded_: list[str] = []

        for feat in X_train.columns:
            arrays = self._prepare_feature_data(feat, X_train, X_val)
            spec = self._fit_feature(feat, arrays, y_train_np, weights_train_np, y_val_np, weights_val_np, time_val_np, feat in minorities, feat in must)
            if spec is None:
                self.excluded_.append(feat)
            else:
                bin_specs[feat] = spec

        self.bin_specs_: dict[str, dict[str, Any]] = bin_specs
        self.transformer_: WOETransformer = WOETransformer(bin_specs=bin_specs).fit(X_train, y_train, weights_train)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        check_is_fitted(self)
        return self.transformer_.transform(X)

    def ungroupable(self) -> list[str]:
        check_is_fitted(self)
        return self.excluded_

    def _evaluate_bin_counts(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        weights_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        weights_val: np.ndarray,
        time_val: np.ndarray,
        is_categorical: bool,
        min_leaf: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rsi_values: list[float] = []
        gini_values: list[float] = []
        for n_leaves in range(2, self.max_bins + 1):
            params = {**_LGBM_PARAMS, "num_leaves": n_leaves, "min_data_in_leaf": min_leaf}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    booster = _train_lgbm(params, x_train, y_train, weights_train, x_val, y_val, weights_val, is_categorical)
                except Exception:
                    rsi_values.append(0.0)
                    gini_values.append(0.0)
                    continue
            val_leaf_predictions = booster.predict(x_val.reshape(-1, 1), num_iteration=1)
            rsi_values.append(_bins_rsi(val_leaf_predictions, y_val, weights_val, time_val, self.stability_threshold))
            gini_values.append(_monthly_gini(val_leaf_predictions, y_val, weights_val, time_val))
        return np.array(rsi_values), np.array(gini_values)

    def _auto_group(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        weights_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        weights_val: np.ndarray,
        time_val: np.ndarray,
        is_categorical: bool,
        is_minority: bool,
        is_must: bool,
        min_leaf: int,
    ) -> GroupingResult:
        rsi_arr, gini_arr = self._evaluate_bin_counts(x_train, y_train, weights_train, x_val, y_val, weights_val, time_val, is_categorical, min_leaf)
        return _select_best_bins(rsi_arr, gini_arr, is_minority, is_must)
