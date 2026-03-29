from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class ScoreResult:
    auc: float
    signs_consistent: bool


@dataclass
class TrainValData:
    X_train: np.ndarray
    y_train: np.ndarray
    weights_train: np.ndarray | None
    X_val: np.ndarray
    y_val: np.ndarray
    weights_val: np.ndarray | None


@dataclass
class StepResult:
    predictors: list[str]
    auc: float
    log_entries: list[dict[str, Any]]


def _corr_matrix(X: np.ndarray, sample: int) -> np.ndarray:
    if len(X) > sample:
        idx = np.random.default_rng(42).choice(len(X), sample, replace=False)
        X = X[idx]
    return np.corrcoef(X.T)


def _max_abs_corr(corr: np.ndarray, idx: list[int]) -> float:
    if len(idx) <= 1:
        return 0.0
    sub = corr[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, 0.0)
    return float(np.abs(sub).max())


def _same_sign(coef: np.ndarray) -> bool:
    return bool(np.abs(np.sum(coef)) == np.sum(np.abs(coef)))


def _fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None, penalty: str, C: float) -> LogisticRegression:
    if penalty == "l1":
        model = LogisticRegression(C=C, solver="saga", l1_ratio=1.0, max_iter=1000)
    else:
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    model.fit(X, y, sample_weight=w)
    return model


def _auc(y: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(roc_auc_score(y, y_pred, sample_weight=w))


def _cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    penalty: str,
    C: float,
    folds: int,
    seed: int,
    stratify: bool,
) -> float:
    kf = (StratifiedKFold if stratify else KFold)(n_splits=folds, shuffle=True, random_state=seed)
    scores = [
        _auc(
            y[va],
            _fit_logit(X[tr], y[tr], w[tr] if w is not None else None, penalty, C).predict_proba(X[va])[:, 1],
            w[va] if w is not None else None,
        )
        for tr, va in kf.split(X, y)
    ]
    return float(np.mean(scores))


class AUCStepwiseLogit(BaseEstimator):
    """Gini-based stepwise logistic regression.

    Selects features by Gini improvement rather than p-values, with optional
    correlation filtering, sign enforcement, and cross-validated scoring.

    Args:
        initial_predictors: Features forced into the model at the start.
        all_predictors: Candidate pool (defaults to all columns in `X`).
        selection_method: ``"forward"``, ``"backward"``, or ``"stepwise"``.
        max_iter: Maximum number of add/remove steps.
        min_increase: Minimum Gini gain required to add a feature.
        max_decrease: Maximum Gini drop allowed before removing a feature.
        max_predictors: Hard cap on model size (0 = unlimited).
        max_correlation: Reject candidates correlated above this with any
            already-selected feature.
        enforce_coef_sign: Reject features that flip a coefficient sign.
        penalty: Regularisation type passed to `LogisticRegression`.
        C: Regularisation strength.
        correlation_sample: Max rows used for the correlation check.
        use_cv: Score via k-fold CV instead of a held-out validation set.
        cv_folds: Number of CV folds.
        cv_seed: Random seed for CV splits.
        cv_stratify: Use stratified folds.

    Attributes:
        predictors_: Ordered list of selected feature names.
        coef_: Coefficients for selected features.
        intercept_: Model intercept.
        progress_: DataFrame logging each add/remove step with Gini deltas.
    """

    def __init__(
        self,
        initial_predictors: list[str] | None = None,
        all_predictors: list[str] | None = None,
        selection_method: str = "stepwise",
        max_iter: int = 1000,
        min_increase: float = 0.005,
        max_decrease: float = 0.0025,
        max_predictors: int = 0,
        max_correlation: float = 1.0,
        enforce_coef_sign: bool = False,
        penalty: str = "l2",
        C: float = 1000.0,
        correlation_sample: int = 10000,
        use_cv: bool = False,
        cv_folds: int = 5,
        cv_seed: int = 42,
        cv_stratify: bool = True,
    ) -> None:
        self.initial_predictors = initial_predictors
        self.all_predictors = all_predictors
        self.selection_method = selection_method
        self.max_iter = max_iter
        self.min_increase = min_increase
        self.max_decrease = max_decrease
        self.max_predictors = max_predictors
        self.max_correlation = max_correlation
        self.enforce_coef_sign = enforce_coef_sign
        self.penalty = penalty
        self.C = C
        self.correlation_sample = correlation_sample
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.cv_stratify = cv_stratify

    def _score(
        self,
        predictors: list[str],
        data: TrainValData,
        col_idx: dict[str, int],
        cache: dict[frozenset[str], ScoreResult],
    ) -> ScoreResult:
        key = frozenset(predictors)
        if key in cache:
            return cache[key]
        col_indices = [col_idx[feat] for feat in predictors]
        model = _fit_logit(data.X_train[:, col_indices], data.y_train, data.weights_train, self.penalty, self.C)
        if self.use_cv:
            auc = _cv_auc(data.X_train[:, col_indices], data.y_train, data.weights_train, self.penalty, self.C, self.cv_folds, self.cv_seed, self.cv_stratify)
        else:
            auc = _auc(data.y_val, model.predict_proba(data.X_val[:, col_indices])[:, 1], data.weights_val)
        result = ScoreResult(auc=auc, signs_consistent=_same_sign(model.coef_.ravel()))
        cache[key] = result
        return result

    def _feasible(self, rec: dict[str, Any], addrm: int, min_inc: float, max_dec: float) -> bool:
        if rec["addrm"] != addrm or rec["used"]:
            return False
        if addrm == 1 and rec["delta"] < min_inc:
            return False
        if addrm == -1 and rec["delta"] < -max_dec:
            return False
        if self.selection_method == "forward":
            if self.enforce_coef_sign and not rec["same_sign"]:
                return False
            if self.max_correlation < 1.0 and rec["max_corr"] > self.max_correlation:
                return False
        return True

    def _prepare_data(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        X_val: pl.DataFrame | None,
        y_val: pl.Series | None,
        weights: pl.Series | None,
        weights_val: pl.Series | None,
    ) -> TrainValData:
        X_train = X.to_numpy().astype(float)
        y_train = y.cast(pl.Float64).to_numpy()
        weights_train = weights.cast(pl.Float64).to_numpy() if weights is not None else None

        if self.use_cv:
            if X_val is not None and y_val is not None:
                X_validation = np.vstack([X_train, X_val.to_numpy().astype(float)])
                y_validation = np.concatenate([y_train, y_val.cast(pl.Float64).to_numpy()])
                weights_validation: np.ndarray | None = (
                    np.concatenate([weights_train, weights_val.cast(pl.Float64).to_numpy()])
                    if weights_train is not None and weights_val is not None
                    else weights_train
                )
            else:
                X_validation, y_validation, weights_validation = X_train, y_train, weights_train
        elif X_val is not None and y_val is not None:
            X_validation = X_val.to_numpy().astype(float)
            y_validation = y_val.cast(pl.Float64).to_numpy()
            weights_validation = weights_val.cast(pl.Float64).to_numpy() if weights_val is not None else None
        else:
            X_validation, y_validation, weights_validation = X_train, y_train, weights_train

        return TrainValData(
            X_train=X_train,
            y_train=y_train,
            weights_train=weights_train,
            X_val=X_validation,
            y_val=y_validation,
            weights_val=weights_validation,
        )

    def _step_forward(
        self,
        candidates: list[str],
        current_preds: list[str],
        current_auc: float,
        iteration: int,
        data: TrainValData,
        col_idx: dict[str, int],
        cache: dict[frozenset[str], ScoreResult],
        corr: np.ndarray,
        min_inc: float,
        max_dec: float,
    ) -> StepResult:
        entries: list[dict[str, Any]] = []
        if self.max_predictors > 0 and len(current_preds) >= self.max_predictors:
            return StepResult(predictors=list(current_preds), auc=current_auc, log_entries=entries)
        for pred in candidates:
            if pred in current_preds:
                continue
            candidate_set = current_preds + [pred]
            scored = self._score(candidate_set, data, col_idx, cache)
            entries.append({
                "iteration": iteration, "addrm": 1, "predictors": candidate_set,
                "n_predictors": len(candidate_set), "auc": scored.auc, "delta": scored.auc - current_auc,
                "used": False, "same_sign": scored.signs_consistent,
                "max_corr": _max_abs_corr(corr, [col_idx[feat] for feat in candidate_set]),
            })
        feasible = sorted(
            [r for r in entries if self._feasible(r, 1, min_inc, max_dec)],
            key=lambda r: r["delta"],
            reverse=True,
        )
        if feasible:
            feasible[0]["used"] = True
            return StepResult(predictors=list(feasible[0]["predictors"]), auc=feasible[0]["auc"], log_entries=entries)
        return StepResult(predictors=list(current_preds), auc=current_auc, log_entries=entries)

    def _step_backward(
        self,
        current_preds: list[str],
        current_auc: float,
        iteration: int,
        data: TrainValData,
        col_idx: dict[str, int],
        cache: dict[frozenset[str], ScoreResult],
        corr: np.ndarray,
        min_inc: float,
        max_dec: float,
    ) -> StepResult:
        entries: list[dict[str, Any]] = []
        if len(current_preds) <= 1:
            return StepResult(predictors=list(current_preds), auc=current_auc, log_entries=entries)
        for pred in current_preds:
            candidate_set = [p for p in current_preds if p != pred]
            scored = self._score(candidate_set, data, col_idx, cache)
            entries.append({
                "iteration": iteration, "addrm": -1, "predictors": candidate_set,
                "n_predictors": len(candidate_set), "auc": scored.auc, "delta": scored.auc - current_auc,
                "used": False, "same_sign": scored.signs_consistent,
                "max_corr": _max_abs_corr(corr, [col_idx[feat] for feat in candidate_set]),
            })
        feasible = sorted(
            [r for r in entries if self._feasible(r, -1, min_inc, max_dec)],
            key=lambda r: r["delta"],
            reverse=True,
        )
        if feasible:
            feasible[0]["used"] = True
            return StepResult(predictors=list(feasible[0]["predictors"]), auc=feasible[0]["auc"], log_entries=entries)
        return StepResult(predictors=list(current_preds), auc=current_auc, log_entries=entries)

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
        weights: pl.Series | None = None,
        weights_val: pl.Series | None = None,
    ) -> "AUCStepwiseLogit":
        col_idx = {c: i for i, c in enumerate(X.columns)}
        initial = list(self.initial_predictors or [])
        candidates = list(self.all_predictors or X.columns)

        data = self._prepare_data(X, y, X_val, y_val, weights, weights_val)
        corr = _corr_matrix(data.X_train, self.correlation_sample)
        max_dec = max(0.0, self.max_decrease)
        min_inc = max(self.min_increase, max_dec + 1e-9)

        cache: dict[frozenset[str], ScoreResult] = {}
        current_preds = list(initial)

        if current_preds:
            initial_score = self._score(current_preds, data, col_idx, cache)
            corr0 = _max_abs_corr(corr, [col_idx[feat] for feat in current_preds])
        else:
            initial_score = ScoreResult(auc=0.5, signs_consistent=True)
            corr0 = 0.0

        records: list[dict[str, Any]] = [{
            "iteration": 0, "addrm": 0, "predictors": list(current_preds),
            "n_predictors": len(current_preds), "auc": initial_score.auc, "delta": 0.0,
            "used": True, "same_sign": initial_score.signs_consistent, "max_corr": corr0,
        }]

        for iteration in range(1, self.max_iter + 1):
            prev = next(r for r in reversed(records) if r["addrm"] == 0)
            current_preds = list(prev["predictors"])
            current_auc = prev["auc"]
            original_preds = list(current_preds)

            if self.selection_method in ("forward", "stepwise"):
                fwd = self._step_forward(candidates, current_preds, current_auc, iteration, data, col_idx, cache, corr, min_inc, max_dec)
                current_preds, current_auc = fwd.predictors, fwd.auc
                records.extend(fwd.log_entries)

            if self.selection_method in ("backward", "stepwise"):
                bwd = self._step_backward(current_preds, current_auc, iteration, data, col_idx, cache, corr, min_inc, max_dec)
                current_preds, current_auc = bwd.predictors, bwd.auc
                records.extend(bwd.log_entries)

            records.append({
                "iteration": iteration, "addrm": 0, "predictors": list(current_preds),
                "n_predictors": len(current_preds), "auc": current_auc, "delta": 0.0,
                "used": True, "same_sign": None, "max_corr": 0.0,
            })

            if current_preds == original_preds:
                break

        self.predictors_: list[str] = current_preds
        if self.predictors_:
            col_indices = [col_idx[feat] for feat in self.predictors_]
            self.model_: LogisticRegression | None = _fit_logit(data.X_train[:, col_indices], data.y_train, data.weights_train, self.penalty, self.C)
            self.coef_: np.ndarray = self.model_.coef_.ravel()
            self.intercept_: float = float(self.model_.intercept_[0])
        else:
            self.model_ = None
            self.coef_ = np.array([])
            rate = float(data.y_train.mean())
            self.intercept_ = float(np.log(rate / (1.0 - rate))) if 0.0 < rate < 1.0 else 0.0

        self.progress_: pl.DataFrame = pl.DataFrame({
            "iteration": [r["iteration"] for r in records],
            "addrm": [r["addrm"] for r in records],
            "n_predictors": [r["n_predictors"] for r in records],
            "auc": [r["auc"] for r in records],
            "delta": [r["delta"] for r in records],
            "predictors": [r["predictors"] for r in records],
        })
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        if not self.predictors_ or self.model_ is None:
            return np.full(len(X), 1.0 / (1.0 + np.exp(-self.intercept_)))
        col_indices = [list(X.columns).index(feat) for feat in self.predictors_]
        return self.model_.predict_proba(X.to_numpy()[:, col_indices])[:, 1]

    def score(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series | None = None) -> float:
        check_is_fitted(self)
        w = weights.cast(pl.Float64).to_numpy() if weights is not None else None
        return _auc(y.cast(pl.Float64).to_numpy(), self.predict(X), w)
