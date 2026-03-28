from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


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
        m = LogisticRegression(C=C, solver="saga", l1_ratio=1.0, max_iter=1000)
    else:
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    m.fit(X, y, sample_weight=w)
    return m


def _auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(roc_auc_score(y, p, sample_weight=w))


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
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        w_tr: np.ndarray | None,
        X_va: np.ndarray,
        y_va: np.ndarray,
        w_va: np.ndarray | None,
        col_idx: dict[str, int],
        cache: dict[frozenset[str], tuple[float, bool]],
    ) -> tuple[float, bool]:
        key = frozenset(predictors)
        if key in cache:
            return cache[key]
        idx = [col_idx[p] for p in predictors]
        m = _fit_logit(X_tr[:, idx], y_tr, w_tr, self.penalty, self.C)
        if self.use_cv:
            score = _cv_auc(X_tr[:, idx], y_tr, w_tr, self.penalty, self.C, self.cv_folds, self.cv_seed, self.cv_stratify)
        else:
            score = _auc(y_va, m.predict_proba(X_va[:, idx])[:, 1], w_va)
        result = (score, _same_sign(m.coef_.ravel()))
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

        X_tr = X.to_numpy().astype(float)
        y_tr = y.cast(pl.Float64).to_numpy()
        w_tr = weights.cast(pl.Float64).to_numpy() if weights is not None else None

        if self.use_cv:
            if X_val is not None and y_val is not None:
                X_va = np.vstack([X_tr, X_val.to_numpy().astype(float)])
                y_va = np.concatenate([y_tr, y_val.cast(pl.Float64).to_numpy()])
                w_va = (
                    np.concatenate([w_tr, weights_val.cast(pl.Float64).to_numpy()])
                    if w_tr is not None and weights_val is not None
                    else w_tr
                )
            else:
                X_va, y_va, w_va = X_tr, y_tr, w_tr
        elif X_val is not None and y_val is not None:
            X_va = X_val.to_numpy().astype(float)
            y_va = y_val.cast(pl.Float64).to_numpy()
            w_va = weights_val.cast(pl.Float64).to_numpy() if weights_val is not None else None
        else:
            X_va, y_va, w_va = X_tr, y_tr, w_tr

        corr = _corr_matrix(X_tr, self.correlation_sample)
        max_dec = max(0.0, self.max_decrease)
        min_inc = max(self.min_increase, max_dec + 1e-9)

        cache: dict[frozenset[str], tuple[float, bool]] = {}
        current_preds = list(initial)

        if current_preds:
            auc0, sgn0 = self._score(current_preds, X_tr, y_tr, w_tr, X_va, y_va, w_va, col_idx, cache)
            corr0 = _max_abs_corr(corr, [col_idx[p] for p in current_preds])
        else:
            auc0, sgn0, corr0 = 0.5, True, 0.0

        records: list[dict[str, Any]] = [{
            "iteration": 0, "addrm": 0, "predictors": list(current_preds),
            "n_predictors": len(current_preds), "auc": auc0, "delta": 0.0,
            "used": True, "same_sign": sgn0, "max_corr": corr0,
        }]

        for iteration in range(1, self.max_iter + 1):
            prev = next(r for r in reversed(records) if r["addrm"] == 0)
            current_preds = list(prev["predictors"])
            current_auc = prev["auc"]
            original_preds = list(current_preds)

            if self.selection_method in ("forward", "stepwise"):
                if self.max_predictors <= 0 or len(current_preds) < self.max_predictors:
                    for pred in [p for p in candidates if p not in current_preds]:
                        cands = current_preds + [pred]
                        auc, sgn = self._score(cands, X_tr, y_tr, w_tr, X_va, y_va, w_va, col_idx, cache)
                        records.append({
                            "iteration": iteration, "addrm": 1, "predictors": cands,
                            "n_predictors": len(cands), "auc": auc, "delta": auc - current_auc,
                            "used": False, "same_sign": sgn,
                            "max_corr": _max_abs_corr(corr, [col_idx[p] for p in cands]),
                        })

                    feasible = sorted(
                        [r for r in records if r["iteration"] == iteration and self._feasible(r, 1, min_inc, max_dec)],
                        key=lambda r: r["delta"], reverse=True,
                    )
                    if feasible:
                        feasible[0]["used"] = True
                        current_preds = list(feasible[0]["predictors"])
                        current_auc = feasible[0]["auc"]

            if self.selection_method in ("backward", "stepwise") and len(current_preds) > 1:
                for pred in current_preds:
                    cands = [p for p in current_preds if p != pred]
                    auc, sgn = self._score(cands, X_tr, y_tr, w_tr, X_va, y_va, w_va, col_idx, cache)
                    records.append({
                        "iteration": iteration, "addrm": -1, "predictors": cands,
                        "n_predictors": len(cands), "auc": auc, "delta": auc - current_auc,
                        "used": False, "same_sign": sgn,
                        "max_corr": _max_abs_corr(corr, [col_idx[p] for p in cands]),
                    })

                feasible = sorted(
                    [r for r in records if r["iteration"] == iteration and self._feasible(r, -1, min_inc, max_dec)],
                    key=lambda r: r["delta"], reverse=True,
                )
                if feasible:
                    feasible[0]["used"] = True
                    current_preds = list(feasible[0]["predictors"])
                    current_auc = feasible[0]["auc"]

            records.append({
                "iteration": iteration, "addrm": 0, "predictors": list(current_preds),
                "n_predictors": len(current_preds), "auc": current_auc, "delta": 0.0,
                "used": True, "same_sign": None, "max_corr": 0.0,
            })

            if current_preds == original_preds:
                break

        self.predictors_: list[str] = current_preds
        if self.predictors_:
            idx = [col_idx[p] for p in self.predictors_]
            self.model_: LogisticRegression | None = _fit_logit(X_tr[:, idx], y_tr, w_tr, self.penalty, self.C)
            self.coef_: np.ndarray = self.model_.coef_.ravel()
            self.intercept_: float = float(self.model_.intercept_[0])
        else:
            self.model_ = None
            self.coef_ = np.array([])
            rate = float(y_tr.mean())
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
        idx = [list(X.columns).index(p) for p in self.predictors_]
        return self.model_.predict_proba(X.to_numpy()[:, idx])[:, 1]

    def score(self, X: pl.DataFrame, y: pl.Series, weights: pl.Series | None = None) -> float:
        check_is_fitted(self)
        w = weights.cast(pl.Float64).to_numpy() if weights is not None else None
        return _auc(y.cast(pl.Float64).to_numpy(), self.predict(X), w)
