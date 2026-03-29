from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: F401
from sklearn.utils.validation import check_is_fitted  # noqa: F401


@dataclass
class AggSpec:
    variable: str
    functions: list[str]
    windows: list[str]
    table: str
    query: str | None = None


@dataclass
class TimeSinceSpec:
    variable: str
    from_: str
    unit: str
    table: str
    query: str | None = None


@dataclass
class RatioSpec:
    numerator: str
    denominator: str


_AGG_EXPRS: dict[str, Callable[[str], pl.Expr]] = {
    "sum":   lambda c: pl.col(c).sum(),
    "mean":  lambda c: pl.col(c).mean(),
    "min":   lambda c: pl.col(c).min(),
    "max":   lambda c: pl.col(c).max(),
    "count": lambda c: pl.col(c).len(),
    "std":   lambda c: pl.col(c).std(),
    "mode":  lambda c: pl.col(c).mode().first(),
}

_UNIT_SCALE: dict[str, float] = {
    "days":   1.0,
    "months": 30.4375,
    "hours":  1.0 / 24.0,
}


def _parse_window_days(w: str) -> float | None:
    if w == "inf":
        return None
    if w.endswith("mo"):
        return int(w[:-2]) * 30.4375
    if w.endswith("d"):
        return float(w[:-1])
    if w.endswith("h"):
        return int(w[:-1]) / 24.0
    raise ValueError(f"Unknown window format: {w}")


def _sanitize_query(q: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", q.lower().replace(" ", "_"))


def _agg_col_name(func: str, var: str, window: str, query: str | None) -> str:
    name = f"{func.upper()}_{var.upper()}_{window}"
    if query:
        name += f"__{_sanitize_query(query)}"
    return name


def _time_since_col_name(from_: str, var: str, unit: str) -> str:
    return f"TIME_SINCE_{from_.upper()}_{var.upper()}_{unit}"
