from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self._agg_specs: list[AggSpec] = []
        self._time_since_specs: list[TimeSinceSpec] = []
        self._ratio_specs: list[RatioSpec] = []
        self._pending_meta: dict[str, str] | None = None

    def _join_tables(self, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
        result = tables[self.primary_]
        for name, tbl in tables.items():
            if name == self.primary_:
                continue
            result = result.join(tbl, on=self.entity_col_, how="left")
        return result

    def fit(
        self,
        tables: dict[str, pl.DataFrame],
        entity_col: str | None = None,
        time_col: str | None = None,
        reference_date: str | None = None,
        primary: str | None = None,
    ) -> "TemporalFeatureEngineer":
        if self._pending_meta is not None:
            meta = self._pending_meta
            entity_col = meta["entity_col"]
            time_col = meta["time_col"]
            reference_date = meta["reference_date"]
            primary = meta["primary"]
        assert entity_col is not None
        assert time_col is not None
        assert reference_date is not None
        assert primary is not None
        self.entity_col_: str = entity_col
        self.time_col_: str = time_col
        self.reference_date_: str = reference_date
        self.primary_: str = primary
        df = self._join_tables(tables)
        self.entities_ = df[self.entity_col_].unique()
        return self

    def transform(self, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
        check_is_fitted(self)
        df = self._join_tables(tables)
        df = df.with_columns(
            (
                pl.lit(self.reference_date_).str.to_date()
                - pl.col(self.time_col_).cast(pl.Date)
            )
            .dt.total_days()
            .alias("TIME_ORDER")
        )
        frames: list[pl.DataFrame] = []
        if self._agg_specs:
            frames.append(self._compute_agg(df))
        if self._time_since_specs:
            frames.append(self._compute_time_since(df))

        entity_frame = pl.DataFrame({self.entity_col_: self.entities_})
        if not frames:
            result = entity_frame
        else:
            result = frames[0]
            for f in frames[1:]:
                result = result.join(f, on=self.entity_col_, how="left")

        if self._ratio_specs:
            ratio_df = self._compute_ratio(result)
            result = result.join(ratio_df, on=self.entity_col_, how="left")

        return entity_frame.join(result, on=self.entity_col_, how="left")

    def fit_transform(  # type: ignore[override]
        self,
        tables: dict[str, pl.DataFrame],
        entity_col: str | None = None,
        time_col: str | None = None,
        reference_date: str | None = None,
        primary: str | None = None,
    ) -> pl.DataFrame:
        return self.fit(tables, entity_col, time_col, reference_date, primary).transform(tables)

    def _compute_agg(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.DataFrame({self.entity_col_: self.entities_})
        for spec in self._agg_specs:
            for window in spec.windows:
                max_days = _parse_window_days(window)
                filtered = df.filter(pl.col("TIME_ORDER") >= 0)
                if max_days is not None:
                    filtered = filtered.filter(pl.col("TIME_ORDER") <= max_days)
                if spec.query:
                    filtered = pl.SQLContext({"t": filtered}, eager=True).execute(
                        f"SELECT * FROM t WHERE {spec.query}"
                    )
                agg_exprs = [
                    _AGG_EXPRS[func](spec.variable).alias(
                        _agg_col_name(func, spec.variable, window, spec.query)
                    )
                    for func in spec.functions
                ]
                agg = filtered.group_by(self.entity_col_).agg(agg_exprs)
                result = result.join(agg, on=self.entity_col_, how="left")
        return result

    def _compute_time_since(self, df: pl.DataFrame) -> pl.DataFrame:
        result = pl.DataFrame({self.entity_col_: self.entities_})
        for spec in self._time_since_specs:
            filtered = df
            if spec.query:
                filtered = pl.SQLContext({"t": filtered}, eager=True).execute(
                    f"SELECT * FROM t WHERE {spec.query}"
                )
            descending = spec.from_ == "last"
            sorted_df = filtered.sort(self.time_col_, descending=descending)
            agg = sorted_df.group_by(self.entity_col_).agg(
                pl.col(spec.variable).first().alias("_ts")
            )
            scale = _UNIT_SCALE[spec.unit]
            col_name = _time_since_col_name(spec.from_, spec.variable, spec.unit)
            agg = agg.with_columns(
                (
                    (
                        pl.lit(self.reference_date_).str.to_date()
                        - pl.col("_ts").cast(pl.Date)
                    )
                    .dt.total_days()
                    .cast(pl.Float64)
                    / scale
                ).alias(col_name)
            ).drop("_ts")
            result = result.join(agg, on=self.entity_col_, how="left")
        return result

    def _compute_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        cols: list[str] = [self.entity_col_]
        for spec in self._ratio_specs:
            col_name = f"RATIO_{spec.numerator}__{spec.denominator}"
            df = df.with_columns(
                pl.when(
                    pl.col(spec.denominator).is_null()
                    | (pl.col(spec.denominator) == 0)
                )
                .then(pl.lit(None))
                .otherwise(pl.col(spec.numerator) / pl.col(spec.denominator))
                .alias(col_name)
            )
            cols.append(col_name)
        return df.select(cols)

    def add_aggregation(
        self,
        variable: str,
        functions: list[str],
        windows: list[str],
        table: str,
        query: str | None = None,
    ) -> "TemporalFeatureEngineer":
        self._agg_specs.append(AggSpec(variable, functions, windows, table, query))
        return self

    def add_time_since(
        self,
        variable: str,
        from_: str,
        unit: str,
        table: str,
        query: str | None = None,
    ) -> "TemporalFeatureEngineer":
        self._time_since_specs.append(TimeSinceSpec(variable, from_, unit, table, query))
        return self

    def add_ratio(self, numerator: str, denominator: str) -> "TemporalFeatureEngineer":
        self._ratio_specs.append(RatioSpec(numerator, denominator))
        return self

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TemporalFeatureEngineer":
        fe = cls()
        fe._pending_meta = cfg["meta"]
        for spec in cfg.get("aggregations", []):
            fe.add_aggregation(
                spec["variable"], spec["functions"], spec["windows"],
                spec["table"], spec.get("query"),
            )
        for spec in cfg.get("time_since", []):
            fe.add_time_since(
                spec["variable"], spec["from"], spec["unit"],
                spec["table"], spec.get("query"),
            )
        for spec in cfg.get("ratios", []):
            fe.add_ratio(spec["numerator"], spec["denominator"])
        return fe
