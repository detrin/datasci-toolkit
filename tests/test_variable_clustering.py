import numpy as np
import polars as pl
import pytest

from datasci_toolkit.variable_clustering import CorrVarClus

RNG = np.random.default_rng(0)

N = 300

# Two correlated groups: (f0, f1, f2) are correlated; (f3, f4) are correlated; f5 is noise
_BASE = RNG.normal(0, 1, N)
_BASE2 = RNG.normal(0, 1, N)
_X = pl.DataFrame({
    "f0": _BASE + RNG.normal(0, 0.1, N),
    "f1": _BASE + RNG.normal(0, 0.1, N),
    "f2": _BASE + RNG.normal(0, 0.2, N),
    "f3": _BASE2 + RNG.normal(0, 0.1, N),
    "f4": _BASE2 + RNG.normal(0, 0.1, N),
    "f5": RNG.normal(0, 1, N),
})
_Y = pl.Series((_BASE > 0).astype(float).tolist())


# --- fit ---

def test_fit_returns_self() -> None:
    cc = CorrVarClus()
    assert cc.fit(_X, _Y) is cc



def test_features_match_input_columns() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    assert set(cc.features_) == set(_X.columns)


def test_labels_length_matches_features() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    assert len(cc.labels_) == len(cc.features_)


def test_cluster_table_has_correct_columns() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    assert set(cc.cluster_table_.columns) == {"feature", "cluster", "gini", "rank"}


def test_cluster_table_rows_match_features() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    assert len(cc.cluster_table_) == len(cc.features_)


def test_gini_values_in_range() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    ginis = cc.cluster_table_["gini"].to_list()
    assert all(0.0 <= g <= 1.0 for g in ginis)


# --- clustering structure ---

def test_correlated_features_share_cluster() -> None:
    cc = CorrVarClus(max_correlation=0.5).fit(_X, _Y)
    tbl = cc.cluster_table_
    c0 = tbl.filter(pl.col("feature") == "f0")["cluster"][0]
    c1 = tbl.filter(pl.col("feature") == "f1")["cluster"][0]
    assert c0 == c1


def test_uncorrelated_groups_in_different_clusters() -> None:
    cc = CorrVarClus(max_correlation=0.5).fit(_X, _Y)
    tbl = cc.cluster_table_
    c0 = tbl.filter(pl.col("feature") == "f0")["cluster"][0]
    c3 = tbl.filter(pl.col("feature") == "f3")["cluster"][0]
    assert c0 != c3


def test_max_clusters_limits_cluster_count() -> None:
    cc = CorrVarClus(max_clusters=2).fit(_X, _Y)
    assert len(set(cc.labels_)) <= 2


def test_strict_max_correlation_produces_more_clusters() -> None:
    cc_strict = CorrVarClus(max_correlation=0.1).fit(_X, _Y)
    cc_loose = CorrVarClus(max_correlation=0.9).fit(_X, _Y)
    assert len(set(cc_strict.labels_)) >= len(set(cc_loose.labels_))


# --- best_features ---

def test_best_features_one_per_cluster() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    n_clusters = len(set(cc.labels_))
    assert len(cc.best_features()) == n_clusters


def test_best_features_are_subset_of_all_features() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    assert set(cc.best_features()).issubset(set(cc.features_))


def test_best_features_have_rank_one() -> None:
    cc = CorrVarClus().fit(_X, _Y)
    best = set(cc.best_features())
    rank_ones = set(cc.cluster_table_.filter(pl.col("rank") == 1)["feature"].to_list())
    assert best == rank_ones


def test_best_feature_has_highest_gini_in_cluster() -> None:
    cc = CorrVarClus(max_clusters=2).fit(_X, _Y)
    tbl = cc.cluster_table_
    for cluster_id in tbl["cluster"].unique().to_list():
        sub = tbl.filter(pl.col("cluster") == cluster_id)
        best_gini = sub.filter(pl.col("rank") == 1)["gini"][0]
        assert best_gini == sub["gini"].max()


# --- constant column handling ---

def test_constant_column_dropped() -> None:
    X_with_const = _X.with_columns(pl.lit(1.0).alias("const"))
    cc = CorrVarClus().fit(X_with_const, _Y)
    assert "const" not in cc.features_


def test_non_constant_columns_retained() -> None:
    X_with_const = _X.with_columns(pl.lit(1.0).alias("const"))
    cc = CorrVarClus().fit(X_with_const, _Y)
    assert len(cc.features_) == len(_X.columns)


# --- sampling ---

def test_sample_param_runs_without_error() -> None:
    cc = CorrVarClus(sample=100).fit(_X, _Y)
    assert len(cc.features_) > 0


def test_sample_larger_than_data_uses_full_data() -> None:
    cc_full = CorrVarClus(sample=0).fit(_X, _Y)
    cc_capped = CorrVarClus(sample=10000).fit(_X, _Y)
    assert set(cc_full.features_) == set(cc_capped.features_)


# --- unfitted guard ---

def test_best_features_unfitted_raises() -> None:
    with pytest.raises(Exception):
        CorrVarClus().best_features()


def test_plot_dendrogram_unfitted_raises() -> None:
    with pytest.raises(Exception):
        CorrVarClus().plot_dendrogram(show=False)


# --- plot_dendrogram (smoke test, no display) ---

def test_plot_dendrogram_runs_without_error(tmp_path) -> None:
    cc = CorrVarClus().fit(_X, _Y)
    cc.plot_dendrogram(show=False, output_file=str(tmp_path / "dendro.png"))
    assert (tmp_path / "dendro.png").exists()
