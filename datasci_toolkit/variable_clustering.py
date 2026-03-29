from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted


class CorrVarClus(BaseEstimator):
    """Hierarchical correlation clustering for variable reduction.

    Groups features into clusters using average-linkage hierarchical clustering
    with a correlation distance metric. Ranks features within each cluster by
    absolute Gini so the most predictive representative can be selected.

    Args:
        max_correlation: Dendrogram cut height. Features correlated above this
            threshold end up in the same cluster.
        max_clusters: Hard cap on number of clusters. Overrides
            ``max_correlation`` when set.
        sample: Subsample rows before clustering for speed on large datasets.
            ``0`` uses all rows.

    Attributes:
        features_: Column names after dropping zero-variance columns.
        labels_: Cluster label per feature (1-indexed).
        Z_: Linkage matrix from ``scipy.cluster.hierarchy.linkage``.
        corr_line_: The correlation threshold used to cut the dendrogram.
        cluster_table_: DataFrame with columns ``feature``, ``cluster``,
            ``gini``, ``rank`` (1 = best in cluster).
    """

    def __init__(
        self,
        max_correlation: float = 0.5,
        max_clusters: int | None = None,
        sample: int = 0,
    ) -> None:
        self.max_correlation = max_correlation
        self.max_clusters = max_clusters
        self.sample = sample

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "CorrVarClus":
        rng = np.random.default_rng(42)
        X_np = X.to_numpy().astype(float)
        y_np = y.cast(pl.Float64).to_numpy()

        if self.sample > 0 and len(X_np) > self.sample:
            idx = rng.choice(len(X_np), self.sample, replace=False)
            X_np, y_np = X_np[idx], y_np[idx]

        keep = X_np.std(axis=0) > 0
        cols = [c for c, k in zip(X.columns, keep) if k]
        X_np = X_np[:, keep]

        X_t = np.nan_to_num(X_np, nan=0.0).T
        Z = linkage(X_t, method="average", metric="correlation")

        clusters = fcluster(Z, 1.0 - self.max_correlation, criterion="distance")
        corr_line = 1.0 - self.max_correlation

        if self.max_clusters is not None:
            clusters_max = fcluster(Z, self.max_clusters, criterion="maxclust")
            if int(clusters_max.max()) < int(clusters.max()):
                clusters = clusters_max
                n = min(self.max_clusters, Z.shape[0])
                corr_line = float(Z[-n, 2])

        ginis = [
            float(abs(2.0 * roc_auc_score(y_np, X_np[:, i]) - 1.0))
            for i in range(len(cols))
        ]

        self.features_: list[str] = cols
        self.labels_: list[int] = clusters.tolist()
        self.Z_: np.ndarray = Z
        self.corr_line_: float = corr_line
        self.cluster_table_: pl.DataFrame = (
            pl.DataFrame({"feature": cols, "cluster": clusters.tolist(), "gini": ginis})
            .with_columns(
                pl.col("gini").rank(method="ordinal", descending=True).over("cluster").alias("rank")
            )
            .sort(["cluster", "gini"], descending=[False, True])
        )
        return self

    def best_features(self) -> list[str]:
        check_is_fitted(self)
        return self.cluster_table_.filter(pl.col("rank") == 1)["feature"].to_list()

    def plot_dendrogram(self, *, output_file: str | None = None, show: bool = True) -> None:
        check_is_fitted(self)
        n = len(self.labels_)
        fig, ax = plt.subplots(figsize=(10, max(4, n // 4)))
        labels = [f"{f}: {c}" for f, c in zip(self.features_, self.labels_)]
        dendrogram(self.Z_, labels=labels, orientation="right", ax=ax)
        ax.axvline(x=self.corr_line_, color="black", linestyle="--")
        ax.set_xlabel("Correlation distance")
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"])
        fig.tight_layout()
        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close(fig)
