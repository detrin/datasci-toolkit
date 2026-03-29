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

    def fit(self, features: pl.DataFrame, target: pl.Series) -> "CorrVarClus":
        random_generator = np.random.default_rng(42)
        features_np = features.to_numpy().astype(float)
        target_np = target.cast(pl.Float64).to_numpy()

        if self.sample > 0 and len(features_np) > self.sample:
            sample_indices = random_generator.choice(len(features_np), self.sample, replace=False)
            features_np, target_np = features_np[sample_indices], target_np[sample_indices]

        nonzero_std_mask = features_np.std(axis=0) > 0
        valid_columns = [c for c, k in zip(features.columns, nonzero_std_mask) if k]
        features_np = features_np[:, nonzero_std_mask]

        features_transposed = np.nan_to_num(features_np, nan=0.0).T
        linkage_matrix = linkage(features_transposed, method="average", metric="correlation")

        clusters = fcluster(linkage_matrix, 1.0 - self.max_correlation, criterion="distance")
        correlation_cutoff = 1.0 - self.max_correlation

        if self.max_clusters is not None:
            clusters_maxclust = fcluster(linkage_matrix, self.max_clusters, criterion="maxclust")
            if int(clusters_maxclust.max()) < int(clusters.max()):
                clusters = clusters_maxclust
                n_extra_clusters = min(self.max_clusters, linkage_matrix.shape[0])
                correlation_cutoff = float(linkage_matrix[-n_extra_clusters, 2])

        gini_scores = [
            float(abs(2.0 * roc_auc_score(target_np, features_np[:, i]) - 1.0))
            for i in range(len(valid_columns))
        ]

        self.features_: list[str] = valid_columns
        self.labels_: list[int] = clusters.tolist()
        self.Z_: np.ndarray = linkage_matrix
        self.corr_line_: float = correlation_cutoff
        self.cluster_table_: pl.DataFrame = (
            pl.DataFrame({"feature": valid_columns, "cluster": clusters.tolist(), "gini": gini_scores})
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
        labels = [f"{feature_name}: {cluster_id}" for feature_name, cluster_id in zip(self.features_, self.labels_)]
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
