import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # CorrVarClus — Correlation-Based Variable Clustering

        Groups features into clusters by correlation structure. Within each
        cluster, features are ranked by Gini so you can pick the most predictive
        representative per cluster — useful for removing redundant predictors
        before logistic regression.

        ## Algorithm

        1. Drop zero-variance columns
        2. Hierarchical clustering with `metric="correlation"`, `method="average"`
        3. Cut the dendrogram at `max_correlation` (or `max_clusters`)
        4. Rank features within each cluster by absolute Gini (descending)

        `best_features()` returns the top-ranked feature from each cluster.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    return np, pl


@app.cell
def _(np, pl):
    rng = np.random.default_rng(0)
    N = 800

    # Group 1: f0, f1, f2 — strongly correlated
    f0 = rng.normal(0, 1, N)
    f1 = f0 + rng.normal(0, 0.2, N)
    f2 = f0 + rng.normal(0, 0.3, N)

    # Group 2: f3, f4 — correlated with each other but not with group 1
    f3 = rng.normal(0, 1, N)
    f4 = f3 + rng.normal(0, 0.2, N)

    # Noise feature
    f5 = rng.normal(0, 1, N)

    # Target: driven by f0 and f3
    logit = 1.2 * f0 + 0.8 * f3 + 0.1 * f5
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=N)).astype(float)

    X = pl.DataFrame({
        "f0": f0.tolist(), "f1": f1.tolist(), "f2": f2.tolist(),
        "f3": f3.tolist(), "f4": f4.tolist(), "f5": f5.tolist(),
    })
    y_s = pl.Series(y.tolist())

    X.head(4)
    return N, X, f0, f1, f2, f3, f4, f5, logit, rng, y, y_s


@app.cell
def _(mo):
    mo.md("## Fitting CorrVarClus")
    return


@app.cell
def _(X, y_s):
    from datasci_toolkit.variable_clustering import CorrVarClus

    cc = CorrVarClus(max_correlation=0.5).fit(X, y_s)
    cc.cluster_table_
    return CorrVarClus, cc


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Best features — one representative per cluster

        The feature with the highest absolute Gini in each cluster.
        Use these as inputs to `AUCStepwiseLogit` to avoid multicollinearity.
        """
    )
    return


@app.cell
def _(cc, mo):
    best = cc.best_features()
    mo.md(f"Best features: `{best}`")
    return (best,)


@app.cell
def _(mo):
    mo.md("## Cluster table — Gini rankings within each cluster")
    return


@app.cell
def _(cc):
    cc.cluster_table_.sort(["cluster", "rank"])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Dendrogram

        Visualises the hierarchical clustering. The horizontal dashed line shows
        the `max_correlation` cut — clusters are formed by cutting at that height.
        """
    )
    return


@app.cell
def _(cc):
    cc.plot_dendrogram(show=False)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Strict threshold — more clusters

        Lower `max_correlation` forces a finer split.
        """
    )
    return


@app.cell
def _(CorrVarClus, X, mo, y_s):
    cc_strict = CorrVarClus(max_correlation=0.2).fit(X, y_s)
    mo.md(
        f"Clusters at 0.5: `{len(cc_strict.cluster_table_['cluster'].unique())}` "
        f"| best: `{cc_strict.best_features()}`"
    )
    return (cc_strict,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Typical workflow

        ```python
        from datasci_toolkit import CorrVarClus, AUCStepwiseLogit

        cc = CorrVarClus(max_correlation=0.5).fit(X_train, y_train)
        reduced_features = cc.best_features()

        model = AUCStepwiseLogit(max_predictors=5).fit(
            X_train.select(reduced_features), y_train,
            X_val=X_val.select(reduced_features), y_val=y_val,
        )
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
