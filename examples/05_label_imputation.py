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
        # Label Imputation — KNNLabelImputer & TargetImputer

        Handles the **missing label problem**: a portion of your population has
        unknown outcomes and can't be used directly for supervised training.

        Common scenarios:
        - Loan applicants who were rejected — their default status is never observed
        - Holdout/control groups in A/B tests
        - Records from before a new monitoring system was deployed

        ## Workflow

        ```
        Labeled data ──► KNNLabelImputer.fit()
        Unlabeled data ─► KNNLabelImputer.predict_proba() → p̂ per unlabeled record
                        ─► KNNLabelImputer.transform()    → rows with imputed target + weight
        ```

        Then combine labeled + imputed unlabeled rows for training.
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

    # 4 features
    N_labeled   = 500
    N_unlabeled = 200

    X_lab = pl.DataFrame({f"f{i}": rng.normal(0, 1, N_labeled).tolist() for i in range(4)})
    y_lab = pl.Series(((X_lab["f0"] + X_lab["f1"]).to_numpy() > 0).astype(float).tolist())

    # Unlabeled comes from a biased region (higher f0 mean — selection bias)
    X_unl = pl.DataFrame({
        "f0": rng.normal(0.5, 1, N_unlabeled).tolist(),
        **{f"f{i}": rng.normal(0, 1, N_unlabeled).tolist() for i in range(1, 4)},
    })

    pl.concat([
        X_lab.with_columns(pl.lit("labeled").alias("split")).head(3),
        X_unl.with_columns(pl.lit("unlabeled").alias("split")).head(3),
    ])
    return N_labeled, N_unlabeled, X_lab, X_unl, rng, y_lab


@app.cell
def _(mo):
    mo.md(
        r"""
        ## KNNLabelImputer

        Finds k nearest labeled neighbors in feature space for each unlabeled
        record. Distance-weighted average of neighbor labels gives P(event).
        """
    )
    return


@app.cell
def _(X_lab, X_unl, pl, y_lab):
    from datasci_toolkit.label_imputation import KNNLabelImputer

    imputer = KNNLabelImputer(n_neighbors=10, method="weighted", metric="minkowski")
    imputer.fit(X_lab, y_lab)

    proba = imputer.predict_proba(X_unl)

    pl.DataFrame({
        "record": list(range(5)),
        "p_event": [round(float(p), 4) for p in proba[:5]],
    })
    return KNNLabelImputer, imputer, proba


@app.cell
def _(mo):
    mo.md(
        r"""
        ### `transform()` — weighted duplication

        `method="weighted"` duplicates each unlabeled record into two rows:
        one with `target=1` (weight = p̂) and one with `target=0` (weight = 1 − p̂).
        This preserves expected event count while keeping all records in training.
        """
    )
    return


@app.cell
def _(X_unl, imputer):
    imputed = imputer.transform(X_unl)
    imputed.head(6)
    return (imputed,)


@app.cell
def _(imputed, mo):
    total_weight = float(imputed["weight"].sum())
    mo.md(f"Total weight of imputed rows: `{total_weight:.1f}` (= {len(imputed) // 2} unlabeled records × 2)")
    return (total_weight,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## TargetImputer — standalone target assignment

        When you already have probabilities from another source (e.g., a
        propensity model) and just need to convert them to training rows.

        Three strategies:

        | Method | Behaviour |
        |---|---|
        | `"weighted"` | Duplicate each row: `(target=1, w=p)` + `(target=0, w=1-p)` |
        | `"randomized"` | Bernoulli draw: single row, `target ∈ {0,1}` |
        | `"cutoff"` | Hard threshold at `cutoff` (default 0.5) |
        """
    )
    return


@app.cell
def _(np, pl):
    from datasci_toolkit.label_imputation import TargetImputer

    proba_ext = np.array([0.1, 0.3, 0.55, 0.7, 0.9])

    results = {}
    for method in ("weighted", "randomized", "cutoff"):
        t = TargetImputer(method=method, seed=0).fit(proba_ext)
        results[method] = t.transform()

    pl.concat([
        v.with_columns(pl.lit(k).alias("method"))
        for k, v in results.items()
    ])
    return TargetImputer, method, proba_ext, results, t


if __name__ == "__main__":
    app.run()
