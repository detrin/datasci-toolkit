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
        # AUCStepwiseLogit — Gini-Based Feature Selection

        Stepwise logistic regression that uses **Gini (2·AUC − 1)** as the
        selection criterion instead of p-values or AIC.

        **Why Gini instead of p-values?**
        P-value stepwise selection optimises for statistical inference, not
        predictive performance. Gini-based selection directly maximises what you
        actually care about. For regulated domains it also supports:

        - `max_correlation` — rejects candidates whose pairwise correlation with
          already-selected features exceeds a threshold
        - `enforce_coef_sign` — rejects features whose coefficient flips sign when
          added (important for scorecard monotonicity)
        - `use_cv` — use cross-validated Gini instead of validation-set Gini
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
    N = 2000

    # 3 informative features + 2 correlated clones + 2 noise features
    f0 = rng.normal(0, 1, N)
    f1 = rng.normal(0, 1, N)
    f2 = rng.normal(0, 1, N)
    f0_clone = f0 + rng.normal(0, 0.1, N)   # nearly identical to f0
    f1_clone = f1 + rng.normal(0, 0.1, N)   # nearly identical to f1
    noise1 = rng.normal(0, 1, N)
    noise2 = rng.normal(0, 1, N)

    logit = 0.8 * f0 + 0.5 * f1 + 0.3 * f2
    y = (1 / (1 + np.exp(-logit)) > rng.uniform(size=N)).astype(float)

    cols = ["f0", "f1", "f2", "f0_clone", "f1_clone", "noise1", "noise2"]
    X_train = pl.DataFrame({
        "f0": f0[:1500].tolist(), "f1": f1[:1500].tolist(),
        "f2": f2[:1500].tolist(), "f0_clone": f0_clone[:1500].tolist(),
        "f1_clone": f1_clone[:1500].tolist(),
        "noise1": noise1[:1500].tolist(), "noise2": noise2[:1500].tolist(),
    })
    X_val = pl.DataFrame({
        "f0": f0[1500:].tolist(), "f1": f1[1500:].tolist(),
        "f2": f2[1500:].tolist(), "f0_clone": f0_clone[1500:].tolist(),
        "f1_clone": f1_clone[1500:].tolist(),
        "noise1": noise1[1500:].tolist(), "noise2": noise2[1500:].tolist(),
    })
    y_train = pl.Series(y[:1500].tolist())
    y_val   = pl.Series(y[1500:].tolist())

    X_train.head(3)
    return (
        N,
        X_train,
        X_val,
        cols,
        f0,
        f0_clone,
        f1,
        f1_clone,
        f2,
        logit,
        noise1,
        noise2,
        rng,
        y,
        y_train,
        y_val,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Stepwise selection with correlation filter

        `max_correlation=0.8` prevents both `f0` and `f0_clone` from being
        selected — they are nearly identical so only the stronger one enters.
        """
    )
    return


@app.cell
def _(X_train, X_val, y_train, y_val):
    from datasci_toolkit.model_selection import AUCStepwiseLogit

    model = AUCStepwiseLogit(
        selection_method="stepwise",
        min_increase=0.002,
        max_correlation=0.8,
        max_predictors=5,
    ).fit(X_train, y_train, X_val=X_val, y_val=y_val)

    model.predictors_
    return AUCStepwiseLogit, model


@app.cell
def _(model, pl):
    pl.DataFrame({
        "predictor": model.predictors_,
        "coefficient": [round(float(c), 4) for c in model.coef_],
    })
    return


@app.cell
def _(mo):
    mo.md("## Selection progress")
    return


@app.cell
def _(model):
    model.progress_.filter(model.progress_["addrm"] == 0)
    return


@app.cell
def _(X_val, mo, model, y_val):
    val_gini = model.score(X_val, y_val)
    mo.md(f"**Validation Gini**: `{val_gini:.4f}`")
    return (val_gini,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Cross-validated selection

        Pass `use_cv=True` to score via k-fold cross-validation on the training
        set rather than a fixed validation split. Slower but avoids overfitting
        the selection to the validation set.
        """
    )
    return


@app.cell
def _(AUCStepwiseLogit, X_train, y_train):
    model_cv = AUCStepwiseLogit(
        selection_method="forward",
        min_increase=0.002,
        max_correlation=0.8,
        use_cv=True,
        cv_folds=5,
        cv_seed=42,
    ).fit(X_train, y_train)

    model_cv.predictors_
    return (model_cv,)


if __name__ == "__main__":
    app.run()
