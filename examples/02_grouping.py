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
        # Grouping — StabilityGrouping & WOETransformer

        **StabilityGrouping**: fits optimal bins constrained by temporal stability.
        Requires a train/validation split and a time series column. Bins that shift
        significantly across time periods are merged, even if that reduces IV.

        **WOETransformer**: applies pre-computed bin specifications and encodes
        features as Weight of Evidence (WOE) values. Fully sklearn-compatible —
        works inside `Pipeline`, `GridSearchCV`, `cross_val_score`.

        ## Typical workflow

        ```
        StabilityGrouping.fit(X_train, y_train, t_train, X_val, y_val, t_val)
               │
               └── .bin_specs_          ← export for audit / BinEditor
               └── .transform(X_test)   ← WOE-encoded output
        ```
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
    rng = np.random.default_rng(42)
    N = 2000

    f0 = rng.normal(0, 1, N)
    f1 = rng.normal(0, 1, N)
    cat = rng.choice(["low", "mid", "high"], N, p=[0.3, 0.4, 0.3])

    event_rate = 1 / (1 + np.exp(-(f0 + 0.5 * f1)))
    target = rng.binomial(1, event_rate).astype(float)
    months = np.repeat(np.arange(8), N // 8)

    df = pl.DataFrame({
        "f0": f0.tolist(),
        "f1": f1.tolist(),
        "target": target.tolist(),
        "month": months.tolist(),
    })
    df.head(5)
    return N, cat, df, event_rate, f0, f1, months, rng, target


@app.cell
def _(mo):
    mo.md("## StabilityGrouping")
    return


@app.cell
def _(df, pl):
    from datasci_toolkit.grouping import StabilityGrouping

    train_sg = df.filter(pl.col("month") < 5)
    val_sg   = df.filter(pl.col("month") >= 5)

    sg = StabilityGrouping(stability_threshold=0.1).fit(
        X_train=train_sg.select(["f0", "f1"]),
        y_train=train_sg["target"],
        t_train=train_sg["month"],
        X_val=val_sg.select(["f0", "f1"]),
        y_val=val_sg["target"],
        t_val=val_sg["month"],
    )
    sg
    return StabilityGrouping, sg, train_sg, val_sg


@app.cell
def _(df, mo, pl, sg):
    test_sg = df.filter(pl.col("month") >= 6)
    out_sg = sg.transform(test_sg.select(["f0", "f1"]))
    mo.md(f"Transformed shape: `{out_sg.shape}`")
    return out_sg, test_sg


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Bin specifications

        After fitting, `bin_specs_` holds the binning definition for each feature.
        These can be saved, audited, and passed into a `BinEditor` for manual
        adjustments or into `WOETransformer` for re-encoding.
        """
    )
    return


@app.cell
def _(pl, sg):
    rows = []
    for feat, spec in sg.bin_specs_.items():
        if spec["dtype"] == "float":
            cuts = [str(round(b, 4)) for b in spec["bins"] if b not in (float("-inf"), float("inf"))]
            rows.append({"feature": feat, "dtype": spec["dtype"], "cuts": ", ".join(cuts)})
        else:
            rows.append({"feature": feat, "dtype": spec["dtype"], "cuts": str(spec["bins"])})
    pl.DataFrame(rows)
    return feat, rows, spec


@app.cell
def _(mo):
    mo.md(
        r"""
        ## WOETransformer — apply saved bin specs

        `WOETransformer` encodes any DataFrame using pre-computed `bin_specs`.
        Pass the specs from `StabilityGrouping.bin_specs_` or from
        `BinEditor.accept()` after manual review.
        """
    )
    return


@app.cell
def _(sg, test_sg, train_sg):
    from datasci_toolkit.grouping import WOETransformer

    woe = WOETransformer(bin_specs=sg.bin_specs_).fit(
        train_sg.select(["f0", "f1"]),
        train_sg["target"],
    )

    train_woe = woe.transform(train_sg.select(["f0", "f1"]))
    test_woe  = woe.transform(test_sg.select(["f0", "f1"]))
    train_woe.head(5)
    return WOETransformer, test_woe, train_woe, woe


if __name__ == "__main__":
    app.run()
