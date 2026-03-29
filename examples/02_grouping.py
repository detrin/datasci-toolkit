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
        # Grouping — WOETransformer & StabilityGrouping

        **WOETransformer**: fits bin specifications on training data and encodes
        features as Weight of Evidence (WOE) values. Fully sklearn-compatible —
        works inside `Pipeline`, `GridSearchCV`, `cross_val_score`.

        **StabilityGrouping**: extends optimal binning with a temporal stability
        constraint. Bins that are unstable across time periods are merged, even if
        that reduces IV. Use when model performance in production matters more than
        in-sample IV.
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

    # Two informative numeric features + one categorical
    f0 = rng.normal(0, 1, N)
    f1 = rng.normal(0, 1, N)
    cat = rng.choice(["low", "mid", "high"], N, p=[0.3, 0.4, 0.3])

    event_rate = 1 / (1 + np.exp(-(f0 + 0.5 * f1)))
    target = rng.binomial(1, event_rate).astype(float)
    months = np.repeat(np.arange(8), N // 8)

    df = pl.DataFrame({
        "f0": f0.tolist(),
        "f1": f1.tolist(),
        "cat": cat.tolist(),
        "target": target.tolist(),
        "month": months.tolist(),
    })
    df.head(5)
    return N, cat, df, event_rate, f0, f1, months, rng, target


@app.cell
def _(mo):
    mo.md("## WOETransformer")
    return


@app.cell
def _(df, pl):
    from datasci_toolkit.grouping import WOETransformer

    train = df.filter(pl.col("month") < 6)
    test  = df.filter(pl.col("month") >= 6)

    woe = WOETransformer(features=["f0", "f1", "cat"]).fit(
        train.select(["f0", "f1", "cat"]),
        train["target"],
    )
    woe
    return WOETransformer, test, train, woe


@app.cell
def _(test, train, woe):
    train_woe = woe.transform(train.select(["f0", "f1", "cat"]))
    test_woe  = woe.transform(test.select(["f0", "f1", "cat"]))
    train_woe.head(5)
    return test_woe, train_woe


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Bin specifications

        After fitting, `bin_specs` holds the binning definition for each feature.
        These can be saved, audited, and passed into a `BinEditor` for manual
        adjustments.
        """
    )
    return


@app.cell
def _(pl, woe):
    rows = []
    for feat, spec in woe.bin_specs.items():
        if spec["dtype"] == "float":
            for i, (woe_val, er) in enumerate(zip(spec["woe"], spec["event_rates"])):
                rows.append({
                    "feature": feat,
                    "bin": i,
                    "woe": round(float(woe_val), 4),
                    "event_rate": round(float(er), 4) if er is not None else None,
                })
    pl.DataFrame(rows)
    return er, feat, i, rows, spec, woe_val


@app.cell
def _(mo):
    mo.md(
        r"""
        ## StabilityGrouping

        Same interface as `WOETransformer` but bins are constrained by temporal
        stability. Internally runs optimal binning, then merges any bin whose
        event rate shifts significantly across months (measured by RSI).
        """
    )
    return


@app.cell
def _(df, pl):
    from datasci_toolkit.grouping import StabilityGrouping

    train_sg = df.filter(pl.col("month") < 6)

    sg = StabilityGrouping(
        features=["f0", "f1"],
        stability_threshold=0.1,
    ).fit(
        train_sg.select(["f0", "f1"]),
        train_sg["target"],
        t=train_sg["month"],
    )
    sg
    return StabilityGrouping, sg, train_sg


@app.cell
def _(df, mo, pl, sg):
    test_sg = df.filter(pl.col("month") >= 6)
    out_sg = sg.transform(test_sg.select(["f0", "f1"]))

    mo.md(f"Transformed shape: `{out_sg.shape}` — {sg.features} → WOE-encoded columns")
    return out_sg, test_sg


if __name__ == "__main__":
    app.run()
