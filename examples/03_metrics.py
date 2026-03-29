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
        # Metrics

        Standard binary classification metrics with polars-native inputs,
        sample weight support, and period-level breakdowns.

        | Function | Description |
        |---|---|
        | `gini` | 2·AUC − 1, optionally weighted |
        | `ks` | Kolmogorov–Smirnov statistic |
        | `lift` | Lift at bottom-perc% by score |
        | `iv` | Information Value for a binned predictor |
        | `BootstrapGini` | Bootstrap CI for Gini |
        | `feature_power` | Gini + IV for every column in a DataFrame |
        | `gini_by_period` | Gini per time period, with optional population mask |
        | `lift_by_period` | Lift per time period |
        | `plot_metric_by_period` | Dual-axis bar+line chart for period metrics |
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
    N = 1000

    score = rng.uniform(0, 1, N)
    y = (score + rng.normal(0, 0.25, N) > 0.5).astype(float)
    periods = pl.Series(np.repeat(np.arange(5), N // 5).tolist())

    y_s  = pl.Series(y.tolist())
    sc_s = pl.Series(score.tolist())
    return N, periods, rng, sc_s, score, y, y_s


@app.cell
def _(mo):
    mo.md("## Point-in-time metrics")
    return


@app.cell
def _(pl, sc_s, y, y_s):
    from datasci_toolkit.metrics import gini, ks, lift, iv, feature_power

    binned = (pl.Series(y.tolist()) > 0.5).cast(pl.Int32)

    pl.DataFrame({
        "metric": ["gini", "ks", "lift@10%", "iv"],
        "value": [
            round(gini(y_s, sc_s), 4),
            round(ks(y_s, sc_s), 4),
            round(lift(y_s, -sc_s, perc=10.0), 4),
            round(iv(y_s, binned), 4),
        ],
    })
    return binned, feature_power, gini, iv, ks, lift


@app.cell
def _(mo):
    mo.md("## Bootstrap Gini — confidence interval")
    return


@app.cell
def _(mo, sc_s, y_s):
    from datasci_toolkit.metrics import BootstrapGini

    bg = BootstrapGini(n_iter=300, ci_level=90.0, seed=0).fit(y_s, sc_s)

    mo.md(
        f"""
        | | Value |
        |---|---|
        | Mean Gini | `{bg.mean_:.4f}` |
        | Std | `{bg.std_:.4f}` |
        | 90% CI | `[{bg.ci_[0]:.4f}, {bg.ci_[1]:.4f}]` |
        """
    )
    return (BootstrapGini, bg)


@app.cell
def _(mo):
    mo.md("## Feature power — Gini + IV across all columns")
    return


@app.cell
def _(feature_power, np, pl, rng, y_s):
    X = pl.DataFrame({
        "strong":  (-pl.Series(rng.normal(0, 1, 1000).tolist())).to_list(),
        "medium":  (np.random.default_rng(1).normal(0, 1, 1000) * 0.5).tolist(),
        "noise":   rng.normal(0, 1, 1000).tolist(),
    })
    feature_power(X, y_s)
    return (X,)


@app.cell
def _(mo):
    mo.md("## Gini and lift by period")
    return


@app.cell
def _(periods, sc_s, y_s):
    from datasci_toolkit.metrics import gini_by_period, lift_by_period

    gini_df = gini_by_period(y_s, sc_s, periods)
    lift_df = lift_by_period(y_s, -sc_s, periods, perc=10.0)

    gini_df.join(lift_df.select(["period", "lift"]), on="period")
    return gini_by_period, gini_df, lift_by_period, lift_df


@app.cell
def _(gini_df, lift_df):
    from datasci_toolkit.metrics import plot_metric_by_period

    periods_list = gini_df["period"].to_list()
    counts       = gini_df["count"].to_list()

    plot_metric_by_period(
        periods_list,
        [gini_df["gini"].to_list(), lift_df["lift"].to_list()],
        counts,
        labels=["Gini", "Lift@10%"],
        ylabel="Score",
        title="Performance by period",
        show=False,
    )
    return (plot_metric_by_period,)


if __name__ == "__main__":
    app.run()
