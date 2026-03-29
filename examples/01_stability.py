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
        # Stability Monitoring — PSI, ESI, StabilityMonitor

        Detect feature drift over time. Two metrics:

        - **PSI** (Population Stability Index): distribution shift between a reference period and later periods.
        - **ESI** (Elena's Stability Index): rank-order stability of a categorical variable's event rates across time.
          PSI can be flat while ESI detects silent reordering drift.
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
    N_MONTHS = 12
    N_PER = 500

    months = np.repeat(np.arange(N_MONTHS), N_PER)

    # feature drifts upward after month 6
    drift = np.where(months >= 6, (months - 5) * 0.15, 0.0)
    feature = rng.normal(drift, 1.0, len(months))

    # categorical feature: category ordering becomes unstable after month 6
    cat_raw = rng.choice(["A", "B", "C", "D"], len(months))
    base_rate = {"A": 0.05, "B": 0.15, "C": 0.25, "D": 0.35}
    flip_rate = {"A": 0.35, "B": 0.25, "C": 0.15, "D": 0.05}
    event_rate = np.array([
        flip_rate[c] if m >= 6 else base_rate[c]
        for c, m in zip(cat_raw, months)
    ])
    target = rng.binomial(1, event_rate).astype(float)

    df = pl.DataFrame({
        "month": months.tolist(),
        "feature": feature.tolist(),
        "cat": cat_raw.tolist(),
        "target": target.tolist(),
        "base": np.ones(len(months), dtype=int).tolist(),
    })
    df
    return N_MONTHS, N_PER, base_rate, cat_raw, df, drift, event_rate, feature, flip_rate, months, rng, target


@app.cell
def _(mo):
    mo.md("## PSI — fit on months 0–2, score months 3–11")
    return


@app.cell
def _(df, pl):
    from datasci_toolkit.stability import PSI

    ref = df.filter(pl.col("month") < 3)["feature"]
    psi = PSI(q=10).fit(ref)

    records = []
    for m in range(3, 12):
        subset = df.filter(pl.col("month") == m)["feature"]
        records.append({"month": m, "psi": round(psi.score(subset), 4)})

    psi_df = pl.DataFrame(records)
    psi_df
    return PSI, m, psi, psi_df, records, ref, subset


@app.cell
def _(df, pl, psi_df):
    from datasci_toolkit.stability import plot_psi_comparison

    months_axis = psi_df["month"].to_list()
    values = [psi_df["psi"].to_list()]
    plot_psi_comparison(months_axis, values, labels=["feature"], title="PSI over time", show=False)
    return (plot_psi_comparison,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## StabilityMonitor — score all features across all months

        Fits on a reference period, scores every subsequent month in one call.
        """
    )
    return


@app.cell
def _(df, pl):
    from datasci_toolkit.stability import StabilityMonitor

    ref_mask = pl.col("month") < 3
    monitor = StabilityMonitor(features=["feature"]).fit(df.filter(ref_mask))
    stability_scores = monitor.score(df.filter(~ref_mask), col_month="month")
    stability_scores.sort("month")
    return StabilityMonitor, monitor, ref_mask, stability_scores


@app.cell
def _(mo):
    mo.md("## ESI — rank-order stability of a categorical variable")
    return


@app.cell
def _(df):
    from datasci_toolkit.stability import ESI

    esi = ESI()
    result = esi.score(
        df,
        var="cat",
        col_target="target",
        col_base="base",
        col_month="month",
    )
    result
    return ESI, esi, result


@app.cell
def _(mo, result):
    mo.md(
        f"""
        **ESI v1** (max-rank ratio): `{result['v1']:.3f}`
        **ESI v2** (rank-product mean): `{result['v2']:.3f}`

        Values near 1.0 = stable ordering. After month 6 the "A"/"D" categories
        swap rank, so both metrics drop well below 1.
        """
    )
    return


if __name__ == "__main__":
    app.run()
