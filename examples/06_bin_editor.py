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
        # BinEditor — Interactive & Headless Binning

        `BinEditor` is a headless state machine for editing bin boundaries.
        It works identically in plain Python scripts, notebooks, and agents.

        `BinEditorWidget` wraps it in an anywidget UI that runs in JupyterLab,
        VS Code notebooks, and Marimo — no backend changes needed.

        ## Architecture

        ```
        WOETransformer.fit() → bin_specs dict
                                     │
                              BinEditor(bin_specs, X, y)   ← headless, testable
                                     │
                           BinEditorWidget(editor)          ← UI layer only
        ```

        The split between editor and widget means agents can call the editor
        directly without any display machinery.
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

    x_num = rng.normal(0, 1, N)
    cats  = rng.choice(["A", "B", "C", "D"], N)
    y     = ((x_num > 0) | (np.isin(cats, ["A", "B"]))).astype(float) * rng.binomial(1, 0.8, N)

    N_MONTHS = 6
    months = np.repeat(np.arange(N_MONTHS), N // N_MONTHS)

    X = pl.DataFrame({"num": x_num.tolist(), "cat": cats.tolist()})
    y_s = pl.Series(y.tolist())
    t_s = pl.Series(months.tolist())

    X.head(4)
    return N, N_MONTHS, X, cats, months, rng, t_s, x_num, y, y_s


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Setting up BinEditor

        `bin_specs` is the dict produced by `WOETransformer.fit()` or
        `StabilityGrouping.fit()`. You can also build it manually.

        For numeric features `bins` is `[-inf, split1, split2, ..., inf]`.
        For categorical features `bins` is a `{category: group_index}` dict.
        """
    )
    return


@app.cell
def _(X, np, t_s, y_s):
    from datasci_toolkit.bin_editor import BinEditor

    bin_specs = {
        "num": {"dtype": "float",    "bins": [-np.inf, -1.0, 0.0, 1.0, np.inf]},
        "cat": {"dtype": "category", "bins": {"A": 0, "B": 0, "C": 1, "D": 2}},
    }

    editor = BinEditor(bin_specs, X, y_s, t=t_s, stability_threshold=0.1)
    editor.features()
    return BinEditor, bin_specs, editor


@app.cell
def _(mo):
    mo.md("## Inspecting state — IV, WOE, event rates")
    return


@app.cell
def _(editor, pl):
    state = editor.state("num")

    pl.DataFrame({
        "bin":        state["bins"][:state["n_bins"]],
        "count":      [round(c) for c in state["counts"][:state["n_bins"]]],
        "event_rate": [round(e, 4) if e is not None else None for e in state["event_rates"][:state["n_bins"]]],
        "woe":        [round(w, 4) for w in state["woe"][:state["n_bins"]]],
    })
    return (state,)


@app.cell
def _(editor, mo, state):
    mo.md(f"**IV** = `{state['iv']:.4f}`  |  **RSI** = `{state['temporal']['rsi']:.4f}`")
    return


@app.cell
def _(mo):
    mo.md("## Editing: split, merge, undo, reset")
    return


@app.cell
def _(editor, pl):
    # Split at 0.5 — adds a new boundary
    s1 = editor.split("num", 0.5)

    pl.DataFrame({
        "bin": s1["bins"][:s1["n_bins"]],
        "woe": [round(w, 4) for w in s1["woe"][:s1["n_bins"]]],
        "iv":  [round(s1["iv"], 4)] * s1["n_bins"],
    })
    return (s1,)


@app.cell
def _(editor, mo):
    # Undo the last split
    s2 = editor.undo("num")
    mo.md(f"After undo — n_bins: `{s2['n_bins']}`, IV: `{s2['iv']:.4f}`")
    return (s2,)


@app.cell
def _(mo):
    mo.md("## Suggested splits — IV-ranked candidates")
    return


@app.cell
def _(editor):
    suggestions = editor.suggest_splits("num", n=5)
    suggestions
    return (suggestions,)


@app.cell
def _(mo):
    mo.md("## Accepting — export final bin specs")
    return


@app.cell
def _(editor):
    # Apply a suggestion and accept
    editor.split("num", suggestions[0])
    final_specs = editor.accept()
    final_specs
    return (final_specs,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## BinEditorWidget — interactive UI

        ```python
        from datasci_toolkit import BinEditorWidget

        widget = BinEditorWidget(editor)
        widget.show()
        ```

        The widget renders in JupyterLab, VS Code notebooks, and Marimo.
        Use the dropdown to switch features, the toolbar to undo/reset/suggest,
        and the merge buttons to remove boundaries. After editing, call
        `widget.result_` to get the accepted bin specs.

        The widget is purely a rendering shell — all state lives in `BinEditor`
        and is fully accessible without the widget.
        """
    )
    return


if __name__ == "__main__":
    app.run()
