from __future__ import annotations

from typing import Any

import numpy as np

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from datasci_toolkit.bin_editor import BinEditor

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _require() -> None:
    if not _AVAILABLE:
        raise ImportError("ipywidgets and matplotlib are required for BinEditorWidget")


class BinEditorWidget:
    def __init__(self, editor: BinEditor) -> None:
        _require()
        self._ed = editor
        self._feat: str = editor.features()[0]
        self._suggestions: list[Any] = []
        self._build()

    def _build(self) -> None:
        W = widgets.Layout

        self._dd = widgets.Dropdown(
            options=self._ed.features(), value=self._feat,
            description="Feature:", layout=W(width="320px"),
        )
        self._dd.observe(self._on_feat, names="value")

        self._btn_undo = widgets.Button(description="Undo", layout=W(width="80px"))
        self._btn_undo.on_click(lambda _: self._do(self._ed.undo, self._feat))

        self._btn_reset = widgets.Button(description="Reset", layout=W(width="80px"))
        self._btn_reset.on_click(lambda _: self._do(self._ed.reset, self._feat))

        self._btn_suggest = widgets.Button(description="Suggest", layout=W(width="80px"))
        self._btn_suggest.on_click(self._on_suggest)

        self._btn_accept = widgets.Button(
            description="✓ Accept All", button_style="success", layout=W(width="120px"),
        )
        self._btn_accept.on_click(self._on_accept)

        self._ft_split = widgets.FloatText(description="Split at:", layout=W(width="200px"))
        self._btn_add = widgets.Button(description="Add", button_style="primary", layout=W(width="70px"))
        self._btn_add.on_click(self._on_add_split)

        self._out_chart = widgets.Output()
        self._out_stability = widgets.Output()
        self._out_ops = widgets.Output()
        self._out_sugg = widgets.Output()
        self._out_info = widgets.Output()

        top = widgets.HBox([self._dd, self._btn_undo, self._btn_reset, self._btn_suggest, self._btn_accept])
        num_row = widgets.HBox([self._ft_split, self._btn_add])

        self._layout = widgets.VBox([top, self._out_chart, self._out_stability, num_row, self._out_ops, self._out_sugg, self._out_info])
        self._render()

    def _on_feat(self, change: dict[str, Any]) -> None:
        self._feat = change["new"]
        self._suggestions = []
        self._render()

    def _do(self, fn: Any, *args: Any) -> None:
        fn(*args)
        self._suggestions = []
        self._render()

    def _on_add_split(self, _: Any) -> None:
        self._ed.split(self._feat, float(self._ft_split.value))
        self._suggestions = []
        self._render()

    def _on_suggest(self, _: Any) -> None:
        self._suggestions = self._ed.suggest_splits(self._feat, n=5)
        self._render_sugg()

    def _on_accept(self, _: Any) -> None:
        self.result_ = self._ed.accept()
        with self._out_info:
            clear_output(wait=True)
            print(f"Accepted {len(self.result_)} features. Access via widget.result_")

    def _render(self) -> None:
        self._render_chart()
        self._render_stability()
        self._render_ops()
        with self._out_sugg:
            clear_output(wait=True)
        with self._out_info:
            clear_output(wait=True)

    def _render_chart(self) -> None:
        state = self._ed.state(self._feat)
        n = state["n_bins"]
        labels = state["bins"][:n]
        counts = np.array(state["counts"][:n], dtype=float)
        er = np.array([v if v is not None else 0.0 for v in state["event_rates"][:n]])
        woe = state["woe"][:n]
        total = counts.sum() or 1.0
        pop = counts / total

        with self._out_chart:
            clear_output(wait=True)
            fig, ax1 = plt.subplots(figsize=(max(6, n * 1.4), 3.5))
            x_pos = np.arange(n)
            colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]
            ax1.bar(x_pos, pop, color=colors, alpha=0.65, width=0.6)
            ax1.set_ylabel("Population share", fontsize=9)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax1.set_ylim(0, max(pop) * 1.35)

            ax2 = ax1.twinx()
            ax2.plot(x_pos, er, "o-", color="crimson", linewidth=2, markersize=5, zorder=5)
            ax2.set_ylabel("Event rate", color="crimson", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="crimson")
            ax2.set_ylim(0, max(er.max() * 1.4, 0.01))

            for xi, w_val, p_val in zip(x_pos, woe, pop):
                ax1.text(xi, p_val + max(pop) * 0.02, f"{w_val:.2f}", ha="center", va="bottom", fontsize=7, color="#1a1a6e", fontweight="bold")

            ax1.set_title(f"{self._feat}   IV = {state['iv']:.4f}   n_bins = {n}", fontsize=10)
            fig.tight_layout()
            plt.show()

    def _render_stability(self) -> None:
        state = self._ed.state(self._feat)
        with self._out_stability:
            clear_output(wait=True)
            if "temporal" not in state:
                return
            temp = state["temporal"]
            months = temp["months"]
            er_by_bin = temp["event_rates"]
            rsi = temp["rsi"]
            n = state["n_bins"]
            labels = state["bins"][:n]

            fig, ax = plt.subplots(figsize=(max(5, len(months) * 0.8), 3.0))
            for i in range(n):
                xs = [m for m, v in zip(months, er_by_bin[i]) if v is not None]
                ys = [v for v in er_by_bin[i] if v is not None]
                if xs:
                    ax.plot(xs, ys, "o-", color=_PALETTE[i % len(_PALETTE)], linewidth=1.5, markersize=4, label=labels[i])
            ax.set_xlabel("Month", fontsize=9)
            ax.set_ylabel("Event rate", fontsize=9)
            ax.set_title(f"{self._feat}  stability  RSI = {rsi:.4f}", fontsize=10)
            ax.legend(fontsize=7, loc="upper right", ncol=max(1, n // 4))
            fig.tight_layout()
            plt.show()

    def _render_ops(self) -> None:
        state = self._ed.state(self._feat)
        with self._out_ops:
            clear_output(wait=True)
            if state["dtype"] == "float":
                splits = state["splits"]
                if splits:
                    btns = []
                    for i, s in enumerate(splits):
                        b = widgets.Button(
                            description=f"✕ {s:.4g}", button_style="warning",
                            layout=widgets.Layout(width="120px"),
                        )
                        b.on_click(lambda _, idx=i: self._do(self._ed.merge, self._feat, idx))
                        btns.append(b)
                    display(widgets.HBox([widgets.Label("Remove boundary:")] + btns))
            else:
                n_g = state["n_bins"]
                if n_g > 1:
                    btns = []
                    for i in range(n_g - 1):
                        cats_a = state["groups"].get(i, [])
                        cats_b = state["groups"].get(i + 1, [])
                        label = f"Merge {i}+{i+1}"
                        if len(cats_a) <= 3 and len(cats_b) <= 3:
                            label = f"[{','.join(cats_a)}] + [{','.join(cats_b)}]"
                        b = widgets.Button(
                            description=label, button_style="warning",
                            layout=widgets.Layout(width="max-content", min_width="120px"),
                        )
                        b.on_click(lambda _, idx=i: self._do(self._ed.merge, self._feat, idx))
                        btns.append(b)
                    display(widgets.HBox([widgets.Label("Merge groups:")] + btns, layout=widgets.Layout(flex_flow="row wrap")))

    def _render_sugg(self) -> None:
        with self._out_sugg:
            clear_output(wait=True)
            if not self._suggestions:
                print("No suggestions.")
                return
            state = self._ed.state(self._feat)
            if state["dtype"] == "float":
                btns = []
                for s in self._suggestions:
                    b = widgets.Button(
                        description=f"+ {s:.4g}", button_style="info",
                        layout=widgets.Layout(width="110px"),
                    )
                    b.on_click(lambda _, val=s: self._apply_sugg_split(val))
                    btns.append(b)
                display(widgets.HBox([widgets.Label("Suggested splits:")] + btns))
            else:
                btns = []
                for a, b_idx in self._suggestions:
                    b = widgets.Button(
                        description=f"Merge {a}+{b_idx}", button_style="info",
                        layout=widgets.Layout(width="110px"),
                    )
                    b.on_click(lambda _, idx=a: self._apply_sugg_merge(idx))
                    btns.append(b)
                display(widgets.HBox([widgets.Label("Suggested merges:")] + btns))

    def _apply_sugg_split(self, value: float) -> None:
        self._ed.split(self._feat, value)
        self._suggestions = []
        self._render()

    def _apply_sugg_merge(self, bin_idx: int) -> None:
        self._ed.merge(self._feat, bin_idx)
        self._suggestions = []
        self._render()

    def show(self, feature: str | None = None) -> None:
        if feature is not None:
            self._feat = feature
            self._dd.value = feature
            self._render()
        display(self._layout)
