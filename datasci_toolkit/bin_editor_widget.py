from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    import anywidget
    import traitlets
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from datasci_toolkit.bin_editor import BinEditor

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

_ESM = r"""
function render({ model, el }) {
  el.style.cssText = "font-family:sans-serif;font-size:13px;";

  function makeBtn(text, bg, fg) {
    const b = document.createElement("button");
    b.textContent = text;
    b.style.cssText = `padding:3px 10px;border-radius:4px;border:none;background:${bg};color:${fg || "#fff"};cursor:pointer;margin:2px;`;
    return b;
  }

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;gap:4px;align-items:center;flex-wrap:wrap;margin-bottom:6px;";

  const dd = document.createElement("select");
  dd.style.cssText = "padding:4px 8px;border-radius:4px;border:1px solid #ccc;";

  const btnUndo    = makeBtn("Undo",         "#6c757d");
  const btnReset   = makeBtn("Reset",        "#6c757d");
  const btnSuggest = makeBtn("Suggest",      "#0d6efd");
  const btnAccept  = makeBtn("✓ Accept All", "#198754");
  toolbar.append(dd, btnUndo, btnReset, btnSuggest, btnAccept);

  const imgChart = document.createElement("img");
  imgChart.style.cssText = "max-width:100%;display:block;";

  const imgStab = document.createElement("img");
  imgStab.style.cssText = "max-width:100%;display:block;margin-top:4px;";

  const splitRow = document.createElement("div");
  splitRow.style.cssText = "display:flex;gap:6px;align-items:center;margin:6px 0;";
  const splitLbl = document.createElement("span");
  splitLbl.textContent = "Split at:";
  const splitInput = document.createElement("input");
  splitInput.type = "number";
  splitInput.placeholder = "value";
  splitInput.style.cssText = "padding:4px;border-radius:4px;border:1px solid #ccc;width:110px;";
  const btnAdd = makeBtn("Add", "#0d6efd");
  splitRow.append(splitLbl, splitInput, btnAdd);

  const opsRow  = document.createElement("div");
  opsRow.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin:4px 0;";

  const suggRow = document.createElement("div");
  suggRow.style.cssText = "display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin:4px 0;";

  const msgEl = document.createElement("div");
  msgEl.style.cssText = "color:#198754;font-weight:bold;margin-top:4px;";

  el.append(toolbar, imgChart, imgStab, splitRow, opsRow, suggRow, msgEl);

  function refresh() {
    const feats  = model.get("features");
    const cur    = model.get("current_feat");
    const dtype  = model.get("feat_dtype");
    const cp     = model.get("chart_png");
    const sp     = model.get("stability_png");
    const merges = model.get("merge_labels");
    const suggs  = model.get("suggestions");

    if (dd.options.length !== feats.length) {
      dd.innerHTML = "";
      feats.forEach(f => {
        const o = document.createElement("option");
        o.value = o.textContent = f;
        dd.append(o);
      });
    }
    dd.value = cur;

    imgChart.src = cp;
    imgChart.style.display = cp ? "block" : "none";
    imgStab.src = sp;
    imgStab.style.display = sp ? "block" : "none";

    splitRow.style.display = dtype === "float" ? "flex" : "none";

    opsRow.innerHTML = "";
    if (merges.length) {
      const lbl = document.createElement("span");
      lbl.textContent = dtype === "float" ? "Remove boundary:" : "Merge groups:";
      opsRow.append(lbl);
      merges.forEach((label, idx) => {
        const b = makeBtn("✕ " + label, "#fd7e14");
        b.onclick = () => model.send({ action: "merge", bin_idx: idx });
        opsRow.append(b);
      });
    }

    suggRow.innerHTML = "";
    if (suggs.length) {
      const lbl = document.createElement("span");
      lbl.textContent = dtype === "float" ? "Suggested splits:" : "Suggested merges:";
      suggRow.append(lbl);
      suggs.forEach(s => {
        let label, act;
        if (dtype === "float") {
          label = "+ " + (+s).toPrecision(4).replace(/\.?0+$/, "");
          act   = { action: "split", value: s };
        } else {
          label = "Merge " + s[0] + "+" + s[1];
          act   = { action: "merge", bin_idx: s[0] };
        }
        const b = makeBtn(label, "#0dcaf0", "#000");
        b.onclick = () => model.send(act);
        suggRow.append(b);
      });
    }

    msgEl.textContent = model.get("message") || "";
  }

  dd.onchange        = () => model.send({ action: "set_feature", feature: dd.value });
  btnUndo.onclick    = () => model.send({ action: "undo" });
  btnReset.onclick   = () => model.send({ action: "reset" });
  btnSuggest.onclick = () => model.send({ action: "suggest" });
  btnAccept.onclick  = () => model.send({ action: "accept" });
  btnAdd.onclick     = () => {
    const v = parseFloat(splitInput.value);
    if (!isNaN(v)) model.send({ action: "split", value: v });
  };
  splitInput.addEventListener("keydown", e => { if (e.key === "Enter") btnAdd.click(); });

  model.on("change", refresh);
  refresh();
}

export default { render };
"""


def _fig_to_b64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


if _AVAILABLE:
    class BinEditorWidget(anywidget.AnyWidget):  # type: ignore[misc]
        _esm = _ESM

        features: Any     = traitlets.List().tag(sync=True)
        current_feat: Any = traitlets.Unicode("").tag(sync=True)
        feat_dtype: Any   = traitlets.Unicode("").tag(sync=True)
        chart_png: Any    = traitlets.Unicode("").tag(sync=True)
        stability_png: Any = traitlets.Unicode("").tag(sync=True)
        merge_labels: Any = traitlets.List().tag(sync=True)
        suggestions: Any  = traitlets.List().tag(sync=True)
        message: Any      = traitlets.Unicode("").tag(sync=True)

        def __init__(self, editor: BinEditor, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._ed = editor
            self.features = editor.features()
            self.current_feat = self.features[0]
            self.on_msg(self._handle_msg)
            self._sync()

        def _handle_msg(self, _: Any, content: dict[str, Any], __: Any) -> None:
            action = content.get("action")
            feat = self.current_feat
            if action == "set_feature":
                self.current_feat = content["feature"]
                self._sync()
            elif action == "split":
                self._ed.split(feat, float(content["value"]))
                self._sync()
            elif action == "merge":
                self._ed.merge(feat, int(content["bin_idx"]))
                self._sync()
            elif action == "undo":
                self._ed.undo(feat)
                self._sync()
            elif action == "reset":
                self._ed.reset(feat)
                self._sync()
            elif action == "suggest":
                raw = self._ed.suggest_splits(feat, n=5)
                state = self._ed.state(feat)
                if state["dtype"] == "float":
                    self.suggestions = [float(v) for v in raw]
                else:
                    self.suggestions = [[int(a), int(b)] for a, b in raw]
            elif action == "accept":
                self.result_ = self._ed.accept()
                self.message = f"Accepted {len(self.result_)} features."

        def _sync(self) -> None:
            state = self._ed.state(self.current_feat)
            self.feat_dtype    = state["dtype"]
            self.chart_png     = self._chart(state)
            self.stability_png = self._stability(state)
            self.merge_labels  = self._merge_labels(state)
            self.suggestions   = []
            self.message       = ""

        def _merge_labels(self, state: dict[str, Any]) -> list[str]:
            if state["dtype"] == "float":
                return [f"{s:.4g}" for s in state["splits"]]
            n_g: int = state["n_bins"]
            groups: dict[Any, list[str]] = state.get("groups", {})
            labels: list[str] = []
            for i in range(n_g - 1):
                cats_a = groups.get(i, groups.get(str(i), []))
                cats_b = groups.get(i + 1, groups.get(str(i + 1), []))
                if len(cats_a) <= 3 and len(cats_b) <= 3:
                    labels.append(f"[{','.join(cats_a)}] + [{','.join(cats_b)}]")
                else:
                    labels.append(f"{i}+{i+1}")
            return labels

        def _bin_labels(self, state: dict[str, Any]) -> list[str]:
            n = state["n_bins"]
            if state["dtype"] == "float":
                return list(state["bins"][:n])
            groups: dict[Any, list[str]] = state.get("groups", {})
            return [
                ",".join(groups.get(i, groups.get(str(i), [f"grp{i}"])))
                for i in range(n)
            ]

        def _chart(self, state: dict[str, Any]) -> str:
            n = state["n_bins"]
            labels = self._bin_labels(state)
            counts = np.array(state["counts"][:n], dtype=float)
            er = np.array([v if v is not None else 0.0 for v in state["event_rates"][:n]])
            woe = state["woe"][:n]
            total = float(counts.sum()) or 1.0
            pop = counts / total

            fig = Figure(figsize=(max(6, n * 1.4), 3.5))
            FigureCanvasAgg(fig)
            ax1 = fig.add_subplot(111)
            x_pos = np.arange(n)
            ax1.bar(x_pos, pop, color=[_PALETTE[i % len(_PALETTE)] for i in range(n)], alpha=0.65, width=0.6)
            ax1.set_ylabel("Population share", fontsize=9)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax1.set_ylim(0, float(max(pop)) * 1.35)
            ax2 = ax1.twinx()
            ax2.plot(x_pos, er, "o-", color="crimson", linewidth=2, markersize=5, zorder=5)
            ax2.set_ylabel("Event rate", color="crimson", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="crimson")
            ax2.set_ylim(0, max(float(er.max()) * 1.4, 0.01))
            for xi, w_val, p_val in zip(x_pos, woe, pop):
                ax1.text(xi, float(p_val) + float(max(pop)) * 0.02, f"{w_val:.2f}", ha="center", va="bottom", fontsize=7, color="#1a1a6e", fontweight="bold")
            ax1.set_title(f"{self.current_feat}   IV = {state['iv']:.4f}   n_bins = {n}", fontsize=10)
            fig.tight_layout()
            return _fig_to_b64(fig)

        def _stability(self, state: dict[str, Any]) -> str:
            if "temporal" not in state:
                return ""
            temp = state["temporal"]
            months = temp["months"]
            er_by_bin = temp["event_rates"]
            rsi = temp["rsi"]
            n = state["n_bins"]
            labels = self._bin_labels(state)

            fig = Figure(figsize=(max(5, len(months) * 0.8), 3.0))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            for i in range(n):
                xs = [m for m, v in zip(months, er_by_bin[i]) if v is not None]
                ys = [v for v in er_by_bin[i] if v is not None]
                if xs:
                    ax.plot(xs, ys, "o-", color=_PALETTE[i % len(_PALETTE)], linewidth=1.5, markersize=4, label=labels[i])
            ax.set_xlabel("Month", fontsize=9)
            ax.set_ylabel("Event rate", fontsize=9)
            ax.set_title(f"{self.current_feat}  stability  RSI = {rsi:.4f}", fontsize=10)
            ax.legend(fontsize=7, loc="upper right", ncol=max(1, n // 4))
            fig.tight_layout()
            return _fig_to_b64(fig)

        def show(self, feature: str | None = None) -> None:
            if feature is not None:
                self.current_feat = feature
                self._sync()
            from IPython.display import display
            display(self)

else:
    class BinEditorWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("anywidget and matplotlib are required for BinEditorWidget")
