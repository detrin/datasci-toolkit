"""Generate _site/index.html listing all exported notebooks."""
from pathlib import Path

NOTEBOOKS = [
    ("01_stability", "Stability Index — ESI & PSI"),
    ("02_grouping", "Grouping — WOETransformer & StabilityGrouping"),
    ("03_metrics", "Metrics — Gini, KS, Lift, IV, Bootstrap CI"),
    ("04_model_selection", "Model Selection — AUCStepwiseLogit"),
    ("05_label_imputation", "Label Imputation — KNNLabelImputer & TargetImputer"),
    ("06_bin_editor", "Bin Editor — Interactive & Headless Binning"),
    ("07_variable_clustering", "Variable Clustering — CorrVarClus"),
]

items = "\n".join(
    f'      <li><a href="{slug}.html">{title}</a></li>'
    for slug, title in NOTEBOOKS
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>datasci-toolkit — Examples</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.5rem; }}
    ul {{ line-height: 2; }}
    a {{ color: #0969da; }}
  </style>
</head>
<body>
  <h1>datasci-toolkit</h1>
  <p>Interactive examples for every module in the library.</p>
  <ul>
{items}
  </ul>
</body>
</html>
"""

out = Path("_site/index.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"Written {out}")
