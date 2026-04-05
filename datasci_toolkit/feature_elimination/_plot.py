from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure


def plot_shap_elimination(report: pl.DataFrame, show: bool = True) -> Figure:
    n_features = report["n_features"].to_list()
    train_mean = report["train_score_mean"].to_numpy()
    train_std = report["train_score_std"].to_numpy()
    val_mean = report["val_score_mean"].to_numpy()
    val_std = report["val_score_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_features, train_mean, label="Train")
    ax.fill_between(n_features, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(n_features, val_mean, label="Validation")
    ax.fill_between(n_features, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Score")
    ax.set_title("SHAP Backward Feature Elimination")
    ax.invert_xaxis()
    ax.legend()

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
