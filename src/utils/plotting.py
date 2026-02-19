"""Plotting helpers for research artifacts."""

from __future__ import annotations

import os
from pathlib import Path

if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = "/tmp"
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


def _finish(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path:
        path = Path(save_path)
        ensure_dir(path.parent)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _clean_series_for_plot(series: pd.Series) -> pd.Series:
    out = pd.Series(series).copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    return out.dropna()


def plot_equity_curve(equity: pd.Series, title: str, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_clean = _clean_series_for_plot(equity)
    if equity_clean.empty:
        ax.text(0.5, 0.5, "No equity data to plot", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(equity_clean.index, equity_clean.values, color="#0B7285", lw=1.8)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    _finish(fig, save_path)


def plot_drawdown(drawdown: pd.Series, title: str, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    drawdown_clean = _clean_series_for_plot(drawdown)
    if drawdown_clean.empty:
        ax.text(0.5, 0.5, "No drawdown data to plot", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(drawdown_clean.index, drawdown_clean.values, color="#C92A2A", lw=1.2)
        ax.fill_between(drawdown_clean.index, drawdown_clean.values, 0, color="#FFA8A8", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    _finish(fig, save_path)


def plot_regime_heatmap(
    heatmap_df: pd.DataFrame,
    title: str,
    save_path: str | Path | None = None,
    fmt: str = ".2f",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = heatmap_df.values.astype(float)
    im = ax.imshow(vals, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(np.arange(heatmap_df.shape[1]))
    ax.set_yticks(np.arange(heatmap_df.shape[0]))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticklabels(heatmap_df.index)
    ax.set_title(title)

    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            ax.text(j, i, format(vals[i, j], fmt), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    _finish(fig, save_path)


def plot_drift_vol_scatter(df: pd.DataFrame, title: str, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["drift"], df["vol"], c=df["net_ret"], cmap="coolwarm", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("Drift")
    ax.set_ylabel("Vol")
    fig.colorbar(scatter, ax=ax, label="Strategy Return")
    ax.grid(alpha=0.2)
    _finish(fig, save_path)
