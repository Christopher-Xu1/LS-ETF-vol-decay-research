"""Report generation utilities for figures, tables, and paper summary blocks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.plotting import plot_drawdown, plot_equity_curve, plot_regime_heatmap


def write_backtest_outputs(
    pair_name: str,
    bt_df: pd.DataFrame,
    metrics: dict[str, float],
    save_dir: str = "reports",
) -> None:
    figures = ensure_dir(Path(save_dir) / "figures")
    tables = ensure_dir(Path(save_dir) / "tables")
    required_cols = {"equity", "drawdown"}
    missing = required_cols.difference(bt_df.columns)
    if missing:
        raise ValueError(f"{pair_name}: missing required backtest columns for reporting: {sorted(missing)}")

    plot_equity_curve(bt_df["equity"], f"{pair_name} Equity Curve", figures / f"{pair_name}_equity.png")
    plot_drawdown(bt_df["drawdown"], f"{pair_name} Drawdown", figures / f"{pair_name}_drawdown.png")

    bt_df.to_csv(tables / f"{pair_name}_backtest.csv", index=True)
    pd.DataFrame([metrics], index=[pair_name]).to_csv(tables / f"{pair_name}_metrics.csv", index=True)


def write_regime_outputs(
    pair_name: str,
    heatmap_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    save_dir: str = "reports",
) -> None:
    figures = ensure_dir(Path(save_dir) / "figures")
    tables = ensure_dir(Path(save_dir) / "tables")

    plot_regime_heatmap(
        heatmap_df,
        title=f"{pair_name} Regime Mean Return",
        save_path=figures / f"{pair_name}_regime_heatmap.png",
    )

    heatmap_df.to_csv(tables / f"{pair_name}_regime_heatmap.csv")
    stats_df.to_csv(tables / f"{pair_name}_regime_stats.csv")


def append_paper_summary(text: str, save_dir: str = "reports") -> None:
    paper_dir = ensure_dir(Path(save_dir) / "paper")
    path = paper_dir / "paper.md"

    if not path.exists():
        path.write_text("# Leveraged Decay Research Notes\n\n", encoding="utf-8")

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(text.strip())
        handle.write("\n")
