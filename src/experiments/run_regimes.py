"""Regime analysis runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.reports import write_regime_outputs
from src.experiments import load_cfg, pair_list
from src.experiments.run_baseline import run as run_baseline
from src.models.regimes import assign_regimes, conditional_stats, pivot_regime_heatmap
from src.utils.plotting import plot_drift_vol_scatter, plot_regime_heatmap


def _q_label_definition(label: str) -> str:
    mapping = {
        "Q1": "lowest quartile (0-25%)",
        "Q2": "second quartile (25-50%)",
        "Q3": "third quartile (50-75%)",
        "Q4": "highest quartile (75-100%)",
    }
    return mapping.get(str(label), "quantile bin")


def _write_heatmap_key_tables(pair_name: str, labeled: pd.DataFrame, save_dir: str) -> None:
    tables_dir = Path(save_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    mu_key = (
        labeled.dropna(subset=["mu_bin", "drift"])
        .groupby("mu_bin", observed=False)["drift"]
        .agg(lower_bound="min", upper_bound="max", n_obs="count")
        .reset_index()
        .rename(columns={"mu_bin": "bin"})
    )
    mu_key["axis"] = "mu_bin (y-axis, drift)"
    mu_key["definition"] = mu_key["bin"].astype(str).map(_q_label_definition)

    vol_key = (
        labeled.dropna(subset=["vol_bin", "vol"])
        .groupby("vol_bin", observed=False)["vol"]
        .agg(lower_bound="min", upper_bound="max", n_obs="count")
        .reset_index()
        .rename(columns={"vol_bin": "bin"})
    )
    vol_key["axis"] = "vol_bin (x-axis, volatility)"
    vol_key["definition"] = vol_key["bin"].astype(str).map(_q_label_definition)

    key = pd.concat([mu_key, vol_key], ignore_index=True)
    key = key[["axis", "bin", "definition", "lower_bound", "upper_bound", "n_obs"]]
    key.to_csv(tables_dir / f"{pair_name}_regime_heatmap_key.csv", index=False)


def run(config_path: str) -> dict[str, pd.DataFrame]:
    cfg = load_cfg(config_path)
    bt_by_pair = run_baseline(config_path)

    out: dict[str, pd.DataFrame] = {}
    ann = int(cfg.get("evaluation", {}).get("annualization", 252))
    save_dir = cfg.get("evaluation", {}).get("save_dir", "reports")

    for pair in pair_list(cfg):
        name = pair["name"]
        bt = bt_by_pair[name].copy()
        labeled = assign_regimes(bt, cfg)

        stats_2d = conditional_stats(labeled, group_col="regime_2d", annualization=ann)
        stats_3d = conditional_stats(labeled, group_col="regime_3d", annualization=ann)
        heatmap = pivot_regime_heatmap(labeled, values_col="net_ret")

        write_regime_outputs(name, heatmap, stats_2d, save_dir=save_dir)
        _write_heatmap_key_tables(name, labeled, save_dir=save_dir)

        rho_states = (
            labeled["regime_3d"]
            .dropna()
            .str.rsplit("_", n=2)
            .str[-2:]
            .str.join("_")
            .dropna()
            .unique()
        )
        for rho_state in sorted(rho_states):
            subset = labeled[labeled["regime_3d"].str.endswith(rho_state)]
            if subset.empty:
                continue
            rho_heat = pivot_regime_heatmap(subset, values_col="net_ret")
            plot_regime_heatmap(
                rho_heat,
                title=f"{name} Regime Mean Return ({rho_state})",
                save_path=Path(save_dir) / "figures" / f"{name}_regime_heatmap_{rho_state}.png",
            )
            rho_heat.to_csv(Path(save_dir) / "tables" / f"{name}_regime_heatmap_{rho_state}.csv")

        scatter_df = labeled[["drift", "vol", "net_ret"]].dropna()
        if not scatter_df.empty:
            plot_drift_vol_scatter(
                scatter_df,
                title=f"{name} Drift vs Vol (color=net return)",
                save_path=Path(save_dir) / "figures" / f"{name}_drift_vol_scatter.png",
            )

        tables_dir = Path(save_dir) / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        labeled.to_csv(tables_dir / f"{name}_with_regimes.csv")
        stats_2d.to_csv(tables_dir / f"{name}_regime2d_stats.csv")
        stats_3d.to_csv(tables_dir / f"{name}_regime3d_stats.csv")

        out[name] = labeled

    return out
