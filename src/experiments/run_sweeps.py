"""Parameter sweep experiments for robustness checks."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.strategy import compute_target_weights
from src.experiments import load_cfg, load_or_build_features, pair_list


SWEEP_GRID = {
    "hedge_lookback_days": [20, 60, 120],
    "hedge_method": ["leverage", "beta", "regression"],
    "rebalance_freq": ["1D", "3D", "7D"],
    "trading_bps": [0.0, 2.0, 5.0],
    "borrow_bps_annual": [0.0, 300.0, 800.0],
}


def run(config_path: str) -> pd.DataFrame:
    cfg = load_cfg(config_path)
    features_by_pair = load_or_build_features(cfg)

    results: list[dict] = []

    keys = list(SWEEP_GRID.keys())
    for values in itertools.product(*[SWEEP_GRID[k] for k in keys]):
        params = dict(zip(keys, values))

        for pair in pair_list(cfg):
            name = pair["name"]
            features = features_by_pair[name]

            strategy_cfg = dict(cfg.get("strategy", {}))
            strategy_cfg["hedge_lookback_days"] = params["hedge_lookback_days"]
            strategy_cfg["hedge_method"] = params["hedge_method"]
            strategy_cfg["rebalance_freq"] = params["rebalance_freq"]

            costs_cfg = dict(cfg.get("costs", {}))
            costs_cfg["trading_bps"] = params["trading_bps"]
            costs_cfg["borrow_bps_annual"] = params["borrow_bps_annual"]

            weights = compute_target_weights(features, pair_cfg=pair, strategy_cfg=strategy_cfg)
            bt = run_backtest(features, weights, costs_cfg=costs_cfg, asset_class=pair.get("asset_class", "equity"))

            ann = int(cfg.get("evaluation", {}).get("annualization", 252))
            m = compute_metrics(bt["net_ret"], bt["equity"], annualization=ann)

            results.append(
                {
                    "pair": name,
                    **params,
                    "sharpe": m.get("sharpe"),
                    "cagr": m.get("cagr"),
                    "max_drawdown": m.get("max_drawdown"),
                    "total_return": m.get("total_return"),
                }
            )

    df = pd.DataFrame(results)
    save_dir = Path(cfg.get("evaluation", {}).get("save_dir", "reports")) / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "sweep_results.csv", index=False)

    top = df.sort_values(["pair", "sharpe"], ascending=[True, False]).groupby("pair", as_index=False).head(10)
    top.to_csv(save_dir / "sweep_top10_by_pair.csv", index=False)

    fig_dir = Path(cfg.get("evaluation", {}).get("save_dir", "reports")) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for pair_name, sub in df.groupby("pair"):
        for param, xlabel in [
            ("borrow_bps_annual", "Borrow (annual bps)"),
            ("trading_bps", "Trading (bps)"),
            ("hedge_lookback_days", "Hedge Lookback (days)"),
        ]:
            series = sub.groupby(param)["sharpe"].mean().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            series.plot(ax=ax, marker="o", color="#0B7285")
            ax.set_title(f"{pair_name}: Avg Sharpe vs {param}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Average Sharpe")
            ax.grid(alpha=0.25)
            fig.savefig(fig_dir / f"{pair_name}_sweep_{param}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    return df
