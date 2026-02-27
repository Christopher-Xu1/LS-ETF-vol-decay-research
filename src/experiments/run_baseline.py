"""Baseline backtest runner for all configured pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, metrics_frame
from src.backtest.reports import write_backtest_outputs
from src.backtest.strategy import compute_target_weights
from src.experiments import load_cfg, load_or_build_features, pair_list


def run(config_path: str) -> dict[str, pd.DataFrame]:
    cfg = load_cfg(config_path)
    features_by_pair = load_or_build_features(cfg)

    metrics_rows: dict[str, dict[str, float]] = {}
    backtests: dict[str, pd.DataFrame] = {}

    for pair in pair_list(cfg):
        name = pair["name"]
        features = features_by_pair[name]
        if features.empty:
            raise ValueError(
                f"{name}: feature panel is empty. "
                "Check data availability/date overlap and lookback windows "
                "(strategy.hedge_lookback_days, regimes lookbacks)."
            )

        weights = compute_target_weights(features, pair_cfg=pair, strategy_cfg=cfg.get("strategy", {}))
        bt = run_backtest(features, weights, costs_cfg=cfg.get("costs", {}), asset_class=pair.get("asset_class", "equity"))
        if bt.empty:
            raise ValueError(
                f"{name}: backtest output is empty after joining features and weights. "
                "Verify processed features and rebalance settings."
            )
        if bt["equity"].dropna().empty:
            raise ValueError(
                f"{name}: equity series contains no finite values. "
                "Verify returns/cost inputs and weight construction."
            )

        ann = int(cfg.get("evaluation", {}).get("annualization", 252))
        metrics = compute_metrics(bt["net_ret"], bt["equity"], annualization=ann)
        if not metrics:
            raise ValueError(f"{name}: metrics are empty; insufficient valid return history for evaluation.")

        backtests[name] = bt
        metrics_rows[name] = metrics
        write_backtest_outputs(name, bt, metrics, save_dir=cfg.get("evaluation", {}).get("save_dir", "reports"))

    summary = metrics_frame(metrics_rows)
    if summary.empty:
        raise ValueError("No baseline metrics generated for any pair.")
    save_dir = Path(cfg.get("evaluation", {}).get("save_dir", "reports")) / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_dir / "baseline_metrics.csv")

    return backtests


def fetch(config_path: str) -> None:
    from src.experiments import fetch_raw_data

    cfg = load_cfg(config_path)
    fetch_raw_data(cfg)


def build(config_path: str) -> None:
    from src.experiments import build_processed_data

    cfg = load_cfg(config_path)
    build_processed_data(cfg)
