"""Stress testing for strategy robustness.

Two stress families are produced:
1) Historical-window stress: evaluate fixed crisis windows from history.
2) Monte Carlo stress: simulate synthetic return paths via permutation or block bootstrap.

Outputs:
- `stress_scenarios.csv`
- `stress_pnl_attribution.csv`
- `stress_monte_carlo_summary.csv`
- `{pair}_mc_path_metrics.csv`
- `{pair}_mc_equity_fan.png`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.constants import LEV_RET_COL, SPOT_RET_COL
from src.experiments import load_cfg, pair_list
from src.experiments.run_baseline import run as run_baseline
from src.utils.dates import days_per_year


def _default_windows(pair_name: str) -> list[tuple[str, str, str]]:
    p = pair_name.lower()
    if "btc" in p or "bitx" in p:
        return [
            ("2022-01-01", "2022-12-31", "crypto_drawdown_2022"),
            ("2024-03-01", "2024-06-30", "crypto_vol_spike_2024"),
        ]
    return [
        ("2020-02-15", "2020-06-30", "covid_crash_2020"),
        ("2022-01-01", "2022-10-31", "growth_selloff_2022"),
    ]


def _configured_windows(cfg: dict, pair_name: str) -> list[tuple[str, str, str]]:
    stress_cfg = cfg.get("stress", {})
    scenarios = stress_cfg.get("scenarios", [])
    if not scenarios:
        return _default_windows(pair_name)

    out: list[tuple[str, str, str]] = []
    pair_lc = pair_name.lower()
    for i, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            continue

        target_pair = str(scenario.get("pair", "all")).lower()
        if target_pair not in {"all", "*", pair_lc}:
            continue

        start = scenario.get("start")
        end = scenario.get("end")
        label = scenario.get("label") or f"scenario_{i + 1}"
        if start and end:
            out.append((str(start), str(end), str(label)))

    return out or _default_windows(pair_name)


def _scenario_drawdown_and_recovery(net_ret: pd.Series) -> tuple[float, float]:
    equity = (1.0 + net_ret.fillna(0.0)).cumprod()
    if equity.empty:
        return float("nan"), float("nan")

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_drawdown = float(dd.min())

    recovery_days = _recovery_days(equity)
    return max_drawdown, recovery_days


def _recovery_days(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1
    trough = dd.idxmin()
    peak_value = running_max.loc[trough]
    post = equity.loc[trough:]
    recovered = post[post >= peak_value]
    if recovered.empty:
        return float("nan")
    return float((recovered.index[0] - trough).days)


def _mc_defaults() -> dict[str, float | int | str | bool]:
    return {
        "enabled": True,
        "method": "permutation",  # permutation | block_bootstrap
        "n_paths": 250,
        "horizon_days": 252,
        "block_size": 5,
        "seed": 42,
        "shock_scale": 1.0,
        "include_costs": True,
    }


def _mc_config(cfg: dict) -> dict[str, float | int | str | bool]:
    merged = _mc_defaults()
    override = cfg.get("stress", {}).get("monte_carlo", {})
    if isinstance(override, dict):
        merged.update(override)

    merged["enabled"] = bool(merged.get("enabled", True))
    merged["method"] = str(merged.get("method", "permutation")).lower()
    merged["n_paths"] = max(1, int(merged.get("n_paths", 250)))
    merged["horizon_days"] = max(2, int(merged.get("horizon_days", 252)))
    merged["block_size"] = max(1, int(merged.get("block_size", 5)))
    merged["seed"] = int(merged.get("seed", 42))
    merged["shock_scale"] = float(merged.get("shock_scale", 1.0))
    merged["include_costs"] = bool(merged.get("include_costs", True))
    return merged


def _sample_return_indices(
    n_obs: int,
    horizon: int,
    method: str,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample index path for Monte Carlo return generation.

    `permutation` preserves the unconditional return distribution exactly within
    each full shuffle, while destroying serial order.

    `block_bootstrap` preserves local serial structure within sampled blocks.
    """
    if n_obs < 2:
        raise ValueError("Need at least 2 observations for Monte Carlo sampling")

    if method == "permutation":
        idx: list[int] = []
        while len(idx) < horizon:
            idx.extend(rng.permutation(n_obs).tolist())
        return np.asarray(idx[:horizon], dtype=int)

    if method == "block_bootstrap":
        starts_max = max(1, n_obs - block_size + 1)
        idx = []
        while len(idx) < horizon:
            s = int(rng.integers(0, starts_max))
            block = list(range(s, min(s + block_size, n_obs)))
            idx.extend(block)
        return np.asarray(idx[:horizon], dtype=int)

    raise ValueError(f"Unsupported Monte Carlo method: {method}")


def _fan_chart(index: pd.Index, equity_paths: np.ndarray, title: str, out_path: Path) -> None:
    if equity_paths.size == 0:
        return

    q05, q25, q50, q75, q95 = np.quantile(equity_paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(index, q05, q95, color="#ADB5BD", alpha=0.35, label="5-95%")
    ax.fill_between(index, q25, q75, color="#868E96", alpha=0.4, label="25-75%")
    ax.plot(index, q50, color="#0B7285", lw=1.8, label="Median")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_historical_windows(
    cfg: dict,
    bt_by_pair: dict[str, pd.DataFrame],
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str | bool]] = []
    attribution_rows: list[dict[str, float | str]] = []

    for pair in pair_list(cfg):
        name = pair["name"]
        bt = bt_by_pair[name]
        for start, end, label in _configured_windows(cfg, name):
            window = bt.loc[(bt.index >= pd.Timestamp(start, tz="UTC")) & (bt.index <= pd.Timestamp(end, tz="UTC"))]
            if window.empty:
                continue

            window_return = float((1.0 + window["net_ret"].fillna(0.0)).prod() - 1.0)
            max_drawdown, recovery_days = _scenario_drawdown_and_recovery(window["net_ret"])
            spot_pnl_total = float(window["spot_pnl"].sum())
            lev_pnl_total = float(window["lev_pnl"].sum())
            raw_pnl_total = float(window["raw_ret"].sum())
            cost_total = float(window["cost_total"].sum())
            net_pnl_total = float(window["net_ret"].sum())
            avg_gross = float(window["gross"].mean())
            rebalance_count = int(window["rebalance"].sum()) if "rebalance" in window.columns else 0
            scheduled_count = int(window["rebalance_scheduled"].sum()) if "rebalance_scheduled" in window.columns else 0
            threshold_count = (
                int(window["rebalance_threshold_trigger"].sum()) if "rebalance_threshold_trigger" in window.columns else 0
            )
            avg_turnover = float(window["turnover"].mean()) if "turnover" in window.columns else 0.0

            rows.append(
                {
                    "pair": name,
                    "scenario": label,
                    "start": start,
                    "end": end,
                    "trading_days": int(window.shape[0]),
                    "window_return": window_return,
                    "max_drawdown": max_drawdown,
                    "recovery_days": recovery_days,
                    "avg_gross": avg_gross,
                    "avg_turnover": avg_turnover,
                    "rebalance_count": rebalance_count,
                    "scheduled_rebalance_count": scheduled_count,
                    "threshold_rebalance_count": threshold_count,
                    "spot_pnl_total": spot_pnl_total,
                    "lev_pnl_total": lev_pnl_total,
                    "raw_pnl_total": raw_pnl_total,
                    "cost_total": cost_total,
                    "net_pnl_total": net_pnl_total,
                }
            )

            attribution_rows.extend(
                [
                    {"pair": name, "scenario": label, "component": "spot_pnl", "value": spot_pnl_total},
                    {"pair": name, "scenario": label, "component": "lev_pnl", "value": lev_pnl_total},
                    {"pair": name, "scenario": label, "component": "trading_cost", "value": float(window["trading_cost"].sum())},
                    {"pair": name, "scenario": label, "component": "slippage_cost", "value": float(window["slippage_cost"].sum())},
                    {"pair": name, "scenario": label, "component": "borrow_cost", "value": float(window["borrow_cost"].sum())},
                    {"pair": name, "scenario": label, "component": "financing_cost", "value": float(window["financing_cost"].sum())},
                    {"pair": name, "scenario": label, "component": "net_pnl", "value": net_pnl_total},
                ]
            )

    scenarios = pd.DataFrame(rows)
    if not scenarios.empty:
        scenarios = scenarios.sort_values(["pair", "start", "scenario"]).reset_index(drop=True)
    scenarios.to_csv(tables_dir / "stress_scenarios.csv", index=False)

    attribution = pd.DataFrame(attribution_rows)
    if not attribution.empty:
        attribution = attribution.sort_values(["pair", "scenario", "component"]).reset_index(drop=True)
    attribution.to_csv(tables_dir / "stress_pnl_attribution.csv", index=False)

    return scenarios, attribution


def _run_monte_carlo(
    cfg: dict,
    bt_by_pair: dict[str, pd.DataFrame],
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    mc = _mc_config(cfg)
    if not mc["enabled"]:
        summary = pd.DataFrame()
        summary.to_csv(tables_dir / "stress_monte_carlo_summary.csv", index=False)
        return summary

    rng = np.random.default_rng(int(mc["seed"]))
    costs_cfg = cfg.get("costs", {})
    summary_rows: list[dict[str, float | int | str | bool]] = []

    for pair in pair_list(cfg):
        name = pair["name"]
        bt = bt_by_pair[name]

        # Strategy targets/metadata reused; return path is stress-simulated.
        needed_weight_cols = [
            "w_spot_target",
            "w_lev_target",
            "scheduled_rebalance",
            "rebalance_threshold_pct",
            "leverage_target_ratio",
        ]
        if not {"w_spot_target", "w_lev_target", SPOT_RET_COL, LEV_RET_COL}.issubset(bt.columns):
            continue

        n_obs = len(bt)
        horizon = min(int(mc["horizon_days"]), n_obs)
        if horizon < 2:
            continue

        idx = bt.index[:horizon]
        weights = bt.loc[idx, [c for c in needed_weight_cols if c in bt.columns]].copy()
        if "scheduled_rebalance" not in weights.columns:
            weights["scheduled_rebalance"] = True
        if "rebalance_threshold_pct" not in weights.columns:
            weights["rebalance_threshold_pct"] = 0.0
        if "leverage_target_ratio" not in weights.columns:
            ratio = (weights["w_lev_target"].abs() / weights["w_spot_target"].abs()).replace([np.inf, -np.inf], np.nan)
            weights["leverage_target_ratio"] = ratio.ffill().bfill()

        base_spot = bt[SPOT_RET_COL].to_numpy()
        base_lev = bt[LEV_RET_COL].to_numpy()
        path_rows = []
        equity_paths = []

        use_costs = costs_cfg if bool(mc["include_costs"]) else {
            "trading_bps": 0.0,
            "slippage_bps": 0.0,
            "borrow_bps_annual": 0.0,
            "financing_rate_annual": 0.0,
        }
        ann = days_per_year(pair.get("asset_class", "equity"))

        for path_id in range(int(mc["n_paths"])):
            sample_idx = _sample_return_indices(
                n_obs=n_obs,
                horizon=horizon,
                method=str(mc["method"]),
                block_size=int(mc["block_size"]),
                rng=rng,
            )

            spot_sim = base_spot[sample_idx] * float(mc["shock_scale"])
            lev_sim = base_lev[sample_idx] * float(mc["shock_scale"])
            sim_features = pd.DataFrame({SPOT_RET_COL: spot_sim, LEV_RET_COL: lev_sim}, index=idx)

            sim_bt = run_backtest(
                features=sim_features,
                weights=weights,
                costs_cfg=use_costs,
                asset_class=pair.get("asset_class", "equity"),
            )
            if sim_bt.empty:
                continue

            eq = sim_bt["equity"].to_numpy()
            equity_paths.append(eq)

            total_return = float(eq[-1] - 1.0)
            max_dd = float(sim_bt["drawdown"].min())
            vol = float(sim_bt["net_ret"].std() * np.sqrt(ann))
            sharpe = float(sim_bt["net_ret"].mean() / sim_bt["net_ret"].std() * np.sqrt(ann)) if sim_bt["net_ret"].std() > 0 else 0.0
            rebalance_count = int(sim_bt["rebalance"].sum()) if "rebalance" in sim_bt.columns else 0
            threshold_count = int(sim_bt["rebalance_threshold_trigger"].sum()) if "rebalance_threshold_trigger" in sim_bt.columns else 0

            path_rows.append(
                {
                    "pair": name,
                    "path_id": path_id,
                    "horizon_days": horizon,
                    "total_return": total_return,
                    "terminal_equity": float(eq[-1]),
                    "max_drawdown": max_dd,
                    "vol": vol,
                    "sharpe": sharpe,
                    "rebalance_count": rebalance_count,
                    "threshold_rebalance_count": threshold_count,
                }
            )

        path_df = pd.DataFrame(path_rows)
        path_df.to_csv(tables_dir / f"{name}_mc_path_metrics.csv", index=False)

        if path_df.empty:
            continue

        summary_rows.append(
            {
                "pair": name,
                "method": str(mc["method"]),
                "n_paths": int(path_df.shape[0]),
                "horizon_days": horizon,
                "shock_scale": float(mc["shock_scale"]),
                "include_costs": bool(mc["include_costs"]),
                "mean_total_return": float(path_df["total_return"].mean()),
                "median_total_return": float(path_df["total_return"].median()),
                "p05_total_return": float(path_df["total_return"].quantile(0.05)),
                "p95_total_return": float(path_df["total_return"].quantile(0.95)),
                "prob_loss": float((path_df["total_return"] < 0.0).mean()),
                "prob_max_dd_gt_10": float((path_df["max_drawdown"] <= -0.10).mean()),
                "prob_max_dd_gt_20": float((path_df["max_drawdown"] <= -0.20).mean()),
                "mean_max_drawdown": float(path_df["max_drawdown"].mean()),
                "mean_rebalance_count": float(path_df["rebalance_count"].mean()),
                "mean_threshold_rebalance_count": float(path_df["threshold_rebalance_count"].mean()),
            }
        )

        if equity_paths:
            arr = np.vstack(equity_paths)
            _fan_chart(
                index=idx,
                equity_paths=arr,
                title=f"{name} Monte Carlo Equity Fan ({mc['method']})",
                out_path=figures_dir / f"{name}_mc_equity_fan.png",
            )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("pair").reset_index(drop=True)
    summary.to_csv(tables_dir / "stress_monte_carlo_summary.csv", index=False)
    return summary


def run(config_path: str) -> pd.DataFrame:
    cfg = load_cfg(config_path)
    bt_by_pair = run_baseline(config_path)
    save_dir = Path(cfg.get("evaluation", {}).get("save_dir", "reports"))
    tables_dir = save_dir / "tables"
    figures_dir = save_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenarios, _ = _run_historical_windows(cfg, bt_by_pair, tables_dir=tables_dir)
    _run_monte_carlo(cfg, bt_by_pair, tables_dir=tables_dir, figures_dir=figures_dir)

    # Return historical-window table to keep notebook/CLI compatibility.
    return scenarios
