"""Backtest execution engine for two-leg portfolios.

The engine simulates live portfolio weights that drift with returns between
rebalances. Rebalance events can be triggered by:
- fixed schedule (`scheduled_rebalance`)
- threshold drift from leverage target ratio (`rebalance_threshold_pct`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.costs import borrow_cost, financing_cost, slippage_cost, trading_cost
from src.constants import LEV_RET_COL, SPOT_RET_COL
from src.utils.dates import days_per_year


def run_backtest(
    features: pd.DataFrame,
    weights: pd.DataFrame,
    costs_cfg: dict,
    asset_class: str,
) -> pd.DataFrame:
    """Run daily PnL simulation with costs and financing.

    Return convention:
    - `raw_ret`: weighted spot + leveraged return before costs
    - `net_ret`: raw_ret minus trading/slippage/borrow/financing costs
    """
    df = features.join(weights, how="inner").copy()
    if df.empty:
        return df

    df[[SPOT_RET_COL, LEV_RET_COL]] = df[[SPOT_RET_COL, LEV_RET_COL]].fillna(0.0)
    df[["w_spot_target", "w_lev_target"]] = df[["w_spot_target", "w_lev_target"]].ffill().fillna(0.0)

    if "scheduled_rebalance" not in df.columns:
        df["scheduled_rebalance"] = True
    if "rebalance_threshold_pct" not in df.columns:
        df["rebalance_threshold_pct"] = 0.0
    if "leverage_target_ratio" not in df.columns:
        ratio = (df["w_lev_target"].abs() / df["w_spot_target"].abs()).replace([np.inf, -np.inf], np.nan)
        df["leverage_target_ratio"] = ratio.ffill().bfill()

    df["scheduled_rebalance"] = df["scheduled_rebalance"].fillna(False).astype(bool)
    df["rebalance_threshold_pct"] = df["rebalance_threshold_pct"].fillna(0.0).astype(float).clip(lower=0.0)
    df["leverage_target_ratio"] = df["leverage_target_ratio"].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    trading_bps = float(costs_cfg.get("trading_bps", 0.0))
    borrow_bps_annual = float(costs_cfg.get("borrow_bps_annual", 0.0))
    slippage_bps = float(costs_cfg.get("slippage_bps", 0.0))
    financing_rate = float(costs_cfg.get("financing_rate_annual", 0.0))

    dpy = days_per_year(asset_class)

    # Live pre-trade weights at the start of each day; initialized flat.
    live_spot = 0.0
    live_lev = 0.0
    equity = 1.0

    rows: list[dict[str, float | bool]] = []
    for i, row in enumerate(df.itertuples()):
        target_spot = float(row.w_spot_target)
        target_lev = float(row.w_lev_target)
        spot_ret = float(getattr(row, SPOT_RET_COL))
        lev_ret = float(getattr(row, LEV_RET_COL))
        scheduled = bool(row.scheduled_rebalance)
        threshold_pct = float(row.rebalance_threshold_pct)
        target_ratio = float(row.leverage_target_ratio) if pd.notna(row.leverage_target_ratio) else np.nan

        # Drift-trigger uses live ratio before today's trade decision.
        live_ratio = np.nan
        if abs(live_spot) > 1e-12:
            live_ratio = abs(live_lev / live_spot)

        ratio_deviation = np.nan
        threshold_trigger = False
        if threshold_pct > 0.0 and np.isfinite(live_ratio) and np.isfinite(target_ratio) and target_ratio > 0:
            ratio_deviation = abs(live_ratio / target_ratio - 1.0)
            threshold_trigger = ratio_deviation >= threshold_pct

        do_rebalance = bool(i == 0 or scheduled or threshold_trigger)
        if do_rebalance:
            turnover = abs(target_spot - live_spot) + abs(target_lev - live_lev)
            live_spot = target_spot
            live_lev = target_lev
        else:
            turnover = 0.0

        # Weights used for today's return.
        exec_spot = live_spot
        exec_lev = live_lev
        spot_pnl = exec_spot * spot_ret
        lev_pnl = exec_lev * lev_ret
        raw_ret = spot_pnl + lev_pnl

        trading_c = trading_cost(turnover, trading_bps)
        slip_c = slippage_cost(turnover, slippage_bps)
        borrow_c = borrow_cost(max(-exec_lev, 0.0), borrow_bps_annual, dpy)
        financing_c = financing_cost(abs(exec_spot) + abs(exec_lev), financing_rate, dpy)
        cost_total = trading_c + slip_c + borrow_c + financing_c
        net_ret = raw_ret - cost_total

        equity *= 1.0 + net_ret

        rows.append(
            {
                "w_spot_live": exec_spot,
                "w_lev_live": exec_lev,
                "turnover": turnover,
                "rebalance": do_rebalance,
                "rebalance_scheduled": scheduled,
                "rebalance_threshold_trigger": threshold_trigger,
                "ratio_deviation_from_target": ratio_deviation,
                "spot_pnl": spot_pnl,
                "lev_pnl": lev_pnl,
                "raw_ret": raw_ret,
                "trading_cost": trading_c,
                "slippage_cost": slip_c,
                "borrow_cost": borrow_c,
                "financing_cost": financing_c,
                "cost_total": cost_total,
                "net_ret": net_ret,
                "equity": equity,
            }
        )

        # Drift weights forward to next day if no rebalance occurs overnight.
        denom = 1.0 + raw_ret
        if denom <= 1e-12:
            live_spot = 0.0
            live_lev = 0.0
        else:
            live_spot = exec_spot * (1.0 + spot_ret) / denom
            live_lev = exec_lev * (1.0 + lev_ret) / denom

    sim = pd.DataFrame(rows, index=df.index)
    for col in sim.columns:
        df[col] = sim[col]

    df["cum_raw"] = (1.0 + df["raw_ret"].fillna(0.0)).cumprod()
    running_max = df["equity"].cummax().replace(0.0, np.nan)
    df["drawdown"] = df["equity"] / running_max - 1.0
    df["gross"] = df["w_spot_live"].abs() + df["w_lev_live"].abs()
    df["net"] = df["w_spot_live"] + df["w_lev_live"]

    return df
