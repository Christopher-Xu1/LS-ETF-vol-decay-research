# Strategy Behavior Notes

This document describes the exact portfolio construction and execution behavior implemented in the code.

## Definitions

- `w_spot_t`: spot/underlying portfolio weight on date `t`
- `w_lev_t`: leveraged ETF portfolio weight on date `t` (typically short)
- `h_t`: hedge ratio multiplier
- `gross_t = |w_spot_t| + |w_lev_t|`
- `net_t = w_spot_t + w_lev_t`

## Hedge Ratio Methods

Implemented in `src/models/hedge_ratio.py`.

- `leverage`: `h_t = 1 / L`, where `L` is instrument leverage (e.g., `3` for TQQQ)
- `beta`: `beta_t = Cov(r_lev, r_spot) / Var(r_spot)`, then `h_t = 1 / beta_t`
- `regression`: `h_t = Cov(r_spot, r_lev) / Var(r_lev)` (minimum-variance hedge)

`h_t` is clamped to stable bounds before weight construction.

## Target Weight Construction

Implemented in `src/backtest/strategy.py`.

1. Create initial spot base:
   - `w_spot_t = gross_exposure / 2`
2. Link short leg:
   - `w_lev_t = -h_t * w_spot_t`
3. Rescale both legs to hit gross target:
   - `gross_t = gross_exposure`
4. Optional directional tilt:
   - `w_spot_t += target_delta`
5. Apply hard constraints:
   - per-leg cap
   - short-leg floor
   - gross cap (`max_gross`)

## Rebalancing

`rebalance_freq` is parsed as a step count (for example, `3D` means every 3rd row).

- Scheduled trigger: on scheduled rows, trade to target weights.
- Threshold trigger: if live ratio `|w_lev / w_spot|` deviates from leverage target ratio by at least
  `rebalance_threshold_pct`, trade to target weights immediately.
- Between trigger events: no trade; weights drift with realized returns.

So hedge signals can update daily, but trades occur only on schedule or threshold events.

## Backtest Return and Cost Model

Implemented in `src/backtest/engine.py`.

- `raw_ret_t = w_spot_live,t * r_spot_t + w_lev_live,t * r_lev_t`
- `turnover_t` is only charged on trade days, measured as distance from live pre-trade weights to targets
- Costs:
  - trading and slippage on turnover
  - borrow on short notional
  - optional financing on gross notional
- `net_ret_t = raw_ret_t - cost_total_t`
- `equity_t = equity_{t-1} * (1 + net_ret_t)`

Opening day turnover is measured against implicit zero initial position, so entry costs are paid on the first tradable day.

## Stress Testing Modes

Implemented in `src/experiments/run_stress.py`.

1. Historical-window stress:
   - Slice baseline backtest into predefined crisis windows.
   - Report window return, max drawdown, recovery days, rebalance activity, and PnL attribution.

2. Monte Carlo stress:
   - Generate synthetic paired return paths from historical returns using:
     - `permutation` , or
     - `block_bootstrap`
   - Re-run full execution logic (including schedule + threshold rebalancing and costs) for each path.
   - Reports distributional risk metrics (5th/95th percentiles, loss probability, drawdown probabilities).
