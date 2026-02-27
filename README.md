# Leveraged Decay Research

This repository backtests long-underlying/short-leveraged-ETF pair strategies to study the viability of convexity decay based strategies. It is structured around two benchmark pairs  `BTC-USD/BITX` for crypto and equitt`QQQ/TQQQ`  and evaluates results by market regime (drift, volatility, autocorrelation), costs, and stress windows.

## Thesis

Daily-reset leveraged ETFs can underperform a simple multiple of the underlying in choppy high-volatility environments. A paired long spot / short leveraged ETF position can capture that decay, but performance is highly regime-dependent and sensitive to borrow and trading costs.

## Repository Layout

- `configs/`: all experiment parameters and pair definitions
- `src/data/`: loading, cleaning, and feature engineering
- `src/models/`: hedge ratio estimation, leveraged math, regime labeling
- `src/backtest/`: weight construction, execution engine, metrics, report exports
- `src/experiments/`: orchestrated workflows (`baseline`, `regimes`, `sweeps`, `stress`)
- `docs/`: implementation notes and behavior specifications
- `reports/`: generated figures and tables
- `tests/`: unit tests for core behavior

## Installation

```bash
python -m pip install -e '.[dev,data]'
```

## CLI Workflows

```bash
python -m src.cli fetch --config configs/default.yaml
python -m src.cli build --config configs/default.yaml
python -m src.cli backtest --config configs/default.yaml
python -m src.cli regimes --config configs/default.yaml
python -m src.cli sweep --config configs/default.yaml
python -m src.cli stress --config configs/default.yaml
python -m src.cli report --config configs/default.yaml
```

Pair-specific runs:

```bash
python -m src.cli backtest --config configs/btc_bitx.yaml
python -m src.cli backtest --config configs/qqq_tqqq.yaml
```

## Strategy Mechanics

For each pair and each date:

1. Estimate hedge ratio `h_t` with one method:
   - `leverage`: fixed `h_t = 1 / L` (instrument leverage)
   - `beta`: rolling inverse beta of leveraged vs spot returns
   - `regression`: rolling minimum-variance hedge ratio
2. Construct target weights:
   - `w_lev_t = -h_t * w_spot_t`
   - rescale to target gross: `|w_spot_t| + |w_lev_t| = gross_exposure`
3. Apply optional directional tilt (`target_delta`) to spot leg.
4. Apply constraints (`max_gross`, per-leg cap, short cap).
5. Execute with:
   - scheduled rebalance cadence (`rebalance_freq`, default weekly `7D`)
   - immediate rebalance if live leg ratio drifts beyond `rebalance_threshold_pct` from leverage target (default `0.05`).

## Stress Testing

`src.cli stress` now runs two complementary stress modes:

1. Historical-window stress:
   - evaluates configured crisis windows (`stress.scenarios`) on the realized backtest.
2. Monte Carlo stress:
   - generates synthetic paired return paths with `stress.monte_carlo.method`:
     - `permutation`
     - `block_bootstrap`
   - reruns the full execution engine per path (including schedule + threshold rebalances and costs).

## Preliminary Results

With current defaults (leverage hedge, weekly rebalance, 5% threshold trigger), both pairs are positive, with materially different risk profiles:

- `qqq_tqqq`: higher consistency and risk-adjusted performance (Sharpe ~2.87, lower realized volatility and shallow drawdowns in sample).
- `btc_bitx`: higher total return with higher volatility and deeper drawdowns, resulting in lower Sharpe (~0.28).

Interpretation:
- The QQQ/TQQQ pair has been more stable under current assumptions.
- The BTC/BITX pair can perform, but path volatility is much higher and risk-adjusted returns are less reliable. Bitcoin bullish rallies are generally more parabolic than QQQ. It's not unusual to see a +10-20% move in a week which leads to massive compounded growth in the BITX leg and high asymmetric drawdown for short position.

## Important Parameter Groups

`configs/default.yaml` is the source of truth.

- `universe.pairs`: pair membership, leverage multiplier, asset class
- `strategy.hedge_method`: `leverage | beta | regression`
- `strategy.hedge_lookback_days`: rolling window for beta/regression hedge
- `strategy.rebalance_freq`: rebalance cadence (`1D`, `3D`, `7D`, ...)
- `strategy.rebalance_threshold_pct`: trigger rebalance when live leg ratio deviates from leverage target by this fraction
- `strategy.gross_exposure`: target gross before caps
- `strategy.max_gross`: hard cap on gross leverage
- `strategy.max_leverage_leg_weight`: per-leg absolute cap
- `strategy.cap_short_weight`: minimum short-leg weight (e.g. `-1.0`)
- `costs.trading_bps`, `costs.slippage_bps`, `costs.borrow_bps_annual`: key implementation frictions
- `regimes.*`: rolling windows and binning thresholds for regime maps

## Warm-Up and Start-of-Trading Behavior

The strategy does not trade until required rolling features exist. Early rows without sufficient lookback are dropped in feature construction. The first tradable row opens the initial position and incurs opening turnover costs.

## Outputs

Generated under `reports/`:

- `reports/figures/`: equity curves, drawdowns, regime heatmaps
- `reports/tables/`: backtest paths, summary metrics, sweeps, stress tables
- Monte Carlo stress artifacts:
  - `reports/tables/stress_monte_carlo_summary.csv`
  - `reports/tables/<pair>_mc_path_metrics.csv`
  - `reports/figures/<pair>_mc_equity_fan.png`
- Regime heatmap axis-key artifacts:
  - `reports/tables/<pair>_regime_heatmap_key.csv` (numeric lower/upper bounds and counts for each quartile bin)

## Reading Regime Heatmaps

- `x-axis = vol_bin`: volatility quantile bin.
- `y-axis = mu_bin`: drift quantile bin.
- Quartile labels:
  - `Q1` = lowest quartile (0-25%)
  - `Q2` = second quartile (25-50%)
  - `Q3` = third quartile (50-75%)
  - `Q4` = highest quartile (75-100%)

## Testing

```bash
python -m pytest
```

## Operational Notes

- Yahoo data can rate-limit (`YFRateLimitError`) and timeout on some runs; rerun fetch/build if needed.
- Borrow assumptions are simplified and static by default.
- Monte Carlo stress can be compute-heavy; lower `stress.monte_carlo.n_paths` for quick iteration.
- This code is for research only, not production execution.

## Disclaimers

- Not investment advice.
- Backtests are model outputs under assumptions and are not guarantees of future performance.
