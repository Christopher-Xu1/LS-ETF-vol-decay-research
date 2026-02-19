# Leveraged Decay Research

This repository backtests long-underlying/short-leveraged-ETF pair strategies to study convexity decay and volatility drag. It is structured around two benchmark pairs (`BTC-USD/BITX` and `QQQ/TQQQ`) and evaluates results by market regime (drift, volatility, autocorrelation), costs, and stress windows.

## Thesis

Daily-reset leveraged ETFs can underperform a simple multiple of the underlying in choppy high-volatility environments. A paired long spot / short leveraged ETF position can capture that decay, but performance is highly regime-dependent and sensitive to borrow and trading costs.

## Repository Layout

- `configs/`: all experiment parameters and pair definitions
- `src/data/`: loading, cleaning, and feature engineering
- `src/models/`: hedge ratio estimation, leveraged math, regime labeling
- `src/backtest/`: weight construction, execution engine, metrics, report exports
- `src/experiments/`: orchestrated workflows (`baseline`, `regimes`, `sweeps`, `stress`)
- `docs/`: implementation notes and behavior specifications
- `reports/`: generated figures/tables/paper snippets
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
- `reports/paper/`: markdown summary notes for writeup integration

## Testing

```bash
python -m pytest
```

## GitHub Publish Checklist

Before pushing to GitHub:

1. Run tests:
   ```bash
   python -m pytest -q
   ```
2. Run a baseline backtest:
   ```bash
   python -m src.cli backtest --config configs/default.yaml
   ```
3. Verify generated artifacts are ignored:
   - `data/raw`, `data/processed`
   - `reports/figures`, `reports/tables`
   - `notebooks/data`, `notebooks/reports`
4. Ensure docs match behavior:
   - `README.md`
   - `docs/STRATEGY_BEHAVIOR.md`
   - config defaults in `configs/default.yaml`

## Operational Notes

- Yahoo data can rate-limit (`YFRateLimitError`) and timeout on some runs; rerun fetch/build if needed.
- Borrow assumptions are simplified and static by default.
- Monte Carlo stress can be compute-heavy; lower `stress.monte_carlo.n_paths` for quick iteration.
- This code is for research only, not production execution.

## Disclaimers

- Not investment advice.
- Backtests are model outputs under assumptions and are not guarantees of future performance.
