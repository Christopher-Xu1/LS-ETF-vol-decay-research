"""Transaction, slippage, and borrow cost functions."""

from __future__ import annotations


def trading_cost(turnover: float, bps: float) -> float:
    return float(turnover) * float(bps) / 10_000.0


def slippage_cost(turnover: float, slippage_bps: float) -> float:
    return float(turnover) * float(slippage_bps) / 10_000.0


def borrow_cost(short_notional: float, borrow_bps_annual: float, days_per_year: int) -> float:
    daily_rate = float(borrow_bps_annual) / 10_000.0 / float(days_per_year)
    return max(float(short_notional), 0.0) * daily_rate


def financing_cost(gross_notional: float, financing_rate_annual: float, days_per_year: int) -> float:
    daily_rate = float(financing_rate_annual) / float(days_per_year)
    return max(float(gross_notional), 0.0) * daily_rate
