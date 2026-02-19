"""Performance and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cvar(series: pd.Series, alpha: float = 0.95) -> float:
    threshold = series.quantile(1 - alpha)
    tail = series[series <= threshold]
    if tail.empty:
        return 0.0
    return float(tail.mean())


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty or equity.empty:
        return {}

    n = len(returns)
    years = n / annualization
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    vol = float(returns.std() * np.sqrt(annualization))
    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(annualization)) if not downside.empty else 0.0

    rf_daily = risk_free_rate / annualization
    excess = returns - rf_daily
    sharpe = float(excess.mean() / returns.std() * np.sqrt(annualization)) if returns.std() > 0 else 0.0
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(annualization))
        if not downside.empty and downside.std() > 0
        else 0.0
    )

    running_max = equity.cummax().replace(0.0, np.nan)
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    metrics = {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "hit_rate": float((returns > 0).mean()),
        "cvar_95": cvar(returns, alpha=0.95),
        "downside_vol": downside_vol,
        "total_return": float(equity.iloc[-1] - 1.0),
    }
    return metrics


def metrics_frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T.sort_index()
