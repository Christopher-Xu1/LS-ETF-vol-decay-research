"""Theory helpers for leveraged ETF return decomposition."""

from __future__ import annotations

import numpy as np


def expected_log_leveraged(mu: float, sigma: float, leverage: float) -> float:
    """Approximate expected log return of daily reset leveraged exposure.

    Parameters are annualized GBM inputs for the underlying process:
    dS / S = mu dt + sigma dW.
    """
    return leverage * mu - 0.5 * (leverage**2) * (sigma**2)


def drag_term(sigma: float, leverage: float) -> float:
    """Volatility drag term in expected log return approximation."""
    return 0.5 * (leverage**2) * (sigma**2)


def trend_term(mu: float, leverage: float) -> float:
    """Trend contribution term in expected log return approximation."""
    return leverage * mu


def simulate_gbm_paths(
    mu: float,
    sigma: float,
    leverage: float,
    n_paths: int,
    n_days: int,
    days_per_year: int = 252,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate spot and leveraged paths from GBM daily returns.

    Parameters
    ----------
    mu, sigma:
        Annualized drift and volatility of the underlying.
    leverage:
        ETF leverage multiplier.
    n_paths, n_days:
        Simulation dimensions.
    days_per_year:
        Scaling for annualized parameters.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / days_per_year

    shocks = rng.standard_normal((n_paths, n_days))
    spot_log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    spot_ret = np.exp(spot_log_ret) - 1.0

    # Daily-reset leveraged simple returns. Clip at -99.9% to avoid invalid states.
    lev_ret = np.clip(leverage * spot_ret, -0.999, None)

    spot_paths = np.cumprod(1.0 + spot_ret, axis=1)
    lev_paths = np.cumprod(1.0 + lev_ret, axis=1)

    return {
        "spot_paths": spot_paths,
        "lev_paths": lev_paths,
        "spot_returns": spot_ret,
        "lev_returns": lev_ret,
    }


def realized_log_return(paths: np.ndarray) -> np.ndarray:
    """Return realized annualized log return per path."""
    terminal = np.maximum(paths[:, -1], 1e-12)
    n_days = paths.shape[1]
    return np.log(terminal) * (252 / n_days)
