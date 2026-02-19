 

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LEV_RET_COL, SPOT_RET_COL


def _rolling_beta_lev_on_spot(spot_ret: pd.Series, lev_ret: pd.Series, window: int) -> pd.Series:
    """Rolling beta of leveraged returns regressed on spot returns."""
    cov = lev_ret.rolling(window).cov(spot_ret)
    var = spot_ret.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def _rolling_var_min_h(spot_ret: pd.Series, lev_ret: pd.Series, window: int) -> pd.Series:
    """Rolling minimum-variance hedge ratio Cov(spot, lev) / Var(lev)."""
    cov = spot_ret.rolling(window).cov(lev_ret)
    var = lev_ret.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def compute_hedge_ratio(
    features: pd.DataFrame,
    method: str,
    leverage: float,
    lookback_days: int = 60,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute hedge ratio series h_t and diagnostics.

    Parameters
    ----------
    features:
        DataFrame containing at least spot/leveraged return columns.
    method:
        One of: leverage, beta, regression.
    leverage:
        Instrument leverage multiplier (used by leverage method).
    lookback_days:
        Rolling estimation window for beta/regression diagnostics.
    """
    method = method.lower()
    spot_ret = features[SPOT_RET_COL]
    lev_ret = features[LEV_RET_COL]

    if method == "leverage":
        # Constant hedge from instrument multiplier, e.g. 1/3 for TQQQ.
        h = pd.Series(1.0 / max(leverage, 1e-9), index=features.index)
        r2 = lev_ret.rolling(lookback_days).corr(spot_ret).pow(2)
        diag = pd.DataFrame({"r2": r2, "method": method}, index=features.index)
        return h, diag

    if method == "beta":
        # Inverse beta: if lev moves ~3x spot, h tends toward ~1/3.
        beta = _rolling_beta_lev_on_spot(spot_ret, lev_ret, lookback_days)
        h = (1.0 / beta.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        r2 = lev_ret.rolling(lookback_days).corr(spot_ret).pow(2)
        diag = pd.DataFrame({"beta": beta, "r2": r2, "method": method}, index=features.index)
        return h.ffill(), diag

    if method == "regression":
        # Minimum-variance hedge for the two-leg return process.
        h = _rolling_var_min_h(spot_ret, lev_ret, lookback_days)
        corr = lev_ret.rolling(lookback_days).corr(spot_ret)
        diag = pd.DataFrame({"r2": corr.pow(2), "method": method}, index=features.index)
        return h.ffill(), diag

    raise ValueError(f"Unknown hedge ratio method: {method}")


def clamp_hedge_ratio(h: pd.Series, lower: float = 0.0, upper: float = 5.0) -> pd.Series:
    """Clamp hedge ratio to stable practical bounds.

    Forward/backward fill prevents missing h_t from blocking weight generation
    once a valid value has been estimated.
    """
    return h.replace([np.inf, -np.inf], np.nan).clip(lower=lower, upper=upper).ffill().bfill()
