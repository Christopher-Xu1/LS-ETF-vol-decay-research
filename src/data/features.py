"""Feature engineering for drift/vol/autocorrelation/hedge inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    FEATURE_COLUMNS,
    LEV_LOGRET_COL,
    LEV_PRICE_COL,
    LEV_RET_COL,
    SPOT_LOGRET_COL,
    SPOT_PRICE_COL,
    SPOT_RET_COL,
)
from src.utils.dates import days_per_year


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _autocorr(x: pd.Series) -> float:
        return float(x.autocorr(lag=lag))

    return series.rolling(window).apply(_autocorr, raw=False)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[SPOT_RET_COL] = out[SPOT_PRICE_COL].pct_change().fillna(0.0)
    out[LEV_RET_COL] = out[LEV_PRICE_COL].pct_change().fillna(0.0)
    out[SPOT_LOGRET_COL] = np.log1p(out[SPOT_RET_COL])
    out[LEV_LOGRET_COL] = np.log1p(out[LEV_RET_COL])
    return out


def build_features(panel: pd.DataFrame, pair_cfg: dict, cfg: dict) -> pd.DataFrame:
    regimes_cfg = cfg.get("regimes", {})
    strat_cfg = cfg.get("strategy", {})

    drift_lb = int(regimes_cfg.get("drift_lookback_days", 30))
    vol_lb = int(regimes_cfg.get("vol_lookback_days", 30))
    rho_lb = int(regimes_cfg.get("window_days", 30))
    rho_lag = int(regimes_cfg.get("autocorr_lag", 1))
    beta_lb = int(strat_cfg.get("hedge_lookback_days", 60))

    ann = days_per_year(pair_cfg.get("asset_class", "equity"))

    out = compute_returns(panel)

    out[FEATURE_COLUMNS["drift"]] = out[SPOT_RET_COL].rolling(drift_lb).mean() * ann
    out[FEATURE_COLUMNS["vol"]] = out[SPOT_RET_COL].rolling(vol_lb).std() * np.sqrt(ann)
    out[FEATURE_COLUMNS["rho"]] = _rolling_autocorr(out[SPOT_RET_COL], window=rho_lb, lag=rho_lag)
    out[FEATURE_COLUMNS["beta"]] = _rolling_beta(
        out[LEV_RET_COL],
        out[SPOT_RET_COL],
        window=beta_lb,
    )

    out = out.dropna(subset=[FEATURE_COLUMNS["drift"], FEATURE_COLUMNS["vol"], FEATURE_COLUMNS["beta"]])
    return out
