"""Project-wide constants and naming conventions."""

from __future__ import annotations

ANNUALIZATION_BY_ASSET_CLASS = {
    "equity": 252,
    "crypto": 365,
}

DEFAULT_ANNUALIZATION = 252

PRICE_COL = "adj_close"
DATE_COL = "date"

RET_COL = "ret"
LOGRET_COL = "logret"

SPOT_PREFIX = "spot"
LEV_PREFIX = "lev"

SPOT_PRICE_COL = f"{SPOT_PREFIX}_{PRICE_COL}"
LEV_PRICE_COL = f"{LEV_PREFIX}_{PRICE_COL}"
SPOT_RET_COL = f"{SPOT_PREFIX}_{RET_COL}"
LEV_RET_COL = f"{LEV_PREFIX}_{RET_COL}"
SPOT_LOGRET_COL = f"{SPOT_PREFIX}_{LOGRET_COL}"
LEV_LOGRET_COL = f"{LEV_PREFIX}_{LOGRET_COL}"

FEATURE_COLUMNS = {
    "drift": "drift",
    "vol": "vol",
    "rho": "rho",
    "beta": "beta",
}

REBALANCE_DEFAULT = "1D"

PAIR_COLUMNS = [
    DATE_COL,
    SPOT_PRICE_COL,
    LEV_PRICE_COL,
]
