"""Data cleaning and alignment for pair-level analysis."""

from __future__ import annotations

import pandas as pd

from src.constants import DATE_COL, LEV_PRICE_COL, PRICE_COL, SPOT_PRICE_COL
from src.utils.dates import align_frames, to_trading_days_index


def validate_prices(df: pd.DataFrame, ticker: str) -> None:
    if PRICE_COL not in df.columns:
        raise ValueError(f"{ticker} missing required column: {PRICE_COL}")
    if (df[PRICE_COL] <= 0).any():
        raise ValueError(f"{ticker} contains non-positive prices")


def clean_price_frame(df: pd.DataFrame, asset_class: str, fill_method: str = "ffill") -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], utc=True)
    out = out.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL)
    out = to_trading_days_index(out, asset_class=asset_class)
    out[PRICE_COL] = out[PRICE_COL].astype(float)
    if fill_method:
        method = fill_method.lower()
        if method == "ffill":
            out[PRICE_COL] = out[PRICE_COL].ffill()
        elif method == "bfill":
            out[PRICE_COL] = out[PRICE_COL].bfill()
        else:
            out[PRICE_COL] = out[PRICE_COL].fillna(method=fill_method)
    return out


def align_pair_data(
    spot: pd.DataFrame,
    lev: pd.DataFrame,
    asset_class: str,
    fill_method: str = "ffill",
    drop_missing: bool = True,
    how: str = "intersection",
) -> pd.DataFrame:
    spot_clean = clean_price_frame(spot, asset_class=asset_class, fill_method=fill_method)[[PRICE_COL]]
    lev_clean = clean_price_frame(lev, asset_class=asset_class, fill_method=fill_method)[[PRICE_COL]]

    spot_aligned, lev_aligned = align_frames([spot_clean, lev_clean], how=how)

    out = pd.concat(
        [
            spot_aligned.rename(columns={PRICE_COL: SPOT_PRICE_COL}),
            lev_aligned.rename(columns={PRICE_COL: LEV_PRICE_COL}),
        ],
        axis=1,
    )

    if drop_missing:
        out = out.dropna()

    out.index.name = DATE_COL
    return out


def prepare_pair_panel(prices: dict[str, pd.DataFrame], pair_cfg: dict, data_cfg: dict) -> pd.DataFrame:
    spot = prices[pair_cfg["spot"]]
    lev = prices[pair_cfg["lev"]]
    validate_prices(spot, pair_cfg["spot"])
    validate_prices(lev, pair_cfg["lev"])

    return align_pair_data(
        spot=spot,
        lev=lev,
        asset_class=pair_cfg.get("asset_class", "equity"),
        fill_method=data_cfg.get("fill_method", "ffill"),
        drop_missing=bool(data_cfg.get("drop_missing", True)),
        how=data_cfg.get("align_calendar", "intersection"),
    )
