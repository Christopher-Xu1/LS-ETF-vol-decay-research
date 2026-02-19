"""Load raw prices from configured sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from src.constants import DATE_COL, PRICE_COL
from src.data.sources import CCXTSource, ManualCSVSource, YahooSource
from src.utils.io import ensure_dir


def choose_source(asset_class: str, source: str | None, external_dir: str = "data/external"):
    source = (source or "auto").lower()
    if source == "manual":
        return ManualCSVSource(external_dir=external_dir)
    if source == "ccxt":
        return CCXTSource()
    if source == "yahoo":
        return YahooSource()

    if asset_class == "crypto":
        return YahooSource()
    return YahooSource()


def unique_tickers(universe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tickers: dict[str, dict[str, Any]] = {}
    for pair in universe.get("pairs", []):
        tickers[pair["spot"]] = {"asset_class": pair.get("asset_class", "equity")}
        tickers[pair["lev"]] = {"asset_class": pair.get("asset_class", "equity")}
    return tickers


def load_prices(
    universe: dict[str, Any],
    start: str,
    end: str | None,
    source: str | None = None,
    external_dir: str = "data/external",
    raw_fallback_dir: str | Path = "data/raw",
    allow_raw_fallback: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load price series for all configured tickers."""
    def _load_fallback(ticker: str) -> pd.DataFrame | None:
        path = Path(raw_fallback_dir) / f"{ticker}.csv"
        if not path.exists():
            return None

        df = pd.read_csv(path)
        if DATE_COL not in df.columns or PRICE_COL not in df.columns:
            return None

        out = df[[DATE_COL, PRICE_COL]].copy()
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], utc=True, errors="coerce")
        out = out.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

        start_ts = pd.Timestamp(start, tz="UTC")
        out = out[out[DATE_COL] >= start_ts]
        if end:
            end_ts = pd.Timestamp(end, tz="UTC")
            out = out[out[DATE_COL] <= end_ts]
        if out.empty:
            return None
        return out.reset_index(drop=True)

    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, Exception] = {}
    for ticker, meta in unique_tickers(universe).items():
        adapter = choose_source(meta["asset_class"], source, external_dir=external_dir)
        try:
            out[ticker] = adapter.fetch(ticker, start=start, end=end)
        except Exception as exc:  # noqa: BLE001 - preserve full upstream error detail
            if allow_raw_fallback:
                fallback = _load_fallback(ticker)
                if fallback is not None:
                    warnings.warn(
                        f"Using cached raw data for {ticker} from {raw_fallback_dir} "
                        f"after live fetch failure: {exc}",
                        RuntimeWarning,
                    )
                    out[ticker] = fallback
                    continue
            errors[ticker] = exc

    if errors:
        detail = "; ".join(f"{ticker}: {err}" for ticker, err in errors.items())
        raise ValueError(f"Failed to load prices for one or more tickers. Details: {detail}")

    return out


def write_raw_prices(prices: dict[str, pd.DataFrame], raw_dir: str | Path = "data/raw") -> None:
    raw_path = ensure_dir(raw_dir)
    for ticker, df in prices.items():
        df.to_csv(raw_path / f"{ticker}.csv", index=False)


def load_raw_prices_from_disk(
    tickers: list[str],
    raw_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    root = Path(raw_dir)
    for ticker in tickers:
        path = root / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        out[ticker] = pd.read_csv(path, parse_dates=["date"])
    return out
