"""Source adapters for market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.constants import DATE_COL, PRICE_COL


class BaseSource(ABC):
    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Return a standardized frame with columns [date, adj_close]."""


class YahooSource(BaseSource):
    @staticmethod
    def _standardize(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(columns=[DATE_COL, PRICE_COL])

        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        out = data[[col]].rename(columns={col: PRICE_COL}).reset_index()
        out.columns = [DATE_COL, PRICE_COL]
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], utc=True)
        return out.reset_index(drop=True)

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for YahooSource") from exc

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            # Fallback: request maximum history to handle cases where the
            # configured start predates ticker inception/listing.
            fallback = yf.download(
                ticker,
                period="max",
                auto_adjust=False,
                progress=False,
            )
            if fallback.empty:
                raise ValueError(
                    f"No data returned for ticker={ticker}. "
                    "Ticker may be unavailable, rate-limited, or unsupported by Yahoo."
                )
            data = fallback

        out_all = self._standardize(data)
        req_start = pd.Timestamp(start, tz="UTC")
        req_end = pd.Timestamp(end, tz="UTC") if end else None

        out = out_all[out_all[DATE_COL] >= req_start]
        if req_end is not None:
            out = out[out[DATE_COL] <= req_end]

        if out.empty:
            available_start = out_all[DATE_COL].min()
            available_end = out_all[DATE_COL].max()
            raise ValueError(
                f"No data overlap for ticker={ticker}. "
                f"Requested range: {req_start.date()} -> {req_end.date() if req_end is not None else 'latest'}. "
                f"Available Yahoo range: {available_start.date()} -> {available_end.date()}."
            )

        return out


class CCXTSource(BaseSource):
    """Fetch daily bars from a CCXT exchange for crypto symbols.

    The ticker is expected as `BASE-QUOTE`, e.g. BTC-USD.
    """

    def __init__(self, exchange_id: str = "kraken") -> None:
        self.exchange_id = exchange_id

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("ccxt is required for CCXTSource") from exc

        symbol = ticker.replace("-", "/")
        exchange_cls = getattr(ccxt, self.exchange_id)
        exchange = exchange_cls()

        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        until = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

        candles = []
        cursor = since
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe="1d", since=cursor, limit=1000)
            if not batch:
                break
            candles.extend(batch)
            last_ts = batch[-1][0]
            cursor = last_ts + 86400000
            if until and cursor > until:
                break
            if len(batch) < 1000:
                break

        if not candles:
            raise ValueError(f"No data returned for symbol={symbol} from {self.exchange_id}")

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df[DATE_COL] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df[PRICE_COL] = df["close"]
        out = df[[DATE_COL, PRICE_COL]].copy()

        if end:
            out = out[out[DATE_COL] <= pd.Timestamp(end, tz="UTC")]
        return out.reset_index(drop=True)


class ManualCSVSource(BaseSource):
    def __init__(self, external_dir: str | Path = "data/external") -> None:
        self.external_dir = Path(external_dir)

    def fetch(self, ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
        path = self.external_dir / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing external CSV: {path}")

        df = pd.read_csv(path)
        if DATE_COL not in df.columns:
            df = df.rename(columns={df.columns[0]: DATE_COL})
        if PRICE_COL not in df.columns:
            if "close" in df.columns:
                df[PRICE_COL] = df["close"]
            else:
                raise ValueError(f"CSV for {ticker} must contain '{PRICE_COL}' or 'close' column")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)
        out = df[[DATE_COL, PRICE_COL]].copy()
        out = out[out[DATE_COL] >= pd.Timestamp(start, tz="UTC")]
        if end:
            out = out[out[DATE_COL] <= pd.Timestamp(end, tz="UTC")]
        return out.reset_index(drop=True)
