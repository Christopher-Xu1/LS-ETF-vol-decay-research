"""Date and calendar utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.constants import ANNUALIZATION_BY_ASSET_CLASS, DATE_COL


def days_per_year(asset_class: str) -> int:
    """Return trading days/year proxy for the given asset class."""
    return ANNUALIZATION_BY_ASSET_CLASS.get(asset_class, 252)


def to_trading_days_index(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """Coerce the frame to a daily time index for the asset class."""
    out = df.copy()
    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], utc=True)
        out = out.set_index(DATE_COL)
    else:
        out.index = pd.to_datetime(out.index, utc=True)

    out = out.sort_index()
    freq = "D" if asset_class == "crypto" else "B"
    idx = pd.date_range(out.index.min(), out.index.max(), freq=freq, tz="UTC")
    out = out.reindex(idx)
    out.index.name = DATE_COL
    return out


def align_frames(
    frames: Iterable[pd.DataFrame],
    how: str = "intersection",
) -> list[pd.DataFrame]:
    """Align a collection of frames by index using intersection or union."""
    frames = [f.copy() for f in frames]
    if not frames:
        return []

    shared = frames[0].index
    for frame in frames[1:]:
        if how == "union":
            shared = shared.union(frame.index)
        else:
            shared = shared.intersection(frame.index)

    return [frame.reindex(shared).sort_index() for frame in frames]


def parse_rebalance_step(freq: str) -> int:
    """Parse simple day-based rebalance frequencies like 1D, 3D, 7D."""
    freq = (freq or "1D").upper()
    if freq.endswith("D"):
        val = freq[:-1] or "1"
        return max(1, int(val))
    return 1
