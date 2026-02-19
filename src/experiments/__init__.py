"""Shared helpers for experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaners import prepare_pair_panel
from src.data.features import build_features
from src.data.loaders import load_prices, load_raw_prices_from_disk, unique_tickers, write_raw_prices
from src.utils.io import ensure_dir, load_config


def load_cfg(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


def pair_list(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg.get("universe", {}).get("pairs", []))


def fetch_raw_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source")
    prices = load_prices(
        universe=cfg.get("universe", {}),
        start=data_cfg.get("start"),
        end=data_cfg.get("end"),
        source=source,
        external_dir="data/external",
    )
    write_raw_prices(prices, raw_dir="data/raw")
    return prices


def _raw_prices_or_fetch(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tickers = list(unique_tickers(cfg.get("universe", {})).keys())
    raw_dir = Path("data/raw")
    available = all((raw_dir / f"{t}.csv").exists() for t in tickers)
    if available:
        return load_raw_prices_from_disk(tickers, raw_dir=raw_dir)
    return fetch_raw_data(cfg)


def build_processed_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    prices = _raw_prices_or_fetch(cfg)
    data_cfg = cfg.get("data", {})

    out: dict[str, pd.DataFrame] = {}
    ensure_dir("data/processed")

    for pair in pair_list(cfg):
        panel = prepare_pair_panel(prices=prices, pair_cfg=pair, data_cfg=data_cfg)
        feats = build_features(panel=panel, pair_cfg=pair, cfg=cfg)
        out[pair["name"]] = feats
        feats.to_csv(Path("data/processed") / f"{pair['name']}_features.csv")

    return out


def load_or_build_features(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    paths = {pair["name"]: Path("data/processed") / f"{pair['name']}_features.csv" for pair in pair_list(cfg)}

    if all(path.exists() for path in paths.values()):
        all_non_empty = True
        for name, path in paths.items():
            out[name] = pd.read_csv(path, index_col=0, parse_dates=True)
            out[name].index = pd.to_datetime(out[name].index, utc=True)
            if out[name].empty:
                all_non_empty = False
        if all_non_empty:
            return out

    return build_processed_data(cfg)
