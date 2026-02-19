"""I/O helpers for config and tabular artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base and return the merged dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and merge with configs/default.yaml when needed."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    default_path = cfg_path.parent / "default.yaml"
    if cfg_path.name != "default.yaml" and default_path.exists():
        with default_path.open("r", encoding="utf-8") as handle:
            default_cfg = yaml.safe_load(handle) or {}
        return deep_update(default_cfg, config)

    return config


def write_table(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=index)
    else:
        df.to_csv(out, index=index)


def read_table(path: str | Path) -> pd.DataFrame:
    in_path = Path(path)
    if in_path.suffix == ".parquet":
        return pd.read_parquet(in_path)
    return pd.read_csv(in_path)
