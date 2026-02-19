"""Regime labeling utilities using drift/volatility/autocorrelation."""

from __future__ import annotations

import pandas as pd


def _split_threshold(series: pd.Series, split: str | float) -> float:
    if isinstance(split, (int, float)):
        return float(split)
    split = split.lower()
    if split == "median":
        return float(series.median())
    if split == "mean":
        return float(series.mean())
    raise ValueError(f"Unsupported split spec: {split}")


def label_regime(mu: float, vol: float, rho: float, splits: dict) -> tuple[str, str]:
    drift_thr = float(splits["drift"])
    vol_thr = float(splits["vol"])
    rho_thr = float(splits.get("rho", 0.0))

    direction = "up" if mu >= drift_thr else "down"
    vol_state = "high_vol" if vol >= vol_thr else "low_vol"
    rho_state = "pos_rho" if rho >= rho_thr else "neg_rho"

    return f"{direction}_{vol_state}", f"{direction}_{vol_state}_{rho_state}"


def _quantile_edges(series: pd.Series, quantiles: list[float]) -> list[float]:
    qs = sorted(set([0.0] + quantiles + [1.0]))
    vals = [float(series.quantile(q)) for q in qs]

    # Make edges strictly increasing for pd.cut.
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1e-9
    return vals


def assign_regimes(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    reg_cfg = cfg.get("regimes", {})

    drift_split = reg_cfg.get("drift_split", "median")
    vol_split = reg_cfg.get("vol_split", "median")
    rho_split = reg_cfg.get("rho_split", 0.0)
    quantiles = list(reg_cfg.get("quantiles", [0.25, 0.5, 0.75]))

    splits = {
        "drift": _split_threshold(out["drift"], drift_split),
        "vol": _split_threshold(out["vol"], vol_split),
        "rho": float(rho_split),
    }

    labels_2d, labels_3d = zip(*(label_regime(mu, vol, rho, splits) for mu, vol, rho in out[["drift", "vol", "rho"]].itertuples(index=False)))
    out["regime_2d"] = labels_2d
    out["regime_3d"] = labels_3d

    mu_edges = _quantile_edges(out["drift"], quantiles)
    vol_edges = _quantile_edges(out["vol"], quantiles)
    mu_labels = [f"Q{i+1}" for i in range(len(mu_edges) - 1)]
    vol_labels = [f"Q{i+1}" for i in range(len(vol_edges) - 1)]

    out["mu_bin"] = pd.cut(out["drift"], bins=mu_edges, labels=mu_labels, include_lowest=True)
    out["vol_bin"] = pd.cut(out["vol"], bins=vol_edges, labels=vol_labels, include_lowest=True)

    return out


def conditional_stats(
    df: pd.DataFrame,
    group_col: str,
    ret_col: str = "net_ret",
    annualization: int = 252,
) -> pd.DataFrame:
    grouped = df.groupby(group_col)[ret_col]
    stats = grouped.agg(["mean", "std", "count"]).rename(columns={"std": "vol"})
    stats["sharpe"] = stats["mean"] / stats["vol"].replace(0.0, pd.NA) * (annualization**0.5)
    return stats.sort_index()


def pivot_regime_heatmap(df: pd.DataFrame, values_col: str = "net_ret") -> pd.DataFrame:
    return df.pivot_table(
        index="mu_bin",
        columns="vol_bin",
        values=values_col,
        aggfunc="mean",
        observed=False,
    )
