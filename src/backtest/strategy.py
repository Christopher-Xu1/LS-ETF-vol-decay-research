"""Weight construction logic for paired convexity-decay strategies.

This module computes *target* weights and rebalance metadata.
The execution engine consumes these targets and simulates:
- scheduled rebalances
- threshold-triggered rebalances
- between-rebalance weight drift from realized returns
"""

from __future__ import annotations

import pandas as pd

from src.models.hedge_ratio import clamp_hedge_ratio, compute_hedge_ratio
from src.utils.dates import parse_rebalance_step


def _rescale_to_gross(w_spot: pd.Series, w_lev: pd.Series, gross_target: float) -> tuple[pd.Series, pd.Series]:
    """Rescale both legs so gross exposure equals `gross_target`.

    Gross is defined as |w_spot| + |w_lev|.
    """
    gross = (w_spot.abs() + w_lev.abs()).replace(0.0, pd.NA)
    scale = (gross_target / gross).fillna(1.0)
    return w_spot * scale, w_lev * scale


def _apply_caps(w_spot: pd.Series, w_lev: pd.Series, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Apply per-leg and portfolio-level risk constraints."""
    max_leg = float(cfg.get("max_leverage_leg_weight", 1.0))
    cap_short = float(cfg.get("cap_short_weight", -1.0))
    max_gross = float(cfg.get("max_gross", 1.5))

    # Individual leg clamps first.
    w_spot = w_spot.clip(-max_leg, max_leg)
    w_lev = w_lev.clip(lower=cap_short, upper=max_leg)

    # If gross still exceeds the portfolio cap, scale both legs together.
    gross = w_spot.abs() + w_lev.abs()
    too_large = gross > max_gross
    if too_large.any():
        scale = max_gross / gross[too_large]
        w_spot.loc[too_large] *= scale
        w_lev.loc[too_large] *= scale

    return w_spot, w_lev


def _rebalance_mask(index: pd.Index, freq: str) -> pd.Series:
    """Return True on rebalance rows determined by frequency."""
    step = parse_rebalance_step(freq)
    mask = pd.Series(False, index=index)
    mask.iloc[::step] = True
    return mask


def compute_target_weights(features: pd.DataFrame, pair_cfg: dict, strategy_cfg: dict) -> pd.DataFrame:
    """Build daily target weights and rebalance-control columns.

    Pipeline:
    1. Estimate hedge ratio h_t.
    2. Build daily target legs from h_t and gross target.
    3. Apply optional directional tilt and hard constraints.
    4. Attach schedule + threshold metadata for execution engine.
    """
    gross_target = float(strategy_cfg.get("gross_exposure", 1.0))
    target_delta = float(strategy_cfg.get("target_delta", 0.0))
    threshold_pct = max(0.0, float(strategy_cfg.get("rebalance_threshold_pct", 0.0)))

    h, diag = compute_hedge_ratio(
        features=features,
        method=strategy_cfg.get("hedge_method", "beta"),
        leverage=float(pair_cfg.get("leverage", 2.0)),
        lookback_days=int(strategy_cfg.get("hedge_lookback_days", 60)),
    )
    h = clamp_hedge_ratio(h)

    # Start with equal gross split before applying hedge linkage.
    w_spot = pd.Series(gross_target / 2.0, index=features.index)
    w_lev = -h * w_spot
    w_spot, w_lev = _rescale_to_gross(w_spot, w_lev, gross_target=gross_target)

    # Directional bias: a positive delta increases net long spot exposure.
    if target_delta != 0.0:
        w_spot = w_spot + target_delta

    w_spot, w_lev = _apply_caps(w_spot, w_lev, strategy_cfg)

    weights = pd.DataFrame(
        {
            "w_spot_target": w_spot,
            "w_lev_target": w_lev,
            "hedge_ratio": h,
            "hedge_r2": diag.get("r2"),
        },
        index=features.index,
    )

    mask = _rebalance_mask(weights.index, strategy_cfg.get("rebalance_freq", "1D"))
    leverage = float(pair_cfg.get("leverage", 2.0))
    leverage_target_ratio = 1.0 / max(leverage, 1e-9)

    out = weights.copy()
    out["scheduled_rebalance"] = mask.astype(bool)
    out["rebalance_threshold_pct"] = threshold_pct
    out["leverage_target_ratio"] = leverage_target_ratio
    out["target_ratio"] = (
        out["w_lev_target"].abs() / out["w_spot_target"].abs()
    ).replace([float("inf"), float("-inf")], pd.NA)
    out["gross_target"] = out["w_spot_target"].abs() + out["w_lev_target"].abs()
    out["net_target"] = out["w_spot_target"] + out["w_lev_target"]
    return out
