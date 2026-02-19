import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.strategy import compute_target_weights


def _sample_features(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    spot_ret = rng.normal(0.0004, 0.012, size=n)
    lev_ret = 2.0 * spot_ret + rng.normal(0.0, 0.01, size=n)

    df = pd.DataFrame(
        {
            "spot_ret": spot_ret,
            "lev_ret": lev_ret,
            "drift": pd.Series(spot_ret, index=idx).rolling(20).mean().fillna(0),
            "vol": pd.Series(spot_ret, index=idx).rolling(20).std().fillna(0.01),
            "rho": pd.Series(spot_ret, index=idx).rolling(20).apply(lambda x: x.autocorr(), raw=False).fillna(0),
            "beta": pd.Series(lev_ret, index=idx).rolling(20).cov(pd.Series(spot_ret, index=idx))
            / pd.Series(spot_ret, index=idx).rolling(20).var(),
        },
        index=idx,
    )
    return df.ffill().fillna(0.0)



def test_backtest_invariants() -> None:
    features = _sample_features()
    pair = {"name": "test_pair", "leverage": 2.0, "asset_class": "equity"}
    strategy_cfg = {
        "hedge_method": "beta",
        "hedge_lookback_days": 20,
        "rebalance_freq": "3D",
        "gross_exposure": 1.0,
        "max_gross": 1.5,
        "max_leverage_leg_weight": 1.0,
        "cap_short_weight": -1.0,
    }
    costs_cfg = {
        "trading_bps": 2.0,
        "slippage_bps": 1.0,
        "borrow_bps_annual": 300.0,
        "financing_rate_annual": 0.0,
    }

    w = compute_target_weights(features, pair_cfg=pair, strategy_cfg=strategy_cfg)
    bt = run_backtest(features, w, costs_cfg=costs_cfg, asset_class="equity")

    assert bt["equity"].notna().all()
    assert (bt["equity"] > 0).all()
    assert (bt["gross"] <= strategy_cfg["max_gross"] + 1e-8).all()
    assert (bt["w_lev_target"] >= strategy_cfg["cap_short_weight"] - 1e-8).all()
