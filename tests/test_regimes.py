import pandas as pd

from src.models.regimes import assign_regimes, label_regime


def test_label_regime_boundaries() -> None:
    splits = {"drift": 0.0, "vol": 0.2, "rho": 0.0}
    r2d, r3d = label_regime(mu=0.01, vol=0.3, rho=0.1, splits=splits)
    assert r2d == "up_high_vol"
    assert r3d == "up_high_vol_pos_rho"



def test_assign_regimes_adds_bins_and_labels() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "drift": [i / 100 for i in range(-5, 5)],
            "vol": [0.1 + i / 100 for i in range(10)],
            "rho": [(-1) ** i * 0.2 for i in range(10)],
            "net_ret": [0.001] * 10,
        },
        index=idx,
    )

    cfg = {"regimes": {"drift_split": "median", "vol_split": "median", "rho_split": 0.0, "quantiles": [0.5]}}
    out = assign_regimes(df, cfg)

    assert "regime_2d" in out.columns
    assert "regime_3d" in out.columns
    assert "mu_bin" in out.columns
    assert "vol_bin" in out.columns
    assert out["mu_bin"].notna().all()
    assert out["vol_bin"].notna().all()
