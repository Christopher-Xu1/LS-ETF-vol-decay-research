import numpy as np
import pandas as pd

from src.experiments import run_stress


def _sample_backtest(start: str = "2020-01-01", periods: int = 120) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    rng = np.random.default_rng(42)

    net_ret = rng.normal(0.0002, 0.01, size=periods)
    raw_ret = net_ret + 0.0003
    spot_pnl = raw_ret * 0.6
    lev_pnl = raw_ret * 0.4
    trading_cost = np.full(periods, 0.0001)
    slippage_cost = np.full(periods, 0.00005)
    borrow_cost = np.full(periods, 0.00008)
    financing_cost = np.zeros(periods)
    cost_total = trading_cost + slippage_cost + borrow_cost + financing_cost

    equity = pd.Series((1.0 + net_ret).cumprod(), index=idx)
    drawdown = equity / equity.cummax() - 1.0

    return pd.DataFrame(
        {
            "net_ret": net_ret,
            "raw_ret": raw_ret,
            "spot_pnl": spot_pnl,
            "lev_pnl": lev_pnl,
            "trading_cost": trading_cost,
            "slippage_cost": slippage_cost,
            "borrow_cost": borrow_cost,
            "financing_cost": financing_cost,
            "cost_total": cost_total,
            "equity": equity,
            "drawdown": drawdown,
            "gross": 1.0,
            "turnover": 0.05,
            "rebalance": 0,
            "rebalance_scheduled": 0,
            "rebalance_threshold_trigger": 0,
        },
        index=idx,
    )


def test_stress_run_emits_scenarios_and_attribution(tmp_path, monkeypatch) -> None:
    cfg = {
        "evaluation": {"save_dir": str(tmp_path)},
        "universe": {"pairs": [{"name": "qqq_tqqq"}]},
        "stress": {
            "scenarios": [
                {"pair": "qqq_tqqq", "start": "2020-02-01", "end": "2020-03-31", "label": "sample_stress"},
            ]
        },
    }
    bt = _sample_backtest()

    monkeypatch.setattr(run_stress, "load_cfg", lambda _: cfg)
    monkeypatch.setattr(run_stress, "pair_list", lambda _cfg: [{"name": "qqq_tqqq"}])
    monkeypatch.setattr(run_stress, "run_baseline", lambda _: {"qqq_tqqq": bt})

    out = run_stress.run("unused.yaml")

    assert not out.empty
    assert {"pair", "scenario", "window_return", "max_drawdown", "recovery_days"}.issubset(out.columns)
    assert {"spot_pnl_total", "lev_pnl_total", "cost_total", "net_pnl_total"}.issubset(out.columns)
    assert out.loc[0, "scenario"] == "sample_stress"

    scenarios_path = tmp_path / "tables" / "stress_scenarios.csv"
    attribution_path = tmp_path / "tables" / "stress_pnl_attribution.csv"
    assert scenarios_path.exists()
    assert attribution_path.exists()

    attribution = pd.read_csv(attribution_path)
    assert set(attribution["component"]) >= {
        "spot_pnl",
        "lev_pnl",
        "trading_cost",
        "slippage_cost",
        "borrow_cost",
        "financing_cost",
        "net_pnl",
    }


def test_stress_run_handles_no_overlapping_windows(tmp_path, monkeypatch) -> None:
    cfg = {
        "evaluation": {"save_dir": str(tmp_path)},
        "universe": {"pairs": [{"name": "qqq_tqqq"}]},
        "stress": {
            "scenarios": [
                {"pair": "qqq_tqqq", "start": "1990-01-01", "end": "1990-12-31", "label": "outside_history"},
            ]
        },
    }
    bt = _sample_backtest(start="2020-01-01", periods=90)

    monkeypatch.setattr(run_stress, "load_cfg", lambda _: cfg)
    monkeypatch.setattr(run_stress, "pair_list", lambda _cfg: [{"name": "qqq_tqqq"}])
    monkeypatch.setattr(run_stress, "run_baseline", lambda _: {"qqq_tqqq": bt})

    out = run_stress.run("unused.yaml")

    assert out.empty
    assert (tmp_path / "tables" / "stress_scenarios.csv").exists()
    assert (tmp_path / "tables" / "stress_pnl_attribution.csv").exists()
