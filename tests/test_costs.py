from src.backtest.costs import borrow_cost, slippage_cost, trading_cost


def test_trading_cost_scaling() -> None:
    assert trading_cost(1.0, 2.0) == 0.0002
    assert trading_cost(0.5, 10.0) == 0.0005



def test_slippage_cost_scaling() -> None:
    assert slippage_cost(1.0, 1.0) == 0.0001



def test_borrow_cost_scaling() -> None:
    val = borrow_cost(short_notional=1.0, borrow_bps_annual=300.0, days_per_year=252)
    assert abs(val - (0.03 / 252)) < 1e-12
