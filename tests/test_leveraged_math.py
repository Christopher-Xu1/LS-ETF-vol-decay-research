import numpy as np

from src.models.leveraged_math import expected_log_leveraged, realized_log_return, simulate_gbm_paths


def test_expected_log_leveraged_matches_simulation_directionally() -> None:
    mu = 0.08
    sigma = 0.3
    leverage = 2.0

    theo = expected_log_leveraged(mu, sigma, leverage)
    sim = simulate_gbm_paths(mu=mu, sigma=sigma, leverage=leverage, n_paths=3000, n_days=252, seed=123)
    realized = realized_log_return(sim["lev_paths"]).mean()

    # Monte Carlo approximation should be close in sign and reasonable magnitude.
    assert np.sign(theo) == np.sign(realized) or abs(realized) < 0.01
    assert abs(theo - realized) < 0.12


def test_higher_vol_increases_drag() -> None:
    mu = 0.08
    low = expected_log_leveraged(mu, sigma=0.1, leverage=3.0)
    high = expected_log_leveraged(mu, sigma=0.5, leverage=3.0)
    assert high < low
