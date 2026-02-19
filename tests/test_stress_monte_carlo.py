import numpy as np
import pytest

from src.experiments.run_stress import _sample_return_indices


def test_sample_return_indices_permutation_properties() -> None:
    rng = np.random.default_rng(123)
    idx = _sample_return_indices(n_obs=20, horizon=15, method="permutation", block_size=5, rng=rng)

    assert len(idx) == 15
    assert (idx >= 0).all() and (idx < 20).all()
    assert len(set(idx.tolist())) == 15


def test_sample_return_indices_block_bootstrap_properties() -> None:
    rng = np.random.default_rng(7)
    idx = _sample_return_indices(n_obs=30, horizon=40, method="block_bootstrap", block_size=4, rng=rng)

    assert len(idx) == 40
    assert (idx >= 0).all() and (idx < 30).all()


def test_sample_return_indices_rejects_unknown_method() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        _sample_return_indices(n_obs=30, horizon=40, method="unknown", block_size=4, rng=rng)
