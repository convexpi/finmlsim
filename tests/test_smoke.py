"""Smoke tests: the generators run, are deterministic from their seed, and return sane shapes."""
import numpy as np
import finmlsim as fms


def test_return_series_shape_and_determinism():
    for name in ("gaussian", "student_t", "garch", "arma"):
        fn = getattr(fms.simulate, name)
        r1 = fn(n=500, seed=0)
        r2 = fn(n=500, seed=0)
        r3 = fn(n=500, seed=1)
        assert isinstance(r1, np.ndarray) and r1.shape == (500,)
        assert np.allclose(r1, r2)          # same seed -> identical
        assert not np.allclose(r1, r3)      # different seed -> different


def test_price_process_and_execution():
    prices, vol = fms.simulate.heston(n=200, seed=1)
    assert len(prices) == len(vol) == 201
    assert np.all(prices > 0) and np.all(vol >= 0)
    ac = fms.simulate.almgren_chriss()
    assert isinstance(ac, dict)


def test_stylized_summary():
    r = fms.simulate.garch(n=1000, seed=0)
    summary = fms.stylized.summary(r)
    assert isinstance(summary, dict) and len(summary) > 0
