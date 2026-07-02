"""
finmlsim.resample — bootstrap utilities for dependent time-series data.

Standard i.i.d. bootstrap (sampling observations with replacement) destroys the serial
dependence that finance data is full of. The block bootstrap and its stationary cousin keep
contiguous chunks of observations together so the resampled series has the same short-range
dependence structure as the original.

These resamplers pair with the multiple-testing diagnostics in :mod:`finmlsim.metrics`
(``deflated_sharpe``, ``min_track_record_length``) — sample size and dependence together
determine how much information a back-test actually contains.
"""

from __future__ import annotations

import numpy as np

__all__ = ["stationary_bootstrap", "circular_block_bootstrap"]


def _rng(seed):
    return np.random.default_rng(seed)


def stationary_bootstrap(x, n_samples=None, mean_block_length=None, seed=0):
    """Politis-Romano (1994) stationary bootstrap.

    Block lengths are drawn from a geometric distribution with mean ``mean_block_length``, so
    the resampled series is itself stationary (unlike the fixed-block bootstrap, which is
    not). Starting indices for each block are drawn uniformly from ``x`` with wraparound. The
    output has length ``n_samples`` (default: ``len(x)``).

    The optimal expected block length grows with the persistence of the series; a common
    rule of thumb is ``mean_block_length = round(n ** (1/3))`` for moderate dependence, but
    automatic selection (Politis-White 2004) requires an estimate of the spectral density at
    zero.

    Parameters
    ----------
    x : array_like
        1-D series to resample.
    n_samples : int, optional
        Length of the resampled series; defaults to ``len(x)``.
    mean_block_length : float, optional
        Mean block length; defaults to ``round(len(x) ** (1/3))``.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Resampled series of length ``n_samples``.
    """
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        raise ValueError("x must be non-empty")
    n_samples = int(n if n_samples is None else n_samples)
    if mean_block_length is None:
        mean_block_length = max(1.0, round(n ** (1 / 3)))
    p = 1.0 / mean_block_length
    rng = _rng(seed)
    out = np.empty(n_samples, dtype=x.dtype)
    i = 0
    while i < n_samples:
        start = int(rng.integers(0, n))
        # Geometric block length, minimum 1.
        block_len = int(rng.geometric(p))
        end = min(i + block_len, n_samples)
        for j in range(i, end):
            out[j] = x[(start + j - i) % n]
        i = end
    return out


def circular_block_bootstrap(x, n_samples=None, block_length=None, seed=0):
    """Fixed-length circular block bootstrap (Politis-Romano 1992).

    Like :func:`stationary_bootstrap` but with deterministic block length ``block_length``,
    with wraparound at the end of ``x``. Simpler and faster than the stationary variant,
    but the resampled series is not itself stationary, which can bias some downstream
    statistics. Use the stationary version when in doubt.
    """
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        raise ValueError("x must be non-empty")
    n_samples = int(n if n_samples is None else n_samples)
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))
    rng = _rng(seed)
    out = np.empty(n_samples, dtype=x.dtype)
    i = 0
    while i < n_samples:
        start = int(rng.integers(0, n))
        end = min(i + block_length, n_samples)
        for j in range(i, end):
            out[j] = x[(start + j - i) % n]
        i = end
    return out
