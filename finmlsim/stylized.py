"""
finmlsim.stylized — measure the stylized facts of a return series.

Tools to quantify what the generators in ``finmlsim.simulate`` (or real data) produce: the
near-zero autocorrelation of returns, the strong autocorrelation of squared returns (volatility
clustering), heavy tails, and (non-)stationarity. Use these to *check* that a simulation exhibits
the empirical regularities you intend to teach.
"""

from __future__ import annotations
import numpy as np

__all__ = ["acf", "excess_kurtosis", "vol_clustering", "summary", "to_prices"]


def acf(x, nlags=20):
    """Autocorrelation function of ``x`` for lags 1..nlags (returns an array of length nlags)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, nlags + 1)])


def excess_kurtosis(x):
    """Excess kurtosis (0 for a Normal; > 0 means heavier tails)."""
    x = np.asarray(x, float)
    z = (x - x.mean()) / x.std()
    return float((z**4).mean() - 3.0)


def vol_clustering(x, nlags=20):
    """A scalar measure of volatility clustering: mean autocorrelation of *squared* returns over
    lags 1..nlags. Near 0 means no clustering; clearly positive means clustering."""
    return float(np.mean(acf(np.asarray(x, float) ** 2, nlags)))


def to_prices(returns, p0=100.0):
    """Convert log returns to a price path starting at ``p0`` (handy for plotting)."""
    return p0 * np.exp(np.cumsum(np.asarray(returns, float)))


def summary(x, nlags=20):
    """Return a dict of stylized-fact diagnostics for a return series."""
    x = np.asarray(x, float)
    a1 = acf(x, nlags)
    return {
        "mean_ann": float(x.mean() * 252),
        "vol_ann": float(x.std() * np.sqrt(252)),
        "excess_kurtosis": excess_kurtosis(x),
        "ret_acf_lag1": float(a1[0]),
        "ret_acf_meanabs": float(np.mean(np.abs(a1))),  # ~0 for real returns
        "sq_ret_acf_mean": vol_clustering(x, nlags),  # >0 if vol clusters
    }
