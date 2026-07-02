"""
finmlsim.data — loaders for small bundled empirical datasets.

Simulators give controlled experiments; empirical loaders let you confirm the same phenomenon
on real data. These functions read the committed Fama-French daily files (the three factors,
the momentum factor, and the 12 industry portfolios) from Kenneth R. French's data library —
publicly available for research and teaching — and return tidy ``pandas`` frames with a
``DatetimeIndex`` and returns in **decimal** (the source is in percent). See
``datasets/fetch_famafrench.py`` in the source repository for how the files are produced.

Everything is offline: the CSVs ship with the package, so figures reproduce without a network.
"""

from __future__ import annotations
import os

_DATA = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "datasets", "famafrench")
)

__all__ = ["ff_factors", "ff_momentum", "industries", "market_return"]


def _load(name):
    import pandas as pd

    df = pd.read_csv(os.path.join(_DATA, name))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").astype(float)
    df = df.mask(df <= -99.99)  # French missing-value sentinels -> NaN
    return df / 100.0  # percent -> decimal


def ff_factors(start=None, end=None):
    """Daily Fama-French factors: ``Mkt-RF``, ``SMB``, ``HML``, ``RF`` (decimal)."""
    return _load("ff_factors_daily.csv").loc[start:end]


def ff_momentum(start=None, end=None):
    """Daily momentum factor ``Mom`` (decimal)."""
    return _load("ff_momentum_daily.csv").loc[start:end]


def industries(start=None, end=None):
    """Daily returns of the 12 Fama-French industry portfolios (decimal)."""
    return _load("ff_12industry_daily.csv").loc[start:end]


def market_return(start=None, end=None):
    """Daily total US market return (``Mkt-RF`` + ``RF``) as a Series (decimal).

    This is the real, value-weighted US equity market — the principled, offline stand-in for a
    broad-market ETF like SPY, with a far longer history.
    """
    f = ff_factors(start, end)
    return (f["Mkt-RF"] + f["RF"]).rename("mkt")
