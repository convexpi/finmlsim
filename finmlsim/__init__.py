"""
finmlsim — scholarly simulators for financial machine learning.

A small, NumPy-only toolkit of citation-anchored simulation models, written for research,
teaching, and self-study. The ``simulate`` module is the heart of the package: generators
covering classical return series (Gaussian, Student-t, ARMA, GARCH, regime-switching,
jump-diffusion, Cont-Bouchaud), continuous-time price processes (Heston, Bates, SABR,
rough Bergomi, Ornstein-Uhlenbeck / Vasicek, fractional Brownian motion), and market
microstructure / execution (Hawkes, Roll, Glosten-Milgrom, Kyle, a Cont-Stoikov limit-order
book, Almgren-Chriss). ``stylized`` measures the empirical facts those generators produce;
``metrics`` evaluates trading strategies; ``resample`` provides block bootstraps for
dependent series.

Quick start::

    import finmlsim as fms
    r = fms.simulate.garch(n=2000, dist="t", seed=0)   # clustering + fat tails
    fms.stylized.summary(r)                             # check the stylized facts
    prices = fms.stylized.to_prices(r)                  # for plotting
"""

from . import simulate, stylized, metrics, data, resample

__version__ = "0.4.1"
__all__ = ["simulate", "stylized", "metrics", "data", "resample"]
