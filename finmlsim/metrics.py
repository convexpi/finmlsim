"""
finmlsim.metrics — evaluation metrics for financial ML.

The tools you reach for when judging a *strategy* rather than a model: risk-adjusted return
(Sharpe), worst peak-to-trough loss (max drawdown), the rank information coefficient and its
consistency (IC-IR), turnover, and the *deflated* Sharpe ratio that haircuts a result for the
number of configurations you searched. These power Weeks 8 (evaluation) and 10 (portfolios).

Conventions: ``returns`` are per-period simple returns unless noted; ``periods_per_year`` defaults
to 252 (trading days). Functions accept array-likes and ignore NaNs where it is sensible to.
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "sharpe",
    "max_drawdown",
    "rank_ic",
    "ic_ir",
    "turnover",
    "deflated_sharpe",
    "min_track_record_length",
]


def _clean(x):
    x = np.asarray(x, float)
    return x[~np.isnan(x)]


def sharpe(returns, periods_per_year=252, rf=0.0):
    """Annualized Sharpe ratio of a per-period return series (excess of ``rf`` per period)."""
    r = _clean(returns) - rf
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / sd)


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown of the cumulative (compounded) return path.

    Returns a negative number (e.g. -0.35 means a 35% drawdown). Takes per-period simple returns.
    """
    r = _clean(returns)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min()) if dd.size else 0.0


def rank_ic(pred, actual):
    """Rank (Spearman) information coefficient between predictions and realized outcomes.

    The finance-standard score for a return forecast: robust to the heavy tails that make
    ordinary correlation unstable. Returns a value in [-1, 1].
    """
    p = np.asarray(pred, float)
    a = np.asarray(actual, float)
    m = ~(np.isnan(p) | np.isnan(a))
    p, a = p[m], a[m]
    if p.size < 3:
        return np.nan
    rp = np.argsort(np.argsort(p))
    ra = np.argsort(np.argsort(a))
    return float(np.corrcoef(rp, ra)[0, 1])


def ic_ir(ic_series, periods_per_year=252):
    """Information ratio of a series of per-period ICs: mean(IC)/std(IC), annualized.

    Measures the *consistency* of a signal across periods, not just its average size.
    """
    ic = _clean(ic_series)
    sd = ic.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * ic.mean() / sd)


def turnover(weights):
    """Average one-period turnover of a weights path of shape (T, n_assets) or (T,) for one asset.

    Turnover at t is sum |w_t - w_{t-1}|; this returns the mean over time. Costs scale with it.
    """
    w = np.asarray(weights, float)
    if w.ndim == 1:
        w = w[:, None]
    return float(np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)))


def deflated_sharpe(
    observed_sharpe, n_trials, n_obs, skew=0.0, kurtosis=3.0, periods_per_year=252
):
    """Deflated Sharpe ratio (López de Prado): the probability the *observed* annualized Sharpe is
    truly positive once you account for how many configurations were tried.

    ``n_trials`` is the number of strategy configurations searched; ``n_obs`` the number of return
    observations; ``skew`` and ``kurtosis`` (non-excess; 3 = normal) describe the returns. Returns a
    probability in [0, 1]; values near 1 mean the result survives the multiple-testing haircut.
    """
    from math import log, sqrt, erf

    sr = observed_sharpe / sqrt(periods_per_year)  # per-period Sharpe
    # Expected maximum Sharpe under the null of n_trials zero-skill trials (variance of SR ~ 1/N
    # for standardized trials; use the standard extreme-value approximation).
    e = 0.5772156649
    emax = sqrt(2 * log(max(n_trials, 2))) - (
        log(log(max(n_trials, 3))) + log(4 * np.pi)
    ) / (2 * sqrt(2 * log(max(n_trials, 2))))
    sr0 = emax / sqrt(periods_per_year)  # threshold per-period Sharpe from search
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurtosis - 1) / 4.0 * sr**2))
    z = (sr - sr0) * sqrt(max(n_obs - 1, 1)) / denom
    return float(0.5 * (1 + erf(z / sqrt(2))))  # standard-normal CDF


def min_track_record_length(
    observed_sharpe,
    target_sharpe=0.0,
    skew=0.0,
    kurtosis=3.0,
    confidence=0.95,
    periods_per_year=252,
):
    """Minimum number of observations needed to conclude, at ``confidence``, that the true Sharpe
    exceeds ``target_sharpe`` (López de Prado's minimum track record length). Returns a count of
    periods (e.g. trading days)."""
    from math import sqrt
    from scipy.stats import norm

    sr = observed_sharpe / sqrt(periods_per_year)
    srt = target_sharpe / sqrt(periods_per_year)
    if sr <= srt:
        return float("inf")
    z = norm.ppf(confidence)
    var = 1 - skew * sr + (kurtosis - 1) / 4.0 * sr**2
    return float(1 + var * (z / (sr - srt)) ** 2)
