"""
finmlsim.simulate — generators for simulated return and price series.

Most generators return a 1-D NumPy array of **log returns** of length ``n`` and take a ``seed``
for reproducibility. A few return prices, paths, event times, or dicts where that fits the
underlying model better; each docstring states the return shape. The point of the package is
pedagogical: by varying a few parameters you can dial each *stylized fact* of real markets on
and off and watch the effect, before confronting empirical data.

Time-series return generators
-----------------------------
- ``gaussian``         : i.i.d. Normal returns (a pure random walk in price).
- ``student_t``        : i.i.d. heavy-tailed returns (fat tails, no clustering).
- ``arma``             : linear autocorrelation in the mean (ARMA).
- ``garch``            : volatility clustering (GARCH(1,1)); optionally with t innovations.
- ``regime_switching`` : two-state (calm/turbulent) Markov volatility.
- ``jump_diffusion``   : Merton-style rare jumps on top of diffusion.
- ``cont_bouchaud``    : simplified herding/percolation agent model (fat tails *and* clustering
                         emerge endogenously).
- ``panel``            : a *cross-section* of many assets driven by a common market factor plus
                         idiosyncratic noise.

Continuous-time price processes
-------------------------------
- ``ornstein_uhlenbeck`` / ``vasicek`` : mean-reverting diffusion (Vasicek 1977).
- ``heston``       : stochastic volatility (Heston 1993); foundational for option pricing.
- ``bates``        : Heston with Merton-style jumps in the price (Bates 1996).
- ``sabr``         : stochastic-α-β-ρ vol model (Hagan et al. 2002); FX / rates vol surfaces.
- ``rbergomi``     : rough volatility (Bayer-Friz-Gatheral 2016); fractional H in (0, 0.5).
- ``fbm`` / ``fgn``: fractional Brownian motion and its increments (Hurst parameter H).

Microstructure and execution
----------------------------
- ``hawkes``           : self-exciting point process (Hawkes 1971); exponential kernel.
- ``roll_bounce``      : Roll (1984) bid-ask bounce — negative serial correlation in tx prices.
- ``glosten_milgrom``  : adverse-selection bid-ask spread (Glosten-Milgrom 1985).
- ``kyle``             : strategic informed-trading and linear price impact (Kyle 1985).
- ``limit_order_book`` : Cont-Stoikov-style continuous-double-auction LOB.
- ``almgren_chriss``   : closed-form optimal-execution trajectories (Almgren-Chriss 2000).

These are teaching models, not production calibrations. See ``finmlsim.stylized`` for tools to
measure the stylized facts each one produces, and ``finmlsim.resample.stationary_bootstrap`` for
the resampling counterpart used in multiple-testing diagnostics.
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "gaussian",
    "student_t",
    "arma",
    "garch",
    "regime_switching",
    "jump_diffusion",
    "cont_bouchaud",
    "panel",
    "ornstein_uhlenbeck",
    "vasicek",
    "heston",
    "bates",
    "sabr",
    "rbergomi",
    "fbm",
    "fgn",
    "hawkes",
    "roll_bounce",
    "glosten_milgrom",
    "kyle",
    "limit_order_book",
    "almgren_chriss",
]


def _rng(seed):
    return np.random.default_rng(seed)


def gaussian(n=1000, mu=0.0003, sigma=0.01, seed=0):
    """I.i.d. Normal log returns. Price is a geometric random walk; no clustering, thin tails."""
    return _rng(seed).normal(mu, sigma, n)


def student_t(n=1000, df=4.0, sigma=0.01, mu=0.0003, seed=0):
    """I.i.d. Student-t log returns: heavy tails (lower ``df`` = fatter), but no volatility clustering.
    Scaled so the innovation has standard deviation ``sigma``."""
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance")
    z = _rng(seed).standard_t(df, n)
    z = z / np.sqrt(df / (df - 2))  # unit variance
    return mu + sigma * z


def arma(n=1000, ar=(0.0,), ma=(0.0,), sigma=0.01, mu=0.0003, seed=0):
    """ARMA log returns — linear autocorrelation in the *mean*.

    ``ar`` and ``ma`` are coefficient tuples (e.g. ``ar=(0.2,)`` for AR(1) with phi=0.2).
    Real return means are nearly uncorrelated, so keep coefficients small; this generator is for
    demonstrating what mean-autocorrelation *would* look like.
    """
    rng = _rng(seed)
    p, q = len(ar), len(ma)
    eps = rng.normal(0.0, sigma, n + max(p, q))
    x = np.zeros(n + max(p, q))
    for t in range(max(p, q), n + max(p, q)):
        x[t] = (
            sum(ar[i] * x[t - 1 - i] for i in range(p))
            + eps[t]
            + sum(ma[j] * eps[t - 1 - j] for j in range(q))
        )
    return mu + x[max(p, q) :]


def garch(
    n=1000, omega=2e-6, alpha=0.1, beta=0.88, mu=0.0003, dist="normal", df=5.0, seed=0
):
    """GARCH(1,1) log returns: volatility clustering.

        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    ``alpha + beta`` near 1 makes volatility persistent (sticky regimes). Set ``dist='t'`` to also
    get heavy-tailed innovations (clustering *and* fat tails — the most realistic single model here).
    """
    rng = _rng(seed)
    if dist == "t":
        z = rng.standard_t(df, n) / np.sqrt(df / (df - 2))
    else:
        z = rng.standard_normal(n)
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / max(1e-12, (1 - alpha - beta))  # unconditional variance
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return mu + r


def regime_switching(
    n=1000,
    sig_calm=0.006,
    sig_turb=0.02,
    p_stay_calm=0.98,
    p_stay_turb=0.95,
    mu_calm=0.0005,
    mu_turb=-0.001,
    seed=0,
):
    """Two-state Markov volatility: a calm regime and a turbulent regime with stickier, larger moves.
    Produces clustering and (mixed across regimes) heavy tails."""
    rng = _rng(seed)
    state = 0  # 0 = calm, 1 = turbulent
    r = np.zeros(n)
    for t in range(n):
        if state == 0:
            r[t] = rng.normal(mu_calm, sig_calm)
            if rng.random() > p_stay_calm:
                state = 1
        else:
            r[t] = rng.normal(mu_turb, sig_turb)
            if rng.random() > p_stay_turb:
                state = 0
    return r


def jump_diffusion(
    n=1000,
    mu=0.0003,
    sigma=0.008,
    jump_prob=0.02,
    jump_mu=-0.01,
    jump_sigma=0.03,
    seed=0,
):
    """Merton-style jump diffusion: Gaussian diffusion plus rare, large (often negative) jumps —
    a clean way to inject crash-like tail events into otherwise thin-tailed returns."""
    rng = _rng(seed)
    diffusion = rng.normal(mu, sigma, n)
    jumps = (rng.random(n) < jump_prob) * rng.normal(jump_mu, jump_sigma, n)
    return diffusion + jumps


def cont_bouchaud(
    n=1000, n_agents=2000, link_prob=None, trade_prob=0.1, sigma=0.004, seed=0
):
    """A simplified **Cont-Bouchaud** herding model.

    Agents are randomly partitioned into clusters (via percolation on a random graph); each period a
    cluster may trade as one, buying or selling together. When the connectivity is near the
    percolation threshold, cluster sizes are power-law distributed, so aggregate order flow — and
    hence returns — exhibits **fat tails and volatility clustering that emerge endogenously**, with
    no GARCH equation imposed. This is a teaching-grade implementation, not a calibration.
    """
    rng = _rng(seed)
    if link_prob is None:
        link_prob = 1.0 / n_agents  # near the percolation threshold
    r = np.zeros(n)
    for t in range(n):
        # Form clusters by a simple random union-find over a sparse random graph.
        parent = np.arange(n_agents)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        n_links = (
            rng.binomial(n_agents, link_prob * n_agents)
            if link_prob * n_agents < 1
            else int(link_prob * n_agents**2)
        )
        n_links = min(n_links, n_agents)  # keep it sparse and cheap
        for _ in range(n_links):
            a, b = rng.integers(0, n_agents, 2)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        roots, sizes = np.unique([find(a) for a in range(n_agents)], return_counts=True)
        # Each cluster trades with some probability, sign +/- equally likely; impact ~ size.
        active = rng.random(len(sizes)) < trade_prob
        signs = rng.choice([-1.0, 1.0], len(sizes))
        order_flow = np.sum(signs[active] * sizes[active])
        r[t] = sigma * order_flow / np.sqrt(n_agents)
    return r


def panel(
    n=1000,
    n_assets=50,
    market_vol=0.01,
    idio_vol=0.015,
    beta_mean=1.0,
    beta_sd=0.3,
    mu_market=0.0003,
    mu_idio_sd=0.0,
    seed=0,
):
    """A cross-section of many assets sharing a common **market factor**.

    Returns a 2-D array of shape ``(n, n_assets)`` of log returns::

        r[t, j] = beta_j * market_t + idio_{t,j}

    where ``market_t`` is one common factor (the "market") and ``idio`` is asset-specific noise.
    Because every asset loads on the same factor, the columns are positively correlated — exactly the
    situation that makes *cross-sectional* normalization (ranking assets against each other at one
    point in time) useful: subtracting the cross-sectional mean each period strips out the common
    market move and isolates the idiosyncratic part. ``mu_idio_sd`` > 0 gives assets heterogeneous
    drifts (handy for survivorship-bias demos). Used in Weeks 3, 9, and 10.
    """
    rng = _rng(seed)
    betas = rng.normal(beta_mean, beta_sd, n_assets)
    drifts = (
        rng.normal(0.0, mu_idio_sd, n_assets) if mu_idio_sd > 0 else np.zeros(n_assets)
    )
    market = rng.normal(mu_market, market_vol, n)
    idio = rng.normal(0.0, idio_vol, (n, n_assets))
    return market[:, None] * betas[None, :] + idio + drifts[None, :]


# ---------------------------------------------------------------------------
# Continuous-time price processes
# ---------------------------------------------------------------------------


def ornstein_uhlenbeck(
    n=1000, x0=0.0, kappa=1.0, theta=0.0, sigma=0.1, dt=1 / 252, seed=0
):
    """Ornstein–Uhlenbeck mean-reverting diffusion.

        dx_t = kappa * (theta - x_t) * dt + sigma * dW_t

    Returns the path ``x`` of length ``n+1`` starting at ``x0``. Uses the exact transition
    (Gaussian) for the discretised process, so the discretisation is unbiased for any ``dt``.
    Building block for pairs trading, short-rate models (Vasicek 1977), and stochastic-vol
    models that need a mean-reverting state variable.
    """
    rng = _rng(seed)
    if kappa <= 0:
        raise ValueError("kappa must be > 0 for a stationary OU process")
    e = np.exp(-kappa * dt)
    mu = theta * (1 - e)
    sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    z = rng.standard_normal(n)
    x = np.empty(n + 1)
    x[0] = x0
    for t in range(n):
        x[t + 1] = mu + e * x[t] + sd * z[t]
    return x


def vasicek(n=1000, r0=0.03, kappa=0.5, theta=0.04, sigma=0.01, dt=1 / 252, seed=0):
    """Vasicek (1977) short-rate model: an OU process applied to nominal rates.

    Same SDE as :func:`ornstein_uhlenbeck`; parameter names re-cast for rates use. Returns the
    short-rate path of length ``n+1``.
    """
    return ornstein_uhlenbeck(
        n=n, x0=r0, kappa=kappa, theta=theta, sigma=sigma, dt=dt, seed=seed
    )


def heston(
    n=1000,
    S0=100.0,
    v0=0.04,
    mu=0.05,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    dt=1 / 252,
    seed=0,
):
    """Heston (1993) stochastic volatility model. Returns prices ``S`` and variance ``v``.

        dS_t / S_t = mu * dt + sqrt(v_t) * dW_t^S
        dv_t       = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v
        corr(dW^S, dW^v) = rho

    Uses Euler with the *full-truncation* scheme of Lord-Koekkoek-van Dijk (2010) for the
    variance, which is the cleanest fix for negative intermediate variances without violating
    the Feller-like positivity on average. Returns ``(S, v)`` each of length ``n+1``.
    """
    rng = _rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w_S = z1
    w_v = rho * z1 + np.sqrt(1 - rho**2) * z2
    S = np.empty(n + 1)
    v = np.empty(n + 1)
    S[0] = S0
    v[0] = v0
    sqrt_dt = np.sqrt(dt)
    for t in range(n):
        v_pos = max(v[t], 0.0)
        S[t + 1] = S[t] * np.exp(
            (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * w_S[t]
        )
        v[t + 1] = (
            v[t] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * w_v[t]
        )
    return S, v


def bates(
    n=1000,
    S0=100.0,
    v0=0.04,
    mu=0.05,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    jump_intensity=0.5,
    jump_mu=-0.05,
    jump_sigma=0.10,
    dt=1 / 252,
    seed=0,
):
    """Bates (1996) stochastic-vol-with-jumps: Heston dynamics for variance and a compensated
    Merton jump component in the price. ``jump_intensity`` is in jumps per unit time. Returns
    ``(S, v)`` each of length ``n+1``.
    """
    rng = _rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w_S = z1
    w_v = rho * z1 + np.sqrt(1 - rho**2) * z2
    # Compensator for the jump component so the drift remains mu.
    kappa_J = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1
    S = np.empty(n + 1)
    v = np.empty(n + 1)
    S[0] = S0
    v[0] = v0
    sqrt_dt = np.sqrt(dt)
    for t in range(n):
        v_pos = max(v[t], 0.0)
        n_j = rng.poisson(jump_intensity * dt)
        J = 0.0
        if n_j > 0:
            J = np.sum(rng.normal(jump_mu, jump_sigma, n_j))
        S[t + 1] = S[t] * np.exp(
            (mu - jump_intensity * kappa_J - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * sqrt_dt * w_S[t]
            + J
        )
        v[t + 1] = (
            v[t] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * w_v[t]
        )
    return S, v


def sabr(n=1000, F0=100.0, alpha0=0.2, beta=0.5, nu=0.4, rho=-0.3, dt=1 / 252, seed=0):
    """SABR stochastic-α-β-ρ model (Hagan et al. 2002), simulated by Euler.

        dF_t      = alpha_t * F_t^beta * dW_t^F
        dalpha_t  = nu * alpha_t * dW_t^alpha
        corr(dW^F, dW^alpha) = rho

    Returns ``(F, alpha)`` each of length ``n+1``. The closed-form Hagan implied-vol expansion
    is the standard analytical companion; this generator is for simulating paths and back-testing
    SABR-implied prices against Monte Carlo. ``beta`` ∈ [0, 1] interpolates between normal
    (β=0) and lognormal (β=1).
    """
    rng = _rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w_F = z1
    w_a = rho * z1 + np.sqrt(1 - rho**2) * z2
    F = np.empty(n + 1)
    a = np.empty(n + 1)
    F[0] = F0
    a[0] = alpha0
    sqrt_dt = np.sqrt(dt)
    for t in range(n):
        F_pos = max(F[t], 1e-12)
        F[t + 1] = F[t] + a[t] * (F_pos**beta) * sqrt_dt * w_F[t]
        a[t + 1] = a[t] * np.exp(-0.5 * nu**2 * dt + nu * sqrt_dt * w_a[t])
    return F, a


def fgn(n=1000, H=0.5, sigma=1.0, seed=0):
    """Fractional Gaussian noise: stationary increments of fractional Brownian motion with
    Hurst exponent ``H`` ∈ (0, 1). H = 0.5 is i.i.d.; H > 0.5 persistent (trending);
    H < 0.5 anti-persistent (mean-reverting). Returns an array of length ``n``.

    Uses the exact Cholesky factorisation of the Toeplitz autocovariance; cost is O(n^2)
    so this generator suits moderate ``n`` (a few thousand). For long paths swap in a
    circulant-embedding (Davies-Harte) implementation.
    """
    rng = _rng(seed)
    k = np.arange(n)
    gamma = 0.5 * (
        np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H)
    )
    # Build the n×n Toeplitz autocovariance matrix and Cholesky-factor it.
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    C = gamma[idx] + 1e-12 * np.eye(n)
    L = np.linalg.cholesky(C)
    return sigma * (L @ rng.standard_normal(n))


def fbm(n=1000, H=0.5, sigma=1.0, seed=0):
    """Fractional Brownian motion of Hurst ``H``: cumulative sum of :func:`fgn`. Returns a
    path of length ``n+1`` starting at zero."""
    increments = fgn(n=n, H=H, sigma=sigma, seed=seed)
    return np.concatenate([[0.0], np.cumsum(increments)])


def rbergomi(n=500, S0=100.0, xi0=0.04, eta=1.5, H=0.1, rho=-0.7, dt=1 / 252, seed=0):
    """Rough Bergomi (Bayer-Friz-Gatheral 2016) — a simulated rough-volatility model.

        ξ_t = ξ_0 * exp(eta * W_t^H - 0.5 * eta^2 * Var(W_t^H))
        dS_t / S_t = sqrt(ξ_t) * dZ_t,   corr(W^H, Z) = rho

    where ``W^H`` is a Riemann-Liouville fBm with Hurst ``H`` < 0.5 (the "rough" regime).
    Returns ``(S, xi)``. This is the simple Riemann-sum version; production implementations
    use the hybrid scheme of Bennedsen-Lunde-Pakkanen (2017) for better small-time accuracy.
    Cost is O(n^2); keep ``n`` modest.
    """
    rng = _rng(seed)
    # Driving fGn for the variance process (correlated with the spot driver).
    fg = fgn(n=n, H=H, sigma=1.0, seed=seed)
    # Build Riemann-Liouville fBm: W^H_t = integral_0^t (t-s)^(H-0.5) dW_s.
    # Approximate with the cumulative-sum-of-power-weighted-increments form.
    w = np.empty(n + 1)
    w[0] = 0.0
    for t in range(1, n + 1):
        w[t] = w[t - 1] + (t * dt) ** (H - 0.5) * fg[t - 1] * np.sqrt(dt)
    var_w = (np.arange(n + 1) * dt) ** (2 * H) / (2 * H) if H > 0 else np.zeros(n + 1)
    xi = xi0 * np.exp(eta * w - 0.5 * eta**2 * var_w)
    # Spot driver: correlated Brownian increments.
    z = rng.standard_normal(n)
    z_corr = rho * (fg / np.sqrt(dt + 1e-12)) + np.sqrt(1 - rho**2) * z
    S = np.empty(n + 1)
    S[0] = S0
    sqrt_dt = np.sqrt(dt)
    for t in range(n):
        S[t + 1] = S[t] * np.exp(
            -0.5 * xi[t] * dt + np.sqrt(max(xi[t], 0.0)) * sqrt_dt * z_corr[t]
        )
    return S, xi


# ---------------------------------------------------------------------------
# Microstructure and execution
# ---------------------------------------------------------------------------


def hawkes(T=1000.0, mu=0.5, alpha=0.8, beta=1.0, seed=0):
    """Univariate Hawkes (1971) self-exciting point process with exponential kernel.

        λ(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

    Generated by Ogata's thinning over [0, T]. Returns the array of event times. Stability
    requires ``alpha < beta`` (branching ratio < 1). Used to model order-flow clustering,
    intraday volatility clustering, and contagion.
    """
    if alpha >= beta:
        raise ValueError("alpha must be < beta for a non-explosive Hawkes process")
    rng = _rng(seed)
    times = []
    lam_bar = mu
    t = 0.0
    while t < T:
        u = rng.random()
        t = t - np.log(u) / lam_bar
        if t >= T:
            break
        # Current intensity at t given history.
        if times:
            lam_t = mu + alpha * np.sum(np.exp(-beta * (t - np.asarray(times))))
        else:
            lam_t = mu
        if rng.random() <= lam_t / lam_bar:
            times.append(t)
            lam_bar = lam_t + alpha  # new upper bound after the jump
        else:
            lam_bar = max(lam_t, mu)
    return np.asarray(times)


def roll_bounce(n=1000, sigma_u=0.005, half_spread=0.001, seed=0):
    """Roll (1984) bid-ask bounce. The unobserved fundamental price follows a random walk;
    each transaction prints at fundamental ± half-spread with equal probability. Returns the
    observed transaction-price log returns of length ``n``, which exhibit the canonical
    negative first-order autocorrelation Cov(r_t, r_{t-1}) = -(half_spread)^2.
    """
    rng = _rng(seed)
    eps = rng.normal(0.0, sigma_u, n + 1)  # fundamental innovations
    p_star = np.cumsum(eps)  # log fundamental price
    q = rng.choice([-1.0, 1.0], n + 1)  # trade direction
    p_obs = p_star + half_spread * q
    return np.diff(p_obs)


def glosten_milgrom(n=1000, V_high=1.1, V_low=0.9, p_high=0.5, alpha=0.3, seed=0):
    """Glosten-Milgrom (1985) sequential-trade model with adverse selection.

    Each period a market maker faces an informed trader (probability ``alpha``) who knows the
    true value ``V`` ∈ {V_high, V_low}, or a noise trader (probability ``1 - alpha``) who buys
    or sells at random. The MM posts bid and ask quotes that are zero-expected-profit *given*
    the trade direction, then a trade occurs and beliefs update by Bayes' rule.

    Returns a dict with arrays of length ``n``: ``ask``, ``bid``, ``spread``, ``trade`` (+1
    buy, -1 sell), and ``belief`` (posterior probability that V = V_high). The true value
    ``V_true`` is drawn once at the start and reported in the dict.
    """
    rng = _rng(seed)
    V_true = V_high if rng.random() < p_high else V_low
    p = p_high
    ask = np.empty(n)
    bid = np.empty(n)
    spread = np.empty(n)
    trade = np.empty(n)
    belief = np.empty(n)
    for t in range(n):
        # Zero-profit MM quotes conditional on trade direction.
        # P(buy) = alpha * P(V=H) + (1-alpha)/2 ; P(sell) symmetric.
        p_buy = alpha * p + (1 - alpha) * 0.5
        p_sell = alpha * (1 - p) + (1 - alpha) * 0.5
        a = (
            alpha * p * V_high + (1 - alpha) * 0.5 * (p * V_high + (1 - p) * V_low)
        ) / p_buy
        b = (
            alpha * (1 - p) * V_low + (1 - alpha) * 0.5 * (p * V_high + (1 - p) * V_low)
        ) / p_sell
        ask[t] = a
        bid[t] = b
        spread[t] = a - b
        belief[t] = p
        # Generate this period's trade.
        if rng.random() < alpha:
            tr = +1 if V_true == V_high else -1  # informed: buy at H, sell at L
        else:
            tr = +1 if rng.random() < 0.5 else -1
        trade[t] = tr
        # Bayes' update of the high-value posterior given the observed trade.
        if tr == +1:
            p = (alpha * p + (1 - alpha) * 0.5 * p) / p_buy
        else:
            p = ((1 - alpha) * 0.5 * p) / p_sell
        p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return {
        "ask": ask,
        "bid": bid,
        "spread": spread,
        "trade": trade,
        "belief": belief,
        "V_true": V_true,
        "V_high": V_high,
        "V_low": V_low,
    }


def kyle(n=1, sigma_v=1.0, sigma_u=1.0, p0=100.0, seed=0):
    """Kyle (1985) single-auction (or repeated-single-auction) model of strategic
    informed trading and linear price impact.

    In a single auction the informed trader knows the liquidating value ``v ~ N(p0, sigma_v^2)``
    and noise traders submit ``u ~ N(0, sigma_u^2)``. The market maker observes only the net
    order flow ``y = x + u`` and sets a price linear in ``y``. The Kyle equilibrium has

        beta  = sigma_u / sigma_v        # informed trading aggressiveness
        lambda = sigma_v / (2 * sigma_u) # market depth / price-impact coefficient
        x* = beta * (v - p0)
        p  = p0 + lambda * y

    Returns a dict with arrays of length ``n``: ``v``, ``u``, ``x``, ``y``, ``price``, plus
    scalar coefficients ``beta`` and ``lambda_``. For ``n > 1`` we repeat the single-period
    equilibrium with independent draws — Kyle's true multi-period continuous-auction extension
    is more elaborate; the single-period version is the workhorse pedagogical case.
    """
    rng = _rng(seed)
    beta = sigma_u / sigma_v
    lam = sigma_v / (2 * sigma_u)
    v = rng.normal(p0, sigma_v, n)
    u = rng.normal(0.0, sigma_u, n)
    x = beta * (v - p0)
    y = x + u
    p = p0 + lam * y
    return {"v": v, "u": u, "x": x, "y": y, "price": p, "beta": beta, "lambda_": lam}


def limit_order_book(
    n_events=10000,
    L=15,
    lam0=8.0,
    k_decay=0.30,
    mu_mkt=1.4,
    theta=0.20,
    tick=0.01,
    mid0=100.0,
    seed=0,
):
    """Cont-Stoikov-style continuous-double-auction LOB simulator.

    Six event types fire as a Poisson superposition: limit-buy / limit-sell arrivals (rate per
    level decays exponentially from the touch, ``lam(d) = lam0 * exp(-k_decay * d)``), market
    buys / sells (rate ``mu_mkt`` each side), and per-share cancellations on each side (rate
    ``theta * book_depth``). The mid moves by half a tick each time the touch shifts.

    Returns a dict with arrays of length ``n_events``: ``mid``, ``best_bid``, ``best_ask``,
    ``spread``, plus ``avg_depth_bid`` and ``avg_depth_ask`` (length ``L``) — the
    time-averaged resting depth at each tick away from the touch.
    """
    rng = _rng(seed)
    bid = np.zeros(L)
    ask = np.zeros(L)
    bid[:] = 50 + 20 * np.arange(L)  # seed the book
    ask[:] = 50 + 20 * np.arange(L)
    mid = mid0
    lam_arr = lam0 * np.exp(-k_decay * np.arange(1, L + 1))
    mid_h = np.empty(n_events)
    bb_h = np.empty(n_events)
    ba_h = np.empty(n_events)
    sp_h = np.empty(n_events)
    depth_b = np.zeros(L)
    depth_a = np.zeros(L)
    for step in range(n_events):
        rates = np.array(
            [
                lam_arr.sum(),
                lam_arr.sum(),
                mu_mkt,
                mu_mkt,
                theta * bid.sum(),
                theta * ask.sum(),
            ]
        )
        u = rng.random() * rates.sum()
        which = int(np.searchsorted(np.cumsum(rates), u))
        if which == 0:
            d = rng.choice(L, p=lam_arr / lam_arr.sum())
            bid[d] += 1
        elif which == 1:
            d = rng.choice(L, p=lam_arr / lam_arr.sum())
            ask[d] += 1
        elif which == 2 and ask[0] > 0:
            ask[0] -= 1
            if ask[0] == 0:
                ask[:-1] = ask[1:]
                ask[-1] = 0
                mid += 0.5 * tick
        elif which == 3 and bid[0] > 0:
            bid[0] -= 1
            if bid[0] == 0:
                bid[:-1] = bid[1:]
                bid[-1] = 0
                mid -= 0.5 * tick
        elif which == 4 and bid.sum() > 0:
            d = rng.choice(L, p=bid / bid.sum())
            bid[d] = max(0, bid[d] - 1)
        elif which == 5 and ask.sum() > 0:
            d = rng.choice(L, p=ask / ask.sum())
            ask[d] = max(0, ask[d] - 1)
        mid_h[step] = mid
        bb_off = int(np.argmax(bid > 0)) if bid.any() else L - 1
        ba_off = int(np.argmax(ask > 0)) if ask.any() else L - 1
        bb_h[step] = mid - 0.5 * tick - bb_off * tick
        ba_h[step] = mid + 0.5 * tick + ba_off * tick
        sp_h[step] = ba_h[step] - bb_h[step]
        depth_b += bid
        depth_a += ask
    return {
        "mid": mid_h,
        "best_bid": bb_h,
        "best_ask": ba_h,
        "spread": sp_h,
        "avg_depth_bid": depth_b / n_events,
        "avg_depth_ask": depth_a / n_events,
    }


def almgren_chriss(X=1.0, T=1.0, N=50, sigma=0.02, eta=2.5e-6, gamma=0.0, lam=1e-6):
    """Almgren-Chriss (2000) optimal-execution trajectory under linear permanent (``gamma``)
    and linear temporary (``eta``) price impact, quadratic risk aversion ``lam``, volatility
    ``sigma``, and ``N`` equal time steps over horizon ``T``. Closed-form solution.

    Returns a dict with the holdings path ``x`` (length ``N+1``), per-period trades ``n_trades``
    (length ``N``), the expected cost ``E_cost`` (cash), and the cost standard deviation
    ``sd_cost``. The cost-variance frontier is traced by varying ``lam`` and re-calling.
    """
    t_grid = np.linspace(0.0, T, N + 1)
    if lam <= 0:
        x = X * (1 - t_grid / T)
    else:
        kappa = np.sqrt(lam * sigma**2 / eta)
        if kappa * T < 1e-3:
            x = X * (1 - t_grid / T)
        else:
            x = X * np.sinh(kappa * (T - t_grid)) / np.sinh(kappa * T)
    n_trades = x[:-1] - x[1:]
    tau = T / N
    x_mid = x[:-1]
    e_cost = 0.5 * gamma * X**2 + (eta / tau) * np.sum(n_trades**2)
    v_cost = sigma**2 * tau * np.sum(x_mid**2)
    return {
        "x": x,
        "n_trades": n_trades,
        "E_cost": float(e_cost),
        "sd_cost": float(np.sqrt(v_cost)),
    }
