# finmlsim

**Scholarly simulators for financial machine learning** — a small, NumPy-only package of
citation-anchored simulation models for research, teaching, and self-study. Twenty-plus
generators across classical return series (Gaussian, Student-t, ARMA, GARCH,
regime-switching, jump-diffusion, Cont–Bouchaud), continuous-time price processes (Heston,
Bates, SABR, rough Bergomi, Ornstein–Uhlenbeck / Vasicek, fractional Brownian motion), and
market microstructure / execution (Hawkes, Roll, Glosten–Milgrom, Kyle, a Cont–Stoikov
limit-order book, Almgren–Chriss). Each model is one short function with a citation in its
docstring.

Dial any *stylized fact* of markets on and off — fat tails, volatility clustering, jumps,
regime switching, long memory, order-flow self-excitement, adverse selection — and watch
what the methods you care about do under those conditions, before you turn them loose on
real data.

```python
import finmlsim as fms

# Volatility clustering plus fat tails
r = fms.simulate.garch(n=2000, dist="t", seed=0)
fms.stylized.summary(r)
prices = fms.stylized.to_prices(r)

# Stochastic-vol pricing with Heston
S, v = fms.simulate.heston(n=252, seed=0)

# Cont-Stoikov limit-order-book dynamics
book = fms.simulate.limit_order_book(n_events=20_000, seed=0)
```

## Install
```bash
pip install finmlsim
# core needs only numpy; analysis extras: pip install "finmlsim[analysis]"
```

## Modules
- **`simulate`** — generators grouped by family:
  - *Return series* (1-D log returns): `gaussian`, `student_t`, `arma`, `garch`,
    `regime_switching`, `jump_diffusion`, `cont_bouchaud`.
  - *Cross-sectional panel*: `panel` (2-D, common-factor structure).
  - *Continuous-time price processes*: `ornstein_uhlenbeck` / `vasicek`,
    `heston` (1993), `bates` (1996), `sabr` (Hagan et al. 2002), `rbergomi`
    (Bayer-Friz-Gatheral 2016), `fbm` / `fgn` (Hurst-parametrized).
  - *Microstructure and execution*: `hawkes` (1971 self-exciting), `roll_bounce` (1984),
    `glosten_milgrom` (1985), `kyle` (1985), `limit_order_book` (Cont-Stoikov style),
    `almgren_chriss` (2000 optimal execution).
  Every generator takes a `seed` for reproducibility.
- **`stylized`** — diagnostics: `acf`, `excess_kurtosis`, `vol_clustering`, `summary`, `to_prices`.
- **`metrics`** — strategy evaluation: `sharpe`, `max_drawdown`, `rank_ic`, `ic_ir`,
  `turnover`, `deflated_sharpe`, `min_track_record_length`.
- **`resample`** — block bootstraps for dependent series: `stationary_bootstrap`
  (Politis-Romano 1994), `circular_block_bootstrap` (Politis-Romano 1992). Pairs with the
  multiple-testing diagnostics in `metrics`.
- **`data`** — loaders for the bundled empirical datasets (Ken French daily factors,
  momentum, and 12 industry portfolios): `ff_factors`, `ff_momentum`, `industries`,
  `market_return`. Offline; see `datasets/famafrench/` in the source repository.

## The stylized facts each return-series generator produces
| Generator | Fat tails | Vol clustering | Mean autocorrelation |
|-----------|:---------:|:--------------:|:--------------------:|
| `gaussian` | – | – | – |
| `student_t` | ✓ | – | – |
| `arma` | – | – | ✓ (by design) |
| `garch` | mild | ✓ | – |
| `garch(dist="t")` | ✓ | ✓ | – |
| `regime_switching` | ✓ (mixed) | ✓ | – |
| `jump_diffusion` | ✓ | – | – |
| `cont_bouchaud` | ✓ (endogenous) | (weak) | – |

## What the continuous-time and microstructure models give you
| Generator | Returns | Key parameter | Use case |
|-----------|---------|---------------|----------|
| `ornstein_uhlenbeck` / `vasicek` | path | `kappa` (mean-reversion rate) | pairs, short rates, vol state |
| `heston` | `(S, v)` | `xi`, `rho`, `kappa` | stochastic vol; smile/skew |
| `bates` | `(S, v)` | + `jump_intensity` | stoch-vol with crash risk |
| `sabr` | `(F, alpha)` | `beta`, `nu`, `rho` | FX / rates vol surfaces |
| `rbergomi` | `(S, xi)` | `H` < 0.5 | rough volatility |
| `fbm` / `fgn` | path / increments | `H` | Hurst-parameterized paths |
| `hawkes` | event times | `alpha`, `beta` | order-flow clustering, contagion |
| `roll_bounce` | tx-price log returns | `half_spread` | bid-ask bounce, negative AC(1) |
| `glosten_milgrom` | dict (ask/bid/spread/belief) | `alpha` (informed prob) | adverse-selection spread |
| `kyle` | dict (v, u, x, y, price) | `sigma_v / sigma_u` | strategic informed trading |
| `limit_order_book` | dict (mid, spread, depth) | `lam0`, `theta`, `mu_mkt` | LOB dynamics, depth, impact |
| `almgren_chriss` | dict (path, costs) | `lam` (risk aversion) | execution trajectories |

These are **teaching and research models**, not production calibrations. Planned additions:
a cost-aware `backtest` module and leak-free `features` helpers.

## License

MIT. See `LICENSE` in the source repository.
