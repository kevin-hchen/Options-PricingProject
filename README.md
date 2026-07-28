# Monte Carlo Options Pricing Engine

A small, well-commented Python engine for pricing options by Monte Carlo — built to learn how derivatives pricing works. Every result is checked against closed-form Black-Scholes.

Pure NumPy + SciPy, no pricing libraries.

## Features

- European, Asian, and barrier options under geometric Brownian motion
- Variance reduction: antithetic and control variates
- Greeks (Delta, Vega) via pathwise and likelihood-ratio methods
- Every price comes with a standard error
- Test suite validating against Black-Scholes

## Quick start

```bash
pip install -r requirements.txt

python examples/01_european_vs_bs.py     # MC converges to Black-Scholes
python examples/02_variance_reduction.py # antithetic + control variates
python examples/03_asian_and_barrier.py  # path-dependent options
python examples/04_greeks.py             # Greeks, two ways

pytest                                   # validated against Black-Scholes
```

Price an option:

```python
from mcpricer import pricers, black_scholes as bs

res = pricers.price_european(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
print(res.price)                             # ~10.45
print(bs.bs_price(100, 100, 0.05, 0.2, 1.0)) # 10.4506  (exact)
print(res.confidence_interval(0.95))         # (lo, hi)
```

## How it works

Under the risk-neutral measure, an option's price is the expected discounted payoff:

```
price = E[ e^(-rT) · payoff(path) ]
```

Simulate many price paths, discount each payoff, average. The error shrinks like `1/√n`, so variance reduction matters — that's most of the interesting code.

## Layout

Read the modules in this order:

| Module | Role |
|---|---|
| [`black_scholes.py`](mcpricer/black_scholes.py) | Analytic prices & Greeks — the ground truth |
| [`gbm.py`](mcpricer/gbm.py) | Simulate GBM paths (+ antithetic sampling) |
| [`payoffs.py`](mcpricer/payoffs.py) | European / Asian / barrier payoffs |
| [`engine.py`](mcpricer/engine.py) | Average, error bars, control variates |
| [`pricers.py`](mcpricer/pricers.py) | One function per option type |
| [`greeks.py`](mcpricer/greeks.py) | Pathwise & likelihood-ratio Greeks |

```
mcpricer/   the engine
examples/   numbered, runnable lessons
tests/      validation against Black-Scholes
```

## Sample output

MC converges to the exact price (10.4506), error shrinks with `√n`:

```
     paths   MC price   std err
    10,000    10.5194    0.1462
   100,000    10.4362    0.0465
 1,000,000    10.4697    0.0147
```

Control variate cuts the Asian error ~24× for the same number of paths:

```
    no control     5.8390    std err 0.01767
   control var     5.8579    std err 0.00074
```

## Next steps

- Digital payoffs — where pathwise Delta fails and likelihood-ratio shines
- More Greeks: Gamma, Rho
- Quasi-Monte Carlo (Sobol) for faster convergence
- Stochastic volatility (Heston); American options (Longstaff-Schwartz)

## References

- Hull, *Options, Futures, and Other Derivatives*
- Glasserman, *Monte Carlo Methods in Financial Engineering*
- Kemna & Vorst (1990) — geometric-Asian closed form

## License

MIT
