"""mcpricer -- a learning-oriented Monte Carlo options pricing engine.

Suggested reading order (each layer builds on the previous):
    black_scholes.py  analytic prices/Greeks -- the ground truth
    gbm.py            simulate price paths (+ antithetic variates)
    payoffs.py        European / Asian / barrier payoffs
    engine.py         average, error bars, control variates
    pricers.py        one function per option type
    greeks.py         pathwise & likelihood-ratio Greeks

Quick start
-----------
>>> from mcpricer import pricers, black_scholes as bs
>>> res = pricers.price_european(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
>>> res.price, bs.bs_price(100, 100, 0.05, 0.2, 1.0)   # MC vs closed form
"""

from . import black_scholes, engine, gbm, greeks, payoffs, pricers

__all__ = ["black_scholes", "engine", "gbm", "greeks", "payoffs", "pricers"]
__version__ = "0.1.0"
