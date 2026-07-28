"""Lesson 4 -- Greeks two ways, checked against Black-Scholes.

Delta and Vega computed by the pathwise and likelihood-ratio methods, side by
side with the analytic values. Watch how, for these smooth payoffs, the
pathwise estimator has the tighter error bar -- the LR method's edge only shows
up on discontinuous payoffs (digitals), where the pathwise method breaks.

Run:  python examples/04_greeks.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcpricer import black_scholes as bs
from mcpricer import greeks

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 500_000


def show(name, res, analytic):
    err = abs(res.price - analytic)
    print(f"{name:>26}: {res.price:>8.4f} +/- {res.std_error:.4f}   "
          f"(analytic {analytic:.4f}, off by {err:.4f})")


print("DELTA (dPrice/dS0) for a European call")
analytic_delta = bs.bs_delta(S0, K, r, sigma, T, "call")
show("pathwise", greeks.pathwise_delta(S0, K, r, sigma, T, n_paths=N,
     rng=np.random.default_rng(5)), analytic_delta)
show("likelihood-ratio", greeks.likelihood_ratio_delta(S0, K, r, sigma, T,
     n_paths=N, rng=np.random.default_rng(5)), analytic_delta)

print("\nVEGA (dPrice/dsigma) for a European call")
analytic_vega = bs.bs_vega(S0, K, r, sigma, T)
show("pathwise", greeks.pathwise_vega(S0, K, r, sigma, T, n_paths=N,
     rng=np.random.default_rng(6)), analytic_vega)
show("likelihood-ratio", greeks.likelihood_ratio_vega(S0, K, r, sigma, T,
     n_paths=N, rng=np.random.default_rng(6)), analytic_vega)

print("\nFor smooth payoffs the pathwise error bars are tighter; the LR method "
      "\nearns its keep on discontinuous payoffs where pathwise fails.")
