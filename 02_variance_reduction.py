"""Lesson 2 -- Variance reduction buys accuracy for free.

Same number of paths, smaller error. We compare three estimators of a European
call:
    plain          i.i.d. sampling
    antithetic     pair each Z with -Z
and then, for an Asian call, add a control variate on top.

Run:  python examples/02_variance_reduction.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcpricer import pricers

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
N = 100_000

print("European call -- effect of antithetic sampling")
print(f"{'method':>14} {'price':>10} {'std err':>10}")
print("-" * 36)
for label, anti in (("plain", False), ("antithetic", True)):
    res = pricers.price_european(S0, K, r, sigma, T, n_paths=N,
                                 antithetic=anti, rng=np.random.default_rng(1))
    print(f"{label:>14} {res.price:>10.4f} {res.std_error:>10.5f}")

print("\nArithmetic Asian call -- effect of a geometric-Asian control variate")
print(f"{'method':>14} {'price':>10} {'std err':>10}")
print("-" * 36)
for label, cv in (("no control", False), ("control var", True)):
    res = pricers.price_asian(S0, K, r, sigma, T, n_paths=N, n_steps=50,
                              use_control_variate=cv,
                              rng=np.random.default_rng(2))
    print(f"{label:>14} {res.price:>10.4f} {res.std_error:>10.5f}")

print("\nBoth tricks leave the price unbiased while shrinking the error bar.")
