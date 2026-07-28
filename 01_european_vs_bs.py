"""Lesson 1 -- Monte Carlo converges to Black-Scholes.

The foundational check: as we add paths, the MC price of a European call homes
in on the exact Black-Scholes value, and its error bar shrinks like 1/sqrt(n).

Run:  python examples/01_european_vs_bs.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcpricer import black_scholes as bs
from mcpricer import pricers

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
analytic = bs.bs_price(S0, K, r, sigma, T, option="call")
print(f"Black-Scholes call price:  {analytic:.4f}\n")

print(f"{'paths':>10} {'MC price':>10} {'std err':>9} {'error':>9}")
print("-" * 42)
for n in (1_000, 10_000, 100_000, 1_000_000):
    res = pricers.price_european(S0, K, r, sigma, T, option="call",
                                 n_paths=n, antithetic=False,
                                 rng=np.random.default_rng(0))
    print(f"{n:>10,} {res.price:>10.4f} {res.std_error:>9.4f} "
          f"{abs(res.price - analytic):>9.4f}")

print("\nNotice: 10x more paths -> the error bar shrinks by ~sqrt(10) ~ 3.2x.")
