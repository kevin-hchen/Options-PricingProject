"""Lesson 3 -- Path-dependent options: Asian and barrier.

These payoffs depend on the *whole* price path, not just its endpoint, so we
simulate full paths. We also demonstrate the in/out barrier identity, a handy
correctness check.

Run:  python examples/03_asian_and_barrier.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcpricer import black_scholes as bs
from mcpricer import pricers

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
euro = bs.bs_price(S0, K, r, sigma, T, option="call")
print(f"Vanilla European call (Black-Scholes): {euro:.4f}\n")

# --- Asian --------------------------------------------------------------
asian = pricers.price_asian(S0, K, r, sigma, T, option="call",
                            n_paths=200_000, n_steps=50,
                            rng=np.random.default_rng(3))
print(f"Arithmetic Asian call: {asian.price:.4f} +/- {asian.std_error:.4f}")
print("  (worth less than the European: averaging damps volatility)\n")

# --- Barrier: in/out parity --------------------------------------------
B = 130.0
out = pricers.price_barrier(S0, K, r, sigma, T, barrier=B, kind="up-and-out",
                            option="call", n_paths=200_000, n_steps=100,
                            rng=np.random.default_rng(4))
inn = pricers.price_barrier(S0, K, r, sigma, T, barrier=B, kind="up-and-in",
                            option="call", n_paths=200_000, n_steps=100,
                            rng=np.random.default_rng(4))
print(f"Up-and-out call (B={B}): {out.price:.4f}")
print(f"Up-and-in  call (B={B}): {inn.price:.4f}")
print(f"in + out = {out.price + inn.price:.4f}   vs vanilla {euro:.4f}")
print("  (they must sum to the vanilla European -- a free correctness check)")
