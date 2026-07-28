"""Sanity checks on the analytic formulas themselves.

Before we trust Black-Scholes as the 'ground truth' for the Monte Carlo tests,
we pin down a few facts that must hold for *any* correct implementation:
put-call parity, and finite-difference agreement of the analytic Greeks.
"""

import numpy as np

from mcpricer import black_scholes as bs

PARAMS = dict(S=100.0, K=105.0, r=0.05, sigma=0.2, T=1.5, q=0.01)


def test_put_call_parity():
    # C - P = S e^{-qT} - K e^{-rT}
    c = bs.bs_price(option="call", **PARAMS)
    p = bs.bs_price(option="put", **PARAMS)
    S, K, r, q, T = (PARAMS[k] for k in ("S", "K", "r", "q", "T"))
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


def test_delta_matches_finite_difference():
    h = 1e-4
    S = PARAMS["S"]
    rest = {k: v for k, v in PARAMS.items() if k != "S"}
    up = bs.bs_price(S=S + h, option="call", **rest)
    dn = bs.bs_price(S=S - h, option="call", **rest)
    fd_delta = (up - dn) / (2 * h)
    assert abs(fd_delta - bs.bs_delta(option="call", **PARAMS)) < 1e-5


def test_vega_matches_finite_difference():
    h = 1e-5
    sigma = PARAMS["sigma"]
    rest = {k: v for k, v in PARAMS.items() if k != "sigma"}
    up = bs.bs_price(sigma=sigma + h, option="call", **rest)
    dn = bs.bs_price(sigma=sigma - h, option="call", **rest)
    fd_vega = (up - dn) / (2 * h)
    assert abs(fd_vega - bs.bs_vega(**PARAMS)) < 1e-4
