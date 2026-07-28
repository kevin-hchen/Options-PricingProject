"""Validate the Monte Carlo pricers against Black-Scholes and known identities.

We use a fixed RNG seed so the tests are deterministic, and we check that the
MC price lands within a few standard errors of the analytic value -- the honest
statistical statement, rather than an arbitrary absolute tolerance.
"""

import numpy as np
import pytest

from mcpricer import black_scholes as bs
from mcpricer import pricers

SEED = 12345
P = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def rng():
    return np.random.default_rng(SEED)


@pytest.mark.parametrize("option", ["call", "put"])
def test_european_matches_black_scholes(option):
    res = pricers.price_european(option=option, n_paths=200_000,
                                 rng=rng(), **P)
    analytic = bs.bs_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"],
                           option=option)
    # Within 4 standard errors -> passes ~99.99% of the time when correct.
    assert abs(res.price - analytic) < 4 * res.std_error


def test_antithetic_reduces_error():
    plain = pricers.price_european(n_paths=50_000, antithetic=False, rng=rng(), **P)
    anti = pricers.price_european(n_paths=50_000, antithetic=True, rng=rng(), **P)
    assert anti.std_error < plain.std_error


def test_asian_cheaper_than_european():
    # Averaging reduces effective volatility, so an Asian call is worth less.
    asian = pricers.price_asian(option="call", n_paths=100_000, n_steps=50,
                                rng=rng(), **P)
    euro = bs.bs_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"], option="call")
    assert asian.price < euro


def test_control_variate_reduces_error():
    with_cv = pricers.price_asian(n_paths=50_000, n_steps=50,
                                  use_control_variate=True, rng=rng(), **P)
    without_cv = pricers.price_asian(n_paths=50_000, n_steps=50,
                                     use_control_variate=False, rng=rng(), **P)
    assert with_cv.std_error < without_cv.std_error


def test_geometric_asian_matches_closed_form():
    res = pricers.price_asian(average="geometric", n_paths=200_000, n_steps=50,
                              antithetic=True, rng=rng(), **P)
    closed = pricers.geometric_asian_price(P["S0"], P["K"], P["r"], P["sigma"],
                                           P["T"], option="call", n_steps=50)
    assert abs(res.price - closed) < 4 * res.std_error


def test_barrier_in_out_parity():
    # up-and-in + up-and-out should reprice the vanilla European call.
    B = 130.0
    common = dict(barrier=B, option="call", n_paths=100_000, n_steps=100)
    # Same seed -> same paths -> the identity holds path-by-path, nearly exact.
    out = pricers.price_barrier(kind="up-and-out", rng=rng(), **common, **P)
    inn = pricers.price_barrier(kind="up-and-in", rng=rng(), **common, **P)
    euro = bs.bs_price(P["S0"], P["K"], P["r"], P["sigma"], P["T"], option="call")
    assert abs((out.price + inn.price) - euro) < 4 * (out.std_error + inn.std_error)
