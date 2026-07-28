"""Validate Monte Carlo Greeks against the analytic Black-Scholes Greeks.

Both the pathwise and likelihood-ratio Delta/Vega should agree with the closed
forms to within a few standard errors.
"""

import numpy as np
import pytest

from mcpricer import black_scholes as bs
from mcpricer import greeks

SEED = 2024
P = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def rng():
    return np.random.default_rng(SEED)


@pytest.mark.parametrize("option", ["call", "put"])
def test_pathwise_delta(option):
    res = greeks.pathwise_delta(option=option, n_paths=400_000, rng=rng(), **P)
    analytic = bs.bs_delta(P["S0"], P["K"], P["r"], P["sigma"], P["T"], option)
    assert abs(res.price - analytic) < 4 * res.std_error


@pytest.mark.parametrize("option", ["call", "put"])
def test_likelihood_ratio_delta(option):
    res = greeks.likelihood_ratio_delta(option=option, n_paths=400_000,
                                        rng=rng(), **P)
    analytic = bs.bs_delta(P["S0"], P["K"], P["r"], P["sigma"], P["T"], option)
    assert abs(res.price - analytic) < 4 * res.std_error


def test_pathwise_vega():
    res = greeks.pathwise_vega(option="call", n_paths=400_000, rng=rng(), **P)
    analytic = bs.bs_vega(P["S0"], P["K"], P["r"], P["sigma"], P["T"])
    assert abs(res.price - analytic) < 4 * res.std_error


def test_likelihood_ratio_vega():
    res = greeks.likelihood_ratio_vega(option="call", n_paths=400_000,
                                       rng=rng(), **P)
    analytic = bs.bs_vega(P["S0"], P["K"], P["r"], P["sigma"], P["T"])
    assert abs(res.price - analytic) < 4 * res.std_error
