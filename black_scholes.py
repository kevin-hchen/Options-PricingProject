"""Analytic Black-Scholes-Merton formulas for European options.

The closed-form ground truth: MC prices must converge to these, and the tests
rely on it. Assumes the underlying follows geometric Brownian motion.

Notation (used throughout the package):
    S      spot price today
    K      strike
    r      risk-free rate (annualised, continuous)
    sigma  volatility (annualised)
    T      time to maturity (years)
    q      dividend yield (default 0.0)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, r, sigma, T, q=0.0):
    """Return the ``d1`` and ``d2`` terms shared by every BS formula.

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    """
    S, K, sigma, T = map(np.asarray, (S, K, sigma, T))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, r, sigma, T, option="call", q=0.0):
    """Black-Scholes price of a European call or put.

    Parameters
    ----------
    option : {"call", "put"}
    """
    d1, d2 = d1_d2(S, K, r, sigma, T, q)
    disc = np.exp(-r * T)      # discount factor  e^{-rT}
    div = np.exp(-q * T)       # dividend factor   e^{-qT}
    if option == "call":
        return S * div * norm.cdf(d1) - K * disc * norm.cdf(d2)
    elif option == "put":
        return K * disc * norm.cdf(-d2) - S * div * norm.cdf(-d1)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


# ---------------------------------------------------------------------------
# Analytic Greeks. We use these to check the Monte Carlo Greeks in greeks.py.
# ---------------------------------------------------------------------------

def bs_delta(S, K, r, sigma, T, option="call", q=0.0):
    """dPrice/dS -- sensitivity to the spot price."""
    d1, _ = d1_d2(S, K, r, sigma, T, q)
    div = np.exp(-q * T)
    if option == "call":
        return div * norm.cdf(d1)
    return div * (norm.cdf(d1) - 1.0)


def bs_vega(S, K, r, sigma, T, q=0.0):
    """dPrice/dsigma -- sensitivity to volatility (same for calls and puts)."""
    d1, _ = d1_d2(S, K, r, sigma, T, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_gamma(S, K, r, sigma, T, q=0.0):
    """d2Price/dS2 -- convexity in the spot (same for calls and puts)."""
    d1, _ = d1_d2(S, K, r, sigma, T, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_rho(S, K, r, sigma, T, option="call", q=0.0):
    """dPrice/dr -- sensitivity to the risk-free rate."""
    _, d2 = d1_d2(S, K, r, sigma, T, q)
    disc = np.exp(-r * T)
    if option == "call":
        return K * T * disc * norm.cdf(d2)
    return -K * T * disc * norm.cdf(-d2)
