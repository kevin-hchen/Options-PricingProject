"""Monte Carlo Greeks: pathwise and likelihood-ratio estimators.

Greeks are price sensitivities (Delta = d/dS0, Vega = d/dsigma). Instead of
noisy bump-and-revalue, both methods differentiate analytically inside the
expectation -- bump-free, lower variance, one simulation. Checked here against
the closed-form Greeks in ``black_scholes.py``.

Setup: with ``Z ~ N(0, 1)``, S_T = S0 * exp[(r - q - sigma^2/2)T + sigma sqrt(T) Z].

1. Pathwise -- differentiate the *payoff*:
       Delta = E[ e^{-rT} * payoff'(S_T) * dS_T/dS0 ]
   Lower variance, but needs a differentiable payoff (fails on digitals).

2. Likelihood-ratio (LR) -- differentiate the *density*:
       Delta = E[ e^{-rT} * payoff(S_T) * score ],  score = d(log density)/d(param)
   Payoff untouched, so it handles discontinuous payoffs; higher variance on
   smooth ones. Score functions are the standard GBM results (Glasserman).
"""

from __future__ import annotations

import numpy as np

from .engine import MCResult, estimate


def _sample_terminal(S0, r, sigma, T, n_paths, q, rng, antithetic):
    """Draw ``Z`` and the matching ``S_T``; we need ``Z`` for the estimators."""
    from .gbm import _draw_normals

    rng = np.random.default_rng() if rng is None else rng
    Z = _draw_normals((n_paths,), rng, antithetic)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return Z, S_T


# ---------------------------------------------------------------------------
# Pathwise estimators (smooth payoffs: European call / put)
# ---------------------------------------------------------------------------

def pathwise_delta(S0, K, r, sigma, T, option="call", q=0.0,
                   n_paths=200_000, antithetic=True, rng=None):
    """Delta = dPrice/dS0 via the pathwise method.

    call:  e^{-rT} * 1{S_T > K} * (S_T / S0)
    put:  -e^{-rT} * 1{S_T < K} * (S_T / S0)
    """
    _, S_T = _sample_terminal(S0, r, sigma, T, n_paths, q, rng, antithetic)
    disc = np.exp(-r * T)
    if option == "call":
        samples = disc * (S_T > K) * (S_T / S0)
    elif option == "put":
        samples = -disc * (S_T < K) * (S_T / S0)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return estimate(samples, antithetic=antithetic)


def pathwise_vega(S0, K, r, sigma, T, option="call", q=0.0,
                  n_paths=200_000, antithetic=True, rng=None):
    """Vega = dPrice/dsigma via the pathwise method.

    Uses ``dS_T/dsigma = S_T * (sqrt(T) Z - sigma T)``. The indicator is the
    same for call and put; only its complement flips, so:
        call:  e^{-rT} * 1{S_T > K} * dS_T/dsigma
        put:   e^{-rT} * 1{S_T < K} * dS_T/dsigma
    """
    Z, S_T = _sample_terminal(S0, r, sigma, T, n_paths, q, rng, antithetic)
    disc = np.exp(-r * T)
    dST_dsigma = S_T * (np.sqrt(T) * Z - sigma * T)
    indicator = (S_T > K) if option == "call" else (S_T < K)
    samples = disc * indicator * dST_dsigma
    return estimate(samples, antithetic=antithetic)


# ---------------------------------------------------------------------------
# Likelihood-ratio estimators (work for any payoff, incl. discontinuous ones)
# ---------------------------------------------------------------------------

def _payoff(S_T, K, option):
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def likelihood_ratio_delta(S0, K, r, sigma, T, option="call", q=0.0,
                           n_paths=200_000, antithetic=True, rng=None):
    """Delta via the likelihood-ratio method.

    Score for S0:  Z / (S0 * sigma * sqrt(T)).
        Delta = E[ e^{-rT} * payoff(S_T) * Z / (S0 sigma sqrt(T)) ]
    """
    Z, S_T = _sample_terminal(S0, r, sigma, T, n_paths, q, rng, antithetic)
    disc = np.exp(-r * T)
    score = Z / (S0 * sigma * np.sqrt(T))
    samples = disc * _payoff(S_T, K, option) * score
    return estimate(samples, antithetic=antithetic)


def likelihood_ratio_vega(S0, K, r, sigma, T, option="call", q=0.0,
                          n_paths=200_000, antithetic=True, rng=None):
    """Vega via the likelihood-ratio method.

    Score for sigma:  (Z^2 - 1) / sigma - Z * sqrt(T).
        Vega = E[ e^{-rT} * payoff(S_T) * ((Z^2 - 1)/sigma - Z sqrt(T)) ]
    """
    Z, S_T = _sample_terminal(S0, r, sigma, T, n_paths, q, rng, antithetic)
    disc = np.exp(-r * T)
    score = (Z**2 - 1.0) / sigma - Z * np.sqrt(T)
    samples = disc * _payoff(S_T, K, option) * score
    return estimate(samples, antithetic=antithetic)
