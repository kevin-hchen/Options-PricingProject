"""High-level pricers: one function per option type.

Glue the lower layers together -- gbm (paths) + payoffs + engine (average) --
so an option prices in a single call. Each returns an :class:`engine.MCResult`,
so you always get an error bar with the price.
"""

from __future__ import annotations

import numpy as np

from . import gbm, payoffs
from .engine import MCResult, control_variate, estimate


def price_european(S0, K, r, sigma, T, option="call", q=0.0,
                   n_paths=100_000, antithetic=True, rng=None):
    """Price a vanilla European option. Converges to ``black_scholes.bs_price``.

    Only the terminal price matters, so we sample ``S_T`` directly -- fast and
    free of discretisation error.
    """
    S_T = gbm.terminal_prices(S0, r, sigma, T, n_paths, q=q, rng=rng,
                              antithetic=antithetic)
    disc_payoff = np.exp(-r * T) * payoffs.european_payoff(S_T, K, option)
    return estimate(disc_payoff, antithetic=antithetic)


def price_asian(S0, K, r, sigma, T, option="call", q=0.0,
                n_paths=100_000, n_steps=50, average="arithmetic",
                antithetic=True, use_control_variate=True, rng=None):
    """Price an Asian (average-price) option.

    The arithmetic average has no closed form, so Monte Carlo is the natural
    tool. When ``use_control_variate`` is on we exploit the geometric-average
    Asian -- which *does* have a closed form -- as a control variate, typically
    cutting the error by an order of magnitude for free.
    """
    paths = gbm.price_paths(S0, r, sigma, T, n_paths, n_steps, q=q, rng=rng,
                            antithetic=antithetic)
    disc = np.exp(-r * T)
    disc_payoff = disc * payoffs.asian_payoff(paths, K, option, average="arithmetic")

    if average == "geometric":
        # Caller explicitly wants the geometric option itself.
        disc_geo = disc * payoffs.asian_payoff(paths, K, option, average="geometric")
        return estimate(disc_geo, antithetic=antithetic)

    if use_control_variate:
        disc_geo = disc * payoffs.asian_payoff(paths, K, option, average="geometric")
        geo_mean = geometric_asian_price(S0, K, r, sigma, T, option, q, n_steps)
        # control_variate is element-wise, so antithetic pairing is preserved.
        adjusted = control_variate(disc_payoff, disc_geo, geo_mean)
        return estimate(adjusted, antithetic=antithetic)

    return estimate(disc_payoff, antithetic=antithetic)


def price_barrier(S0, K, r, sigma, T, barrier, kind="up-and-out",
                  option="call", q=0.0, n_paths=100_000, n_steps=100,
                  antithetic=True, rng=None):
    """Price a barrier option with discrete monitoring at the grid points.

    More steps -> the discrete barrier better approximates a continuously
    monitored one. As a sanity check, an *in* plus its matching *out* option
    should reprice the vanilla European.
    """
    paths = gbm.price_paths(S0, r, sigma, T, n_paths, n_steps, q=q, rng=rng,
                            antithetic=antithetic)
    disc_payoff = np.exp(-r * T) * payoffs.barrier_payoff(
        paths, K, barrier, option=option, kind=kind)
    return estimate(disc_payoff, antithetic=antithetic)


def geometric_asian_price(S0, K, r, sigma, T, option="call", q=0.0, n_steps=50):
    """Closed-form price of a discretely-monitored geometric-average Asian.

    - geometric average of lognormals is lognormal -> BS-style formula with an
      adjusted drift and volatility
    - matched to the ``n_steps`` grid, so it's an exact control mean for
      ``price_asian`` (not an approximation)

    Reference: Kemna & Vorst (1990), discrete version.
    """
    from scipy.stats import norm

    n = n_steps
    # Monitoring times t_i = i * T / n, i = 1..n.
    t = np.arange(1, n + 1) * (T / n)

    # Effective volatility and drift of the log geometric average.
    # Var of the average of the log-prices, times 1/T conventions folded in:
    sig2 = sigma**2 / (n**2 * T) * np.sum((2 * np.arange(1, n + 1) - 1) * t[::-1])
    sigma_g = np.sqrt(sig2)
    mu = (r - q - 0.5 * sigma**2) * t.mean() + 0.5 * sigma_g**2 * T

    d1 = (np.log(S0 / K) + (mu + 0.5 * sigma_g**2 * T)) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)
    disc = np.exp(-r * T)
    forward = S0 * np.exp(mu)
    if option == "call":
        return disc * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
