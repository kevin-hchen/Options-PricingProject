"""Simulating geometric Brownian motion (GBM) price paths.

Everything downstream starts here: draw random paths, then average discounted
payoffs over them.

We use the exact SDE solution (no discretisation error):

    S_t = S_0 * exp[ (r - q - sigma^2 / 2) t + sigma * sqrt(t) * Z ],  Z ~ N(0, 1)

- European options need only the terminal price  -> ``terminal_prices``
- Path-dependent options need the whole path      -> ``price_paths``

Antithetic variates: for every draw ``Z``, also use ``-Z``. Both are valid
N(0, 1), so the estimate stays unbiased, but the paired paths are negatively
correlated, shrinking the variance of their average.
"""

from __future__ import annotations

import numpy as np


def _draw_normals(shape, rng, antithetic):
    """Draw standard-normal samples, optionally as antithetic pairs.

    With ``antithetic=True`` the first half of the samples are i.i.d. N(0,1)
    and the second half are their negation, so ``Z`` and ``-Z`` are paired.
    """
    if not antithetic:
        return rng.standard_normal(shape)

    n = shape[0]
    if n % 2 != 0:
        raise ValueError("antithetic sampling needs an even number of paths")
    half = rng.standard_normal((n // 2,) + tuple(shape[1:]))
    return np.concatenate([half, -half], axis=0)


def terminal_prices(S0, r, sigma, T, n_paths, q=0.0, rng=None, antithetic=False):
    """Sample the terminal price ``S_T`` directly (one draw per path).

    This is all a European option needs -- its payoff depends only on ``S_T``,
    so we skip the intermediate steps and sample the exact lognormal law.

    Returns
    -------
    ndarray of shape (n_paths,)
    """
    rng = np.random.default_rng() if rng is None else rng
    Z = _draw_normals((n_paths,), rng, antithetic)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def price_paths(S0, r, sigma, T, n_paths, n_steps, q=0.0, rng=None, antithetic=False):
    """Simulate full price paths on a uniform time grid.

    Needed for path-dependent options: an Asian option averages the price
    along the path, a barrier option checks whether the path crossed a level.

    Returns
    -------
    ndarray of shape (n_paths, n_steps + 1)
        Column 0 is ``S0`` for every path; column ``j`` is the price at time
        ``j * T / n_steps``. Because GBM has independent lognormal increments
        we can build the path by exponentiating a cumulative sum -- still exact
        at the grid points, no Euler discretisation error.
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = T / n_steps
    Z = _draw_normals((n_paths, n_steps), rng, antithetic)

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_increments = drift + diffusion                       # (n_paths, n_steps)

    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths
