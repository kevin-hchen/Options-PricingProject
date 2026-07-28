"""Option payoff functions.

Turn simulated prices into the cash paid at maturity. Kept separate from the
simulation and averaging so the same MC machinery prices any payoff here.

Each function is vectorised: arrays in, one payoff per path out.
"""

from __future__ import annotations

import numpy as np


def european_payoff(S_T, K, option="call"):
    """Vanilla European payoff, a function of the terminal price only.

    call: max(S_T - K, 0)      put: max(K - S_T, 0)
    """
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    elif option == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def asian_payoff(paths, K, option="call", average="arithmetic"):
    """Asian (average-price) payoff: uses the average price along each path.

    - averaging damps volatility -> worth less than the European
    - arithmetic average has no closed form under GBM -> MC shines here
    - geometric average *does* have one -> a great control variate

    Parameters
    ----------
    paths : ndarray (n_paths, n_steps + 1)   full paths from ``gbm.price_paths``
    average : {"arithmetic", "geometric"}
    """
    # Exclude the known S0 at column 0 so it does not bias the average.
    body = paths[:, 1:]
    if average == "arithmetic":
        avg = body.mean(axis=1)
    elif average == "geometric":
        avg = np.exp(np.log(body).mean(axis=1))
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    return european_payoff(avg, K, option)


def barrier_payoff(paths, K, barrier, option="call", kind="up-and-out"):
    """Barrier payoff: knocks in or out when a level is touched.

    Discrete monitoring -- the barrier is checked only at grid points; finer
    grids approach the continuously-monitored price.

    - out: pays the vanilla payoff *unless* the barrier is breached (else 0)
    - in:  pays *only if* the barrier is breached
    - by construction, in + out = vanilla

    Parameters
    ----------
    barrier : float   the level B
    kind : {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
    """
    S_T = paths[:, -1]
    vanilla = european_payoff(S_T, K, option)

    hi = paths.max(axis=1)
    lo = paths.min(axis=1)
    if kind.startswith("up"):
        breached = hi >= barrier
    elif kind.startswith("down"):
        breached = lo <= barrier
    else:
        raise ValueError("kind must start with 'up' or 'down'")

    if kind.endswith("out"):
        alive = ~breached
    elif kind.endswith("in"):
        alive = breached
    else:
        raise ValueError("kind must end with 'out' or 'in'")

    return np.where(alive, vanilla, 0.0)
