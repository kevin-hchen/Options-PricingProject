"""The generic Monte Carlo estimator.

Once you have discounted payoffs, pricing is just averaging:

    price = E[ e^{-rT} * payoff ]  ~=  sample mean of the discounted payoffs

- Law of Large Numbers: the mean converges to the true price
- Central Limit Theorem: its error shrinks like 1 / sqrt(n)
  (halving the error costs 4x the paths -> variance reduction matters)

This module turns payoff samples into a price + error bar (``estimate``) and
implements the control-variate trick (``control_variate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """A Monte Carlo price with its statistical uncertainty.

    - price      mean of the discounted payoffs
    - std_error  standard error of that mean; true price is within
                 ``price +/- 1.96 * std_error`` ~95% of the time
    - n_paths    number of simulated paths
    """

    price: float
    std_error: float
    n_paths: int

    def confidence_interval(self, level=0.95):
        """Return a two-sided normal confidence interval (lo, hi)."""
        from scipy.stats import norm

        z = norm.ppf(0.5 + level / 2.0)
        return (self.price - z * self.std_error, self.price + z * self.std_error)

    def __repr__(self):
        return (f"MCResult(price={self.price:.6f}, "
                f"std_error={self.std_error:.6f}, n_paths={self.n_paths})")


def estimate(discounted_payoffs, antithetic=False):
    """Turn discounted payoffs into an ``MCResult`` (mean + standard error).

    The mean is the same either way; the standard error needs care:

    - independent samples: std_error = sample_std / sqrt(n), the i.i.d. formula
    - antithetic samples: the pairs are negatively correlated, so the i.i.d.
      formula is *wrong*. Instead, average each pair, then take the standard
      error over the n/2 pair-means -- this credits the variance reduction.
      Pair layout (first half vs. second half) matches ``gbm._draw_normals``.
    """
    x = np.asarray(discounted_payoffs, dtype=float)
    n = x.size

    if antithetic:
        if n % 2 != 0:
            raise ValueError("antithetic estimate needs an even number of samples")
        half = n // 2
        pair_means = 0.5 * (x[:half] + x[half:])
        price = pair_means.mean()
        std_error = pair_means.std(ddof=1) / np.sqrt(half)
        return MCResult(price=price, std_error=std_error, n_paths=n)

    price = x.mean()
    std_error = x.std(ddof=1) / np.sqrt(n)
    return MCResult(price=price, std_error=std_error, n_paths=n)


def control_variate(payoffs, control, control_mean):
    """Apply the control-variate technique; return adjusted samples.

    Given a correlated control ``C`` whose true mean ``E[C]`` we know exactly:

        Y* = Y - beta * (C - E[C]),   beta = Cov(Y, C) / Var(C)

    - same mean as Y (unbiased), smaller variance when Y and C move together
    - variance drops by a factor 1 - corr(Y, C)^2 (95% correlated -> ~10x)
    - classic pairing: arithmetic Asian priced against the geometric Asian,
      which has a closed-form mean

    Parameters
    ----------
    payoffs : ndarray       discounted payoffs Y
    control : ndarray       discounted control samples C
    control_mean : float    known true value of E[C]

    Returns the adjusted samples Y*; feed them straight into ``estimate``.
    """
    payoffs = np.asarray(payoffs, dtype=float)
    control = np.asarray(control, dtype=float)

    cov = np.cov(payoffs, control, ddof=1)      # 2x2 covariance matrix
    beta = cov[0, 1] / cov[1, 1]
    return payoffs - beta * (control - control_mean)
