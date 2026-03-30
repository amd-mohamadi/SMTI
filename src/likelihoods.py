"""
Likelihood functions for MT inversion: polarity and amplitude ratios.

These implement the same mathematics as MTfit.probability.polarity_ln_pdf
and MTfit.probability.amplitude_ratio_ln_pdf, but use only NumPy/SciPy.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.special import erf
from scipy.stats import norm as gaussian


_SMALL_NUMBER = 1e-24


def gaussian_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Gaussian cumulative distribution function with small-error safeguards.
    """
    if not isinstance(sigma, (float, int, np.floating)):
        sigma = np.asarray(sigma, dtype=float)
        sigma[sigma == 0] = _SMALL_NUMBER
    elif sigma == 0:
        sigma = _SMALL_NUMBER
    return gaussian.cdf(x, loc=mu, scale=sigma)


def ratio_pdf(
    z: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    corr: float = 0.0,
) -> np.ndarray:
    """
    Ratio pdf for Z = X / Y, following Hinkley (1969).

    Parameters correspond to those in MTfit.probability.ratio_pdf, but this
    implementation omits the C-extension path.
    """
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(mu_x, np.ndarray) and mu_x.ndim == 3:
        if isinstance(mu_y, np.ndarray) and mu_y.ndim == 2:
            mu_y = np.expand_dims(mu_y, 1)
    if isinstance(mu_y, np.ndarray) and mu_y.ndim == 3:
        if isinstance(mu_x, np.ndarray) and mu_x.ndim == 2:
            mu_x = np.expand_dims(mu_x, 1)
        if isinstance(z, np.ndarray) and z.ndim == 2:
            z = np.expand_dims(z, 1)
        if isinstance(sigma_x, np.ndarray) and sigma_x.ndim == 2:
            sigma_x = np.expand_dims(sigma_x, 1)
        if isinstance(sigma_y, np.ndarray) and sigma_y.ndim == 2:
            sigma_y = np.expand_dims(sigma_y, 1)

    z_2 = z * z
    sigma_x_2 = sigma_x * sigma_x
    sigma_xy = sigma_x * sigma_y
    sigma_y_2 = sigma_y * sigma_y
    mu_x_2 = mu_x * mu_x

    if corr > 0:
        a = np.sqrt(z_2 / sigma_x_2 - 2 * corr * z / sigma_xy + 1.0 / sigma_y_2)
    else:
        a = np.sqrt(z_2 / sigma_x_2 + 1.0 / sigma_y_2)
    a_2 = a * a
    b = (mu_x * z) / sigma_x_2 - (corr * sigma_x_2 / sigma_xy) + (mu_y / sigma_y_2)
    c = (mu_x_2 / sigma_x_2) + (mu_y * mu_y / sigma_y_2)
    if corr > 0:
        c -= 2 * corr * (mu_x * mu_y) / sigma_xy

    d = np.exp(((b * b) - (c * a_2)) / (2.0 * (1.0 - corr * corr) * a_2))
    p = (b * d) / (np.sqrt(2.0 * np.pi) * sigma_xy * a * a_2)
    p = p * (
        gaussian_cdf(b / (np.sqrt(1.0 - corr * corr) * a), 0.0, 1.0)
        - gaussian_cdf(-b / (np.sqrt(1.0 - corr * corr) * a), 0.0, 1.0)
    )
    p += (
        np.sqrt(1.0 - corr * corr)
        / (np.pi * sigma_xy * a_2)
        * np.exp(-c / (2.0 * (1.0 - corr * corr)))
    )

    p = np.asarray(p, dtype=float)
    p[np.isnan(p)] = 0.0
    return p


def polarity_ln_pdf(
    a: np.ndarray,
    mt: np.ndarray,
    sigma: np.ndarray,
    incorrect_polarity_probability: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    """
    Log-likelihood for polarity observations.

    Parameters follow MTfit.probability.polarity_ln_pdf, but this implementation
    omits the C-extension path and returns a NumPy array of log-probabilities.
    """
    a = np.asarray(a, dtype=float)
    mt = np.asarray(mt, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if sigma.ndim != 1:
        raise TypeError("sigma is expected to be a one-dimensional numpy array")
    if a.ndim != 3:
        raise TypeError("a is expected to be a three-dimensional numpy array")
    if mt.ndim != 2:
        raise TypeError(
            "mt is expected to be a two-dimensional numpy array of MT six-vectors"
        )

    sigma = sigma.copy()
    sigma[sigma == 0] = _SMALL_NUMBER

    if mt.shape[0] != a.shape[-1]:
        mt = mt.T

    X = np.tensordot(a, mt, axes=1)

    while sigma.ndim < 3:
        sigma = np.expand_dims(sigma, 1)

    if isinstance(incorrect_polarity_probability, (float, int)):
        incorrect = incorrect_polarity_probability
    else:
        incorrect = np.asarray(incorrect_polarity_probability, dtype=float)
        while incorrect.ndim < 3:
            incorrect = np.expand_dims(incorrect, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(incorrect, (float, int)):
            ln_p = np.log(
                (0.5 * (1.0 + erf(X / (np.sqrt(2.0) * sigma))) * (1.0 - incorrect))
                + (
                    0.5
                    * (1.0 + erf(-X / (np.sqrt(2.0) * sigma)))
                    * incorrect
                )
            )
        else:
            ln_p = np.log(
                (0.5 * (1.0 + erf(X / (np.sqrt(2.0) * sigma))) * (1.0 - incorrect))
                + (
                    0.5
                    * (1.0 + erf(-X / (np.sqrt(2.0) * sigma)))
                    * incorrect
                )
            )

    ln_p = np.asarray(ln_p, dtype=float)
    ln_p[np.isnan(ln_p)] = -np.inf
    ln_p_total = np.sum(ln_p, axis=0)
    ln_p_total[np.isnan(ln_p_total)] = -np.inf
    return ln_p_total


def amplitude_ratio_ln_pdf(
    ratio: np.ndarray,
    mt: np.ndarray,
    a_x: np.ndarray,
    a_y: np.ndarray,
    percentage_error_x: np.ndarray,
    percentage_error_y: np.ndarray,
) -> np.ndarray:
    """
    Log-likelihood for amplitude ratio observations.

    Equivalent in spirit to MTfit.probability.amplitude_ratio_ln_pdf but using
    only Python/NumPy/SciPy (no C-extension).
    """
    a_x = np.asarray(a_x, dtype=float)
    a_y = np.asarray(a_y, dtype=float)
    mt = np.asarray(mt, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    percentage_error_x = np.asarray(percentage_error_x, dtype=float)
    percentage_error_y = np.asarray(percentage_error_y, dtype=float)

    if a_x.ndim != 3:
        raise TypeError("a_x is expected to be a three-dimensional numpy array")
    if a_y.ndim != 3:
        raise TypeError("a_y is expected to be a three-dimensional numpy array")
    if mt.ndim != 2:
        raise TypeError("mt is expected to be a two-dimensional numpy array")
    if ratio.ndim != 1:
        raise TypeError("ratio is expected to be a one-dimensional numpy array")
    if percentage_error_x.ndim != 1:
        raise TypeError("percentage_error_x is expected to be a one-dimensional array")
    if percentage_error_y.ndim != 1:
        raise TypeError("percentage_error_y is expected to be a one-dimensional array")

    percentage_error_x = percentage_error_x.copy()
    percentage_error_y = percentage_error_y.copy()
    percentage_error_x[percentage_error_x == 0] = _SMALL_NUMBER
    percentage_error_y[percentage_error_y == 0] = _SMALL_NUMBER
    percentage_error_x = np.abs(percentage_error_x)
    percentage_error_y = np.abs(percentage_error_y)

    if mt.shape[0] != a_x.shape[-1]:
        mt = mt.T

    mu_x = np.tensordot(a_x, mt, axes=1)
    mu_y = np.tensordot(a_y, mt, axes=1)
    mu_x = np.abs(mu_x)
    mu_y = np.abs(mu_y)

    while percentage_error_x.ndim < 3:
        percentage_error_x = np.expand_dims(percentage_error_x, 1)
    while percentage_error_y.ndim < 3:
        percentage_error_y = np.expand_dims(percentage_error_y, 1)
    while ratio.ndim < 3:
        ratio = np.expand_dims(ratio, 1)

    numerator_error = percentage_error_x * mu_x
    denominator_error = percentage_error_y * mu_y

    with np.errstate(divide="ignore", invalid="ignore"):
        p = ratio_pdf(ratio, mu_x, mu_y, numerator_error, denominator_error) + ratio_pdf(
            -ratio, mu_x, mu_y, numerator_error, denominator_error
        )
        ln_p = np.log(p)

    ln_p = np.asarray(ln_p, dtype=float)
    ln_p[np.isnan(ln_p)] = -np.inf
    ln_p_total = np.sum(ln_p, axis=0)
    ln_p_total[np.isnan(ln_p_total)] = -np.inf
    return ln_p_total


