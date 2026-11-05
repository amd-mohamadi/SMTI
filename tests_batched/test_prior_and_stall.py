"""Prior tails and the tempering-stall flag, in the unbatched and batched samplers."""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from src import inversion_blackjax_batched as B  # noqa: E402
from src.inversion_blackjax import InversionBlackJAX  # noqa: E402
from src.tape import Tape_MT6  # noqa: E402
from src.utilities import generate_synthetic_event_data  # noqa: E402

OPTIONS = ["PPolarity", "P/SHAmplitudeRatio"]
ZERO = ("gamma", "delta", "kappa", "h", "sigma")


@pytest.fixture(scope="module")
def event():
    rng = np.random.default_rng(11)
    true_mt6 = np.asarray(
        Tape_MT6(
            np.array([0.05]), np.array([0.1]), np.array([1.0]),
            np.array([0.6]), np.array([0.3]),
        ),
        dtype=float,
    ).reshape(-1)
    return generate_synthetic_event_data(
        OPTIONS,
        true_mt6,
        rng.uniform(0.0, 360.0, 14),
        rng.uniform(20.0, 160.0, 14),
    )


def make_inversion(event, **overrides):
    settings = dict(
        inversion_options=OPTIONS,
        num_particles=120,
        num_mcmc_steps=3,
        num_chains=2,
        dc=False,
        random_seed=7,
        mcmc_kernel="mwg_irmh",
        location_samples_n=1,
        azimuth_error=0.0,
        takeoff_error=0.0,
    )
    settings.update(overrides)
    return InversionBlackJAX(event, **settings)


def batched_logprior():
    logprior, _ = B._make_model(
        {},
        has_pol=False,
        has_ar=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=1.0,
    )
    return logprior


def at(**values):
    return {
        name: jnp.asarray(values.get(name, 0.0), dtype=jnp.float64) for name in ZERO
    }


@pytest.mark.parametrize("name", ["kappa", "h", "sigma"])
def test_orientation_tail_keeps_the_exact_density(name):
    logprior = batched_logprior()
    u = 1000.0
    expected = float(logprior(at())) + 2.0 * np.log(2.0) - u
    assert float(logprior(at(**{name: u}))) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", ["gamma", "delta"])
def test_beta_tail_keeps_the_exact_density(name):
    logprior = batched_logprior()
    u = 1000.0
    # Beta(3, 3) in sigmoid space falls as exp(-3 |u|) in both tails.
    expected = float(logprior(at())) + 6.0 * np.log(2.0) - 3.0 * u
    assert float(logprior(at(**{name: u}))) == pytest.approx(expected, abs=1e-9)
    assert float(logprior(at(**{name: -u}))) == pytest.approx(expected, abs=1e-9)


def test_nan_position_has_zero_prior_density():
    assert float(batched_logprior()(at(kappa=jnp.nan))) == -np.inf


def test_unbatched_iteration_limit_is_reported_as_a_stall(event):
    result = make_inversion(event, max_smc_iterations=2, num_chains=1).forward()
    assert result.final_beta < 1.0
    assert result.tempering_stalled is True


def test_unbatched_full_run_is_not_a_stall(event):
    result = make_inversion(event, max_smc_iterations=100, num_chains=1).forward()
    assert result.final_beta == pytest.approx(1.0)
    assert result.tempering_stalled is False


def test_batched_run_reports_final_beta(event):
    results = B.run_batched(make_inversion(event), [("e", event)], smc_dtype="float64")
    result = results["e"]
    assert result.final_beta == pytest.approx(1.0)
    assert result.tempering_stalled is False
