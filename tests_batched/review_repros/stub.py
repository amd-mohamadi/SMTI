"""Tiny CPU harness for the M1 integration review repros.

Builds a duck-typed stand-in for ``InversionBlackJAX`` whose
``_prepare_event_arrays`` returns a *synthetic* prep dict, so bucketing /
driver-integration behaviour can be exercised in <1 s per run without the
step-4 filter, read_data, or the real 44x62 arrays.

Only the attributes ``inversion_blackjax_batched`` actually reads are provided;
``_stack_chain_results`` is borrowed unchanged from the production class.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import sys

SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
if SMTI_DEV not in sys.path:
    sys.path.insert(0, SMTI_DEV)

import numpy as np  # noqa: E402

from src.inversion_blackjax import InversionBlackJAX  # noqa: E402
import src.inversion_blackjax_batched as bx  # noqa: E402

# prepare_batch calls the module-level import directly; the synthetic events
# carry no station geometry, so stub it out (it only feeds _prepare_event_arrays,
# which we override).
bx.build_location_samples_from_errors = lambda *a, **k: None


def make_prep(n_pol=6, n_ar=8, n_loc=3, seed=0, has_pol=True, has_ar=True,
              scalar_incorrect=True, error_pol_val=0.2, log_sigma_val=0.3):
    """A synthetic ``_prepare_event_arrays`` output.

    ``error_pol_val`` / ``log_sigma_val`` set how informative the event is,
    which controls how big the adaptive-tempered beta increments are.
    """
    rng = np.random.default_rng(seed)
    prep = {
        "has_pol": has_pol,
        "has_ar": has_ar,
        "station_smooth_ar": False,
        "location_samples": None,
        "a_pol": None,
        "error_pol": None,
        "pol_obs": None,
        "incorrect_prob": 0.05,
        "a1_ar": None,
        "a2_ar": None,
        "amp_ratio_obs": None,
        "log_ratio_sigma": None,
        "ar_station_context": None,
    }
    if has_pol:
        prep["a_pol"] = rng.normal(size=(n_pol, n_loc, 6))
        prep["error_pol"] = np.full(n_pol, error_pol_val)
        prep["pol_obs"] = np.ones(n_pol)
        prep["incorrect_prob"] = (
            0.05 if scalar_incorrect else np.full(n_pol, 0.05)
        )
    if has_ar:
        prep["a1_ar"] = rng.normal(size=(n_ar, n_loc, 6))
        prep["a2_ar"] = rng.normal(size=(n_ar, n_loc, 6))
        prep["amp_ratio_obs"] = np.exp(rng.normal(size=n_ar) * 0.3)
        prep["log_ratio_sigma"] = np.full(n_ar, log_sigma_val)
    return prep


class StubInv:
    """Duck-typed InversionBlackJAX exposing only what the batched path uses."""

    _stack_chain_results = InversionBlackJAX._stack_chain_results

    def __init__(self, num_particles=40, num_chains=2, num_mcmc_steps=1,
                 mechanism_steps=1, max_smc_iterations=60, random_seed=1234):
        self.smc_method = "adaptive_tempered"
        self.mcmc_kernel = "rmh"
        self.adapt_proposal = True
        self.amp_ratio_noise_mode = "global"
        self.dc = False
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.num_chains = num_chains
        self.num_particles = num_particles
        self.num_mcmc_steps = num_mcmc_steps
        self.mechanism_steps = mechanism_steps
        self.max_smc_iterations = max_smc_iterations
        self.min_tempering_increment = 1e-4
        self.tempering_stall_patience = 3
        self.smc_target_ess_ratio = 0.9
        self.gamma_beta_prior = (3.0, 3.0)
        self.delta_beta_prior = (3.0, 3.0)
        self.amp_ratio_sigma_prior = 2.0
        self.location_samples_n = 3
        self.azimuth_error = 5
        self.takeoff_error = 10

    def _filter_event_by_options(self, event):
        return event

    def _prepare_event_arrays(self, event, location_samples=None):
        # the "event" IS the prep dict in these repros
        return event


QUIET = lambda _msg: None
