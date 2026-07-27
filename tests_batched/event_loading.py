"""Shared event-loading helpers for the M1 batched-SMC tests.

Loads production events exactly the way ``m0_spike/prepare_eq00124_arrays.py``
and ``tests_batched/test_batched_core.py`` do: the step-4 station filter from
``smti_inversion/4_run_inversion.py`` (imported by path, read-only) followed by
``src.data_loader.read_data`` with the production keyword arguments.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

SMTI_DEV = os.environ.get("SMTI_ROOT", "/s0/data/CAPE/smti_workflow/SMTI_dev")
WORKFLOW = "/s0/data/CAPE/smti_workflow/smti_inversion"
M0_SPIKE = os.path.join(WORKFLOW, "m0_spike")
DATA_ROOT = "/s0/data/CAPE/smti_workflow/inversion_data/data"
VELOCITY_MODEL_PATH = "/s0/data/CAPE/smti_workflow/cape_surface_corrected.tvel"

for _p in (SMTI_DEV, M0_SPIKE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INVERSION_OPTIONS = [
    "PPolarity",
    "SHPolarity",
    "P/SHAmplitudeRatio",
    "P/SVAmplitudeRatio",
]
READ_DATA_KW = dict(
    min_p_phase_score=0.55,
    min_s_phase_score=0.55,
    min_ppl_score=0.6,
    min_shpl_score=0.7,
    min_svpl_score=0.7,
    p_polarity_phase="Pz",
)
LOCATION_SAMPLES_N = 20
AZIMUTH_ERROR = 5
TAKEOFF_ERROR = 10
SEED = 20240216

EVENT_IDS_ALL = sorted(
    d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))
)

_driver4 = None


def _filter_step4(dat_path, include_stacked_stations=True):
    global _driver4
    if _driver4 is None:
        spec = importlib.util.spec_from_file_location(
            "driver4", os.path.join(WORKFLOW, "4_run_inversion.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["driver4"] = mod
        cwd = os.getcwd()
        os.chdir(WORKFLOW)
        if WORKFLOW not in sys.path:
            sys.path.insert(0, WORKFLOW)
        try:
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
        _driver4 = mod
    return _driver4._filter_step4_data_dat(
        dat_path, include_stacked_stations=include_stacked_stations
    )


def load_event(event_id: str):
    """Return the production ``DataDict`` for ``event_id``."""
    from src.data_loader import read_data

    dat_path = os.path.join(DATA_ROOT, event_id, "data.dat")
    filtered_path, _summary = _filter_step4(dat_path)
    try:
        return read_data(
            filtered_path,
            inversion_options=INVERSION_OPTIONS,
            velocity_model_path=VELOCITY_MODEL_PATH,
            **READ_DATA_KW,
        )
    finally:
        try:
            os.unlink(filtered_path)
        except OSError:
            pass


def count_obs(data, n_loc=LOCATION_SAMPLES_N, seed=SEED):
    """Exact ``(N_pol, N_ar)`` as the likelihood sees them (post station-intersection)."""
    from src.data_prep import (
        amplitude_ratio_matrix,
        build_location_samples_from_errors,
        polarity_matrix,
    )

    ls = build_location_samples_from_errors(
        data,
        rng=np.random.default_rng(seed),
        n_samples=n_loc,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
    )
    a_pol, _err, _inc = polarity_matrix(data, location_samples=ls)
    a1, _a2, _amp, _e1, _e2 = amplitude_ratio_matrix(data, location_samples=ls)
    n_pol = 0 if a_pol is False or a_pol is None else int(np.asarray(a_pol).shape[0])
    n_ar = 0 if a1 is False or a1 is None else int(np.asarray(a1).shape[0])
    return n_pol, n_ar


def make_inversion(
    data,
    num_particles,
    num_chains,
    num_mcmc_steps=5,
    cls=None,
    max_smc_iterations=60,
    chain_execution="sequential",
):
    """Production-settings :class:`InversionBlackJAX` (same as the stage-2 tests)."""
    from src.inversion_blackjax import InversionBlackJAX

    cls = cls or InversionBlackJAX
    return cls(
        data,
        inversion_options=INVERSION_OPTIONS,
        num_particles=num_particles,
        dc=False,
        gamma_beta_prior=(3.0, 3.0),
        delta_beta_prior=(3.0, 3.0),
        amp_ratio_sigma_prior=2.0,
        amp_ratio_noise_mode="global",
        random_seed=SEED,
        location_samples_n=LOCATION_SAMPLES_N,
        azimuth_error=AZIMUTH_ERROR,
        takeoff_error=TAKEOFF_ERROR,
        mcmc_kernel="rmh",
        num_mcmc_steps=num_mcmc_steps,
        mechanism_steps=5,
        smc_target_ess_ratio=0.9,
        max_smc_iterations=max_smc_iterations,
        num_chains=num_chains,
        chain_execution=chain_execution,
    )
