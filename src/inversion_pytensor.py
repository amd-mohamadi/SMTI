"""
PyTensor-based MT inversion using PyMC's SMC sampler.

This module provides a new Inversion class that uses pure PyTensor operations
instead of custom Python callbacks. This enables:
- True parallelization across cores
- Compilation by Nutpie/NumPyro for 10-100x speedup
- Better integration with PyMC ecosystem

Key differences from inversion.py:
- No custom Op or pm.Potential
- Uses standard PyMC distributions (pm.Bernoulli, pm.LogNormal)
- Forward model is pure PyTensor (compilable)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from .data_prep import (
    amplitude_matrix,
    amplitude_ratio_matrix,
    build_location_samples_from_errors,
    polarity_matrix,
)
from .tape_pytensor import pt_Tape_MT6
from .forward_model_pytensor import pt_station_angles


DataDict = Dict[str, Any]
_SMALL_NUMBER = 1e-10


@dataclass
class InversionResult:
    """
    Container for a single-event inversion result.

    Attributes
    ----------
    mt6 : np.ndarray
        Moment tensor six-vectors, shape (6, n_samples).
    gamma, delta, kappa, h, sigma : np.ndarray
        Tape parameters for each sample, shape (n_samples,).
    ln_p : np.ndarray
        Log-likelihood for each sample, shape (n_samples,) (placeholder, not computed).
    weights : np.ndarray
        Normalised posterior weights for each sample, shape (n_samples,).
    idata : arviz.InferenceData
        Full PyMC inference data object with diagnostics.
    """

    mt6: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    kappa: np.ndarray
    h: np.ndarray
    sigma: np.ndarray
    ln_p: Optional[np.ndarray]
    weights: np.ndarray
    idata: Any  # arviz.InferenceData


class InversionPyTensor:
    """
    PyTensor-based SMC inversion in Tape parameterisation using PyMC.

    This version uses pure PyTensor operations for the forward model and
    standard PyMC distributions for likelihoods, enabling true parallelization
    and compilation.

    Parameters
    ----------
    data : dict or list of dict
        Event data dictionary or list of dictionaries in the same format as
        MTfit's example_data (keys like 'PPolarity', 'P/SHAmplitudeRatio', etc.).
    inversion_options : str or sequence of str, optional
        String or sequence of strings specifying which data types to use,
        e.g. 'PPolarity' or ['PPolarity', 'P/SHAmplitudeRatio'].
    draws : int, default 2000
        Number of SMC posterior draws (per chain) to return per event.
    chains : int, default 4
        Number of SMC chains (parallelization now works!).
    cores : int, optional
        Number of CPU cores to use. If ``None``, defaults to number of chains.
    dc : bool, default False
        If True, restricts the source to double-couple (DC) space by fixing
        the Tape source-type parameters ``gamma = 0`` and ``delta = 0``.
    random_seed : int or RandomState, optional
        Seed or sequence of seeds forwarded to :func:`pymc.sample_smc`.
    progressbar : bool, default True
        Whether to show PyMC's progress bar.
    compute_convergence_checks : bool, default True
        Passed through to :func:`pymc.sample_smc`.
    smc_kernel : default pm.smc.kernels.IMH
        SMC kernel class.
    smc_kernel_kwargs : dict, optional
        Optional dict of keyword arguments forwarded to ``smc_kernel``.
    amp_ratio_sigma_prior : float, default 0.5
        Scale parameter for HalfCauchy prior on log-ratio uncertainty.
    gamma_beta_prior : tuple, default (2.0, 2.0)
        Alpha, beta parameters for Beta prior on Tape source-type parameter
        ``gamma`` on the unit interval before mapping to Tape range.
    delta_beta_prior : tuple, default (2.0, 2.0)
        Alpha, beta parameters for Beta prior on Tape source-type parameter
        ``delta`` on the unit interval before mapping to Tape range.
    location_samples_n : int, default 20
        Number of station-angle location samples to draw per event when
        ``Azimuth_err`` and ``TakeOff_err`` are provided in the input data.
    azimuth_error : float, optional
        Global 1σ uncertainty (in degrees) applied to all station azimuths
        when constructing location samples. If ``None``, per-station
        ``Azimuth_err`` fields (if present) are used instead.
    takeoff_error : float, optional
        Global 1σ uncertainty (in degrees) applied to all station takeoff
        angles when constructing location samples. If ``None``, per-station
        ``TakeOff_err`` fields (if present) are used instead.

    Notes
    -----
    - This implementation uses standard PyMC distributions instead of custom Ops
    - Parallelization with ``cores`` parameter now works effectively
    - Can be compiled with Nutpie for additional speedup
    - Simplified likelihood models compared to full MTfit implementation
    """

    def __init__(
        self,
        data: Union[DataDict, Sequence[DataDict]],
        inversion_options: Optional[Union[str, Sequence[str]]] = None,
        draws: int = 2_000,
        chains: int = 4,
        cores: Optional[int] = None,
        dc: bool = False,
        random_seed: Optional[Union[int, np.random.Generator, np.random.RandomState]] = 68,
        progressbar: bool = True,
        compute_convergence_checks: bool = True,
        smc_kernel: Any = pm.smc.kernels.IMH,
        smc_kernel_kwargs: Optional[Dict[str, Any]] = None,
        amp_ratio_sigma_prior: float = 0.5,
        location_samples_n: int = 20,
        azimuth_error: Optional[float] = None,
        takeoff_error: Optional[float] = None,
        gamma_beta_prior: tuple = (1.0, 1.0),
        delta_beta_prior: tuple = (1.0, 1.0),
        safe_sampling: bool = False,
        **kwargs: Any,
    ) -> None:
        # Normalise data to a list of events
        if isinstance(data, dict):
            self.data: List[DataDict] = [data]
        else:
            self.data = list(data)

        # Normalise inversion options
        if inversion_options is None:
            self.inversion_options: Optional[List[str]] = None
        elif isinstance(inversion_options, str):
            self.inversion_options = [inversion_options]
        else:
            self.inversion_options = list(inversion_options)

        # Backwards compatibility
        if "max_samples" in kwargs and draws == 2_000:
            draws = int(kwargs.pop("max_samples"))

        self.draws = int(draws)
        self.chains = int(chains)
        self.cores = int(cores) if cores is not None else self.chains
        # If True, restricts sampling to double-couple (DC) source types by
        # fixing the Tape source-type parameters (gamma, delta) to zero.
        # If False, gamma and delta are sampled over the full prior range.
        self.dc = bool(dc)

        # Store random seed and an internal RNG for fallbacks
        if isinstance(random_seed, (np.random.Generator, np.random.RandomState)):
            self.random_seed = random_seed
            self.rng = np.random.default_rng()
        else:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(random_seed)

        self.progressbar = bool(progressbar)
        self.compute_convergence_checks = bool(compute_convergence_checks)
        self.smc_kernel = smc_kernel
        self.smc_kernel_kwargs = smc_kernel_kwargs or {}

        # Prior hyperparameters
        self.amp_ratio_sigma_prior = amp_ratio_sigma_prior
        self.gamma_beta_prior = gamma_beta_prior
        self.delta_beta_prior = delta_beta_prior

        # Number of location samples for station angle uncertainty
        self.location_samples_n = int(location_samples_n)
        # Global station angle uncertainties (degrees); if None, fall back to
        # any per-station Azimuth_err/TakeOff_err fields in the data.
        self.azimuth_error = azimuth_error
        self.takeoff_error = takeoff_error

        # If True, runs SMC chains sequentially in a loop to avoid deadlocks
        # in ProcessPoolExecutor reported in PyMC > 5.12.0.
        self.safe_sampling = bool(safe_sampling)

        # Debug/diagnostic fields (set during forward())
        self.last_log_ratio_sigma_base: Optional[np.ndarray] = None

        self.results: Union[InversionResult, List[InversionResult], None] = None

    def forward(self) -> Union[InversionResult, List[InversionResult]]:
        """
        Run the SMC-based forward model and likelihood evaluation.

        Returns
        -------
        InversionResult or list of InversionResult
            Results for each event. For a single-event input, a single
            InversionResult is returned for convenience.
        """
        event_results: List[InversionResult] = []

        for event in self.data:
            filtered = self._filter_event_by_options(event)

            # Build data matrices for polarities, absolute amplitudes and amplitude ratios
            has_pol = any(
                "polarity" in key.lower() and "prob" not in key.lower()
                for key in filtered.keys()
            )
            has_amp = any(
                "amplitude" in key.lower() and "amplituderatio" not in key.lower()
                for key in filtered.keys()
            )
            has_ar = any(
                "amplituderatio" in key.lower() or "amplitude_ratio" in key.lower()
                for key in filtered.keys()
            )

            if not (has_pol or has_amp or has_ar):
                raise ValueError(
                    "No supported data types found in event for the given inversion_options."
                )

            # Pre-sample location scatter from station angle errors, if provided.
            location_samples = build_location_samples_from_errors(
                filtered,
                rng=self.rng,
                n_samples=self.location_samples_n,
                azimuth_error=self.azimuth_error,
                takeoff_error=self.takeoff_error,
            )

            # Prepare polarity data
            if has_pol:
                a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(
                    filtered, location_samples=location_samples
                )
                a_pol_arr = np.asarray(a_polarity, dtype=float)
                if a_pol_arr.ndim != 3:
                    raise RuntimeError(
                        "Polarity matrix is expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_pol_obs, n_loc_samples_pol, _ = a_pol_arr.shape
                # Flatten location samples so each (obs, location) pair becomes
                # an independent row, mirroring the NumPy likelihood behaviour.
                a_pol_np = a_pol_arr.reshape(n_pol_obs * n_loc_samples_pol, 6)

                error_pol_np = np.asarray(error_polarity, dtype=float).reshape(-1)
                if error_pol_np.size != n_pol_obs:
                    raise RuntimeError(
                        "Polarity error length does not match number of observations."
                    )
                if n_loc_samples_pol > 1:
                    error_pol_flat = np.repeat(error_pol_np, n_loc_samples_pol)
                else:
                    error_pol_flat = error_pol_np

                # Get observed polarities (sign of amplitude, 0=down,1=up)
                pol_obs_base = self._extract_polarity_observations(filtered)
                if pol_obs_base.size != n_pol_obs:
                    raise RuntimeError(
                        "Number of polarity observations does not match angle matrix."
                    )
                if n_loc_samples_pol > 1:
                    pol_observations = np.repeat(pol_obs_base, n_loc_samples_pol)
                else:
                    pol_observations = pol_obs_base

                if isinstance(incorrect_polarity_prob, (float, int)):
                    incorrect_prob = incorrect_polarity_prob
                else:
                    inc_np = np.asarray(incorrect_polarity_prob, dtype=float).reshape(-1)
                    if inc_np.size != n_pol_obs:
                        raise RuntimeError(
                            "IncorrectPolarityProbability length does not match "
                            "number of observations."
                        )
                    incorrect_prob = (
                        np.repeat(inc_np, n_loc_samples_pol)
                        if n_loc_samples_pol > 1
                        else inc_np
                    )

            else:
                a_pol_np = None
                error_pol_flat = None
                pol_observations = None
                incorrect_prob = 0.0

            # Prepare absolute amplitude data
            if has_amp:
                (
                    a_amp,
                    amplitude_amp,
                    perc_err_amp,
                ) = amplitude_matrix(filtered, location_samples=location_samples)

                a_amp_arr = np.asarray(a_amp, dtype=float)
                if a_amp_arr.ndim != 3:
                    raise RuntimeError(
                        "Amplitude matrices are expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_amp_obs, n_loc_samples_amp, _ = a_amp_arr.shape
                a_amp_np = a_amp_arr.reshape(n_amp_obs * n_loc_samples_amp, 6)

                amp_base = np.asarray(amplitude_amp, dtype=float).reshape(-1)
                perc_err_amp_np = np.asarray(perc_err_amp, dtype=float).reshape(-1)
                if amp_base.size != n_amp_obs or perc_err_amp_np.size != n_amp_obs:
                    raise RuntimeError(
                        "Amplitude arrays do not match number of observations."
                    )

                # Treat fractional amplitude errors as log-space sigmas
                log_amp_sigma_base = np.abs(perc_err_amp_np)
                self.last_log_amp_sigma_base = log_amp_sigma_base.copy()

                if n_loc_samples_amp > 1:
                    amp_obs = np.repeat(amp_base, n_loc_samples_amp)
                    log_amp_sigma = np.repeat(log_amp_sigma_base, n_loc_samples_amp)
                else:
                    amp_obs = amp_base
                    log_amp_sigma = log_amp_sigma_base
            else:
                a_amp_np = None
                amp_obs = None
                log_amp_sigma = None
                self.last_log_amp_sigma_base = None

            # Prepare amplitude ratio data
            if has_ar:
                (
                    a1_ar,
                    a2_ar,
                    amplitude_ratio,
                    perc_err1,
                    perc_err2,
                ) = amplitude_ratio_matrix(filtered, location_samples=location_samples)

                a1_arr = np.asarray(a1_ar, dtype=float)
                a2_arr = np.asarray(a2_ar, dtype=float)
                if a1_arr.ndim != 3 or a2_arr.ndim != 3:
                    raise RuntimeError(
                        "Amplitude-ratio matrices are expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_ar_obs, n_loc_samples_ar, _ = a1_arr.shape
                a1_ar_np = a1_arr.reshape(n_ar_obs * n_loc_samples_ar, 6)
                a2_ar_np = a2_arr.reshape(n_ar_obs * n_loc_samples_ar, 6)

                amp_ratio_base = np.asarray(amplitude_ratio, dtype=float).reshape(-1)
                perc_err1_np = np.asarray(perc_err1, dtype=float).reshape(-1)
                perc_err2_np = np.asarray(perc_err2, dtype=float).reshape(-1)
                if (
                    amp_ratio_base.size != n_ar_obs
                    or perc_err1_np.size != n_ar_obs
                    or perc_err2_np.size != n_ar_obs
                ):
                    raise RuntimeError(
                        "Amplitude-ratio arrays do not match number of observations."
                    )

                # Combined error for log-ratio (per observation)
                # For LogNormal, we need sigma in log-space
                # Approximate from percentage errors
                log_ratio_sigma_base = np.sqrt(perc_err1_np**2 + perc_err2_np**2)
                # Expose per-observation log-ratio uncertainties for debugging.
                self.last_log_ratio_sigma_base = log_ratio_sigma_base.copy()
                if n_loc_samples_ar > 1:
                    amp_ratio_obs = np.repeat(amp_ratio_base, n_loc_samples_ar)
                    log_ratio_sigma = np.repeat(
                        log_ratio_sigma_base, n_loc_samples_ar
                    )
                else:
                    amp_ratio_obs = amp_ratio_base
                    log_ratio_sigma = log_ratio_sigma_base

            else:
                a1_ar_np = a2_ar_np = None
                amp_ratio_obs = None
                log_ratio_sigma = None
                self.last_log_ratio_sigma_base = None

            # Build PyMC model with PyTensor forward model
            with pm.Model() as model:
                # ===== Priors =====
                if self.dc:
                    # Double-couple constraint: fix source-type parameters
                    # gamma and delta to zero (pure DC, no ISO/CLVD).
                    gamma = pt.as_tensor(0.0)
                    delta = pt.as_tensor(0.0)
                else:
                    # gamma = pm.Uniform("gamma", lower=-np.pi / 6.0, upper=np.pi / 6.0)
                    # delta = pm.Uniform("delta", lower=-np.pi / 2.0, upper=np.pi / 2.0)
                    # Beta priors on [0, 1]
                    gamma_alpha, gamma_beta = self.gamma_beta_prior
                    delta_alpha, delta_beta = self.delta_beta_prior
                    gamma_raw = pm.Beta("gamma_raw", alpha=gamma_alpha, beta=gamma_beta)
                    delta_raw = pm.Beta("delta_raw", alpha=delta_alpha, beta=delta_beta)

                    # Map to original ranges:
                    # gamma ∈ [-π/6, π/6], delta ∈ [-π/2, π/2]
                    gamma = pm.Deterministic(
                        "gamma",
                        -np.pi / 6.0 + gamma_raw * (2.0 * np.pi / 6.0),
                    )
                    delta = pm.Deterministic(
                        "delta",
                        -np.pi / 2.0 + delta_raw * (np.pi),
                    )

                kappa = pm.Uniform("kappa", lower=0.0, upper=2.0 * np.pi)
                h = pm.Uniform("h", lower=0.0, upper=1.0)  # cos(dip)
                sigma_tape = pm.Uniform("sigma", lower=-np.pi / 2.0, upper=np.pi / 2.0)

                # ===== Forward Model (Pure PyTensor) =====
                mt6 = pt_Tape_MT6(gamma, delta, kappa, h, sigma_tape)

                # ===== Polarity Likelihood =====
                if has_pol and pol_observations is not None:
                    # Predicted signed amplitude for each (observation, location) pair.
                    # a_pol_np already includes the measured sign (see polarity_matrix),
                    # so X_pol = d_i * (g_i · mt6), where d_i ∈ {+1, -1} is the observed polarity.
                    X_pol = pt.dot(pt.as_tensor(a_pol_np), mt6)

                    # Recover geometry-only amplitude μ_i = g_i · mt6 from X_pol and the
                    # observed polarity sign d_i (encoded as 0/1 in pol_observations).
                    pol_obs_np = pol_observations.astype("float64")
                    sign_obs_np = 2.0 * pol_obs_np - 1.0  # {0,1} -> {-1,+1}
                    sign_obs = pt.as_tensor(sign_obs_np)
                    mu_geom = sign_obs * X_pol

                    # Measurement uncertainty per observation (avoid zero variance).
                    sigma_pol = pt.as_tensor(error_pol_flat.reshape(-1))
                    sigma_pol = pt.maximum(sigma_pol, _SMALL_NUMBER)

                    # Base probability that the true polarity is "up":
                    # f_i = Φ(μ_i / σ_i)
                    base_prob = 0.5 * (
                        1.0 + pt.erf(mu_geom / (pt.sqrt(2.0) * sigma_pol))
                    )

                    # Incorrect-polarity probability (scalar or per-observation array).
                    if isinstance(incorrect_prob, (float, int)):
                        inc_tensor = pt.as_tensor(
                            np.array(incorrect_prob, dtype="float64")
                        )
                    else:
                        inc_tensor = pt.as_tensor(
                            np.asarray(incorrect_prob, dtype=float).reshape(-1)
                        )

                    # Final probability that the observed polarity is "up" (1):
                    # p_up = inc + (1 - 2*inc) * Φ(μ / σ)
                    p_up = inc_tensor + (1.0 - 2.0 * inc_tensor) * base_prob

                    pm.Bernoulli(
                        "polarity_obs",
                        p=p_up,
                        observed=pol_observations,
                    )

                # ===== Absolute Amplitude Likelihood =====
                if has_amp and a_amp_np is not None and amp_obs is not None:
                    # Predicted absolute amplitudes
                    amp_pred = pt.abs(pt.dot(pt.as_tensor(a_amp_np), mt6))  # (N_obs_total,)

                    amp_obs_tensor = pt.as_tensor(amp_obs, dtype="float64")
                    log_amp_sigma_tensor = pt.as_tensor(log_amp_sigma, dtype="float64")
                    log_amp_sigma_tensor = pt.maximum(
                        log_amp_sigma_tensor, _SMALL_NUMBER
                    )

                    pm.LogNormal(
                        "amp_obs",
                        mu=pt.log(amp_pred + _SMALL_NUMBER),
                        sigma=log_amp_sigma_tensor,
                        observed=amp_obs_tensor,
                    )

                # ===== Amplitude Ratio Likelihood =====
                if has_ar:
                    # Predicted amplitudes
                    # a1_ar_np is (N_obs, 6), mt6 is (6,)
                    amp1_pred = pt.abs(pt.dot(pt.as_tensor(a1_ar_np), mt6))  # (N_obs,)
                    amp2_pred = pt.abs(pt.dot(pt.as_tensor(a2_ar_np), mt6))  # (N_obs,)

                    # Predicted log-ratio
                    log_ratio_pred = pt.log(amp1_pred + _SMALL_NUMBER) - pt.log(amp2_pred + _SMALL_NUMBER)

                    # Uncertainty parameter (per observation)
                    # Use HalfCauchy prior for overall scale
                    sigma_ar = pm.HalfCauchy("sigma_amp_ratio", beta=self.amp_ratio_sigma_prior)

                    # Combined uncertainty (measurement + model)
                    sigma_combined = pt.sqrt(
                        pt.as_tensor(log_ratio_sigma)**2 + sigma_ar**2
                    )

                    # LogNormal likelihood on the observed ratio
                    pm.LogNormal(
                        "amp_ratio_obs",
                        mu=log_ratio_pred,
                        sigma=sigma_combined,
                        observed=amp_ratio_obs,
                    )

                # ===== Sampling =====
                if self.random_seed is None:
                    seed: Any = int(self.rng.integers(2**31 - 1))
                else:
                    seed = self.random_seed

                if self.safe_sampling and self.chains > 1:
                    # The SMC hang is often caused by progress bar + ProcessPoolExecutor.
                    # Disabling progressbar while keeping parallel execution usually works.
                    print(f"Running {self.chains} SMC chains with safe_sampling (progressbar disabled)...")
                    idata = pm.sample_smc(
                        draws=self.draws,
                        chains=self.chains,
                        kernel=self.smc_kernel,
                        random_seed=seed,
                        cores=self.cores,
                        compute_convergence_checks=self.compute_convergence_checks,
                        return_inferencedata=True,
                        progressbar=False,  # Disable progress bar to avoid hang
                        **self.smc_kernel_kwargs,
                    )

                else:
                    idata = pm.sample_smc(
                        draws=self.draws,
                        chains=self.chains,
                        kernel=self.smc_kernel,
                        random_seed=seed,
                        cores=self.cores,
                        compute_convergence_checks=self.compute_convergence_checks,
                        return_inferencedata=True,
                        progressbar=self.progressbar,
                        **self.smc_kernel_kwargs,
                    )

            # Extract posterior samples
            post = idata.posterior
            if "gamma" in post.data_vars and "delta" in post.data_vars:
                gamma_s = np.asarray(post["gamma"]).reshape(-1)
                delta_s = np.asarray(post["delta"]).reshape(-1)
                kappa_s = np.asarray(post["kappa"]).reshape(-1)
                h_s = np.asarray(post["h"]).reshape(-1)
                sigma_s = np.asarray(post["sigma"]).reshape(-1)
                n_samp = gamma_s.size
            else:
                # DC case: gamma = delta = 0, only orientation parameters are sampled.
                kappa_s = np.asarray(post["kappa"]).reshape(-1)
                h_s = np.asarray(post["h"]).reshape(-1)
                sigma_s = np.asarray(post["sigma"]).reshape(-1)
                n_samp = kappa_s.size
                gamma_s = np.zeros(n_samp, dtype=float)
                delta_s = np.zeros(n_samp, dtype=float)

            # Convert posterior Tape samples to MT6 vectors
            # Use NumPy version for post-processing
            from .tape import Tape_MT6
            mt6_samples = Tape_MT6(gamma_s, delta_s, kappa_s, h_s, sigma_s)
            mt6_samples = np.asarray(mt6_samples, dtype=float)
            if mt6_samples.ndim == 1:
                mt6_samples = mt6_samples.reshape(6, n_samp)
            elif mt6_samples.shape != (6, n_samp):
                mt6_samples = mt6_samples.reshape(6, n_samp)

            # Weights (SMC returns approximately posterior-distributed samples)
            if n_samp > 0:
                weights = np.ones(n_samp, dtype=float) / float(n_samp)
            else:
                weights = np.zeros(0, dtype=float)

            result = InversionResult(
                mt6=mt6_samples,
                gamma=gamma_s,
                delta=delta_s,
                kappa=kappa_s,
                h=h_s,
                sigma=sigma_s,
                ln_p=None,  # Not computed in this version
                weights=weights,
                idata=idata,
            )
            event_results.append(result)

        self.results = event_results[0] if len(event_results) == 1 else event_results
        return self.results

    def _filter_event_by_options(self, event: DataDict) -> DataDict:
        """
        Return a shallow copy of the event containing only keys listed
        in inversion_options (plus 'UID' if present).
        """
        if self.inversion_options is None:
            return dict(event)

        filtered: DataDict = {}
        for key, value in event.items():
            if key == "UID" or key in self.inversion_options:
                filtered[key] = value
        return filtered

    def _extract_polarity_observations(self, data: DataDict) -> np.ndarray:
        """
        Extract observed polarity signs (0=down, 1=up) from data dictionary.

        Parameters
        ----------
        data : dict
            Event data dictionary

        Returns
        -------
        np.ndarray
            Observed polarities as 0/1 array, shape (N_obs,)
        """
        polarities = []

        for key in sorted(
            k for k in data.keys() if "polarity" in k.lower() and "prob" not in k.lower()
        ):
            measured = np.asarray(data[key]["Measured"], dtype=float).flatten()
            # Convert -1/+1 to 0/1
            pol_01 = ((measured + 1) / 2).astype(int)
            polarities.append(pol_01)

        if polarities:
            return np.concatenate(polarities)
        else:
            return np.array([], dtype=int)


class InversionPyTensorNuts:
    """
    PyTensor-based NUTS inversion in Tape parameterisation using nutpie.

    This version uses pure PyTensor operations for the forward model and
    standard PyMC distributions for likelihoods, with nutpie for efficient
    NUTS sampling via Rust/JAX backends.

    Parameters
    ----------
    data : dict or list of dict
        Event data dictionary or list of dictionaries in the same format as
        MTfit's example_data (keys like 'PPolarity', 'P/SHAmplitudeRatio', etc.).
    inversion_options : str or sequence of str, optional
        String or sequence of strings specifying which data types to use,
        e.g. 'PPolarity' or ['PPolarity', 'P/SHAmplitudeRatio'].
    draws : int, default 2000
        Number of posterior draws (per chain) to return per event.
    tune : int, default 1000
        Number of tuning (warm-up) steps for NUTS.
    chains : int, default 4
        Number of NUTS chains (parallelization).
    cores : int, optional
        Number of CPU cores to use. If ``None``, defaults to number of chains.
    dc : bool, default False
        If True, restricts the source to double-couple (DC) space by fixing
        the Tape source-type parameters ``gamma = 0`` and ``delta = 0``.
    random_seed : int or RandomState, optional
        Seed forwarded to the sampler.
    progressbar : bool, default True
        Whether to show progress bar.
    amp_ratio_sigma_prior : float, default 0.5
        Scale parameter for HalfCauchy prior on log-ratio uncertainty.
    gamma_beta_prior : tuple, default (1.0, 1.0)
        Alpha, beta parameters for Beta prior on Tape source-type parameter
        ``gamma`` on the unit interval before mapping to Tape range.
    delta_beta_prior : tuple, default (1.0, 1.0)
        Alpha, beta parameters for Beta prior on Tape source-type parameter
        ``delta`` on the unit interval before mapping to Tape range.
    location_samples_n : int, default 20
        Number of station-angle location samples to draw per event when
        ``Azimuth_err`` and ``TakeOff_err`` are provided in the input data.
    azimuth_error : float, optional
        Global 1σ uncertainty (in degrees) applied to all station azimuths.
    takeoff_error : float, optional
        Global 1σ uncertainty (in degrees) applied to all station takeoff angles.

    Notes
    -----
    - This implementation uses nutpie for efficient NUTS sampling
    - Requires nutpie to be installed: ``pip install nutpie``
    - Can achieve significant speedups over PyMC's default sampler
    """

    def __init__(
        self,
        data: Union[DataDict, Sequence[DataDict]],
        inversion_options: Optional[Union[str, Sequence[str]]] = None,
        draws: int = 2_000,
        tune: int = 1_000,
        chains: int = 4,
        cores: Optional[int] = None,
        dc: bool = False,
        random_seed: Optional[Union[int, np.random.Generator, np.random.RandomState]] = 68,
        progressbar: bool = True,
        amp_ratio_sigma_prior: float = 0.5,
        location_samples_n: int = 20,
        azimuth_error: Optional[float] = None,
        takeoff_error: Optional[float] = None,
        gamma_beta_prior: tuple = (1.0, 1.0),
        delta_beta_prior: tuple = (1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        # Normalise data to a list of events
        if isinstance(data, dict):
            self.data: List[DataDict] = [data]
        else:
            self.data = list(data)

        # Normalise inversion options
        if inversion_options is None:
            self.inversion_options: Optional[List[str]] = None
        elif isinstance(inversion_options, str):
            self.inversion_options = [inversion_options]
        else:
            self.inversion_options = list(inversion_options)

        # Backwards compatibility
        if "max_samples" in kwargs and draws == 2_000:
            draws = int(kwargs.pop("max_samples"))

        self.draws = int(draws)
        self.tune = int(tune)
        self.chains = int(chains)
        self.cores = int(cores) if cores is not None else self.chains
        self.dc = bool(dc)

        # Store random seed and an internal RNG for fallbacks
        if isinstance(random_seed, (np.random.Generator, np.random.RandomState)):
            self.random_seed = random_seed
            self.rng = np.random.default_rng()
        else:
            self.random_seed = random_seed
            self.rng = np.random.default_rng(random_seed)

        self.progressbar = bool(progressbar)

        # Prior hyperparameters
        self.amp_ratio_sigma_prior = amp_ratio_sigma_prior
        self.gamma_beta_prior = gamma_beta_prior
        self.delta_beta_prior = delta_beta_prior

        # Number of location samples for station angle uncertainty
        self.location_samples_n = int(location_samples_n)
        self.azimuth_error = azimuth_error
        self.takeoff_error = takeoff_error

        # Debug/diagnostic fields (set during forward())
        self.last_log_ratio_sigma_base: Optional[np.ndarray] = None

        self.results: Union[InversionResult, List[InversionResult], None] = None

    def forward(self) -> Union[InversionResult, List[InversionResult]]:
        """
        Run the NUTS-based forward model and likelihood evaluation using nutpie.

        Returns
        -------
        InversionResult or list of InversionResult
            Results for each event. For a single-event input, a single
            InversionResult is returned for convenience.
        """
        import nutpie

        event_results: List[InversionResult] = []

        for event in self.data:
            filtered = self._filter_event_by_options(event)

            # Build data matrices for polarities, absolute amplitudes and amplitude ratios
            has_pol = any(
                "polarity" in key.lower() and "prob" not in key.lower()
                for key in filtered.keys()
            )
            has_amp = any(
                "amplitude" in key.lower() and "amplituderatio" not in key.lower()
                for key in filtered.keys()
            )
            has_ar = any(
                "amplituderatio" in key.lower() or "amplitude_ratio" in key.lower()
                for key in filtered.keys()
            )

            if not (has_pol or has_amp or has_ar):
                raise ValueError(
                    "No supported data types found in event for the given inversion_options."
                )

            # Pre-sample location scatter from station angle errors, if provided.
            location_samples = build_location_samples_from_errors(
                filtered,
                rng=self.rng,
                n_samples=self.location_samples_n,
                azimuth_error=self.azimuth_error,
                takeoff_error=self.takeoff_error,
            )

            # Prepare polarity data
            if has_pol:
                a_polarity, error_polarity, incorrect_polarity_prob = polarity_matrix(
                    filtered, location_samples=location_samples
                )
                a_pol_arr = np.asarray(a_polarity, dtype=float)
                if a_pol_arr.ndim != 3:
                    raise RuntimeError(
                        "Polarity matrix is expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_pol_obs, n_loc_samples_pol, _ = a_pol_arr.shape
                a_pol_np = a_pol_arr.reshape(n_pol_obs * n_loc_samples_pol, 6)

                error_pol_np = np.asarray(error_polarity, dtype=float).reshape(-1)
                if error_pol_np.size != n_pol_obs:
                    raise RuntimeError(
                        "Polarity error length does not match number of observations."
                    )
                if n_loc_samples_pol > 1:
                    error_pol_flat = np.repeat(error_pol_np, n_loc_samples_pol)
                else:
                    error_pol_flat = error_pol_np

                pol_obs_base = self._extract_polarity_observations(filtered)
                if pol_obs_base.size != n_pol_obs:
                    raise RuntimeError(
                        "Number of polarity observations does not match angle matrix."
                    )
                if n_loc_samples_pol > 1:
                    pol_observations = np.repeat(pol_obs_base, n_loc_samples_pol)
                else:
                    pol_observations = pol_obs_base

                if isinstance(incorrect_polarity_prob, (float, int)):
                    incorrect_prob = incorrect_polarity_prob
                else:
                    inc_np = np.asarray(incorrect_polarity_prob, dtype=float).reshape(-1)
                    if inc_np.size != n_pol_obs:
                        raise RuntimeError(
                            "IncorrectPolarityProbability length does not match "
                            "number of observations."
                        )
                    incorrect_prob = (
                        np.repeat(inc_np, n_loc_samples_pol)
                        if n_loc_samples_pol > 1
                        else inc_np
                    )

            else:
                a_pol_np = None
                error_pol_flat = None
                pol_observations = None
                incorrect_prob = 0.0

            # Prepare absolute amplitude data
            if has_amp:
                (
                    a_amp,
                    amplitude_amp,
                    perc_err_amp,
                ) = amplitude_matrix(filtered, location_samples=location_samples)

                a_amp_arr = np.asarray(a_amp, dtype=float)
                if a_amp_arr.ndim != 3:
                    raise RuntimeError(
                        "Amplitude matrices are expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_amp_obs, n_loc_samples_amp, _ = a_amp_arr.shape
                a_amp_np = a_amp_arr.reshape(n_amp_obs * n_loc_samples_amp, 6)

                amp_base = np.asarray(amplitude_amp, dtype=float).reshape(-1)
                perc_err_amp_np = np.asarray(perc_err_amp, dtype=float).reshape(-1)
                if amp_base.size != n_amp_obs or perc_err_amp_np.size != n_amp_obs:
                    raise RuntimeError(
                        "Amplitude arrays do not match number of observations."
                    )

                log_amp_sigma_base = np.abs(perc_err_amp_np)
                self.last_log_amp_sigma_base = log_amp_sigma_base.copy()

                if n_loc_samples_amp > 1:
                    amp_obs = np.repeat(amp_base, n_loc_samples_amp)
                    log_amp_sigma = np.repeat(log_amp_sigma_base, n_loc_samples_amp)
                else:
                    amp_obs = amp_base
                    log_amp_sigma = log_amp_sigma_base
            else:
                a_amp_np = None
                amp_obs = None
                log_amp_sigma = None
                self.last_log_amp_sigma_base = None

            # Prepare amplitude ratio data
            if has_ar:
                (
                    a1_ar,
                    a2_ar,
                    amplitude_ratio,
                    perc_err1,
                    perc_err2,
                ) = amplitude_ratio_matrix(filtered, location_samples=location_samples)

                a1_arr = np.asarray(a1_ar, dtype=float)
                a2_arr = np.asarray(a2_ar, dtype=float)
                if a1_arr.ndim != 3 or a2_arr.ndim != 3:
                    raise RuntimeError(
                        "Amplitude-ratio matrices are expected to have shape "
                        "(N_obs, N_loc_samples, 6)."
                    )
                n_ar_obs, n_loc_samples_ar, _ = a1_arr.shape
                a1_ar_np = a1_arr.reshape(n_ar_obs * n_loc_samples_ar, 6)
                a2_ar_np = a2_arr.reshape(n_ar_obs * n_loc_samples_ar, 6)

                amp_ratio_base = np.asarray(amplitude_ratio, dtype=float).reshape(-1)
                perc_err1_np = np.asarray(perc_err1, dtype=float).reshape(-1)
                perc_err2_np = np.asarray(perc_err2, dtype=float).reshape(-1)
                if (
                    amp_ratio_base.size != n_ar_obs
                    or perc_err1_np.size != n_ar_obs
                    or perc_err2_np.size != n_ar_obs
                ):
                    raise RuntimeError(
                        "Amplitude-ratio arrays do not match number of observations."
                    )

                log_ratio_sigma_base = np.sqrt(perc_err1_np**2 + perc_err2_np**2)
                self.last_log_ratio_sigma_base = log_ratio_sigma_base.copy()
                if n_loc_samples_ar > 1:
                    amp_ratio_obs = np.repeat(amp_ratio_base, n_loc_samples_ar)
                    log_ratio_sigma = np.repeat(
                        log_ratio_sigma_base, n_loc_samples_ar
                    )
                else:
                    amp_ratio_obs = amp_ratio_base
                    log_ratio_sigma = log_ratio_sigma_base

            else:
                a1_ar_np = a2_ar_np = None
                amp_ratio_obs = None
                log_ratio_sigma = None
                self.last_log_ratio_sigma_base = None

            # Build PyMC model with PyTensor forward model
            with pm.Model() as model:
                # ===== Priors =====
                if self.dc:
                    gamma = pt.as_tensor(0.0)
                    delta = pt.as_tensor(0.0)
                else:
                    gamma_alpha, gamma_beta = self.gamma_beta_prior
                    delta_alpha, delta_beta = self.delta_beta_prior
                    gamma_raw = pm.Beta("gamma_raw", alpha=gamma_alpha, beta=gamma_beta)
                    delta_raw = pm.Beta("delta_raw", alpha=delta_alpha, beta=delta_beta)

                    gamma = pm.Deterministic(
                        "gamma",
                        -np.pi / 6.0 + gamma_raw * (2.0 * np.pi / 6.0),
                    )
                    delta = pm.Deterministic(
                        "delta",
                        -np.pi / 2.0 + delta_raw * (np.pi),
                    )

                kappa = pm.Uniform("kappa", lower=0.0, upper=2.0 * np.pi)
                h = pm.Uniform("h", lower=0.0, upper=1.0)
                sigma_tape = pm.Uniform("sigma", lower=-np.pi / 2.0, upper=np.pi / 2.0)

                # ===== Forward Model (Pure PyTensor) =====
                mt6 = pt_Tape_MT6(gamma, delta, kappa, h, sigma_tape)

                # ===== Polarity Likelihood =====
                if has_pol and pol_observations is not None:
                    X_pol = pt.dot(pt.as_tensor(a_pol_np), mt6)

                    pol_obs_np = pol_observations.astype("float64")
                    sign_obs_np = 2.0 * pol_obs_np - 1.0
                    sign_obs = pt.as_tensor(sign_obs_np)
                    mu_geom = sign_obs * X_pol

                    sigma_pol = pt.as_tensor(error_pol_flat.reshape(-1))
                    sigma_pol = pt.maximum(sigma_pol, _SMALL_NUMBER)

                    base_prob = 0.5 * (
                        1.0 + pt.erf(mu_geom / (pt.sqrt(2.0) * sigma_pol))
                    )

                    if isinstance(incorrect_prob, (float, int)):
                        inc_tensor = pt.as_tensor(
                            np.array(incorrect_prob, dtype="float64")
                        )
                    else:
                        inc_tensor = pt.as_tensor(
                            np.asarray(incorrect_prob, dtype=float).reshape(-1)
                        )

                    p_up = inc_tensor + (1.0 - 2.0 * inc_tensor) * base_prob

                    pm.Bernoulli(
                        "polarity_obs",
                        p=p_up,
                        observed=pol_observations,
                    )

                # ===== Absolute Amplitude Likelihood =====
                if has_amp and a_amp_np is not None and amp_obs is not None:
                    amp_pred = pt.abs(pt.dot(pt.as_tensor(a_amp_np), mt6))

                    amp_obs_tensor = pt.as_tensor(amp_obs, dtype="float64")
                    log_amp_sigma_tensor = pt.as_tensor(log_amp_sigma, dtype="float64")
                    log_amp_sigma_tensor = pt.maximum(
                        log_amp_sigma_tensor, _SMALL_NUMBER
                    )

                    pm.LogNormal(
                        "amp_obs",
                        mu=pt.log(amp_pred + _SMALL_NUMBER),
                        sigma=log_amp_sigma_tensor,
                        observed=amp_obs_tensor,
                    )

                # ===== Amplitude Ratio Likelihood =====
                if has_ar:
                    amp1_pred = pt.abs(pt.dot(pt.as_tensor(a1_ar_np), mt6))
                    amp2_pred = pt.abs(pt.dot(pt.as_tensor(a2_ar_np), mt6))

                    log_ratio_pred = pt.log(amp1_pred + _SMALL_NUMBER) - pt.log(amp2_pred + _SMALL_NUMBER)

                    sigma_ar = pm.HalfCauchy("sigma_amp_ratio", beta=self.amp_ratio_sigma_prior)

                    sigma_combined = pt.sqrt(
                        pt.as_tensor(log_ratio_sigma)**2 + sigma_ar**2
                    )

                    pm.LogNormal(
                        "amp_ratio_obs",
                        mu=log_ratio_pred,
                        sigma=sigma_combined,
                        observed=amp_ratio_obs,
                    )

                # ===== Sampling with nutpie =====
                if self.random_seed is None:
                    seed: Any = int(self.rng.integers(2**31 - 1))
                else:
                    seed = self.random_seed if isinstance(self.random_seed, int) else int(self.rng.integers(2**31 - 1))

                # Compile the model with nutpie
                compiled_model = nutpie.compile_pymc_model(model)

                # Sample using nutpie
                idata = nutpie.sample(
                    compiled_model,
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    seed=seed,
                    progress_bar=self.progressbar,
                )

            # Extract posterior samples
            post = idata.posterior
            if "gamma" in post.data_vars and "delta" in post.data_vars:
                gamma_s = np.asarray(post["gamma"]).reshape(-1)
                delta_s = np.asarray(post["delta"]).reshape(-1)
                kappa_s = np.asarray(post["kappa"]).reshape(-1)
                h_s = np.asarray(post["h"]).reshape(-1)
                sigma_s = np.asarray(post["sigma"]).reshape(-1)
                n_samp = gamma_s.size
            else:
                kappa_s = np.asarray(post["kappa"]).reshape(-1)
                h_s = np.asarray(post["h"]).reshape(-1)
                sigma_s = np.asarray(post["sigma"]).reshape(-1)
                n_samp = kappa_s.size
                gamma_s = np.zeros(n_samp, dtype=float)
                delta_s = np.zeros(n_samp, dtype=float)

            # Convert posterior Tape samples to MT6 vectors
            from .tape import Tape_MT6
            mt6_samples = Tape_MT6(gamma_s, delta_s, kappa_s, h_s, sigma_s)
            mt6_samples = np.asarray(mt6_samples, dtype=float)
            if mt6_samples.ndim == 1:
                mt6_samples = mt6_samples.reshape(6, n_samp)
            elif mt6_samples.shape != (6, n_samp):
                mt6_samples = mt6_samples.reshape(6, n_samp)

            # Weights (NUTS returns equally weighted posterior samples)
            if n_samp > 0:
                weights = np.ones(n_samp, dtype=float) / float(n_samp)
            else:
                weights = np.zeros(0, dtype=float)

            result = InversionResult(
                mt6=mt6_samples,
                gamma=gamma_s,
                delta=delta_s,
                kappa=kappa_s,
                h=h_s,
                sigma=sigma_s,
                ln_p=None,
                weights=weights,
                idata=idata,
            )
            event_results.append(result)

        self.results = event_results[0] if len(event_results) == 1 else event_results
        return self.results

    def _filter_event_by_options(self, event: DataDict) -> DataDict:
        """
        Return a shallow copy of the event containing only keys listed
        in inversion_options (plus 'UID' if present).
        """
        if self.inversion_options is None:
            return dict(event)

        filtered: DataDict = {}
        for key, value in event.items():
            if key == "UID" or key in self.inversion_options:
                filtered[key] = value
        return filtered

    def _extract_polarity_observations(self, data: DataDict) -> np.ndarray:
        """
        Extract observed polarity signs (0=down, 1=up) from data dictionary.

        Parameters
        ----------
        data : dict
            Event data dictionary

        Returns
        -------
        np.ndarray
            Observed polarities as 0/1 array, shape (N_obs,)
        """
        polarities = []

        for key in sorted(
            k for k in data.keys() if "polarity" in k.lower() and "prob" not in k.lower()
        ):
            measured = np.asarray(data[key]["Measured"], dtype=float).flatten()
            # Convert -1/+1 to 0/1
            pol_01 = ((measured + 1) / 2).astype(int)
            polarities.append(pol_01)

        if polarities:
            return np.concatenate(polarities)
        else:
            return np.array([], dtype=int)
