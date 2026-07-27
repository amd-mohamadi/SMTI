"""
BlackJAX-based SMC inversion for moment tensors.

This is an experimental alternative to the PyMC/PyTensor-based InversionPyTensor.
It uses BlackJAX's adaptive tempered SMC sampler with JAX for JIT compilation.

Key advantages:
- Pure JAX implementation supports GPU or CPU execution
- Faster compilation (no PyTensor compile cache issues)
- All sampling logic is JIT-compiled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import multiprocessing as mp
import queue
import re
import shutil
import sys
import time
import warnings

import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.scipy as jsp
from jax import random

# BlackJAX imports
import blackjax
from blackjax.smc.resampling import systematic
from blackjax.smc import inner_kernel_tuning
import blackjax.mcmc.random_walk as blackjax_rw

# Local imports
from .blockwise_rmh import (
    default_rmh_blocks,
    build_blockwise_scale_overrides,
)
from .mwg_kernel import (
    build_mwg_kernel,
    build_mwg_parameter_update_fn,
    default_mwg_blocks,
    default_inner_steps,
)
from .tape_jax import jax_Tape_MT6, jax_Tape_MT6_batch
from .data_prep import (
    polarity_matrix,
    amplitude_ratio_matrix,
    build_location_samples_from_errors,
)


DataDict = Dict[str, Any]
_SMALL_NUMBER = 1e-10

# Above this argument value ``expm1`` is already astronomically larger than the
# ``eps`` floor, so the two branches of :func:`softplus_inv` agree to well
# below float64 round-off (relative difference ~1e-16 at x = 20).
_SOFTPLUS_INV_LARGE_X = 20.0


def softplus_inv(x, eps: float = 1e-6):
    """Numerically stable inverse of ``softplus``, i.e. ``log(exp(x) - 1)``.

    Behaviourally equivalent to the original formulation
    ``log(expm1(max(x, eps)) + eps)`` (identical to <1e-12 relative in float64),
    but overflow-free: ``expm1`` returns ``+inf`` for ``x >~ 89`` in float32
    (and ``x >~ 710`` in float64), which turned ~1.5% of the initial
    ``sigma_amp_ratio`` particles into ``+inf`` in float32 and silently froze
    the noise block for a whole run (M0 cross-cutting finding 2).

    For large ``x`` the identity ``log(expm1(x)) = x + log1p(-exp(-x))`` is used,
    which never evaluates ``exp`` of a large positive number.  The ``eps`` floor
    on ``x`` and the ``+ eps`` inside the log (which matters only for very small
    ``x``) are preserved.

    .. note::

       "Behaviourally equivalent" is not "bit-identical": over the actual
       initial-particle distribution ~0.2% of float64 values (those on the
       ``x > 20`` branch) shift by 1 ulp (max |diff| ~7e-15).  MCMC
       accept/reject amplifies last-bit differences chaotically, so a seeded
       run started after this change does **not** reproduce a pre-change run
       particle-for-particle -- it reproduces it statistically.  Archived
       reference posteriors (e.g. the eq00124 numbers in
       ``202607_q_factor_diagnosis.md``) need re-baselining after the merge.
    """
    xc = jnp.maximum(x, eps)
    is_large = xc > _SOFTPLUS_INV_LARGE_X
    # Guard both branches so neither can produce inf/nan for the *other*
    # branch's inputs (jnp.where evaluates both sides).
    x_small = jnp.where(is_large, jnp.asarray(1.0, xc.dtype), xc)
    x_large = jnp.where(is_large, xc, jnp.asarray(_SOFTPLUS_INV_LARGE_X, xc.dtype))
    small = jnp.log(jnp.expm1(x_small) + eps)
    large = x_large + jnp.log1p(-jnp.exp(-x_large))
    return jnp.where(is_large, large, small)


def _blackjax_process_chain_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one BlackJAX chain in a spawned CPU process."""
    import jax

    if payload.get("chain_device") == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    sampler = InversionBlackJAX(
        data=payload["event"],
        inversion_options=payload["inversion_options"],
        num_particles=payload["num_particles"],
        dc=payload["dc"],
        random_seed=payload["seed"],
        location_samples_n=payload["location_samples_n"],
        azimuth_error=payload["azimuth_error"],
        takeoff_error=payload["takeoff_error"],
        gamma_beta_prior=payload["gamma_beta_prior"],
        delta_beta_prior=payload["delta_beta_prior"],
        amp_ratio_sigma_prior=payload["amp_ratio_sigma_prior"],
        amp_ratio_noise_mode=payload["amp_ratio_noise_mode"],
        amp_ratio_station_log_scale=payload["amp_ratio_station_log_scale"],
        amp_ratio_smooth_length_deg=payload["amp_ratio_smooth_length_deg"],
        amp_ratio_smooth_k=payload["amp_ratio_smooth_k"],
        amp_ratio_smooth_strength=payload["amp_ratio_smooth_strength"],
        num_mcmc_steps=payload["num_mcmc_steps"],
        mcmc_kernel=payload["mcmc_kernel"],
        rmh_proposal_scale=payload["rmh_proposal_scale"],
        mechanism_steps=payload["mechanism_steps"],
        nuts_adapt_steps=payload["nuts_adapt_steps"],
        nuts_initial_step_size=payload["nuts_initial_step_size"],
        nuts_target_acceptance=payload["nuts_target_acceptance"],
        nuts_max_num_doublings=payload["nuts_max_num_doublings"],
        smc_target_ess_ratio=payload["smc_target_ess_ratio"],
        max_smc_iterations=payload["max_smc_iterations"],
        use_nuts_adaptation=payload["use_nuts_adaptation"],
        adapt_proposal=payload["adapt_proposal"],
        min_tempering_increment=payload["min_tempering_increment"],
        tempering_stall_patience=payload["tempering_stall_patience"],
        smc_method=payload["smc_method"],
        ps_target_ess=payload["ps_target_ess"],
        num_chains=1,
        chain_execution="sequential",
        chain_device=payload.get("chain_device", "default"),
    )
    progress_queue = payload.get("progress_queue")

    def _progress_callback(message: str) -> None:
        if progress_queue is not None:
            progress_queue.put(
                {"chain_idx": payload["chain_idx"], "text": str(message)}
            )

    result = sampler._invert_single_event(
        payload["event"],
        location_samples=payload["location_samples"],
        progress_callback=_progress_callback,
    )
    return sampler._serialize_chain_result(result)


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
        Log-likelihood for each sample (placeholder).
    weights : np.ndarray
        Normalized posterior weights for each sample.
    idata : Any
        Placeholder for compatibility (None for BlackJAX).
    """

    mt6: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray
    kappa: np.ndarray
    h: np.ndarray
    sigma: np.ndarray
    ln_p: Optional[np.ndarray]
    weights: np.ndarray
    idata: Any
    sigma_amp_ratio: Optional[np.ndarray] = None
    sigma_amp_ratio_global: Optional[np.ndarray] = None
    sigma_amp_ratio_station: Optional[np.ndarray] = None
    sigma_amp_ratio_station_names: Optional[np.ndarray] = None
    num_chains: int = 1
    #: True when SMC stopped on the tempering-stall guard instead of reaching
    #: beta = 1 (set per chain by the batched path; stacked results report
    #: True if any chain stalled).
    tempering_stalled: bool = False


class InversionBlackJAX:
    """
    BlackJAX-based SMC inversion in Tape parameterization.

    This is an experimental alternative to InversionPyTensor that uses
    BlackJAX's adaptive tempered SMC sampler for faster inference.

    Parameters
    ----------
    data : dict or list of dict
        Event data dictionary with keys like 'PPolarity', 'P/SHAmplitudeRatio', etc.
    inversion_options : str or sequence of str, optional
        Which data types to use, e.g. 'PPolarity' or ['PPolarity', 'P/SHAmplitudeRatio'].
    num_particles : int, default 2000
        Number of SMC particles.
    dc : bool, default False
        If True, fix gamma=delta=0 (double-couple constraint).
    random_seed : int, optional
        Random seed for reproducibility.
    location_samples_n : int, default 5
        Number of location samples for station angle uncertainty.
    azimuth_error : float, optional
        Global azimuth error in degrees.
    takeoff_error : float, optional
        Global takeoff angle error in degrees.
    gamma_beta_prior : tuple, default (1.0, 1.0)
        Beta prior parameters for gamma.
    delta_beta_prior : tuple, default (1.0, 1.0)
        Beta prior parameters for delta.
    num_chains : int, default 4
        Number of chains. By default, BlackJAX runs four chains.
    chain_execution : {"sequential", "process"}, default "process"
        Chain execution backend. By default, multi-chain runs use one process per chain.
    chain_device : {"default", "cpu"}, default "cpu"
        Device routing for the chain backend. By default, the multi-chain backend targets CPU.
    """

    def __init__(
        self,
        data: Union[DataDict, Sequence[DataDict]],
        inversion_options: Optional[Union[str, Sequence[str]]] = None,
        num_particles: int = 2000,
        dc: bool = False,
        random_seed: Optional[int] = 42,
        location_samples_n: int = 5,
        azimuth_error: Optional[float] = None,
        takeoff_error: Optional[float] = None,
        gamma_beta_prior: tuple = (1.0, 1.0),
        delta_beta_prior: tuple = (1.0, 1.0),
        amp_ratio_sigma_prior: float = 5.0,
        amp_ratio_noise_mode: str = "global",
        amp_ratio_station_log_scale: float = 0.3,
        amp_ratio_smooth_length_deg: float = 30.0,
        amp_ratio_smooth_k: int = 4,
        amp_ratio_smooth_strength: float = 1.0,
        num_mcmc_steps: int = 5,
        mcmc_kernel: str = "rmh",
        rmh_proposal_scale: float = 0.1,
        nuts_adapt_steps: int = 500,
        nuts_initial_step_size: float = 1e-2,
        nuts_target_acceptance: float = 0.8,
        nuts_max_num_doublings: int = 10,
        smc_target_ess_ratio: float = 0.9,
        max_smc_iterations: int = 60,
        use_nuts_adaptation: bool = True,
        adapt_proposal: bool = True,
        min_tempering_increment: float = 1e-4,
        tempering_stall_patience: int = 8,
        smc_method: str = "adaptive_tempered",
        ps_target_ess: float = 5.0,
        mechanism_steps: int = 3,
        **kwargs: Any,
    ) -> None:
        # Normalize data to a list
        if isinstance(data, dict):
            self.data: List[DataDict] = [data]
        else:
            self.data = list(data)

        # Normalize inversion options
        if inversion_options is None:
            self.inversion_options: Optional[List[str]] = None
        elif isinstance(inversion_options, str):
            self.inversion_options = [inversion_options]
        else:
            self.inversion_options = list(inversion_options)

        self.num_particles = int(num_particles)
        self.dc = bool(dc)
        self.random_seed = (
            int(random_seed)
            if random_seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        self.location_samples_n = int(location_samples_n)
        self.azimuth_error = azimuth_error
        self.takeoff_error = takeoff_error
        self.gamma_beta_prior = gamma_beta_prior
        self.delta_beta_prior = delta_beta_prior
        self.amp_ratio_sigma_prior = amp_ratio_sigma_prior
        self.amp_ratio_noise_mode = str(amp_ratio_noise_mode).lower()
        if self.amp_ratio_noise_mode not in {"global", "station_smooth"}:
            raise ValueError(
                "amp_ratio_noise_mode must be one of {'global', 'station_smooth'}"
            )
        self.amp_ratio_station_log_scale = float(amp_ratio_station_log_scale)
        if self.amp_ratio_station_log_scale <= 0.0:
            raise ValueError("amp_ratio_station_log_scale must be > 0.")
        self.amp_ratio_smooth_length_deg = float(amp_ratio_smooth_length_deg)
        if self.amp_ratio_smooth_length_deg <= 0.0:
            raise ValueError("amp_ratio_smooth_length_deg must be > 0.")
        self.amp_ratio_smooth_k = int(amp_ratio_smooth_k)
        if self.amp_ratio_smooth_k < 0:
            raise ValueError("amp_ratio_smooth_k must be >= 0.")
        self.amp_ratio_smooth_strength = float(amp_ratio_smooth_strength)
        if self.amp_ratio_smooth_strength < 0.0:
            raise ValueError("amp_ratio_smooth_strength must be >= 0.")
        self.num_mcmc_steps = num_mcmc_steps
        self.mcmc_kernel = str(mcmc_kernel).lower()
        if self.mcmc_kernel not in {"rmh", "nuts"}:
            raise ValueError("mcmc_kernel must be one of {'rmh', 'nuts'}")
        self.rmh_proposal_scale = float(rmh_proposal_scale)
        self.nuts_adapt_steps = nuts_adapt_steps
        self.nuts_initial_step_size = nuts_initial_step_size
        self.nuts_target_acceptance = nuts_target_acceptance
        self.nuts_max_num_doublings = nuts_max_num_doublings
        self.smc_target_ess_ratio = smc_target_ess_ratio
        if not (0.0 < self.smc_target_ess_ratio <= 1.0):
            raise ValueError("smc_target_ess_ratio must be in (0, 1].")
        self.max_smc_iterations = max_smc_iterations
        self.use_nuts_adaptation = use_nuts_adaptation
        self.adapt_proposal = bool(adapt_proposal)
        self.min_tempering_increment = float(min_tempering_increment)
        self.tempering_stall_patience = int(tempering_stall_patience)
        self.smc_method = str(smc_method).lower()
        if self.smc_method not in {"adaptive_tempered", "adaptive_persistent"}:
            raise ValueError(
                "smc_method must be one of {'adaptive_tempered', 'adaptive_persistent'}"
            )
        self.ps_target_ess = float(ps_target_ess)
        self.mechanism_steps = int(mechanism_steps)
        self.num_chains = int(kwargs.pop("num_chains", 4))
        if self.num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        chain_execution = kwargs.pop("chain_execution", None)
        if chain_execution is None:
            chain_execution = "process" if self.num_chains > 1 else "sequential"
        self.chain_execution = str(chain_execution).lower()
        if self.chain_execution not in {"sequential", "process"}:
            raise ValueError("chain_execution must be one of {'sequential', 'process'}")
        self.chain_device = str(kwargs.pop("chain_device", "cpu")).lower()
        if self.chain_device not in {"default", "cpu"}:
            raise ValueError("chain_device must be one of {'default', 'cpu'}")
        chain_cores = kwargs.pop("chain_cores", None)
        if chain_cores is None:
            self.chain_cores = (
                self.num_chains if self.chain_execution == "process" else None
            )
        else:
            self.chain_cores = int(chain_cores)
            if self.chain_cores <= 0:
                raise ValueError("chain_cores must be >= 1")
        if self.chain_execution == "process":
            if self.num_chains <= 1:
                raise ValueError("chain_execution='process' requires num_chains > 1")
            if self.chain_device != "cpu":
                raise ValueError(
                    "chain_execution='process' currently requires chain_device='cpu'"
                )

        self.rng = np.random.default_rng(self.random_seed)
        self.results: Optional[InversionResult] = None

    def forward(self) -> Union[InversionResult, List[InversionResult]]:
        """
        Run the BlackJAX SMC-based inversion.

        Returns
        -------
        InversionResult or list of InversionResult
        """
        event_results: List[InversionResult] = []

        for event in self.data:
            filtered = self._filter_event_by_options(event)
            if self.num_chains > 1:
                result = self._invert_multi_chain(filtered)
            else:
                result = self._invert_single_event(filtered)
            event_results.append(result)

        self.results = event_results[0] if len(event_results) == 1 else event_results
        return self.results

    def _filter_event_by_options(self, event: DataDict) -> DataDict:
        """Filter event to only include requested data types."""
        if self.inversion_options is None:
            return dict(event)

        filtered: DataDict = {}
        for key, value in event.items():
            if key == "UID" or key in self.inversion_options:
                filtered[key] = value
        return filtered

    @staticmethod
    def _serialize_chain_result(result: InversionResult) -> Dict[str, Any]:
        return {
            "mt6": np.asarray(result.mt6, dtype=float),
            "gamma": np.asarray(result.gamma, dtype=float),
            "delta": np.asarray(result.delta, dtype=float),
            "kappa": np.asarray(result.kappa, dtype=float),
            "h": np.asarray(result.h, dtype=float),
            "sigma": np.asarray(result.sigma, dtype=float),
            "weights": np.asarray(result.weights, dtype=float),
            "sigma_amp_ratio": (
                None
                if result.sigma_amp_ratio is None
                else np.asarray(result.sigma_amp_ratio, dtype=float)
            ),
            "sigma_amp_ratio_global": (
                None
                if result.sigma_amp_ratio_global is None
                else np.asarray(result.sigma_amp_ratio_global, dtype=float)
            ),
            "sigma_amp_ratio_station": (
                None
                if result.sigma_amp_ratio_station is None
                else np.asarray(result.sigma_amp_ratio_station, dtype=float)
            ),
            "sigma_amp_ratio_station_names": (
                None
                if result.sigma_amp_ratio_station_names is None
                else np.asarray(result.sigma_amp_ratio_station_names, dtype=object)
            ),
            "tempering_stalled": bool(result.tempering_stalled),
        }

    @staticmethod
    def _deserialize_chain_result(payload: Dict[str, Any]) -> InversionResult:
        return InversionResult(
            mt6=np.asarray(payload["mt6"], dtype=float),
            gamma=np.asarray(payload["gamma"], dtype=float),
            delta=np.asarray(payload["delta"], dtype=float),
            kappa=np.asarray(payload["kappa"], dtype=float),
            h=np.asarray(payload["h"], dtype=float),
            sigma=np.asarray(payload["sigma"], dtype=float),
            ln_p=None,
            weights=np.asarray(payload["weights"], dtype=float),
            idata=None,
            sigma_amp_ratio=(
                None
                if payload.get("sigma_amp_ratio") is None
                else np.asarray(payload["sigma_amp_ratio"], dtype=float)
            ),
            sigma_amp_ratio_global=(
                None
                if payload.get("sigma_amp_ratio_global") is None
                else np.asarray(payload["sigma_amp_ratio_global"], dtype=float)
            ),
            sigma_amp_ratio_station=(
                None
                if payload.get("sigma_amp_ratio_station") is None
                else np.asarray(payload["sigma_amp_ratio_station"], dtype=float)
            ),
            sigma_amp_ratio_station_names=(
                None
                if payload.get("sigma_amp_ratio_station_names") is None
                else np.asarray(payload["sigma_amp_ratio_station_names"], dtype=object)
            ),
            tempering_stalled=bool(payload.get("tempering_stalled", False)),
        )

    @staticmethod
    def _summarize_process_status(status: str) -> str:
        text = " ".join(str(status).split())
        if not text:
            return "waiting"

        lower = text.lower()
        if lower.startswith("smc completed in"):
            match = re.search(r"completed in\s+([0-9.]+)s", text, re.IGNORECASE)
            if match is not None:
                return f"done {float(match.group(1)):.1f}s"
            return "done"

        if lower.startswith("smc tempering stalled"):
            return "stalled"

        if lower.startswith("initial particle log-likelihood stats"):
            return text.replace("Initial particle log-likelihood stats: ", "init ll ")

        if lower.startswith("init"):
            return text

        match = re.search(
            r"(?:stage\s*=\s*|stage\s+)(\d+)(?::)?\s+beta\s*=\s*([0-9.]+)",
            text,
            re.IGNORECASE,
        )
        if match is not None:
            stage = int(match.group(1))
            beta = float(match.group(2))
            beta_token = f"beta={beta:.4f}"
            ess_match = re.search(r"\bESS\s*=\s*([0-9.]+)", text)
            pess_match = re.search(r"\bpESS\s*=\s*([0-9.]+)", text)
            acc_match = re.search(r"\bacc\s*=\s*([0-9.]+)", text)
            ess_token = ""
            if ess_match is not None:
                ess_token = f" ESS={int(float(ess_match.group(1)))}"
            elif pess_match is not None:
                ess_token = f" pESS={int(float(pess_match.group(1)))}"
            acc_token = ""
            if acc_match is not None:
                acc_token = f" acc={float(acc_match.group(1)):.3f}"
            return f"S{stage} {beta_token}{ess_token}{acc_token}"

        return text

    @staticmethod
    def _truncate_process_status(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return "." * max_len
        return text[: max_len - 3] + "..."

    @classmethod
    def _format_process_status_line(
        cls,
        statuses: Dict[int, str],
        num_chains: int,
        max_columns: Optional[int] = None,
    ) -> str:
        if max_columns is None:
            max_columns = shutil.get_terminal_size(fallback=(120, 20)).columns

        lines = []
        for chain_idx in range(num_chains):
            status = cls._summarize_process_status(statuses.get(chain_idx, "waiting"))
            line = f"C{chain_idx + 1} {status}"
            lines.append(cls._truncate_process_status(line, max_columns))
        return "\n".join(lines)

    def _build_process_chain_payloads(
        self,
        event: DataDict,
        chain_seeds: List[int],
        location_samples: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for chain_idx, seed in enumerate(chain_seeds):
            payloads.append(
                {
                    "chain_idx": chain_idx,
                    "seed": seed,
                    "event": event,
                    "location_samples": location_samples,
                    "inversion_options": self.inversion_options,
                    "num_particles": self.num_particles,
                    "dc": self.dc,
                    "location_samples_n": self.location_samples_n,
                    "azimuth_error": self.azimuth_error,
                    "takeoff_error": self.takeoff_error,
                    "gamma_beta_prior": self.gamma_beta_prior,
                    "delta_beta_prior": self.delta_beta_prior,
                    "amp_ratio_sigma_prior": self.amp_ratio_sigma_prior,
                    "amp_ratio_noise_mode": self.amp_ratio_noise_mode,
                    "amp_ratio_station_log_scale": self.amp_ratio_station_log_scale,
                    "amp_ratio_smooth_length_deg": self.amp_ratio_smooth_length_deg,
                    "amp_ratio_smooth_k": self.amp_ratio_smooth_k,
                    "amp_ratio_smooth_strength": self.amp_ratio_smooth_strength,
                    "num_mcmc_steps": self.num_mcmc_steps,
                    "mcmc_kernel": self.mcmc_kernel,
                    "rmh_proposal_scale": self.rmh_proposal_scale,
                    "mechanism_steps": self.mechanism_steps,
                    "nuts_adapt_steps": self.nuts_adapt_steps,
                    "nuts_initial_step_size": self.nuts_initial_step_size,
                    "nuts_target_acceptance": self.nuts_target_acceptance,
                    "nuts_max_num_doublings": self.nuts_max_num_doublings,
                    "smc_target_ess_ratio": self.smc_target_ess_ratio,
                    "max_smc_iterations": self.max_smc_iterations,
                    "use_nuts_adaptation": self.use_nuts_adaptation,
                    "adapt_proposal": self.adapt_proposal,
                    "min_tempering_increment": self.min_tempering_increment,
                    "tempering_stall_patience": self.tempering_stall_patience,
                    "smc_method": self.smc_method,
                    "ps_target_ess": self.ps_target_ess,
                    "chain_device": self.chain_device,
                }
            )
        return payloads

    def _run_multi_chain_process(
        self, chain_payloads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        ctx = mp.get_context("spawn")
        max_workers = min(self.num_chains, self.chain_cores or self.num_chains)
        status_by_chain: Dict[int, str] = {
            payload["chain_idx"]: f"starting seed={payload['seed']}"
            for payload in chain_payloads
        }
        initial_block = self._format_process_status_line(
            status_by_chain, self.num_chains
        )
        print(initial_block, flush=True)
        printed_lines = initial_block.count("\n") + 1
        with ctx.Manager() as manager:
            progress_queue = manager.Queue()
            with ctx.Pool(processes=max_workers) as pool:
                async_results = []
                for payload in chain_payloads:
                    worker_payload = dict(payload)
                    worker_payload["progress_queue"] = progress_queue
                    async_results.append(
                        (
                            payload["chain_idx"],
                            pool.apply_async(
                                _blackjax_process_chain_worker, (worker_payload,)
                            ),
                        )
                    )

                ordered: List[Optional[Dict[str, Any]]] = [None] * len(async_results)

                def _print_progress_block(block: str, final: bool = False) -> None:
                    nonlocal printed_lines
                    if sys.stdout.isatty() and printed_lines:
                        print(f"\033[{printed_lines}F\033[J", end="")
                    print(block, flush=True)
                    printed_lines = block.count("\n") + 1
                    if final:
                        printed_lines = 0

                def _drain_progress_queue() -> None:
                    updated = False
                    while True:
                        try:
                            msg = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        status_by_chain[int(msg["chain_idx"])] = str(msg["text"])
                        updated = True
                    if updated:
                        block = self._format_process_status_line(
                            status_by_chain, self.num_chains
                        )
                        _print_progress_block(block)

                while True:
                    _drain_progress_queue()
                    if all(async_result.ready() for _, async_result in async_results):
                        break
                    time.sleep(0.05)

                _drain_progress_queue()
                if status_by_chain:
                    block = self._format_process_status_line(
                        status_by_chain, self.num_chains
                    )
                    _print_progress_block(block, final=True)

                for chain_idx, async_result in async_results:
                    try:
                        ordered[chain_idx] = async_result.get()
                    except Exception as exc:
                        raise RuntimeError(f"chain {chain_idx} failed: {exc}") from exc
        return [payload for payload in ordered if payload is not None]

    def _invert_multi_chain(self, event: DataDict) -> InversionResult:
        """Run multiple independent SMC chains and stack results."""
        ss = np.random.SeedSequence(self.random_seed)
        chain_seeds = [int(s) for s in ss.generate_state(self.num_chains)]

        # Use the same station-angle realization for every chain so R-hat
        # compares chains targeting the same posterior.
        location_samples = build_location_samples_from_errors(
            event,
            rng=self.rng,
            n_samples=self.location_samples_n,
            azimuth_error=self.azimuth_error,
            takeoff_error=self.takeoff_error,
        )

        chain_payloads = self._build_process_chain_payloads(
            event, chain_seeds, location_samples
        )

        chain_results: List[InversionResult] = []
        original_seed = self.random_seed
        original_rng = self.rng

        if self.chain_execution == "process":
            serialized_results = self._run_multi_chain_process(chain_payloads)
            chain_results = [
                self._deserialize_chain_result(payload)
                for payload in serialized_results
            ]
        else:
            for chain_idx, seed in enumerate(chain_seeds):
                print(
                    f"\n{'=' * 40} Chain {chain_idx + 1}/{self.num_chains} "
                    f"(seed={seed}) {'=' * 40}"
                )
                self.random_seed = seed
                self.rng = np.random.default_rng(seed)
                chain_results.append(
                    self._invert_single_event(event, location_samples=location_samples)
                )

        self.random_seed = original_seed
        self.rng = original_rng

        return self._stack_chain_results(chain_results)

    def _stack_chain_results(
        self, chain_results: List[InversionResult]
    ) -> InversionResult:
        """Stack per-chain results into one chain-major :class:`InversionResult`.

        Extracted verbatim from the tail of :meth:`_invert_multi_chain` so the
        batched path (``inversion_blackjax_batched``) produces exactly the same
        structure: ``mt6`` ``(chains, 6, n_samples)``, ``gamma`` ...
        ``(chains, n_samples)``, ``num_chains`` set.  For persistent sampling
        chains may have different sample counts, so shorter chains are padded
        with zero-weight repeats first.
        """
        if self.smc_method == "adaptive_persistent":
            # Persistent sampling: chains may have different sample counts.
            # Pad shorter chains to match the longest, using zero weights
            # for padded samples so ArviZ can compute R-hat across chains.
            max_n = max(r.gamma.size for r in chain_results)

            def _pad_1d(arr, target_len):
                if arr.size == target_len:
                    return arr
                pad_val = arr[-1]  # repeat last sample (weight=0)
                return np.concatenate([arr, np.full(target_len - arr.size, pad_val)])

            def _pad_mt6(mt6, target_len):
                if mt6.shape[1] == target_len:
                    return mt6
                pad = np.tile(mt6[:, -1:], (1, target_len - mt6.shape[1]))
                return np.concatenate([mt6, pad], axis=1)

            def _pad_weights(w, target_len):
                if w.size == target_len:
                    return w
                return np.concatenate([w, np.zeros(target_len - w.size)])

            def _pad_sample_axis(arr, target_len):
                if arr.shape[0] == target_len:
                    return arr
                pad = np.repeat(arr[-1:, ...], target_len - arr.shape[0], axis=0)
                return np.concatenate([arr, pad], axis=0)

            padded = []
            for r in chain_results:
                padded.append(
                    InversionResult(
                        mt6=_pad_mt6(r.mt6, max_n),
                        gamma=_pad_1d(r.gamma, max_n),
                        delta=_pad_1d(r.delta, max_n),
                        kappa=_pad_1d(r.kappa, max_n),
                        h=_pad_1d(r.h, max_n),
                        sigma=_pad_1d(r.sigma, max_n),
                        ln_p=None,
                        weights=_pad_weights(r.weights, max_n),
                        idata=None,
                        sigma_amp_ratio=(
                            _pad_1d(r.sigma_amp_ratio, max_n)
                            if r.sigma_amp_ratio is not None
                            else None
                        ),
                        sigma_amp_ratio_global=(
                            _pad_1d(r.sigma_amp_ratio_global, max_n)
                            if r.sigma_amp_ratio_global is not None
                            else None
                        ),
                        sigma_amp_ratio_station=(
                            _pad_sample_axis(r.sigma_amp_ratio_station, max_n)
                            if r.sigma_amp_ratio_station is not None
                            else None
                        ),
                        sigma_amp_ratio_station_names=r.sigma_amp_ratio_station_names,
                        tempering_stalled=r.tempering_stalled,
                    )
                )
            chain_results = padded

        return InversionResult(
            mt6=np.stack([r.mt6 for r in chain_results], axis=0),
            gamma=np.stack([r.gamma for r in chain_results], axis=0),
            delta=np.stack([r.delta for r in chain_results], axis=0),
            kappa=np.stack([r.kappa for r in chain_results], axis=0),
            h=np.stack([r.h for r in chain_results], axis=0),
            sigma=np.stack([r.sigma for r in chain_results], axis=0),
            ln_p=None,
            weights=np.stack([r.weights for r in chain_results], axis=0),
            idata=None,
            sigma_amp_ratio=(
                np.stack([r.sigma_amp_ratio for r in chain_results], axis=0)
                if chain_results[0].sigma_amp_ratio is not None
                else None
            ),
            sigma_amp_ratio_global=(
                np.stack([r.sigma_amp_ratio_global for r in chain_results], axis=0)
                if chain_results[0].sigma_amp_ratio_global is not None
                else None
            ),
            sigma_amp_ratio_station=(
                np.stack([r.sigma_amp_ratio_station for r in chain_results], axis=0)
                if chain_results[0].sigma_amp_ratio_station is not None
                else None
            ),
            sigma_amp_ratio_station_names=chain_results[0].sigma_amp_ratio_station_names,
            num_chains=self.num_chains,
            tempering_stalled=any(bool(r.tempering_stalled) for r in chain_results),
        )

    @staticmethod
    def _build_amp_ratio_station_context(
        metadata: Dict[str, np.ndarray],
        smooth_k: int,
        smooth_length_deg: float,
    ) -> Dict[str, np.ndarray]:
        station_names_raw = [
            str(name)
            for name in np.asarray(metadata["station_names"], dtype=object).reshape(-1)
        ]
        station_az_raw = np.asarray(metadata["azimuth"], dtype=float).reshape(-1)
        station_to_raw = np.asarray(metadata["takeoff"], dtype=float).reshape(-1)

        n_obs = len(station_names_raw)
        if n_obs == 0:
            raise ValueError("station_smooth amplitude-ratio noise requires observations.")
        if station_az_raw.shape[0] != n_obs or station_to_raw.shape[0] != n_obs:
            raise ValueError(
                "Amplitude-ratio station metadata must align with observations."
            )

        station_to_index: Dict[str, int] = {}
        station_names: List[str] = []
        station_azimuth: List[float] = []
        station_takeoff: List[float] = []
        ratio_station_index = np.empty(n_obs, dtype=np.int32)

        for obs_idx, name in enumerate(station_names_raw):
            station_idx = station_to_index.get(name)
            if station_idx is None:
                station_idx = len(station_names)
                station_to_index[name] = station_idx
                station_names.append(name)
                station_azimuth.append(float(station_az_raw[obs_idx]))
                station_takeoff.append(float(station_to_raw[obs_idx]))
            ratio_station_index[obs_idx] = station_idx

        station_azimuth_arr = np.asarray(station_azimuth, dtype=float)
        station_takeoff_arr = np.asarray(station_takeoff, dtype=float)
        n_station = station_azimuth_arr.shape[0]

        edge_i: List[int] = []
        edge_j: List[int] = []
        edge_w: List[float] = []
        k_eff = min(int(smooth_k), max(n_station - 1, 0))
        if n_station > 1 and k_eff > 0:
            az_rad = np.deg2rad(station_azimuth_arr)
            takeoff_rad = np.deg2rad(station_takeoff_arr)
            vectors = np.column_stack(
                [
                    np.sin(takeoff_rad) * np.cos(az_rad),
                    np.sin(takeoff_rad) * np.sin(az_rad),
                    np.cos(takeoff_rad),
                ]
            )
            dot = np.clip(vectors @ vectors.T, -1.0, 1.0)
            distances = np.arccos(dot)
            np.fill_diagonal(distances, np.inf)

            length_rad = np.deg2rad(float(smooth_length_deg))
            for i in range(n_station):
                for j in np.argsort(distances[i])[:k_eff]:
                    distance = distances[i, j]
                    if not np.isfinite(distance):
                        continue
                    edge_i.append(i)
                    edge_j.append(int(j))
                    edge_w.append(float(np.exp(-(distance**2) / (2.0 * length_rad**2))))

        return {
            "station_names": np.asarray(station_names, dtype=object),
            "ratio_station_index": ratio_station_index,
            "station_azimuth": station_azimuth_arr,
            "station_takeoff": station_takeoff_arr,
            "edge_i": np.asarray(edge_i, dtype=np.int32),
            "edge_j": np.asarray(edge_j, dtype=np.int32),
            "edge_w": np.asarray(edge_w, dtype=float),
        }

    def _prepare_event_arrays(
        self,
        event: DataDict,
        location_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the numpy data arrays / flags one event's likelihood needs.

        Pure extraction of the data-preparation block that used to live inline
        at the top of :meth:`_invert_single_event` — same computations, same
        order, same dtypes.  Kept side-effect free (apart from consuming
        ``self.rng`` when ``location_samples`` is None, exactly as before) so
        the batched GPU path can reuse it per event.

        Returns a dict with keys:
          ``has_pol``, ``has_ar``, ``station_smooth_ar`` (flags),
          ``location_samples``,
          ``a_pol`` (N_pol, N_loc, 6), ``error_pol`` (N_pol,), ``pol_obs``
          (N_pol,), ``incorrect_prob`` (float or (N_pol,)),
          ``a1_ar``/``a2_ar`` (N_ar, N_loc, 6), ``amp_ratio_obs`` (N_ar,),
          ``log_ratio_sigma`` (N_ar,), ``ar_station_context``.
        ``pol_obs`` is not used by the JAX likelihood (the sign is already
        folded into ``a_pol``) but is returned for completeness/diagnostics.
        """
        has_pol = any(
            "polarity" in key.lower() and "prob" not in key.lower()
            for key in event.keys()
        )
        has_ar = any(
            "amplituderatio" in key.lower() or "amplitude_ratio" in key.lower()
            for key in event.keys()
        )
        station_smooth_ar = has_ar and self.amp_ratio_noise_mode == "station_smooth"

        if not (has_pol or has_ar):
            raise ValueError("No supported data types found in event.")

        # Build location samples for station angle uncertainty
        if location_samples is None:
            location_samples = build_location_samples_from_errors(
                event,
                rng=self.rng,
                n_samples=self.location_samples_n,
                azimuth_error=self.azimuth_error,
                takeoff_error=self.takeoff_error,
            )

        # Prepare polarity data
        if has_pol:
            a_pol_arr, error_polarity, incorrect_polarity_prob = polarity_matrix(
                event, location_samples=location_samples
            )
            a_pol_np = np.asarray(a_pol_arr, dtype=np.float64)
            error_pol_np = np.asarray(error_polarity, dtype=np.float64).reshape(-1)

            # Extract observed polarities
            pol_obs = self._extract_polarity_observations(event)

            if isinstance(incorrect_polarity_prob, (float, int)):
                incorrect_prob = float(incorrect_polarity_prob)
            else:
                incorrect_prob = np.asarray(
                    incorrect_polarity_prob, dtype=np.float64
                ).flatten()
        else:
            a_pol_np = None
            error_pol_np = None
            pol_obs = None
            incorrect_prob = 0.0

        # Prepare amplitude ratio data
        if has_ar:
            if station_smooth_ar:
                (
                    a1_ar,
                    a2_ar,
                    amplitude_ratio,
                    perc_err1,
                    perc_err2,
                    ar_metadata,
                ) = amplitude_ratio_matrix(
                    event,
                    location_samples=location_samples,
                    return_metadata=True,
                )
                ar_station_context = self._build_amp_ratio_station_context(
                    ar_metadata,
                    smooth_k=self.amp_ratio_smooth_k,
                    smooth_length_deg=self.amp_ratio_smooth_length_deg,
                )
            else:
                (
                    a1_ar,
                    a2_ar,
                    amplitude_ratio,
                    perc_err1,
                    perc_err2,
                ) = amplitude_ratio_matrix(event, location_samples=location_samples)
                ar_station_context = None

            a1_ar_np = np.asarray(a1_ar, dtype=np.float64)
            a2_ar_np = np.asarray(a2_ar, dtype=np.float64)
            amp_ratio_obs = np.asarray(amplitude_ratio, dtype=np.float64).reshape(-1)
            perc_err1_np = np.asarray(perc_err1, dtype=np.float64).reshape(-1)
            perc_err2_np = np.asarray(perc_err2, dtype=np.float64).reshape(-1)
            log_ratio_sigma = np.sqrt(perc_err1_np**2 + perc_err2_np**2)
        else:
            a1_ar_np = a2_ar_np = None
            amp_ratio_obs = None
            log_ratio_sigma = None
            ar_station_context = None

        return {
            "has_pol": has_pol,
            "has_ar": has_ar,
            "station_smooth_ar": station_smooth_ar,
            "location_samples": location_samples,
            "a_pol": a_pol_np,
            "error_pol": error_pol_np,
            "pol_obs": pol_obs,
            "incorrect_prob": incorrect_prob,
            "a1_ar": a1_ar_np,
            "a2_ar": a2_ar_np,
            "amp_ratio_obs": amp_ratio_obs,
            "log_ratio_sigma": log_ratio_sigma,
            "ar_station_context": ar_station_context,
        }

    def _invert_single_event(
        self,
        event: DataDict,
        location_samples: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Any] = None,
    ) -> InversionResult:
        """Run SMC inversion for a single event."""
        start_time = time.time()

        def _emit_progress(message: str) -> None:
            if progress_callback is None:
                print(message)
            else:
                progress_callback(message)

        # ``jax_enable_x64`` is process-global: a process that disabled it to run
        # the batched sampler in float32 (see ``inversion_blackjax_batched``'s
        # module docstring) would silently run THIS float64 sampler in single
        # precision, and the result is cast back to float64 by numpy so nothing
        # downstream would reveal it.  Warn loudly rather than fail, since the
        # unbatched path is also the documented fp32 fallback.
        if not jax.config.jax_enable_x64:
            warnings.warn(
                "_invert_single_event is running with jax_enable_x64=False: the "
                "production float64 sampler will be evaluated in float32. Run "
                "the batched fp32 path in a separate process if float64 "
                "fallback results are required.",
                RuntimeWarning,
                stacklevel=2,
            )
            _emit_progress(
                "WARNING: jax_enable_x64 is False -- this inversion runs in float32."
            )

        _emit_progress("Preparing data and likelihood matrices...")

        # --- Prepare Data ---
        prepared = self._prepare_event_arrays(
            event, location_samples=location_samples
        )
        has_pol = prepared["has_pol"]
        has_ar = prepared["has_ar"]
        station_smooth_ar = prepared["station_smooth_ar"]
        location_samples = prepared["location_samples"]
        a_pol_np = prepared["a_pol"]
        error_pol_np = prepared["error_pol"]
        pol_obs = prepared["pol_obs"]
        incorrect_prob = prepared["incorrect_prob"]
        a1_ar_np = prepared["a1_ar"]
        a2_ar_np = prepared["a2_ar"]
        amp_ratio_obs = prepared["amp_ratio_obs"]
        log_ratio_sigma = prepared["log_ratio_sigma"]
        ar_station_context = prepared["ar_station_context"]

        # --- Build JAX Log-Density Functions ---

        # Convert data to JAX arrays
        if has_pol:
            a_pol_jax = jnp.asarray(a_pol_np)
            error_pol_jax = jnp.asarray(error_pol_np)
            incorrect_prob_jax = (
                jnp.asarray(incorrect_prob)
                if isinstance(incorrect_prob, np.ndarray)
                else incorrect_prob
            )

        if has_ar:
            a1_ar_jax = jnp.asarray(a1_ar_np)
            a2_ar_jax = jnp.asarray(a2_ar_np)
            amp_ratio_obs_jax = jnp.asarray(amp_ratio_obs)
            log_ratio_sigma_jax = jnp.asarray(log_ratio_sigma)
            if station_smooth_ar:
                ratio_station_index_jax = jnp.asarray(
                    ar_station_context["ratio_station_index"]
                )
                edge_i_jax = jnp.asarray(ar_station_context["edge_i"])
                edge_j_jax = jnp.asarray(ar_station_context["edge_j"])
                edge_w_jax = jnp.asarray(ar_station_context["edge_w"])
                n_amp_ratio_stations = int(
                    ar_station_context["station_names"].shape[0]
                )
                has_smooth_edges = bool(ar_station_context["edge_i"].size > 0)
            else:
                ratio_station_index_jax = None
                edge_i_jax = edge_j_jax = edge_w_jax = None
                n_amp_ratio_stations = 0
                has_smooth_edges = False
        else:
            ratio_station_index_jax = None
            edge_i_jax = edge_j_jax = edge_w_jax = None
            n_amp_ratio_stations = 0
            has_smooth_edges = False

        dc = self.dc
        amp_ratio_sigma_prior = self.amp_ratio_sigma_prior
        amp_ratio_station_log_scale = self.amp_ratio_station_log_scale
        amp_ratio_smooth_strength = self.amp_ratio_smooth_strength
        amp_ratio_noise_fields = (
            ("sigma_amp_ratio_global", "sigma_amp_ratio_station_log_offset")
            if station_smooth_ar
            else ("sigma_amp_ratio",)
        )
        eps = 1e-6

        def _logit(p):
            p = jnp.clip(p, eps, 1.0 - eps)
            return jnp.log(p) - jnp.log1p(-p)

        def _softplus_inv(x):
            # Stable, overflow-free; see module-level ``softplus_inv``.
            return softplus_inv(x, eps)

        def _unconstrained_to_params(position):
            params = {}

            if dc:
                # Fix gamma/delta in DC case
                ref = position["kappa"]
                params["gamma"] = jnp.zeros_like(ref)
                params["delta"] = jnp.zeros_like(ref)
            else:
                ug = position["gamma"]
                ud = position["delta"]
                g_raw = jnn.sigmoid(ug)
                d_raw = jnn.sigmoid(ud)
                params["gamma"] = -jnp.pi / 6.0 + g_raw * (jnp.pi / 3.0)
                params["delta"] = -jnp.pi / 2.0 + d_raw * (jnp.pi)

            uk = position["kappa"]
            uh = position["h"]
            us = position["sigma"]

            k_raw = jnn.sigmoid(uk)
            h_raw = jnn.sigmoid(uh)
            s_raw = jnn.sigmoid(us)

            params["kappa"] = k_raw * (2.0 * jnp.pi)
            params["h"] = h_raw
            params["sigma"] = -jnp.pi / 2.0 + s_raw * (jnp.pi)

            if station_smooth_ar:
                u_ar = position["sigma_amp_ratio_global"]
                sigma_global = jnn.softplus(u_ar) + eps
                z_station = position["sigma_amp_ratio_station_log_offset"]
                sigma_station = sigma_global[..., None] * jnp.exp(z_station)
                params["sigma_amp_ratio"] = sigma_global
                params["sigma_amp_ratio_global"] = sigma_global
                params["sigma_amp_ratio_station_log_offset"] = z_station
                params["sigma_amp_ratio_station"] = sigma_station
            elif has_ar:
                u_ar = position["sigma_amp_ratio"]
                sigma_ar = jnn.softplus(u_ar) + eps
                params["sigma_amp_ratio"] = sigma_ar
                params["sigma_amp_ratio_global"] = sigma_ar

            return params

        def logprior_fn(position):
            """Log-prior in unconstrained space (includes Jacobians)."""
            log_p = 0.0

            if not dc:
                ug = position["gamma"]
                ud = position["delta"]

                g_raw = jnn.sigmoid(ug)
                d_raw = jnn.sigmoid(ud)

                # Beta priors on raw variables
                g_a, g_b = self.gamma_beta_prior
                d_a, d_b = self.delta_beta_prior

                log_beta_g = (
                    (g_a - 1.0) * jnp.log(g_raw + eps)
                    + (g_b - 1.0) * jnp.log1p(-g_raw + eps)
                    - jsp.special.betaln(g_a, g_b)
                )
                log_beta_d = (
                    (d_a - 1.0) * jnp.log(d_raw + eps)
                    + (d_b - 1.0) * jnp.log1p(-d_raw + eps)
                    - jsp.special.betaln(d_a, d_b)
                )

                # Jacobians for transforms (range constants dropped)
                log_jac_g = jnp.log(g_raw + eps) + jnp.log1p(-g_raw + eps)
                log_jac_d = jnp.log(d_raw + eps) + jnp.log1p(-d_raw + eps)

                log_p += log_beta_g + log_jac_g
                log_p += log_beta_d + log_jac_d

            # Uniform priors with Jacobians for kappa, h, sigma
            uk = position["kappa"]
            uh = position["h"]
            us = position["sigma"]

            k_raw = jnn.sigmoid(uk)
            h_raw = jnn.sigmoid(uh)
            s_raw = jnn.sigmoid(us)

            # kappa ~ Uniform(0, 2π) in physical space.
            # We parameterize via k_raw = sigmoid(uk) and kappa = 2π * k_raw.
            log_unif_kappa = -jnp.log(2.0 * jnp.pi)
            log_jac_kappa = (
                jnp.log(2.0 * jnp.pi) + jnp.log(k_raw + eps) + jnp.log1p(-k_raw + eps)
            )
            log_p += log_unif_kappa + log_jac_kappa
            # h ~ Uniform(0, 1)
            log_p += jnp.log(h_raw + eps) + jnp.log1p(-h_raw + eps)
            # sigma ~ Uniform(-π/2, π/2) in physical space (range π).
            # We parameterize via s_raw = sigmoid(us) and sigma = -π/2 + π*s_raw.
            log_unif_sigma = -jnp.log(jnp.pi)
            log_jac_sigma = (
                jnp.log(jnp.pi) + jnp.log(s_raw + eps) + jnp.log1p(-s_raw + eps)
            )
            log_p += log_unif_sigma + log_jac_sigma

            # Prior for sigma_amp_ratio (HalfCauchy + Jacobian of softplus)
            if station_smooth_ar:
                u_ar = position["sigma_amp_ratio_global"]
                sigma_ar = jnn.softplus(u_ar) + eps
                beta_hc = amp_ratio_sigma_prior
                log_p_hc = (
                    jnp.log(2.0)
                    - jnp.log(jnp.pi * beta_hc)
                    - jnp.log(1.0 + (sigma_ar / beta_hc) ** 2)
                )
                log_p += log_p_hc + jnn.log_sigmoid(u_ar)

                z_station = position["sigma_amp_ratio_station_log_offset"]
                z_scaled = z_station / amp_ratio_station_log_scale
                log_p += -0.5 * jnp.mean(z_scaled**2)
                if has_smooth_edges:
                    edge_diff = z_station[edge_i_jax] - z_station[edge_j_jax]
                    log_p += (
                        -0.5
                        * amp_ratio_smooth_strength
                        * jnp.mean(edge_w_jax * (edge_diff / amp_ratio_station_log_scale) ** 2)
                    )
            elif has_ar:
                u_ar = position["sigma_amp_ratio"]
                sigma_ar = jnn.softplus(u_ar) + eps
                beta_hc = amp_ratio_sigma_prior
                log_p_hc = (
                    jnp.log(2.0)
                    - jnp.log(jnp.pi * beta_hc)
                    - jnp.log(1.0 + (sigma_ar / beta_hc) ** 2)
                )
                log_p += log_p_hc + jnn.log_sigmoid(u_ar)

            log_p = jnp.clip(log_p, -1.0e3, 1.0e3)
            return jnp.nan_to_num(log_p, nan=-1.0e3, posinf=1.0e3, neginf=-1.0e3)

        def loglikelihood_fn(position):
            """Combined log-likelihood for polarity and amplitude ratio."""
            params = _unconstrained_to_params(position)
            gamma = params["gamma"]
            delta = params["delta"]
            kappa = params["kappa"]
            h = params["h"]
            sigma = params["sigma"]

            # Compute MT6 from Tape parameters
            mt6 = jax_Tape_MT6(gamma, delta, kappa, h, sigma)

            log_lik = 0.0

            # Polarity likelihood
            if has_pol:
                # a_pol_jax: (N_obs, N_loc_samp, 6)
                # mt6: (6,)
                # X_pol: (N_obs, N_loc_samp)
                X_pol = jnp.tensordot(a_pol_jax, mt6, axes=[[2], [0]])

                # Polarity probability (probit model)
                sigma_pol = jnp.maximum(error_pol_jax, _SMALL_NUMBER)
                sigma_pol_bc = sigma_pol[:, None]  # (N_obs, 1)

                base_prob = 0.5 * (
                    1.0 + jax.scipy.special.erf(X_pol / (jnp.sqrt(2.0) * sigma_pol_bc))
                )

                # Account for incorrect polarity probability
                if isinstance(incorrect_prob_jax, float):
                    p_sample = (
                        incorrect_prob_jax
                        + (1.0 - 2.0 * incorrect_prob_jax) * base_prob
                    )
                else:
                    inc_bc = incorrect_prob_jax[:, None]
                    p_sample = inc_bc + (1.0 - 2.0 * inc_bc) * base_prob

                # Marginalize over location samples
                p_avg = jnp.mean(p_sample, axis=1)
                log_lik += jnp.sum(jnp.log(p_avg + _SMALL_NUMBER))

            # Amplitude ratio likelihood
            if has_ar:
                # a1_ar_jax: (N_obs, N_loc_samp, 6)
                amp1_pred = jnp.abs(jnp.tensordot(a1_ar_jax, mt6, axes=[[2], [0]]))
                amp2_pred = jnp.abs(jnp.tensordot(a2_ar_jax, mt6, axes=[[2], [0]]))

                # Predicted log-ratio
                log_ratio_pred = jnp.log(amp1_pred + _SMALL_NUMBER) - jnp.log(
                    amp2_pred + _SMALL_NUMBER
                )

                # Observed log-ratio
                log_ratio_obs = jnp.log(jnp.maximum(amp_ratio_obs_jax, _SMALL_NUMBER))[
                    :, None
                ]

                # Log-normal likelihood
                # Combine measurement variance with sigma_amp_ratio variance
                if station_smooth_ar:
                    sigma_ar = params["sigma_amp_ratio_station"][ratio_station_index_jax]
                else:
                    sigma_ar = params["sigma_amp_ratio"]
                sigma_combined = jnp.sqrt(log_ratio_sigma_jax**2 + sigma_ar**2)
                sigma_bc = sigma_combined[:, None]

                resid = log_ratio_obs - log_ratio_pred
                # Full log-normal PDF: includes -log(x) Jacobian term
                # (matches PyTensor version in inversion_pytensor.py)
                log_prob_samples = (
                    -0.5 * (resid / sigma_bc) ** 2
                    - jnp.log(sigma_bc)
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - jnp.log(amp_ratio_obs_jax[:, None])
                )

                # Marginalize
                n_samp = a1_ar_jax.shape[1]
                log_prob_marginal = jax.scipy.special.logsumexp(
                    log_prob_samples, axis=1
                ) - jnp.log(n_samp)
                log_lik += jnp.sum(log_prob_marginal)

            log_lik = jnp.clip(log_lik, -1.0e3, 1.0e3)
            return jnp.nan_to_num(log_lik, nan=-1.0e3, posinf=1.0e3, neginf=-1.0e3)

        # --- Initialize Particles (unconstrained space) ---
        _emit_progress("Initializing particles and evaluating initial likelihood...")
        key = random.PRNGKey(self.random_seed)

        def init_particle(key):
            n_keys = 5 if (not dc) else 3
            if station_smooth_ar:
                n_keys += 2
            elif has_ar:
                n_keys += 1
            keys = random.split(key, n_keys)
            idx = 0

            params_u = {}

            if not dc:
                g_a, g_b = self.gamma_beta_prior
                d_a, d_b = self.delta_beta_prior
                g_raw = random.beta(keys[idx], g_a, g_b)
                idx += 1
                d_raw = random.beta(keys[idx], d_a, d_b)
                idx += 1
                params_u["gamma"] = _logit(g_raw)
                params_u["delta"] = _logit(d_raw)

            k_raw = random.uniform(keys[idx], minval=eps, maxval=1.0 - eps)
            idx += 1
            h_raw = random.uniform(keys[idx], minval=eps, maxval=1.0 - eps)
            idx += 1
            s_raw = random.uniform(keys[idx], minval=eps, maxval=1.0 - eps)
            idx += 1

            params_u["kappa"] = _logit(k_raw)
            params_u["h"] = _logit(h_raw)
            params_u["sigma"] = _logit(s_raw)

            if station_smooth_ar:
                u = random.uniform(keys[idx], minval=eps, maxval=1.0 - eps)
                idx += 1
                sigma_ar = amp_ratio_sigma_prior * jnp.tan(jnp.pi * u / 2.0)
                sigma_ar = jnp.clip(sigma_ar, 1e-6, 100.0)
                params_u["sigma_amp_ratio_global"] = _softplus_inv(sigma_ar)
                params_u["sigma_amp_ratio_station_log_offset"] = (
                    random.normal(
                        keys[idx],
                        shape=(n_amp_ratio_stations,),
                        dtype=jnp.float64,
                    )
                    * amp_ratio_station_log_scale
                    * 0.1
                )
            elif has_ar:
                u = random.uniform(keys[idx], minval=eps, maxval=1.0 - eps)
                sigma_ar = amp_ratio_sigma_prior * jnp.tan(jnp.pi * u / 2.0)
                sigma_ar = jnp.clip(sigma_ar, 1e-6, 100.0)
                params_u["sigma_amp_ratio"] = _softplus_inv(sigma_ar)

            return params_u

        key, init_key = random.split(key)
        init_keys = random.split(init_key, self.num_particles)
        initial_particles = jax.vmap(init_particle)(init_keys)

        def logdensity_fn(position):
            val = logprior_fn(position) + loglikelihood_fn(position)
            val = jnp.clip(val, -1.0e3, 1.0e3)
            return jnp.nan_to_num(val, nan=-1.0e3, posinf=1.0e3, neginf=-1.0e3)

        init_ll = np.asarray(jax.vmap(loglikelihood_fn)(initial_particles))
        finite_mask = np.isfinite(init_ll)
        finite_frac = float(np.mean(finite_mask)) if init_ll.size > 0 else 0.0
        ll_min = (
            float(np.min(init_ll[finite_mask])) if np.any(finite_mask) else float("nan")
        )
        ll_max = (
            float(np.max(init_ll[finite_mask])) if np.any(finite_mask) else float("nan")
        )
        _emit_progress(
            "Initial particle log-likelihood stats: "
            f"finite={finite_frac:.3f}, min={ll_min:.3f}, max={ll_max:.3f}"
        )

        use_blockwise = False  # Only enabled for RMH with adapt_proposal

        if self.mcmc_kernel == "nuts":
            _emit_progress("Adapting NUTS parameters (window adaptation)...")
            init_position = jax.tree.map(lambda x: x[0], initial_particles)
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                logdensity_fn,
                is_mass_matrix_diagonal=True,
                initial_step_size=self.nuts_initial_step_size,
                target_acceptance_rate=self.nuts_target_acceptance,
                max_num_doublings=self.nuts_max_num_doublings,
            )

            if self.use_nuts_adaptation:
                key, adapt_key = random.split(key)
                adapt_results, _ = warmup.run(
                    adapt_key, init_position, num_steps=self.nuts_adapt_steps
                )
                nuts_params = adapt_results.parameters
                step_size = nuts_params["step_size"]
                inverse_mass_matrix = nuts_params["inverse_mass_matrix"]
                max_num_doublings = int(
                    nuts_params.get("max_num_doublings", self.nuts_max_num_doublings)
                )
            else:
                from jax.flatten_util import ravel_pytree

                dim = ravel_pytree(init_position)[0].shape[0]
                step_size = jnp.array(self.nuts_initial_step_size)
                inverse_mass_matrix = jnp.ones((dim,))
                max_num_doublings = int(self.nuts_max_num_doublings)

            # NOTE: `max_num_doublings` must be a static Python int for JIT.
            # Do NOT pass it via `mcmc_parameters` (would become a tracer).
            mcmc_parameters = blackjax.smc.extend_params(
                {
                    "step_size": step_size,
                    "inverse_mass_matrix": inverse_mass_matrix,
                }
            )

            nuts_kernel = blackjax.nuts.build_kernel()

            def mcmc_step_fn(
                rng_key, state, logdensity_fn, step_size, inverse_mass_matrix
            ):
                return nuts_kernel(
                    rng_key,
                    state,
                    logdensity_fn,
                    step_size,
                    inverse_mass_matrix,
                    max_num_doublings=max_num_doublings,
                )

            mcmc_init_fn = blackjax.nuts.init
            _emit_progress("Using BlackJAX SMC rejuvenation kernel: NUTS")
        else:
            rmh_kernel = blackjax_rw.build_additive_step()
            mcmc_init_fn = blackjax.rmh.init
            use_blockwise = self.adapt_proposal

            if use_blockwise:
                # --- Metropolis-Within-Gibbs (MWG) kernel ---
                blocks = default_mwg_blocks(
                    dc=dc,
                    has_ar=has_ar,
                    noise_fields=amp_ratio_noise_fields,
                )
                field_names = list(initial_particles.keys())

                initial_stds_by_field = {
                    name: float(jnp.std(initial_particles[name]))
                    for name in field_names
                }

                inner_steps = default_inner_steps(mechanism_steps=self.mechanism_steps)

                mwg_step_fn = build_mwg_kernel(
                    blocks=blocks,
                    inner_steps_per_block=inner_steps,
                )
                mcmc_parameter_update_fn = build_mwg_parameter_update_fn(
                    blocks=blocks,
                    field_names=field_names,
                    initial_stds=initial_stds_by_field,
                )

                initial_scales = build_blockwise_scale_overrides(
                    blocks=blocks,
                    initial_stds=initial_stds_by_field,
                    current_stds=initial_stds_by_field,
                )
                initial_parameter_value = {
                    name: jnp.asarray(scale)[None]
                    for name, scale in initial_scales.items()
                }

                block_names_str = ", ".join(
                    f"{b.name}(x{inner_steps.get(b.name, 1)})" for b in blocks
                )
                _emit_progress(f"Using MWG kernel: blocks=[{block_names_str}]")
            else:
                # --- Standard isotropic RMH ---
                proposal_scale = jnp.array(self.rmh_proposal_scale, dtype=jnp.float64)
                mcmc_parameters = blackjax.smc.extend_params(
                    {"proposal_scale": proposal_scale}
                )

                def mcmc_step_fn(rng_key, state, logdensity_fn, proposal_scale):
                    random_step = blackjax_rw.normal(proposal_scale)
                    return rmh_kernel(rng_key, state, logdensity_fn, random_step)

                _emit_progress(
                    f"Using BlackJAX SMC rejuvenation kernel: RMH (proposal_scale={float(proposal_scale):.4f})"
                )

        # --- SMC Sampler Construction ---
        use_persistent = self.smc_method == "adaptive_persistent"
        target_ess = self.smc_target_ess_ratio

        if use_persistent:
            if use_blockwise:
                # Use MWG with fixed scales (no adaptation, but multiple inner steps)
                fixed_scales = {
                    name: jnp.asarray(scale) for name, scale in initial_scales.items()
                }

                def mcmc_step_fn_persistent(
                    rng_key, state, logdensity_fn, _unused_scale
                ):
                    return mwg_step_fn(rng_key, state, logdensity_fn, **fixed_scales)

                mcmc_parameters = blackjax.smc.extend_params(
                    {"_unused_scale": jnp.array(0.0)}
                )
                mcmc_step_fn = mcmc_step_fn_persistent
                _emit_progress(
                    "Note: persistent sampling uses MWG with fixed scales (no adaptation)"
                )

            smc = blackjax.adaptive_persistent_sampling_smc(
                logprior_fn,
                loglikelihood_fn,
                self.max_smc_iterations,
                mcmc_step_fn,
                mcmc_init_fn,
                mcmc_parameters,
                systematic,
                target_ess=self.ps_target_ess,
                num_mcmc_steps=self.num_mcmc_steps,
            )
        elif use_blockwise:
            smc = inner_kernel_tuning.as_top_level_api(
                smc_algorithm=blackjax.adaptive_tempered_smc,
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                mcmc_step_fn=mwg_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                resampling_fn=systematic,
                mcmc_parameter_update_fn=mcmc_parameter_update_fn,
                initial_parameter_value=initial_parameter_value,
                target_ess=target_ess,
                num_mcmc_steps=self.num_mcmc_steps,
            )
        else:
            smc = blackjax.adaptive_tempered_smc(
                logprior_fn=logprior_fn,
                loglikelihood_fn=loglikelihood_fn,
                mcmc_step_fn=mcmc_step_fn,
                mcmc_init_fn=mcmc_init_fn,
                mcmc_parameters=mcmc_parameters,
                resampling_fn=systematic,
                target_ess=target_ess,
                num_mcmc_steps=self.num_mcmc_steps,
            )

        state = smc.init(initial_particles)
        step = jax.jit(smc.step)

        def _get_smc_state(st):
            """Unwrap StateWithParameterOverride if needed."""
            return st.sampler_state if hasattr(st, "sampler_state") else st

        if use_persistent:
            from blackjax.smc.persistent_sampling import compute_persistent_ess

        method_name = (
            "adaptive persistent SMC" if use_persistent else "adaptive tempered SMC"
        )
        _emit_progress(
            f"Running BlackJAX {method_name} ({self.mcmc_kernel.upper()})..."
        )
        smc_st = _get_smc_state(state)
        prev_beta = float(smc_st.tempering_param)
        stall_count = 0
        tempering_stalled = False
        for iteration in range(self.max_smc_iterations):
            smc_st = _get_smc_state(state)
            if use_persistent:
                at_target = float(smc_st.tempering_param) >= 1.0 - 1e-6
                if at_target:
                    p_ess = float(
                        compute_persistent_ess(
                            jnp.log(smc_st.persistent_weights),
                            normalize_weights=True,
                        )
                    )
                    if p_ess >= self.ps_target_ess * self.num_particles:
                        break
            else:
                if float(smc_st.tempering_param) >= 1.0 - 1e-6:
                    break
            key, subkey = random.split(key)
            state, info = step(subkey, state)
            smc_st = _get_smc_state(state)
            curr_beta = float(smc_st.tempering_param)
            if not np.isfinite(curr_beta):
                raise RuntimeError(
                    "BlackJAX SMC produced non-finite tempering parameter (beta=NaN/Inf). "
                    "This indicates non-finite log-density values during sampling."
                )
            delta_beta = curr_beta - prev_beta
            prev_beta = curr_beta

            if use_persistent:
                p_ess = float(
                    compute_persistent_ess(
                        jnp.log(smc_st.persistent_weights),
                        normalize_weights=True,
                    )
                )
                ess_str = f"pESS={p_ess:.0f}"
            else:
                ess = float(1.0 / jnp.sum(smc_st.weights**2))
                ess_str = f"ESS={ess:.0f}/{self.num_particles}"

            acc_rate = None
            update_info = getattr(info, "update_info", None)
            if update_info is not None and hasattr(update_info, "acceptance_rate"):
                try:
                    acc_rate = float(np.asarray(update_info.acceptance_rate).mean())
                except Exception:
                    acc_rate = None
            if acc_rate is None and hasattr(info, "acceptance_rate"):
                try:
                    acc_rate = float(np.asarray(info.acceptance_rate).mean())
                except Exception:
                    acc_rate = None

            if acc_rate is None:
                _emit_progress(
                    f"  Stage {iteration + 1}: beta={curr_beta:.4f} (d={delta_beta:.5f}), {ess_str}"
                )
            else:
                _emit_progress(
                    f"  Stage {iteration + 1}: beta={curr_beta:.4f} (d={delta_beta:.5f}), "
                    f"{ess_str}, acc={acc_rate:.3f}"
                )

            if delta_beta < self.min_tempering_increment:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= self.tempering_stall_patience:
                tempering_stalled = True
                _emit_progress(
                    "SMC tempering stalled "
                    f"(delta_beta < {self.min_tempering_increment:.1e} for {stall_count} stages). "
                    "Stopping early."
                )
                break

        elapsed = time.time() - start_time
        smc_st = _get_smc_state(state)
        _emit_progress(
            f"SMC completed in {elapsed:.2f}s with beta={float(smc_st.tempering_param):.4f}"
        )

        # --- Extract Results ---
        if use_persistent:
            from blackjax.smc.persistent_sampling import remove_padding

            smc_st = remove_padding(smc_st)

            # persistent_particles: {field: (n_iter+1, n_particles)}
            # persistent_weights:   (n_iter+1, n_particles)
            # Flatten across iterations to get all accumulated samples
            flat_particles = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]), smc_st.persistent_particles
            )
            particles_phys = _unconstrained_to_params(flat_particles)

            raw_pw = np.asarray(smc_st.persistent_weights).reshape(-1)
            final_weights = raw_pw / raw_pw.sum()
        else:
            particles_phys = _unconstrained_to_params(smc_st.particles)
            final_weights = np.asarray(smc_st.weights)

        gamma_s = np.asarray(particles_phys["gamma"])
        delta_s = np.asarray(particles_phys["delta"])
        kappa_s = np.asarray(particles_phys["kappa"])
        h_s = np.asarray(particles_phys["h"])
        sigma_s = np.asarray(particles_phys["sigma"])
        n_samp = gamma_s.size

        from .tape import Tape_MT6

        if has_ar:
            sigma_ar_s = np.asarray(particles_phys["sigma_amp_ratio"])
            sigma_ar_global_s = np.asarray(particles_phys["sigma_amp_ratio_global"])
            if station_smooth_ar:
                sigma_ar_station_s = np.asarray(
                    particles_phys["sigma_amp_ratio_station"]
                )
                sigma_ar_station_names = np.asarray(
                    ar_station_context["station_names"], dtype=object
                )
            else:
                sigma_ar_station_s = None
                sigma_ar_station_names = None
        else:
            sigma_ar_s = None
            sigma_ar_global_s = None
            sigma_ar_station_s = None
            sigma_ar_station_names = None

        mt6_samples = Tape_MT6(gamma_s, delta_s, kappa_s, h_s, sigma_s)
        mt6_samples = np.asarray(mt6_samples, dtype=float)
        if mt6_samples.ndim == 1:
            mt6_samples = mt6_samples.reshape(6, n_samp)
        elif mt6_samples.shape != (6, n_samp):
            mt6_samples = mt6_samples.reshape(6, n_samp)

        return InversionResult(
            mt6=mt6_samples,
            gamma=gamma_s,
            delta=delta_s,
            kappa=kappa_s,
            h=h_s,
            sigma=sigma_s,
            ln_p=None,
            weights=final_weights,
            idata=None,
            sigma_amp_ratio=sigma_ar_s,
            sigma_amp_ratio_global=sigma_ar_global_s,
            sigma_amp_ratio_station=sigma_ar_station_s,
            sigma_amp_ratio_station_names=sigma_ar_station_names,
            tempering_stalled=tempering_stalled,
        )

    def _extract_polarity_observations(self, data: DataDict) -> np.ndarray:
        """Extract observed polarity signs (0=down, 1=up) from data dictionary."""
        polarities = []
        for key in sorted(
            k
            for k in data.keys()
            if "polarity" in k.lower() and "prob" not in k.lower()
        ):
            measured = np.asarray(data[key]["Measured"], dtype=float).flatten()
            pol_01 = ((measured + 1) / 2).astype(int)
            polarities.append(pol_01)

        if polarities:
            return np.concatenate(polarities)
        else:
            return np.array([], dtype=int)
