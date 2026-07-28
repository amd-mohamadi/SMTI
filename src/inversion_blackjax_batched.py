"""
Batched-event SMC for :class:`~src.inversion_blackjax.InversionBlackJAX`.

Implements sections 3.2-3.6 of ``202607_gpu_batched_smc_plan.md`` (milestone 1):
one XLA program runs ``B = (events in bucket) x (chains)`` independent
adaptive-tempered SMC samplers under ``jax.vmap``.  Each batch entry keeps its
own data, its own RNG seed and its own tempering schedule; the host loop only
reads back ``(B,)`` vectors of lambda / ESS / acceptance per stage.

Public API
----------
``prepare_batch(inv, events)``
    Build the padded, masked, stacked :class:`Bucket` list.
``run_batched(inv, events, smc_dtype=...)``
    Sample every bucket and return ``{event_id: InversionResult}`` with exactly
    the structure the unbatched ``forward()`` produces (chain-stacked when
    ``inv.num_chains > 1``), so ``4_run_inversion`` post-processing is unchanged.

Scope (v1, matching what production actually uses)
--------------------------------------------------
``smc_method='adaptive_tempered'``, ``mcmc_kernel`` in
``{'rmh', 'irmh', 'mwg_irmh'}`` with ``adapt_proposal=True`` (MWG and/or
GMM-IRMH via ``inner_kernel_tuning``), ``amp_ratio_noise_mode='global'``,
``dc=False``.  NUTS and other configs raise :class:`NotImplementedError`;
callers should fall back to the unbatched path.

Precision
---------
``smc_dtype='float32'`` casts every data array and every initial particle to
float32 and uses :mod:`src.tape_jax_dtype` (also used for float64, where it is
bit-identical to ``src.tape_jax``).  For the float32 path to stay in float32
end-to-end, the *process* must have x64 disabled before any array is created::

    import jax
    from src.inversion_blackjax import InversionBlackJAX        # forces x64 on
    from src.inversion_blackjax_batched import run_batched
    jax.config.update("jax_enable_x64", False)                  # ... turn it off
    # ... only now build arrays / the sampler

(``src.tape_jax`` sets ``jax_enable_x64=True`` at import; importing
``InversionBlackJAX`` at all therefore flips it on, which is why the flag has to
be reset afterwards rather than simply "not importing tape_jax".)

.. warning::

   ``jax_enable_x64`` is a **process-global** switch.  A process that turns it
   off to run the batched sampler in float32 also silently demotes the
   *unbatched* production sampler (``_invert_single_event``, the documented
   fallback for unsupported configurations) to single precision --
   ``src/tape_jax.py``'s hardcoded ``dtype=jnp.float64`` literals are downcast
   with only a ``UserWarning``, and the resulting ``InversionResult`` is cast
   back to float64 by numpy so nothing downstream reveals it.
   ``_invert_single_event`` now emits a warning in that situation, but the
   robust arrangement for M2 is to run the fp32 batched sampler in a
   **dedicated subprocess** (as ``tests_batched/test_batched_core.py`` does)
   and keep the driver process x64-enabled for fallback events.
   Requesting ``smc_dtype='float64'`` in an x64-disabled process raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.scipy as jsp
from jax import random

import blackjax
from blackjax.smc import inner_kernel_tuning
from blackjax.smc.resampling import systematic

from .blockwise_rmh import build_blockwise_scale_overrides
from .mwg_kernel import (
    build_mwg_kernel,
    build_mwg_parameter_update_fn,
    default_inner_steps,
    default_mwg_blocks,
)
from .irmh_kernel import (
    build_gmm_parameter_update_fn,
    build_hybrid_mwg_irmh_step_fn,
    build_irmh_step_fn,
    gmm_params_to_mcmc_dict,
    initial_gmm_parameter_value,
    DiagGMM,
)
from .data_prep import build_location_samples_from_errors
from .tape import Tape_MT6
from .tape_jax_dtype import jax_Tape_MT6
# Imported at module level (before any array exists) so the fp32 recipe above
# stays valid: this is the only place the batched path touches ``tape_jax``'s
# global x64 switch.
from .inversion_blackjax import InversionResult, softplus_inv


DataDict = Dict[str, Any]

_SMALL_NUMBER = 1e-10
_EPS = 1e-6
_LAMBDA_DONE = 1.0 - 1e-6

#: Station counts are rounded up to a multiple of this before stacking, so a
#: handful of distinct XLA programs covers a whole catalogue (plan 3.2).
PAD_MULTIPLE = 8

#: Particle field order (matches ``init_particle`` in ``_invert_single_event``).
FIELDS_NO_AR = ("gamma", "delta", "kappa", "h", "sigma")
FIELDS_AR = FIELDS_NO_AR + ("sigma_amp_ratio",)


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchEntry:
    """One independent SMC run: a single chain of a single event."""

    event_id: str
    event_index: int
    chain_index: int
    seed: int


@dataclass
class Bucket:
    """A group of batch entries that share one padded shape (one XLA program).

    Attributes
    ----------
    n_pol, n_ar : int
        Padded observation counts (multiples of :data:`PAD_MULTIPLE`).
    n_loc : int
        Number of location samples (batch-uniform config constant).
    has_pol, has_ar : bool
        Which likelihood terms are active for every entry in this bucket.
    entries : list of BatchEntry
        Length ``B``; entry ``b`` corresponds to row ``b`` of every array in
        ``data``.
    data : dict of np.ndarray
        Stacked, padded, masked arrays with leading axis ``B``.
    n_active : int or None
        Number of *real* entries.  ``None`` means "all of them".  Chunking a
        bucket to a fixed ``max_batch`` (plan 3.2) repeats entry 0 to fill the
        last chunk so its XLA shapes match the full chunks; rows
        ``n_active:`` are those repeats and their results are discarded.
    """

    n_pol: int
    n_ar: int
    n_loc: int
    has_pol: bool
    has_ar: bool
    entries: List[BatchEntry] = field(default_factory=list)
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    n_active: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def active(self) -> int:
        """Number of rows whose results are kept."""
        return len(self.entries) if self.n_active is None else int(self.n_active)

    @property
    def key(self) -> Tuple[int, int, int, bool, bool]:
        return (self.n_pol, self.n_ar, self.n_loc, self.has_pol, self.has_ar)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Bucket(n_pol={self.n_pol}, n_ar={self.n_ar}, n_loc={self.n_loc}, "
            f"B={self.size})"
        )


# ---------------------------------------------------------------------------
# scope guard
# ---------------------------------------------------------------------------
_BATCHED_MCMC_KERNELS = frozenset({"rmh", "irmh", "mwg_irmh"})


def _check_supported(inv: Any) -> None:
    """Raise NotImplementedError for configurations the batched path lacks."""
    if inv.smc_method != "adaptive_tempered":
        raise NotImplementedError(
            "Batched SMC supports smc_method='adaptive_tempered' only; got "
            f"'{inv.smc_method}'. Use the unbatched path for persistent sampling."
        )
    kernel = str(getattr(inv, "mcmc_kernel", "rmh")).lower()
    if kernel not in _BATCHED_MCMC_KERNELS:
        raise NotImplementedError(
            "Batched SMC supports mcmc_kernel in "
            f"{sorted(_BATCHED_MCMC_KERNELS)}; got '{inv.mcmc_kernel}'. "
            "NUTS window adaptation is not vmapped."
        )
    if not inv.adapt_proposal:
        raise NotImplementedError(
            "Batched SMC supports adapt_proposal=True (MWG / hybrid IRMH "
            "parameter adaptation) only; got adapt_proposal=False."
        )
    if inv.amp_ratio_noise_mode != "global":
        raise NotImplementedError(
            "Batched SMC supports amp_ratio_noise_mode='global' only; got "
            f"'{inv.amp_ratio_noise_mode}'. station_smooth needs padded "
            "per-station segment sums (plan section 6)."
        )
    if inv.dc:
        raise NotImplementedError(
            "Batched SMC supports dc=False only; the double-couple constraint "
            "is not wired into the batched model."
        )


def _dummy_gmm_parameter_value(
    n_components: int, dim: int, dtype=jnp.float32
) -> Dict[str, jnp.ndarray]:
    """Placeholders with correct shapes for the step-path program build.

    Live GMM parameters after stage 0 come from ``parameter_override`` updated
    by ``build_gmm_parameter_update_fn``; these dummies are only structural.
    """
    k = int(n_components)
    d = int(dim)
    log_w = jnp.full((k,), -jnp.log(jnp.asarray(k, dtype=dtype)), dtype=dtype)
    means = jnp.zeros((k, d), dtype=dtype)
    log_stds = jnp.zeros((k, d), dtype=dtype)
    return gmm_params_to_mcmc_dict(
        DiagGMM(log_weights=log_w, means=means, log_stds=log_stds)
    )


# ---------------------------------------------------------------------------
# padding helpers
# ---------------------------------------------------------------------------
def _round_up(n: int, multiple: int = PAD_MULTIPLE) -> int:
    if n <= 0:
        return 0
    return int(-(-int(n) // int(multiple)) * int(multiple))


def _pad_event_arrays(
    prep: Dict[str, Any], n_pol: int, n_ar: int, n_loc: int
) -> Dict[str, np.ndarray]:
    """Pad one event's arrays to the bucket shape and build the masks.

    Padding values are chosen so padded rows evaluate to *finite* garbage that
    the mask then zeroes out (never NaN):

    * ``a_pol`` rows 0.0    -> ``X_pol = 0`` -> ``base_prob = 0.5``
    * ``error_pol`` 1.0     -> no division by zero
    * ``incorrect_prob`` 0.0
    * ``a1_ar``/``a2_ar`` rows 0.0 -> ``log_ratio_pred = 0``
    * ``amp_ratio_obs`` 1.0 -> ``log(obs) = 0`` and ``-log(obs) = 0``
    * ``log_ratio_sigma`` 1.0 -> ``sigma_combined`` finite and positive
    """
    out: Dict[str, np.ndarray] = {}

    if prep["has_pol"]:
        a_pol = np.asarray(prep["a_pol"], dtype=np.float64)
        n = a_pol.shape[0]
        if n > n_pol:
            raise ValueError(f"polarity count {n} exceeds bucket width {n_pol}")
        padded = np.zeros((n_pol, n_loc, 6), dtype=np.float64)
        padded[:n] = a_pol
        out["a_pol"] = padded

        mask = np.zeros(n_pol, dtype=np.float64)
        mask[:n] = 1.0
        out["pol_mask"] = mask

        err = np.ones(n_pol, dtype=np.float64)
        err[:n] = np.asarray(prep["error_pol"], dtype=np.float64).reshape(-1)
        out["error_pol"] = err

        # ``incorrect_prob`` is a python float for a global value and a
        # per-observation array otherwise; normalise to per-observation so a
        # bucket can mix the two.  The math is identical either way.
        inc_raw = prep["incorrect_prob"]
        inc = np.zeros(n_pol, dtype=np.float64)
        inc[:n] = np.asarray(inc_raw, dtype=np.float64).reshape(-1)
        out["incorrect_prob"] = inc

    if prep["has_ar"]:
        a1 = np.asarray(prep["a1_ar"], dtype=np.float64)
        a2 = np.asarray(prep["a2_ar"], dtype=np.float64)
        n = a1.shape[0]
        if n > n_ar:
            raise ValueError(f"amplitude-ratio count {n} exceeds bucket width {n_ar}")
        for name, arr in (("a1_ar", a1), ("a2_ar", a2)):
            padded = np.zeros((n_ar, n_loc, 6), dtype=np.float64)
            padded[:n] = arr
            out[name] = padded

        mask = np.zeros(n_ar, dtype=np.float64)
        mask[:n] = 1.0
        out["ar_mask"] = mask

        obs = np.ones(n_ar, dtype=np.float64)
        obs[:n] = np.asarray(prep["amp_ratio_obs"], dtype=np.float64).reshape(-1)
        out["amp_ratio_obs"] = obs

        sig = np.ones(n_ar, dtype=np.float64)
        sig[:n] = np.asarray(prep["log_ratio_sigma"], dtype=np.float64).reshape(-1)
        out["log_ratio_sigma"] = sig

    return out


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------
def prepare_batch(
    inv: Any,
    events: Sequence[Tuple[str, DataDict]],
    pad_multiple: int = PAD_MULTIPLE,
    per_event_seeds: bool = False,
) -> List[Bucket]:
    """Turn ``(event_id, event)`` pairs into shape-bucketed, stacked batches.

    Per event: one station-angle realisation is drawn (shared by all of that
    event's chains, mirroring ``_invert_multi_chain``), the numpy likelihood
    arrays are built with ``inv._prepare_event_arrays`` and padded to the
    bucket shape ``(ceil(N_pol/8)*8, ceil(N_ar/8)*8)``.  Batch entries are
    ``event x chain``; chain seeds mirror ``forward()``'s branch exactly, so a
    batched chain and its unbatched twin start from the same particles:
    ``np.random.SeedSequence(inv.random_seed).generate_state(num_chains)`` for
    ``num_chains > 1`` (as ``_invert_multi_chain`` does) and plain
    ``inv.random_seed`` for ``num_chains == 1`` (as ``_invert_single_event``
    does).

    ``event_id`` must be unique across ``events``: it keys the result dict of
    :func:`run_batched`, so duplicates would silently merge two events' chains
    into one over-long chain axis.  Pad/repeat entries needed to fill a fixed
    batch width must therefore carry a distinct id (or be added with
    ``max_batch``, which does the repetition internally).

    Parameters
    ----------
    inv : InversionBlackJAX
        Configured sampler; used for its settings, its RNG and its data-prep.
    events : sequence of (event_id, DataDict)
        Events to invert.  ``event_id`` keys the result dict of
        :func:`run_batched`.
    pad_multiple : int
        Bucket granularity for the observation counts.
    per_event_seeds : bool
        ``False`` (default) reproduces ``forward()`` exactly: every event of the
        batch gets the SAME chain seeds, so entry ``b`` is bit-identical to its
        unbatched twin -- which is what the equivalence tests check, but which
        also means two events in one batch share their initial particle cloud
        and their whole SMC key stream (common random numbers, correlated
        Monte-Carlo error).  ``True`` gives event ``i`` its own stream via
        ``SeedSequence(inv.random_seed).spawn(len(events))[i]``, which is what a
        catalogue-scale driver wants; event-vs-unbatched parity is then only
        statistical.

    Returns
    -------
    list of Bucket
    """
    _check_supported(inv)

    ids = [str(eid) for eid, _ev in events]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(
            "duplicate event_id(s) in a batch: "
            f"{dups}. Event ids key the result dict, so duplicates would merge "
            "two events' chains into one result; give repeat/pad entries a "
            "distinct id (or use run_batched(max_batch=...) which pads "
            "internally)."
        )

    if per_event_seeds:
        # One independent stream per event: entry (event i, chain c) shares
        # nothing with (event j, chain c). Uses SeedSequence.spawn, which is
        # designed exactly for this.
        children = np.random.SeedSequence(inv.random_seed).spawn(len(events))
        chain_seeds_per_event = [
            [int(s) for s in child.generate_state(max(1, int(inv.num_chains)))]
            for child in children
        ]
    elif inv.num_chains == 1:
        # forward() calls _invert_single_event directly, which seeds with
        # random.PRNGKey(self.random_seed) -- no SeedSequence derivation.
        chain_seeds_per_event = None
        chain_seeds = [int(inv.random_seed)]
    else:
        ss = np.random.SeedSequence(inv.random_seed)
        chain_seeds_per_event = None
        chain_seeds = [int(s) for s in ss.generate_state(inv.num_chains)]

    buckets: Dict[Tuple[int, int, int, bool, bool], Bucket] = {}
    pending: Dict[Tuple[int, int, int, bool, bool], List[Dict[str, np.ndarray]]] = {}

    for event_index, (event_id, event) in enumerate(events):
        filtered = inv._filter_event_by_options(event)

        # One station-angle realization per event, shared by its chains.
        # The RNG is derived from (random_seed, UID) so the realization does
        # not depend on batch composition and matches the unbatched path.
        location_samples = build_location_samples_from_errors(
            filtered,
            rng=inv._location_rng(filtered),
            n_samples=inv.location_samples_n,
            azimuth_error=inv.azimuth_error,
            takeoff_error=inv.takeoff_error,
        )
        prep = inv._prepare_event_arrays(filtered, location_samples=location_samples)
        if prep["station_smooth_ar"]:
            raise NotImplementedError(
                "Batched SMC does not support station_smooth amplitude-ratio "
                f"noise (event {event_id})."
            )

        has_pol = bool(prep["has_pol"])
        has_ar = bool(prep["has_ar"])
        n_pol_raw = int(prep["a_pol"].shape[0]) if has_pol else 0
        n_ar_raw = int(prep["a1_ar"].shape[0]) if has_ar else 0
        n_loc = int(prep["a_pol"].shape[1] if has_pol else prep["a1_ar"].shape[1])

        key = (
            _round_up(n_pol_raw, pad_multiple),
            _round_up(n_ar_raw, pad_multiple),
            n_loc,
            has_pol,
            has_ar,
        )
        if key not in buckets:
            buckets[key] = Bucket(
                n_pol=key[0], n_ar=key[1], n_loc=n_loc, has_pol=has_pol, has_ar=has_ar
            )
            pending[key] = []

        padded = _pad_event_arrays(prep, key[0], key[1], n_loc)
        event_chain_seeds = (
            chain_seeds if chain_seeds_per_event is None
            else chain_seeds_per_event[event_index]
        )
        for chain_index, seed in enumerate(event_chain_seeds):
            buckets[key].entries.append(
                BatchEntry(
                    event_id=str(event_id),
                    event_index=event_index,
                    chain_index=chain_index,
                    seed=int(seed),
                )
            )
            pending[key].append(padded)

    for key, bucket in buckets.items():
        rows = pending[key]
        bucket.data = {
            name: np.stack([row[name] for row in rows], axis=0) for name in rows[0]
        }
    return list(buckets.values())


# ---------------------------------------------------------------------------
# batched model (plan 3.3)
# ---------------------------------------------------------------------------
def _make_unconstrained_to_params(has_ar: bool) -> Callable:
    """Elementwise unconstrained -> physical map (identical to the unbatched one).

    Every operation is elementwise in the particle, so the returned function can
    be applied directly to ``(B, P)`` arrays as well as to single particles.
    """

    def _unconstrained_to_params(position):
        params = {}
        g_raw = jnn.sigmoid(position["gamma"])
        d_raw = jnn.sigmoid(position["delta"])
        params["gamma"] = -jnp.pi / 6.0 + g_raw * (jnp.pi / 3.0)
        params["delta"] = -jnp.pi / 2.0 + d_raw * (jnp.pi)

        k_raw = jnn.sigmoid(position["kappa"])
        h_raw = jnn.sigmoid(position["h"])
        s_raw = jnn.sigmoid(position["sigma"])
        params["kappa"] = k_raw * (2.0 * jnp.pi)
        params["h"] = h_raw
        params["sigma"] = -jnp.pi / 2.0 + s_raw * (jnp.pi)

        if has_ar:
            sigma_ar = jnn.softplus(position["sigma_amp_ratio"]) + _EPS
            params["sigma_amp_ratio"] = sigma_ar
            params["sigma_amp_ratio_global"] = sigma_ar
        return params

    return _unconstrained_to_params


def _make_model(
    data: Dict[str, Any],
    has_pol: bool,
    has_ar: bool,
    gamma_beta_prior: Tuple[float, float],
    delta_beta_prior: Tuple[float, float],
    amp_ratio_sigma_prior: float,
):
    """Build ``(logprior_fn, loglikelihood_fn)`` for ONE batch entry.

    ``data`` holds that entry's padded arrays; under ``jax.vmap`` they are
    traced slices of the stacked ``(B, ...)`` arrays, so one trace serves the
    whole batch.  The math is the unbatched math of ``_invert_single_event``
    with ``jnp.sum(x)`` replaced by ``jnp.sum(mask * x)``.
    """
    unconstrained_to_params = _make_unconstrained_to_params(has_ar)

    def logprior_fn(position):
        log_p = 0.0

        ug = position["gamma"]
        ud = position["delta"]
        g_raw = jnn.sigmoid(ug)
        d_raw = jnn.sigmoid(ud)

        g_a, g_b = gamma_beta_prior
        d_a, d_b = delta_beta_prior
        log_beta_g = (
            (g_a - 1.0) * jnp.log(g_raw + _EPS)
            + (g_b - 1.0) * jnp.log1p(-g_raw + _EPS)
            - jsp.special.betaln(g_a, g_b)
        )
        log_beta_d = (
            (d_a - 1.0) * jnp.log(d_raw + _EPS)
            + (d_b - 1.0) * jnp.log1p(-d_raw + _EPS)
            - jsp.special.betaln(d_a, d_b)
        )
        log_jac_g = jnp.log(g_raw + _EPS) + jnp.log1p(-g_raw + _EPS)
        log_jac_d = jnp.log(d_raw + _EPS) + jnp.log1p(-d_raw + _EPS)
        log_p += log_beta_g + log_jac_g
        log_p += log_beta_d + log_jac_d

        k_raw = jnn.sigmoid(position["kappa"])
        h_raw = jnn.sigmoid(position["h"])
        s_raw = jnn.sigmoid(position["sigma"])

        log_unif_kappa = -jnp.log(2.0 * jnp.pi)
        log_jac_kappa = (
            jnp.log(2.0 * jnp.pi) + jnp.log(k_raw + _EPS) + jnp.log1p(-k_raw + _EPS)
        )
        log_p += log_unif_kappa + log_jac_kappa
        log_p += jnp.log(h_raw + _EPS) + jnp.log1p(-h_raw + _EPS)
        log_unif_sigma = -jnp.log(jnp.pi)
        log_jac_sigma = (
            jnp.log(jnp.pi) + jnp.log(s_raw + _EPS) + jnp.log1p(-s_raw + _EPS)
        )
        log_p += log_unif_sigma + log_jac_sigma

        if has_ar:
            u_ar = position["sigma_amp_ratio"]
            sigma_ar = jnn.softplus(u_ar) + _EPS
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
        params = unconstrained_to_params(position)
        mt6 = jax_Tape_MT6(
            params["gamma"],
            params["delta"],
            params["kappa"],
            params["h"],
            params["sigma"],
        )

        log_lik = 0.0

        if has_pol:
            a_pol = data["a_pol"]
            pol_mask = data["pol_mask"]
            X_pol = jnp.tensordot(a_pol, mt6, axes=[[2], [0]])

            sigma_pol = jnp.maximum(data["error_pol"], _SMALL_NUMBER)
            sigma_pol_bc = sigma_pol[:, None]

            base_prob = 0.5 * (
                1.0 + jax.scipy.special.erf(X_pol / (jnp.sqrt(2.0) * sigma_pol_bc))
            )
            inc_bc = data["incorrect_prob"][:, None]
            p_sample = inc_bc + (1.0 - 2.0 * inc_bc) * base_prob

            p_avg = jnp.mean(p_sample, axis=1)
            log_lik += jnp.sum(pol_mask * jnp.log(p_avg + _SMALL_NUMBER))

        if has_ar:
            a1_ar = data["a1_ar"]
            a2_ar = data["a2_ar"]
            ar_mask = data["ar_mask"]
            amp_ratio_obs = data["amp_ratio_obs"]

            amp1_pred = jnp.abs(jnp.tensordot(a1_ar, mt6, axes=[[2], [0]]))
            amp2_pred = jnp.abs(jnp.tensordot(a2_ar, mt6, axes=[[2], [0]]))

            log_ratio_pred = jnp.log(amp1_pred + _SMALL_NUMBER) - jnp.log(
                amp2_pred + _SMALL_NUMBER
            )
            log_ratio_obs = jnp.log(jnp.maximum(amp_ratio_obs, _SMALL_NUMBER))[:, None]

            sigma_ar = params["sigma_amp_ratio"]
            sigma_combined = jnp.sqrt(data["log_ratio_sigma"] ** 2 + sigma_ar**2)
            sigma_bc = sigma_combined[:, None]

            resid = log_ratio_obs - log_ratio_pred
            log_prob_samples = (
                -0.5 * (resid / sigma_bc) ** 2
                - jnp.log(sigma_bc)
                - 0.5 * jnp.log(2 * jnp.pi)
                - jnp.log(amp_ratio_obs[:, None])
            )

            n_samp = a1_ar.shape[1]
            log_prob_marginal = jax.scipy.special.logsumexp(
                log_prob_samples, axis=1
            ) - jnp.log(n_samp)
            log_lik += jnp.sum(ar_mask * log_prob_marginal)

        log_lik = jnp.clip(log_lik, -1.0e3, 1.0e3)
        return jnp.nan_to_num(log_lik, nan=-1.0e3, posinf=1.0e3, neginf=-1.0e3)

    return logprior_fn, loglikelihood_fn


def _make_init_particle(
    has_ar: bool,
    gamma_beta_prior: Tuple[float, float],
    delta_beta_prior: Tuple[float, float],
    amp_ratio_sigma_prior: float,
) -> Callable:
    """Single-particle initialiser, identical to ``_invert_single_event``'s."""

    def _logit(p):
        p = jnp.clip(p, _EPS, 1.0 - _EPS)
        return jnp.log(p) - jnp.log1p(-p)

    def init_particle(key):
        n_keys = 6 if has_ar else 5
        keys = random.split(key, n_keys)

        g_a, g_b = gamma_beta_prior
        d_a, d_b = delta_beta_prior
        params_u = {
            "gamma": _logit(random.beta(keys[0], g_a, g_b)),
            "delta": _logit(random.beta(keys[1], d_a, d_b)),
            "kappa": _logit(random.uniform(keys[2], minval=_EPS, maxval=1.0 - _EPS)),
            "h": _logit(random.uniform(keys[3], minval=_EPS, maxval=1.0 - _EPS)),
            "sigma": _logit(random.uniform(keys[4], minval=_EPS, maxval=1.0 - _EPS)),
        }
        if has_ar:
            u = random.uniform(keys[5], minval=_EPS, maxval=1.0 - _EPS)
            sigma_ar = amp_ratio_sigma_prior * jnp.tan(jnp.pi * u / 2.0)
            sigma_ar = jnp.clip(sigma_ar, 1e-6, 100.0)
            params_u["sigma_amp_ratio"] = softplus_inv(sigma_ar, _EPS)
        return params_u

    return init_particle


def _build_smc(
    data: Dict[str, Any],
    stds: Dict[str, Any],
    *,
    has_pol: bool,
    has_ar: bool,
    gamma_beta_prior,
    delta_beta_prior,
    amp_ratio_sigma_prior: float,
    mechanism_steps: int,
    num_mcmc_steps: int,
    target_ess: float,
    mcmc_kernel: str = "rmh",
    irmh_n_components: int = 3,
    irmh_steps: int = 1,
    irmh_em_iters: int = 8,
    particles: Optional[Dict[str, Any]] = None,
    gmm_rng_key: Optional[jax.Array] = None,
):
    """Production SMC algorithm for one batch entry (traced under ``vmap``).

    Mirrors ``_invert_single_event`` for ``mcmc_kernel`` in
    ``{'rmh', 'irmh', 'mwg_irmh'}`` with ``adapt_proposal=True``.  ``stds``
    holds per-entry *traced* scalars instead of the Python floats the
    unbatched path uses (M0 cross-cutting finding 5).

    ``particles`` (optional) is used only to fit the stage-0 GMM for IRMH
    kernels.  The step-path program passes ``particles=None`` and uses dummy
    GMM leaves; live GMM params come from ``parameter_override`` after each
    stage.
    """
    logprior_fn, loglikelihood_fn = _make_model(
        data,
        has_pol=has_pol,
        has_ar=has_ar,
        gamma_beta_prior=gamma_beta_prior,
        delta_beta_prior=delta_beta_prior,
        amp_ratio_sigma_prior=amp_ratio_sigma_prior,
    )

    field_names = list(stds.keys())
    blocks = default_mwg_blocks(
        dc=False, has_ar=has_ar, noise_fields=("sigma_amp_ratio",)
    )
    inner_steps = default_inner_steps(mechanism_steps=mechanism_steps)
    initial_scales = build_blockwise_scale_overrides(
        blocks=blocks, initial_stds=stds, current_stds=stds
    )
    scale_params = {
        name: jnp.asarray(scale)[None] for name, scale in initial_scales.items()
    }
    kernel = str(mcmc_kernel).lower()

    if kernel == "rmh":
        mcmc_step_fn = build_mwg_kernel(
            blocks=blocks, inner_steps_per_block=inner_steps
        )
        mcmc_parameter_update_fn = build_mwg_parameter_update_fn(
            blocks=blocks,
            field_names=field_names,
            initial_stds=stds,
        )
        initial_parameter_value = dict(scale_params)
    elif kernel in {"irmh", "mwg_irmh"}:
        dim = len(field_names)
        if particles is not None:
            key = gmm_rng_key if gmm_rng_key is not None else random.PRNGKey(0)
            gmm_params = initial_gmm_parameter_value(
                particles,
                field_names,
                n_components=int(irmh_n_components),
                n_iters=int(irmh_em_iters),
                rng_key=key,
            )
        else:
            # Structural dummies for the step-path XLA program.
            sample_dtype = jnp.result_type(
                *[jnp.asarray(v) for v in stds.values()]
            )
            gmm_params = _dummy_gmm_parameter_value(
                int(irmh_n_components), dim, dtype=sample_dtype
            )

        mwg_scale_update = build_mwg_parameter_update_fn(
            blocks=blocks,
            field_names=field_names,
            initial_stds=stds,
        )
        if kernel == "mwg_irmh":
            mcmc_step_fn = build_hybrid_mwg_irmh_step_fn(
                field_names=field_names,
                blocks=blocks,
                inner_steps_per_block=inner_steps,
                n_components=int(irmh_n_components),
                n_irmh_steps=int(irmh_steps),
            )
            mcmc_parameter_update_fn = build_gmm_parameter_update_fn(
                field_names=field_names,
                n_components=int(irmh_n_components),
                n_iters=int(irmh_em_iters),
                mwg_update_fn=mwg_scale_update,
            )
            initial_parameter_value = {**scale_params, **gmm_params}
        else:
            mcmc_step_fn = build_irmh_step_fn(
                field_names=field_names,
                n_components=int(irmh_n_components),
                n_irmh_steps=int(irmh_steps),
            )
            mcmc_parameter_update_fn = build_gmm_parameter_update_fn(
                field_names=field_names,
                n_components=int(irmh_n_components),
                n_iters=int(irmh_em_iters),
                mwg_update_fn=None,
            )
            initial_parameter_value = dict(gmm_params)
    else:
        raise NotImplementedError(
            f"Batched _build_smc does not support mcmc_kernel={mcmc_kernel!r}"
        )

    return inner_kernel_tuning.as_top_level_api(
        smc_algorithm=blackjax.adaptive_tempered_smc,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=mcmc_step_fn,
        mcmc_init_fn=blackjax.rmh.init,
        resampling_fn=systematic,
        mcmc_parameter_update_fn=mcmc_parameter_update_fn,
        initial_parameter_value=initial_parameter_value,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )


def _sampler_state(state):
    """Unwrap ``StateWithParameterOverride`` if present."""
    return state.sampler_state if hasattr(state, "sampler_state") else state


# ---------------------------------------------------------------------------
# batched driver (plan 3.4)
# ---------------------------------------------------------------------------
def _initial_particles(
    init_particle: Callable, seeds: Sequence[int], num_particles: int, dtype
):
    """``(B, P)`` per field, one independent chain per seed.

    The key stream matches ``_invert_single_event`` exactly
    (``key = PRNGKey(seed)``; ``key, init_key = split(key)``;
    ``init_keys = split(init_key, P)``), so entry ``b`` starts from the same
    particles its unbatched twin would.  The leftover ``key`` per entry is
    returned and drives that entry's SMC stages.
    """
    per_entry = []
    run_keys = []
    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key, init_key = random.split(key)
        init_keys = random.split(init_key, num_particles)
        particles = jax.vmap(init_particle)(init_keys)
        per_entry.append(jax.tree.map(lambda x: x.astype(dtype), particles))
        run_keys.append(key)
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_entry)
    return stacked, jnp.stack(run_keys, axis=0)


#: Cache of jitted (init, step, read_back) triples.  ``jax.jit`` caches on the
#: *function object*, so building fresh closures per call -- as the first
#: implementation did -- recompiled the whole XLA program on every
#: :func:`run_batched` call even for identical shapes and configuration
#: (~21 s per fp32 program on an A40).  The closures depend only on the
#: quantities in the key below; data, stds, keys and particles are arguments.
_PROGRAM_CACHE: Dict[Tuple, Tuple[Callable, Callable, Callable]] = {}


def clear_program_cache() -> None:
    """Drop the cached jitted batched programs (frees their XLA executables)."""
    _PROGRAM_CACHE.clear()


def _hashable(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(float(v) for v in np.asarray(value).ravel())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _batched_programs(smc_kwargs: Dict[str, Any]):
    """Return cached ``(batched_init, batched_step, read_back)`` for a config."""
    key = tuple(sorted((k, _hashable(v)) for k, v in smc_kwargs.items()))
    cached = _PROGRAM_CACHE.get(key)
    if cached is not None:
        return cached

    # Split static kwargs: IRMH init fits a GMM on ``parts``; the step path
    # rebuilds with particles=None (dummy GMM leaves only).
    def init_fn(data, stds_e, parts, gmm_key):
        return _build_smc(
            data, stds_e, particles=parts, gmm_rng_key=gmm_key, **smc_kwargs
        ).init(parts)

    def step_fn(data, stds_e, key_, state):
        return _build_smc(data, stds_e, particles=None, **smc_kwargs).step(
            key_, state
        )

    @jax.jit
    def read_back(state, info):
        st = _sampler_state(state)
        lam = st.tempering_param
        ess = 1.0 / jnp.sum(st.weights**2, axis=-1)
        update_info = getattr(info, "update_info", info)
        acc = getattr(update_info, "acceptance_rate", None)
        if acc is None:
            acc = jnp.zeros_like(lam)
        else:
            acc = jnp.mean(jnp.reshape(acc, (lam.shape[0], -1)), axis=-1)
        return lam, ess, acc

    programs = (
        jax.jit(jax.vmap(init_fn, in_axes=(0, 0, 0, 0))),
        jax.jit(jax.vmap(step_fn, in_axes=(0, 0, 0, 0))),
        read_back,
    )
    _PROGRAM_CACHE[key] = programs
    return programs


@jax.jit
def _freeze_entries(new_state, old_state, keep_old):
    """Per-entry select: rows of ``keep_old`` revert to ``old_state``.

    ``keep_old`` is a ``(B,)`` boolean array; every leaf of the (vmapped) SMC
    state has ``B`` as its leading axis, so the mask is broadcast over each
    leaf's remaining axes.  This is what makes a finished / stalled / failed
    entry stop evolving while the rest of the batch keeps stepping.
    """

    def _sel(new, old):
        new = jnp.asarray(new)
        old = jnp.asarray(old)
        if new.ndim == 0:
            return new
        mask = keep_old.reshape((-1,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, old, new)

    return jax.tree.map(_sel, new_state, old_state)


def _run_bucket(
    inv: Any,
    bucket: Bucket,
    dtype,
    progress: Callable[[str], None],
    label: str,
) -> Dict[str, Any]:
    """Run one bucket to completion as a single vmapped XLA program.

    Entries are *frozen* the moment they finish: once an entry reaches
    ``lambda >= 1``, trips the tempering-stall guard, or produces a non-finite
    lambda, its slice of the SMC state stops being updated (the vmapped step
    still runs for it -- masking the work away would need a different program --
    but its result is discarded and the previous state is kept).  Consequences:

    * a finished entry is bit-identical to its unbatched twin regardless of what
      else shares the bucket (no extra lambda=1 rejuvenation stages);
    * a stalled entry returns exactly the state the unbatched loop's ``break``
      would return, and its final lambda stays < 1, so ``tempering_stalled``
      describes the samples actually returned;
    * one entry with a NaN/Inf lambda no longer aborts the whole bucket: it is
      marked failed (``out['failed']``) and every other entry finishes normally.
    """
    B = bucket.size
    P = int(inv.num_particles)
    has_pol, has_ar = bucket.has_pol, bucket.has_ar
    fields = FIELDS_AR if has_ar else FIELDS_NO_AR

    data_b = {k: jnp.asarray(v, dtype=dtype) for k, v in bucket.data.items()}

    init_particle = _make_init_particle(
        has_ar=has_ar,
        gamma_beta_prior=inv.gamma_beta_prior,
        delta_beta_prior=inv.delta_beta_prior,
        amp_ratio_sigma_prior=inv.amp_ratio_sigma_prior,
    )
    particles, run_keys = _initial_particles(
        init_particle, [e.seed for e in bucket.entries], P, dtype
    )
    # Per-entry init stds stay TRACED (B,) arrays -- no float() on tracers.
    stds = {name: jnp.std(particles[name], axis=-1) for name in fields}

    smc_kwargs = dict(
        has_pol=has_pol,
        has_ar=has_ar,
        gamma_beta_prior=tuple(float(v) for v in inv.gamma_beta_prior),
        delta_beta_prior=tuple(float(v) for v in inv.delta_beta_prior),
        amp_ratio_sigma_prior=float(inv.amp_ratio_sigma_prior),
        mechanism_steps=int(inv.mechanism_steps),
        num_mcmc_steps=int(inv.num_mcmc_steps),
        target_ess=float(inv.smc_target_ess_ratio),
        mcmc_kernel=str(getattr(inv, "mcmc_kernel", "rmh")).lower(),
        irmh_n_components=int(getattr(inv, "irmh_n_components", 3)),
        irmh_steps=int(getattr(inv, "irmh_steps", 1)),
        irmh_em_iters=int(getattr(inv, "irmh_em_iters", 8)),
    )
    batched_init, batched_step, read_back = _batched_programs(smc_kwargs)

    # Stage-0 GMM EM-init key, derived per entry exactly like the unbatched
    # worker (``PRNGKey(self.random_seed ^ 0xA11CE)`` with the chain seed).
    gmm_keys = jnp.stack(
        [random.PRNGKey(int(e.seed) ^ 0xA11CE) for e in bucket.entries]
    )

    t0 = time.time()
    state = batched_init(data_b, stds, particles, gmm_keys)
    lam = np.asarray(_sampler_state(state).tempering_param, dtype=np.float64)
    t_init = time.time() - t0
    progress(f"{label} init+compile {t_init:.1f}s (B={B}, P={P})")

    stall = np.zeros(B, dtype=int)
    stalled = np.zeros(B, dtype=bool)
    failed = np.zeros(B, dtype=bool)
    done = lam >= _LAMBDA_DONE
    stages = 0
    keys = run_keys

    t0 = time.time()
    for stage in range(int(inv.max_smc_iterations)):
        if bool(np.all(done)):
            break
        split_keys = jax.vmap(random.split)(keys)
        keys, sub = split_keys[:, 0], split_keys[:, 1]
        prev_state = state
        state, info = batched_step(data_b, stds, sub, state)
        lam_new, ess, acc = read_back(state, info)
        del info
        lam_new = np.asarray(lam_new, dtype=np.float64)
        ess = np.asarray(ess, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        stages = stage + 1

        bad = ~np.isfinite(lam_new)
        # Entries finished BEFORE this stage (plus anything that just went
        # non-finite) keep the state they had; only live entries advance.
        keep_old = done | bad
        if bool(np.any(keep_old)):
            state = _freeze_entries(state, prev_state, jnp.asarray(keep_old))
            lam_new = np.where(keep_old, lam, lam_new)
        del prev_state

        if bool(np.any(bad)):
            failed |= bad
            progress(
                f"{label} stage {stages:3d}: ERROR non-finite tempering parameter "
                f"for entries {np.flatnonzero(bad).tolist()}; they are frozen at "
                "their previous state and reported as failed."
            )

        delta = lam_new - lam
        live = ~done
        stall = np.where(delta < inv.min_tempering_increment, stall + 1, 0)
        lam = lam_new
        at_target = lam >= _LAMBDA_DONE
        newly_stalled = (
            (stall >= inv.tempering_stall_patience) & ~at_target & ~failed & live
        )
        stalled |= newly_stalled
        done = at_target | stalled | failed

        live_now = ~done
        ess_min = float(ess[live_now].min()) if bool(np.any(live_now)) else float(ess.min())
        acc_mean = float(np.mean(acc[live_now])) if bool(np.any(live_now)) else float(np.mean(acc))
        progress(
            f"{label} stage {stages:3d}: lambda min={lam.min():.4f} "
            f"med={float(np.median(lam)):.4f}  finished={int(done.sum())}/{B}  "
            f"ESS min={ess_min:.0f}/{P}  acc={acc_mean:.3f}"
        )

    wall = time.time() - t0
    # ``tempering_stalled`` describes the samples returned: an entry is stalled
    # iff its FINAL lambda is short of the target (frozen there by the guard or
    # by max_smc_iterations), never merely because the guard fired at some point.
    stalled_final = (lam < _LAMBDA_DONE) & ~failed
    n_stalled = int(stalled_final.sum())
    n_failed = int(failed.sum())
    progress(
        f"{label} completed in {wall:.2f}s ({stages} stages), "
        f"lambda min={lam.min():.4f}"
        + (f", {n_stalled}/{B} entries stalled" if n_stalled else "")
        + (f", {n_failed}/{B} entries FAILED" if n_failed else "")
    )

    final = _sampler_state(state)
    unconstrained_to_params = _make_unconstrained_to_params(has_ar)
    params = unconstrained_to_params(final.particles)
    params_np = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    weights_np = np.asarray(final.weights, dtype=np.float64)

    # Per-sample data log-likelihood under the same model used for SMC.
    # Vmap over batch entries, then over particles within each entry.
    def _ll_one_entry(data_one, particles_one):
        _logprior, loglikelihood_fn = _make_model(
            data_one,
            has_pol=has_pol,
            has_ar=has_ar,
            gamma_beta_prior=tuple(float(v) for v in inv.gamma_beta_prior),
            delta_beta_prior=tuple(float(v) for v in inv.delta_beta_prior),
            amp_ratio_sigma_prior=float(inv.amp_ratio_sigma_prior),
        )
        del _logprior
        return jax.vmap(loglikelihood_fn)(particles_one)

    ln_p_np = np.asarray(
        jax.vmap(_ll_one_entry)(data_b, final.particles), dtype=np.float64
    )

    return {
        "params": params_np,
        "weights": weights_np,
        "ln_p": ln_p_np,
        "lambda": lam,
        "stalled": stalled_final,
        "stall_guard_fired": stalled,
        "failed": failed,
        "stages": stages,
        "wall": wall,
        "t_init": t_init,
        "particle_dtype": np.asarray(final.particles["gamma"]).dtype,
    }


def _entry_result(
    params: Dict[str, np.ndarray],
    weights: np.ndarray,
    b: int,
    stalled: bool,
    ln_p: Optional[np.ndarray] = None,
):
    """Build one ``InversionResult`` from row ``b`` of a bucket's output."""
    gamma_s = np.asarray(params["gamma"][b], dtype=float)
    delta_s = np.asarray(params["delta"][b], dtype=float)
    kappa_s = np.asarray(params["kappa"][b], dtype=float)
    h_s = np.asarray(params["h"][b], dtype=float)
    sigma_s = np.asarray(params["sigma"][b], dtype=float)
    n_samp = gamma_s.size

    mt6 = np.asarray(
        Tape_MT6(gamma_s, delta_s, kappa_s, h_s, sigma_s), dtype=float
    ).reshape(6, n_samp)

    if "sigma_amp_ratio" in params:
        sigma_ar_s = np.asarray(params["sigma_amp_ratio"][b], dtype=float)
        sigma_ar_global_s = np.asarray(params["sigma_amp_ratio_global"][b], dtype=float)
    else:
        sigma_ar_s = None
        sigma_ar_global_s = None

    ln_p_b = None
    if ln_p is not None:
        ln_p_b = np.asarray(ln_p[b], dtype=float)

    return InversionResult(
        mt6=mt6,
        gamma=gamma_s,
        delta=delta_s,
        kappa=kappa_s,
        h=h_s,
        sigma=sigma_s,
        ln_p=ln_p_b,
        weights=np.asarray(weights[b], dtype=float),
        idata=None,
        sigma_amp_ratio=sigma_ar_s,
        sigma_amp_ratio_global=sigma_ar_global_s,
        sigma_amp_ratio_station=None,
        sigma_amp_ratio_station_names=None,
        tempering_stalled=bool(stalled),
    )


# ---------------------------------------------------------------------------
# run_batched
# ---------------------------------------------------------------------------
def _chunk_bucket(bucket: Bucket, max_batch: Optional[int]) -> List[Bucket]:
    """Split a bucket into chunks of at most ``max_batch`` entries.

    The last chunk is padded back up to ``max_batch`` by repeating its first
    entry (plan 3.2) so every chunk of a bucket has identical XLA shapes and
    shares one compiled program; the repeats are marked inactive and their
    results are discarded.
    """
    if max_batch is None or bucket.size <= int(max_batch):
        return [bucket]
    max_batch = int(max_batch)
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1")

    chunks: List[Bucket] = []
    for start in range(0, bucket.size, max_batch):
        idx = list(range(start, min(start + max_batch, bucket.size)))
        n_active = len(idx)
        if n_active < max_batch:
            idx = idx + [idx[0]] * (max_batch - n_active)
        chunks.append(
            Bucket(
                n_pol=bucket.n_pol,
                n_ar=bucket.n_ar,
                n_loc=bucket.n_loc,
                has_pol=bucket.has_pol,
                has_ar=bucket.has_ar,
                entries=[bucket.entries[i] for i in idx],
                data={k: v[idx] for k, v in bucket.data.items()},
                n_active=n_active,
            )
        )
    return chunks


def _safe_callback(callback: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap a progress callback so it can never abort an in-flight bucket."""
    state = {"broken": False}

    def emit(msg: str) -> None:
        if state["broken"]:
            return
        try:
            callback(msg)
        except Exception as exc:  # noqa: BLE001 - progress must never be fatal
            state["broken"] = True
            try:
                print(
                    f"WARNING: progress_callback raised {type(exc).__name__}: {exc}; "
                    "further progress lines are suppressed.",
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                pass

    return emit


def run_batched(
    inv: Any,
    events: Sequence[Tuple[str, DataDict]],
    smc_dtype: str = "float64",
    progress_callback: Optional[Callable[[str], None]] = None,
    pad_multiple: int = PAD_MULTIPLE,
    max_batch: Optional[int] = None,
    on_bucket_error: str = "skip",
    return_failures: bool = False,
    per_event_seeds: bool = False,
):
    """Invert many events at once as bucketed, vmapped SMC programs.

    Parameters
    ----------
    inv : InversionBlackJAX
        Configured sampler.  Supports ``adaptive_tempered`` + global AR-noise +
        ``dc=False`` with ``mcmc_kernel`` in ``{'rmh', 'irmh', 'mwg_irmh'}``;
        anything else raises :class:`NotImplementedError`.
    events : sequence of (event_id, DataDict)
        Events to invert.
    smc_dtype : {'float64', 'float32'}
        Sampling precision.  ``'float32'`` casts all data arrays and initial
        particles to float32 (~9x faster on an A40, M0 spike 4); see the module
        docstring for the required process setup.
    progress_callback : callable, optional
        Receives one progress line per SMC stage.  Defaults to ``print``.
        Exceptions raised by the callback are swallowed (a failing log sink must
        not destroy an in-flight bucket).
    pad_multiple : int
        Bucket granularity for observation counts.
    max_batch : int, optional
        Hard cap on the batch width ``B`` of one XLA program.  Buckets larger
        than this are split into chunks of exactly ``max_batch`` entries (the
        last chunk padded with a repeat of its first entry, results discarded),
        which keeps device memory bounded and lets every chunk of a bucket reuse
        one compiled program.  ``None`` (default) means "one program per bucket,
        however wide it is" -- fine for a handful of events, but a catalogue of
        same-shaped events would build a single enormous program, so drivers
        should pass ``max_batch = gpu_batch_events * inv.num_chains``.
    on_bucket_error : {'skip', 'raise'}
        What to do when a bucket raises (OOM, XLA error, ...).  ``'skip'``
        (default) records every event of that bucket as failed and continues, so
        the results of the buckets that did complete survive.  ``'raise'``
        re-raises after attaching ``exc.partial_results`` / ``exc.failures``.
    return_failures : bool
        If True return ``(results, failures)`` where ``failures`` maps
        ``event_id -> reason`` for every event that could not be inverted.
    per_event_seeds : bool
        Give every event of the batch its own RNG stream instead of the shared,
        ``forward()``-identical one (see :func:`prepare_batch`).  Drivers that
        invert a catalogue should pass ``True``: with ``False`` the events of
        one batch share their initial particles and key stream, so their
        Monte-Carlo errors are correlated.  Default ``False`` keeps bit-level
        equivalence with the unbatched path (what the M1 tests assert).

    Returns
    -------
    dict or (dict, dict)
        ``{event_id: InversionResult}``.  With ``inv.num_chains > 1`` each
        result is chain-stacked by ``InversionBlackJAX._stack_chain_results``,
        i.e. structurally identical to what ``forward()`` returns today
        (``mt6`` ``(chains, 6, P)``, ``gamma`` ``(chains, P)``, ...).  Events
        whose bucket failed are absent from the dict.
    """
    if smc_dtype not in ("float64", "float32"):
        raise ValueError("smc_dtype must be one of {'float64', 'float32'}")
    if on_bucket_error not in ("skip", "raise"):
        raise ValueError("on_bucket_error must be one of {'skip', 'raise'}")
    if smc_dtype == "float64" and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "smc_dtype='float64' requires jax_enable_x64=True; this process has "
            "x64 disabled, so every array would be silently demoted to float32 "
            "and the 'float64' results would be single precision. Re-enable x64 "
            "(or run the fp32 batched sampler in its own process -- see the "
            "module docstring)."
        )
    dtype = jnp.float32 if smc_dtype == "float32" else jnp.float64

    emit = _safe_callback(progress_callback if progress_callback is not None else print)
    if smc_dtype == "float32" and jax.config.jax_enable_x64:
        emit(
            "NOTE: smc_dtype='float32' in an x64-enabled process. This works "
            "(particles stay float32), but some intermediates may be promoted "
            "to float64 and the fp32 speedup is lost. Disable x64 after imports "
            "and before any array is created (see module docstring)."
        )

    buckets = prepare_batch(
        inv, events, pad_multiple=pad_multiple, per_event_seeds=per_event_seeds
    )
    chunks: List[Bucket] = []
    for bucket in buckets:
        chunks.extend(_chunk_bucket(bucket, max_batch))

    per_entry: Dict[Tuple[int, str], List[Tuple[int, InversionResult]]] = {}
    failures: Dict[str, str] = {}
    for i, bucket in enumerate(chunks):
        pad_note = "" if bucket.active == bucket.size else f" (+{bucket.size - bucket.active} pad)"
        label = (
            f"[bucket {i + 1}/{len(chunks)} N_pol={bucket.n_pol} "
            f"N_ar={bucket.n_ar} B={bucket.size}{pad_note}]"
        )
        try:
            out = _run_bucket(inv, bucket, dtype, emit, label)
        except Exception as exc:  # noqa: BLE001 - isolate one bucket's failure
            reason = f"{type(exc).__name__}: {exc}"
            for entry in bucket.entries[: bucket.active]:
                failures[entry.event_id] = reason
            emit(f"{label} FAILED: {reason}")
            if on_bucket_error == "raise":
                exc.partial_results = _assemble(inv, per_entry, failures)  # type: ignore[attr-defined]
                exc.failures = dict(failures)  # type: ignore[attr-defined]
                raise
            continue

        actual = np.dtype(out["particle_dtype"])
        if actual != np.dtype(smc_dtype):
            emit(
                f"{label} WARNING: requested smc_dtype='{smc_dtype}' but the SMC "
                f"particles are {actual}."
            )
        for b, entry in enumerate(bucket.entries[: bucket.active]):
            if bool(out["failed"][b]):
                failures[entry.event_id] = (
                    "non-finite tempering parameter in chain "
                    f"{entry.chain_index}"
                )
                continue
            per_entry.setdefault((entry.event_index, entry.event_id), []).append(
                (
                    entry.chain_index,
                    _entry_result(
                        out["params"],
                        out["weights"],
                        b,
                        bool(out["stalled"][b]),
                        ln_p=out.get("ln_p"),
                    ),
                )
            )

    results = _assemble(inv, per_entry, failures)
    if failures:
        emit(
            f"{len(failures)} event(s) failed and are absent from the results: "
            + ", ".join(sorted(failures))
        )
    return (results, failures) if return_failures else results


def _assemble(
    inv: Any,
    per_entry: Dict[Tuple[int, str], List[Tuple[int, InversionResult]]],
    failures: Dict[str, str],
) -> Dict[str, InversionResult]:
    """Regroup per-(event, chain) results into ``{event_id: InversionResult}``.

    Keyed on ``(event_index, event_id)`` so that even if two entries somehow
    share an id (``prepare_batch`` rejects that up front) their chains cannot be
    merged into one over-long chain axis.
    """
    results: Dict[str, InversionResult] = {}
    for (_event_index, event_id), chains in per_entry.items():
        if event_id in failures:
            continue
        chain_results = [r for _idx, r in sorted(chains, key=lambda t: t[0])]
        if len(chain_results) != int(inv.num_chains):
            failures[event_id] = (
                f"got {len(chain_results)} chain result(s), expected "
                f"{int(inv.num_chains)}"
            )
            continue
        if inv.num_chains > 1:
            results[event_id] = inv._stack_chain_results(chain_results)
        else:
            results[event_id] = chain_results[0]
    return results
