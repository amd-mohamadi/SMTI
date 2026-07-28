"""
Independence Metropolis-Hastings kernel with a diagonal GMM proposal.

Fits a fixed-K diagonal Gaussian mixture to the current unconstrained particle
cloud (pure JAX EM) and uses BlackJAX IRMH so particles can jump between
posterior basins. Designed to compose with the existing MWG blockwise RMH
kernel (hybrid rejuvenation) and to stay pure-JAX for batched ``vmap``.

Notes on BlackJAX 1.3 acceptance correction
-------------------------------------------
``proposal_logdensity_fn(new_state, prev_state)`` is invoked from
``build_rmh_transition_energy`` as part of the *asymmetric* MH ratio. Empirical
check against a known independent proposal shows BlackJAX expects this callable
to return ``log q(prev_state)`` for an independent ``q`` (i.e. the log-density
of the *second* argument). We follow that convention so the accept probability
matches standard IRMH.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import random

import blackjax.mcmc.random_walk as blackjax_rw

from .mwg_kernel import MWGInfo, build_mwg_kernel


# ---------------------------------------------------------------------------
# Unconstrained dict <-> vector
# ---------------------------------------------------------------------------

def pack_particles(
    particles: Dict[str, jnp.ndarray],
    field_names: Sequence[str],
) -> jnp.ndarray:
    """Stack particle fields into ``(n_particles, D)``."""
    cols = [jnp.reshape(particles[name], (-1, 1)) for name in field_names]
    return jnp.concatenate(cols, axis=1)


def unpack_vector(
    z: jnp.ndarray,
    field_names: Sequence[str],
    template: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Unpack a single ``(D,)`` vector into an unconstrained position dict."""
    out = {}
    for i, name in enumerate(field_names):
        out[name] = jnp.reshape(z[i], template[name].shape)
    return out


def unpack_particles(
    Z: jnp.ndarray,
    field_names: Sequence[str],
    template: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Unpack ``(n, D)`` into a particle dict with leading sample axis."""
    out = {}
    for i, name in enumerate(field_names):
        out[name] = Z[:, i].reshape((-1,) + template[name].shape)
    return out


# ---------------------------------------------------------------------------
# Diagonal GMM (pure JAX)
# ---------------------------------------------------------------------------

class DiagGMM(NamedTuple):
    """Diagonal Gaussian mixture parameters."""
    log_weights: jnp.ndarray  # (K,)
    means: jnp.ndarray        # (K, D)
    log_stds: jnp.ndarray     # (K, D)


def _log_gauss_diag(x: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    """Log N(x; mean, diag(exp(log_std)**2)). ``x`` is (D,)."""
    z = (x - mean) / jnp.exp(log_std)
    return -0.5 * jnp.sum(z * z + 2.0 * log_std + jnp.log(2.0 * jnp.pi))


def gmm_log_prob(x: jnp.ndarray, gmm: DiagGMM) -> jnp.ndarray:
    """Log density of a single point ``x`` (D,) under the mixture."""
    def one_comp(k):
        return gmm.log_weights[k] + _log_gauss_diag(x, gmm.means[k], gmm.log_stds[k])

    log_comp = jax.vmap(one_comp)(jnp.arange(gmm.means.shape[0]))
    return jax.scipy.special.logsumexp(log_comp)


def gmm_log_prob_batch(X: jnp.ndarray, gmm: DiagGMM) -> jnp.ndarray:
    """Log density for each row of ``X`` (n, D)."""
    return jax.vmap(lambda x: gmm_log_prob(x, gmm))(X)


def sample_gmm(rng_key: jax.Array, gmm: DiagGMM) -> jnp.ndarray:
    """Draw one sample ``(D,)`` from the diagonal GMM."""
    key_c, key_x = random.split(rng_key)
    # Categorical over components
    comp = random.categorical(key_c, gmm.log_weights)
    mean = gmm.means[comp]
    std = jnp.exp(gmm.log_stds[comp])
    return mean + std * random.normal(key_x, shape=mean.shape, dtype=mean.dtype)


def fit_diag_gmm(
    rng_key: jax.Array,
    X: jnp.ndarray,
    n_components: int = 3,
    n_iters: int = 8,
    weight_floor: float = 1e-3,
    var_floor: float = 1e-4,
) -> DiagGMM:
    """
    Fit a diagonal GMM to rows of ``X`` (n, D) with fixed-iteration EM.

    Fully JAX-traceable (``lax.scan``), suitable for use inside a jitted
    ``mcmc_parameter_update_fn`` and under ``vmap``.
    """
    n, d = X.shape
    k = int(n_components)
    key_init, _ = random.split(rng_key)

    # Robust init: place component means at evenly spaced quantiles along the
    # highest-variance axis (stable under vmap; avoids all-mass collapse when
    # random point picks land in one blob). Jitter with a small RNG offset so
    # different keys still diversify.
    col_std = jnp.std(X, axis=0)
    axis = jnp.argmax(col_std)
    order = jnp.argsort(X[:, axis])
    # Quantile indices in [0, n-1]
    qs = (jnp.arange(k, dtype=X.dtype) + 0.5) / jnp.asarray(k, dtype=X.dtype)
    q_idx = jnp.clip((qs * n).astype(jnp.int32), 0, n - 1)
    means0 = X[order[q_idx]]
    jitter = 0.05 * col_std * random.normal(key_init, shape=means0.shape, dtype=X.dtype)
    means0 = means0 + jitter
    # Per-component std starts from global / sqrt(K)
    global_std = jnp.maximum(col_std, jnp.sqrt(var_floor))
    log_stds0 = jnp.log(
        jnp.broadcast_to(global_std / jnp.sqrt(jnp.asarray(k, dtype=X.dtype)), (k, d))
    )
    log_w0 = jnp.full((k,), -jnp.log(jnp.asarray(k, dtype=X.dtype)))

    def em_step(carry, _):
        log_w, means, log_stds = carry
        # E-step: responsibilities (n, K)
        def log_joint(i, j):
            return log_w[j] + _log_gauss_diag(X[i], means[j], log_stds[j])

        # (n, K)
        log_resp = jax.vmap(
            lambda i: jax.vmap(lambda j: log_joint(i, j))(jnp.arange(k))
        )(jnp.arange(n))
        log_norm = jax.scipy.special.logsumexp(log_resp, axis=1, keepdims=True)
        resp = jnp.exp(log_resp - log_norm)  # (n, K)

        # M-step
        Nk = jnp.sum(resp, axis=0) + 1e-8  # (K,)
        means_new = (resp.T @ X) / Nk[:, None]
        # Weighted variance per component/dim
        diff = X[None, :, :] - means_new[:, None, :]  # (K, n, D)
        var = jnp.sum(resp.T[:, :, None] * diff * diff, axis=1) / Nk[:, None]
        var = jnp.maximum(var, var_floor)
        log_stds_new = 0.5 * jnp.log(var)

        w = Nk / jnp.sum(Nk)
        w = jnp.maximum(w, weight_floor)
        w = w / jnp.sum(w)
        log_w_new = jnp.log(w)
        return (log_w_new, means_new, log_stds_new), None

    (log_w, means, log_stds), _ = jax.lax.scan(
        em_step, (log_w0, means0, log_stds0), xs=None, length=int(n_iters)
    )
    return DiagGMM(log_weights=log_w, means=means, log_stds=log_stds)


def fit_diag_gmm_from_particles(
    rng_key: jax.Array,
    particles: Dict[str, jnp.ndarray],
    field_names: Sequence[str],
    n_components: int = 3,
    n_iters: int = 8,
) -> DiagGMM:
    """Pack particles and fit a diagonal GMM."""
    X = pack_particles(particles, field_names)
    return fit_diag_gmm(
        rng_key, X, n_components=n_components, n_iters=n_iters
    )


# ---------------------------------------------------------------------------
# IRMH step (BlackJAX) parameterized by GMM leaves
# ---------------------------------------------------------------------------

class IRMHInfo(NamedTuple):
    """Info for a pure IRMH rejuvenation step (BlackJAX-compatible fields)."""
    acceptance_rate: jnp.ndarray
    is_accepted: jnp.ndarray
    proposal: jnp.ndarray
    irmh_acceptance_rate: jnp.ndarray
    # Present so scale-adaptation update fns that look for MWG fields tolerate IRMH-only.
    block_acceptance_rates: jnp.ndarray


class HybridInfo(NamedTuple):
    """Info for hybrid IRMH + MWG rejuvenation."""
    acceptance_rate: jnp.ndarray
    is_accepted: jnp.ndarray
    proposal: jnp.ndarray
    irmh_acceptance_rate: jnp.ndarray
    block_acceptance_rates: jnp.ndarray


def _gmm_from_kwargs(n_components: int, dim: int, **params) -> DiagGMM:
    """Rebuild DiagGMM from flat kwargs used as mcmc_parameters."""
    # Shared parameters arrive with leading particle-axis dim 1.
    log_w = jnp.reshape(params["gmm_log_weights"], (n_components,))
    means = jnp.reshape(params["gmm_means"], (n_components, dim))
    log_stds = jnp.reshape(params["gmm_log_stds"], (n_components, dim))
    return DiagGMM(log_weights=log_w, means=means, log_stds=log_stds)


def build_irmh_step_fn(
    field_names: Sequence[str],
    n_components: int,
    n_irmh_steps: int = 1,
):
    """
    Build an IRMH ``mcmc_step_fn`` for BlackJAX SMC.

    Signature::

        step(rng_key, state, logdensity_fn, **mcmc_params) -> (state, IRMHInfo)

    Expected ``mcmc_params`` keys (shared across particles, leading dim 1)::

        gmm_log_weights, gmm_means, gmm_log_stds
    """
    irmh_kernel = blackjax_rw.build_irmh()
    field_names = tuple(field_names)
    dim = len(field_names)

    def step_fn(rng_key, state, logdensity_fn, **params):
        gmm = _gmm_from_kwargs(n_components, dim, **params)
        template = state.position

        def proposal_distribution(key):
            z = sample_gmm(key, gmm)
            return unpack_vector(z, field_names, template)

        def _position_to_vector(position):
            return jnp.stack(
                [jnp.reshape(position[name], ()) for name in field_names]
            )

        def proposal_logdensity_fn(new_state, prev_state):
            # See module docstring: BlackJAX wants log q(prev) for independent q.
            return gmm_log_prob(_position_to_vector(prev_state.position), gmm)

        def one_irmh(carry, key):
            st = carry
            st, info = irmh_kernel(
                key, st, logdensity_fn, proposal_distribution, proposal_logdensity_fn
            )
            return st, (info.acceptance_rate, info.is_accepted)

        keys = random.split(rng_key, int(n_irmh_steps))
        state, (acc_probs, accepted) = jax.lax.scan(one_irmh, state, keys)
        # Realized accept fraction, not the mean MH acceptance *probability*
        # (BlackJAX RWInfo.acceptance_rate is p_accept; averaging that
        # overstates how often the chain actually moved).
        accept_frac = jnp.mean(accepted.astype(acc_probs.dtype))
        # Dummy proposal leaf for RWInfo-compat (not used by SMC)
        dummy_prop = jnp.asarray(0.0)
        return state, IRMHInfo(
            acceptance_rate=accept_frac,
            is_accepted=jnp.any(accepted),
            proposal=dummy_prop,
            irmh_acceptance_rate=accept_frac,
            block_acceptance_rates=jnp.zeros((1,), dtype=acc_probs.dtype),
        )

    return step_fn


def build_hybrid_mwg_irmh_step_fn(
    field_names: Sequence[str],
    blocks,
    inner_steps_per_block: Optional[Dict[str, int]] = None,
    n_components: int = 3,
    n_irmh_steps: int = 1,
):
    """
    Hybrid rejuvenation: ``n_irmh_steps`` independence MH, then MWG local moves.

    ``mcmc_params`` must include both GMM leaves and per-block scales.
    """
    irmh_step = build_irmh_step_fn(
        field_names=field_names,
        n_components=n_components,
        n_irmh_steps=n_irmh_steps,
    )
    mwg_step = build_mwg_kernel(
        blocks=blocks, inner_steps_per_block=inner_steps_per_block
    )
    block_names = tuple(b.name for b in blocks)
    gmm_keys = ("gmm_log_weights", "gmm_means", "gmm_log_stds")

    def step_fn(rng_key, state, logdensity_fn, **params):
        key_i, key_m = random.split(rng_key)
        gmm_params = {k: params[k] for k in gmm_keys}
        # Shared mcmc_parameters arrive with leading axis 1; MWG scales must be
        # scalars (or broadcastable without introducing a batch dim on accept).
        block_params = {
            k: jnp.reshape(params[k], ()) for k in block_names
        }

        state, irmh_info = irmh_step(key_i, state, logdensity_fn, **gmm_params)
        state, mwg_info = mwg_step(key_m, state, logdensity_fn, **block_params)

        # Overall accept rate: average of IRMH and mean MWG block rates
        overall = 0.5 * (
            irmh_info.irmh_acceptance_rate + mwg_info.acceptance_rate
        )
        return state, HybridInfo(
            acceptance_rate=overall,
            is_accepted=mwg_info.is_accepted,
            proposal=mwg_info.proposal,
            irmh_acceptance_rate=irmh_info.irmh_acceptance_rate,
            block_acceptance_rates=mwg_info.block_acceptance_rates,
        )

    return step_fn


# ---------------------------------------------------------------------------
# Parameter update (fit GMM; optionally keep MWG scales)
# ---------------------------------------------------------------------------

def gmm_params_to_mcmc_dict(gmm: DiagGMM) -> Dict[str, jnp.ndarray]:
    """Shared (leading dim 1) GMM leaves for ``inner_kernel_tuning``."""
    return {
        "gmm_log_weights": gmm.log_weights[None, ...],
        "gmm_means": gmm.means[None, ...],
        "gmm_log_stds": gmm.log_stds[None, ...],
    }


def initial_gmm_parameter_value(
    particles: Dict[str, jnp.ndarray],
    field_names: Sequence[str],
    n_components: int = 3,
    n_iters: int = 8,
    rng_key: Optional[jax.Array] = None,
) -> Dict[str, jnp.ndarray]:
    """Fit an initial GMM on the particle cloud for stage-0 MCMC params."""
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    gmm = fit_diag_gmm_from_particles(
        rng_key, particles, field_names, n_components=n_components, n_iters=n_iters
    )
    return gmm_params_to_mcmc_dict(gmm)


def build_gmm_parameter_update_fn(
    field_names: Sequence[str],
    n_components: int = 3,
    n_iters: int = 8,
    mwg_update_fn=None,
):
    """
    ``mcmc_parameter_update_fn`` that refits the GMM after each SMC stage.

    If ``mwg_update_fn`` is provided (hybrid mode), its scale dict is merged
    with the GMM leaves.
    """
    field_names = tuple(field_names)

    def mcmc_parameter_update_fn(rng_key, state, info):
        gmm = fit_diag_gmm_from_particles(
            rng_key,
            state.particles,
            field_names,
            n_components=n_components,
            n_iters=n_iters,
        )
        out = gmm_params_to_mcmc_dict(gmm)
        if mwg_update_fn is not None:
            scales = mwg_update_fn(rng_key, state, info)
            out = {**scales, **out}
        return out

    return mcmc_parameter_update_fn
