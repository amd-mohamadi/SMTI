"""Unit tests for the diagonal-GMM independence MH kernel (no full SMC)."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from src.irmh_kernel import (
    DiagGMM,
    build_hybrid_mwg_irmh_step_fn,
    build_irmh_step_fn,
    fit_diag_gmm,
    gmm_log_prob,
    gmm_params_to_mcmc_dict,
    pack_particles,
    sample_gmm,
    unpack_vector,
)
from src.mwg_kernel import default_mwg_blocks, default_inner_steps
from src.blockwise_rmh import RMHBlock
import blackjax.mcmc.random_walk as blackjax_rw


def test_fit_diag_gmm_two_blobs():
    key = random.PRNGKey(0)
    n = 400
    k1, k2 = random.split(key)
    a = random.normal(k1, (n // 2, 2)) + jnp.array([-3.0, 0.0])
    b = random.normal(k2, (n // 2, 2)) + jnp.array([3.0, 0.0])
    X = jnp.concatenate([a, b], axis=0)

    gmm = fit_diag_gmm(random.PRNGKey(1), X, n_components=2, n_iters=15)
    means = np.asarray(gmm.means)
    # Sorted by first coordinate
    order = np.argsort(means[:, 0])
    means = means[order]
    assert means[0, 0] < -1.5, means
    assert means[1, 0] > 1.5, means
    # Weights roughly balanced
    w = np.exp(np.asarray(gmm.log_weights))[order]
    assert 0.3 < w[0] < 0.7 and 0.3 < w[1] < 0.7, w


def test_gmm_log_prob_ranks_correctly():
    gmm = DiagGMM(
        log_weights=jnp.log(jnp.array([0.5, 0.5])),
        means=jnp.array([[0.0, 0.0], [5.0, 0.0]]),
        log_stds=jnp.zeros((2, 2)),
    )
    lp0 = float(gmm_log_prob(jnp.array([0.0, 0.0]), gmm))
    lp5 = float(gmm_log_prob(jnp.array([5.0, 0.0]), gmm))
    lp_mid = float(gmm_log_prob(jnp.array([2.5, 0.0]), gmm))
    assert lp0 > lp_mid
    assert lp5 > lp_mid


def test_sample_and_pack_roundtrip():
    fields = ("gamma", "delta", "kappa")
    particles = {
        "gamma": jnp.linspace(-1, 1, 10),
        "delta": jnp.linspace(-2, 2, 10),
        "kappa": jnp.linspace(0, 3, 10),
    }
    X = pack_particles(particles, fields)
    assert X.shape == (10, 3)
    template = {k: particles[k][0] for k in fields}
    pos = unpack_vector(X[3], fields, template)
    assert float(pos["gamma"]) == float(particles["gamma"][3])


def test_irmh_step_jittable_and_moves():
    fields = ("x", "y")
    # Target: mixture of two well-separated Gaussians (unnormalized)
    def logdensity_fn(position):
        p = jnp.stack([position["x"], position["y"]])
        m1 = jnp.array([-2.0, 0.0])
        m2 = jnp.array([2.0, 0.0])
        lp1 = -0.5 * jnp.sum((p - m1) ** 2)
        lp2 = -0.5 * jnp.sum((p - m2) ** 2)
        return jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]))

    # Proposal GMM matches the target modes
    gmm = DiagGMM(
        log_weights=jnp.log(jnp.array([0.5, 0.5])),
        means=jnp.array([[-2.0, 0.0], [2.0, 0.0]]),
        log_stds=jnp.zeros((2, 2)),
    )
    params = gmm_params_to_mcmc_dict(gmm)
    # Squeeze leading dim for direct step (SMC adds it via extend_params;
    # our step_fn reshapes anyway)
    step = build_irmh_step_fn(fields, n_components=2, n_irmh_steps=5)
    init = blackjax_rw.init({"x": jnp.array(-2.0), "y": jnp.array(0.0)}, logdensity_fn)

    @jax.jit
    def run(key, st):
        return step(key, st, logdensity_fn, **params)

    st2, info = run(random.PRNGKey(0), init)
    assert jnp.isfinite(st2.logdensity)
    acc = float(info.acceptance_rate)
    assert 0.0 <= acc <= 1.0
    # Over many steps from left mode, should sometimes land near right mode
    st = init
    landed_right = False
    key = random.PRNGKey(42)
    for i in range(40):
        key, sub = random.split(key)
        st, _ = run(sub, st)
        if float(st.position["x"]) > 0.5:
            landed_right = True
            break
    assert landed_right, f"IRMH never jumped; last x={float(st.position['x'])}"


def test_irmh_asymmetric_proposal_unbiased():
    """Regression: the IRMH proposal correction must cancel proposal asymmetry.

    Symmetric 50/50 bimodal target sampled with a deliberately *asymmetric*
    (90/10) GMM proposal. With the correct acceptance ratio
    ``[p(x') q(x)] / [p(x) q(x')]`` the long-run mode occupancy is ~0.5/0.5;
    with a missing or sign-flipped ``proposal_logdensity_fn`` it sits near
    0.9 or 0.1. (The mode-jump test alone cannot detect that failure.)
    """
    fields = ("x", "y")

    def logdensity_fn(position):
        p = jnp.stack([position["x"], position["y"]])
        m1 = jnp.array([-2.0, 0.0])
        m2 = jnp.array([2.0, 0.0])
        lp1 = -0.5 * jnp.sum((p - m1) ** 2)
        lp2 = -0.5 * jnp.sum((p - m2) ** 2)
        return jax.scipy.special.logsumexp(jnp.stack([lp1, lp2]))

    gmm = DiagGMM(
        log_weights=jnp.log(jnp.array([0.9, 0.1])),
        means=jnp.array([[-2.0, 0.0], [2.0, 0.0]]),
        log_stds=jnp.zeros((2, 2)),
    )
    params = gmm_params_to_mcmc_dict(gmm)
    step = build_irmh_step_fn(fields, n_components=2, n_irmh_steps=1)
    init = blackjax_rw.init({"x": jnp.array(-2.0), "y": jnp.array(0.0)}, logdensity_fn)

    def scan_body(st, key):
        st, _ = step(key, st, logdensity_fn, **params)
        return st, st.position["x"]

    keys = random.split(random.PRNGKey(7), 8000)
    _, xs = jax.jit(lambda s, k: jax.lax.scan(scan_body, s, k))(init, keys)
    frac_right = float(np.mean(np.asarray(xs[1000:]) > 0.0))
    assert 0.42 < frac_right < 0.58, (
        f"right-mode occupancy {frac_right:.3f}; proposal correction broken"
    )


def test_irmh_info_realized_acceptance():
    """acceptance_rate must be the realized accept fraction (k / n_steps)."""
    fields = ("x", "y")

    def logdensity_fn(position):
        return -0.5 * (position["x"] ** 2 + position["y"] ** 2)

    gmm = DiagGMM(
        log_weights=jnp.log(jnp.array([0.5, 0.5])),
        means=jnp.zeros((2, 2)),
        log_stds=jnp.zeros((2, 2)),
    )
    params = gmm_params_to_mcmc_dict(gmm)
    n_steps = 5
    step = build_irmh_step_fn(fields, n_components=2, n_irmh_steps=n_steps)
    init = blackjax_rw.init({"x": jnp.array(0.0), "y": jnp.array(0.0)}, logdensity_fn)
    _, info = jax.jit(lambda k, s: step(k, s, logdensity_fn, **params))(
        random.PRNGKey(3), init
    )
    acc = float(info.acceptance_rate)
    assert 0.0 <= acc <= 1.0
    # Realized fraction over n_steps indicators is a multiple of 1/n_steps;
    # a mean of continuous accept *probabilities* almost surely is not.
    assert abs(acc * n_steps - round(acc * n_steps)) < 1e-6, acc
    assert bool(info.is_accepted) == (acc > 0.0)


def test_hybrid_step_composes():
    fields = ("gamma", "delta", "kappa", "h", "sigma")
    blocks = (
        RMHBlock("source_type", ("gamma", "delta")),
        RMHBlock("mechanism", ("kappa", "h", "sigma")),
    )
    inner = default_inner_steps(mechanism_steps=1)
    step = build_hybrid_mwg_irmh_step_fn(
        field_names=fields,
        blocks=blocks,
        inner_steps_per_block=inner,
        n_components=2,
        n_irmh_steps=1,
    )

    def logdensity_fn(position):
        # Mildly peaked isotropic prior-like density in unconstrained space
        return -0.5 * sum(jnp.square(position[n]) for n in fields)

    gmm = DiagGMM(
        log_weights=jnp.log(jnp.array([0.5, 0.5])),
        means=jnp.zeros((2, len(fields))),
        log_stds=jnp.zeros((2, len(fields))),
    )
    params = gmm_params_to_mcmc_dict(gmm)
    # Block scales
    params["source_type"] = jnp.array(0.5)[None]
    params["mechanism"] = jnp.array(0.5)[None]

    pos = {n: jnp.array(0.1) for n in fields}
    st = blackjax_rw.init(pos, logdensity_fn)
    st2, info = jax.jit(lambda k, s: step(k, s, logdensity_fn, **params))(
        random.PRNGKey(0), st
    )
    assert jnp.isfinite(st2.logdensity)
    assert info.block_acceptance_rates.shape[0] == 2
    assert 0.0 <= float(info.irmh_acceptance_rate) <= 1.0


if __name__ == "__main__":
    test_fit_diag_gmm_two_blobs()
    print("fit OK")
    test_gmm_log_prob_ranks_correctly()
    print("log_prob OK")
    test_sample_and_pack_roundtrip()
    print("pack OK")
    test_irmh_step_jittable_and_moves()
    print("irmh OK")
    test_irmh_asymmetric_proposal_unbiased()
    print("asymmetric-proposal occupancy OK")
    test_irmh_info_realized_acceptance()
    print("realized acceptance OK")
    test_hybrid_step_composes()
    print("hybrid OK")
    print("ALL PASS")
