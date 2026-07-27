"""
Dtype-agnostic JAX Tape -> MT6 conversion.

``src/tape_jax.py`` force-enables ``jax_enable_x64`` at import time and pins
every array literal to ``jnp.float64``.  That is correct for the production
float64 path but makes a float32 sampler impossible: importing it flips x64 on
process-wide, and the hardcoded literals upcast float32 particles back to
float64 inside the likelihood.

This module is a faithful copy of ``src.tape_jax.jax_Tape_MT6`` whose constant
arrays follow the dtype of the *inputs* instead.  With float64 inputs it is
bit-identical to the original (verified in
``tests_batched/test_batched_core.py``); with float32 inputs it stays in
float32.  It has no import-time side effects, so an fp32 process can use it
without ever pulling in ``src.tape_jax``.

Only ``jax_Tape_MT6`` is reproduced -- it is the single tape entry point the
BlackJAX likelihood uses.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


_SQRT2 = math.sqrt(2.0)
_SQRT3 = math.sqrt(3.0)
_INV_SQRT6 = 1.0 / math.sqrt(6.0)


def jax_Tape_MT6(gamma, delta, kappa, h, sigma):
    """Tape (gamma, delta, kappa, h, sigma) -> normalised MT six-vector.

    Same math as ``src.tape_jax.jax_Tape_MT6`` (``jax_GD_E`` -> ``jax_SDR_TNP``
    -> ``jax_FP_TNP`` -> ``jax_Tape_MT33`` -> ``jax_MT33_MT6``), inlined so the
    intermediate literals can adopt the caller's dtype.

    Parameters are scalars (or broadcastable arrays); the returned array has
    shape ``(6,)`` and the promoted dtype of the inputs.
    """
    dt = jnp.result_type(gamma, delta, kappa, h, sigma)

    # --- jax_GD_E ---
    U = _INV_SQRT6 * jnp.asarray(
        [
            [_SQRT3, 0.0, -_SQRT3],
            [-1.0, 2.0, -1.0],
            [_SQRT2, _SQRT2, _SQRT2],
        ],
        dtype=dt,
    )
    phi = (jnp.pi / 2.0) - delta
    X = jnp.asarray(
        [
            jnp.cos(gamma) * jnp.sin(phi),
            jnp.sin(gamma) * jnp.sin(phi),
            jnp.cos(phi),
        ],
        dtype=dt,
    )
    E = U.T @ X

    # --- jax_SDR_TNP(kappa, arccos(h), sigma) ---
    strike, dip, rake = kappa, jnp.arccos(h), sigma
    N1 = jnp.asarray(
        [
            (jnp.cos(strike) * jnp.cos(rake))
            + (jnp.sin(strike) * jnp.cos(dip) * jnp.sin(rake)),
            (jnp.sin(strike) * jnp.cos(rake))
            - (jnp.cos(strike) * jnp.cos(dip) * jnp.sin(rake)),
            -jnp.sin(dip) * jnp.sin(rake),
        ],
        dtype=dt,
    )
    N2 = jnp.asarray(
        [
            -jnp.sin(strike) * jnp.sin(dip),
            jnp.cos(strike) * jnp.sin(dip),
            -jnp.cos(dip),
        ],
        dtype=dt,
    )

    # --- jax_FP_TNP ---
    T = N1 + N2
    T = T / jnp.linalg.norm(T)
    P = N1 - N2
    P = P / jnp.linalg.norm(P)
    N = -jnp.cross(T, P)

    # --- jax_Tape_MT33 ---
    D = jnp.diag(E)
    L = jnp.column_stack([T, N, P])
    MT33 = L @ D @ L.T

    # --- jax_MT33_MT6 ---
    mt6 = jnp.asarray(
        [
            MT33[0, 0],
            MT33[1, 1],
            MT33[2, 2],
            _SQRT2 * MT33[0, 1],
            _SQRT2 * MT33[0, 2],
            _SQRT2 * MT33[1, 2],
        ],
        dtype=dt,
    )
    norm = jnp.sqrt(jnp.sum(mt6 * mt6))
    return mt6 / jnp.maximum(norm, 1e-10)
