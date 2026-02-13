"""
JAX-based Tape parameterization for moment tensors.

This is a JAX port of tape.py, enabling JIT compilation and use with BlackJAX.
All functions use jax.numpy and are compatible with jax.jit and jax.vmap.
"""

from __future__ import annotations

# Enable float64 support in JAX (must be before importing jax.numpy)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial


@jax.jit
def jax_GD_E(gamma: float, delta: float) -> jnp.ndarray:
    """
    Convert Tape parameters (gamma, delta) to eigenvalues (E1, E2, E3).
    
    Parameters
    ----------
    gamma : float
        Source-type parameter in [-π/6, π/6]
    delta : float  
        Source-type parameter in [-π/2, π/2]
    
    Returns
    -------
    jnp.ndarray
        Shape (3,) eigenvalues
    """
    U = (1.0 / jnp.sqrt(6.0)) * jnp.array([
        [jnp.sqrt(3.0), 0.0, -jnp.sqrt(3.0)],
        [-1.0, 2.0, -1.0],
        [jnp.sqrt(2.0), jnp.sqrt(2.0), jnp.sqrt(2.0)],
    ], dtype=jnp.float64)
    
    phi = (jnp.pi / 2.0) - delta
    X = jnp.array([
        jnp.cos(gamma) * jnp.sin(phi),
        jnp.sin(gamma) * jnp.sin(phi),
        jnp.cos(phi),
    ], dtype=jnp.float64)
    
    return U.T @ X


@jax.jit
def jax_FP_TNP(normal: jnp.ndarray, slip: jnp.ndarray):
    """
    Convert fault normal and slip vectors to T, N, P axes.
    
    Returns
    -------
    tuple
        (T, N, P) axes, each shape (3,)
    """
    T = normal + slip
    T = T / jnp.linalg.norm(T)
    
    P = normal - slip
    P = P / jnp.linalg.norm(P)
    
    N = -jnp.cross(T, P)
    
    return T, N, P


@jax.jit
def jax_SDR_TNP(strike: float, dip: float, rake: float):
    """
    Convert strike, dip, rake (radians, N-E-D system) to T, N, P axes.
    
    Parameters
    ----------
    strike : float
        Strike angle in radians
    dip : float
        Dip angle in radians
    rake : float
        Rake angle in radians
    
    Returns
    -------
    tuple
        (T, N, P) axes, each shape (3,)
    """
    N1 = jnp.array([
        (jnp.cos(strike) * jnp.cos(rake))
        + (jnp.sin(strike) * jnp.cos(dip) * jnp.sin(rake)),
        (jnp.sin(strike) * jnp.cos(rake))
        - (jnp.cos(strike) * jnp.cos(dip) * jnp.sin(rake)),
        -jnp.sin(dip) * jnp.sin(rake),
    ], dtype=jnp.float64)
    
    N2 = jnp.array([
        -jnp.sin(strike) * jnp.sin(dip),
        jnp.cos(strike) * jnp.sin(dip),
        -jnp.cos(dip),
    ], dtype=jnp.float64)
    
    return jax_FP_TNP(N1, N2)


@jax.jit
def jax_MT33_MT6(MT33: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a 3x3 moment tensor to a normalized six-vector.
    
    Six-vector ordering:
        [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz]
    """
    mxx = MT33[0, 0]
    myy = MT33[1, 1]
    mzz = MT33[2, 2]
    mxy = MT33[0, 1]
    mxz = MT33[0, 2]
    myz = MT33[1, 2]
    
    mt6 = jnp.array([
        mxx,
        myy,
        mzz,
        jnp.sqrt(2.0) * mxy,
        jnp.sqrt(2.0) * mxz,
        jnp.sqrt(2.0) * myz,
    ], dtype=jnp.float64)
    
    norm = jnp.sqrt(jnp.sum(mt6 * mt6))
    # Avoid division by zero
    mt6_normalized = mt6 / jnp.maximum(norm, 1e-10)
    
    return mt6_normalized


@jax.jit
def jax_Tape_MT33(
    gamma: float,
    delta: float,
    kappa: float,
    h: float,
    sigma: float,
) -> jnp.ndarray:
    """
    Convert Tape parameters to a 3x3 moment tensor.
    
    Parameters
    ----------
    gamma : float
        Source-type parameter in [-π/6, π/6]
    delta : float
        Source-type parameter in [-π/2, π/2]
    kappa : float
        Strike angle in [0, 2π]
    h : float
        cos(dip) in [0, 1]
    sigma : float
        Rake angle in [-π/2, π/2]
    
    Returns
    -------
    jnp.ndarray
        Shape (3, 3) moment tensor
    """
    E = jax_GD_E(gamma, delta)
    D = jnp.diag(E)
    T, N, P = jax_SDR_TNP(kappa, jnp.arccos(h), sigma)
    
    # Stack T, N, P as columns: shape (3, 3)
    L = jnp.column_stack([T, N, P])
    
    return L @ D @ L.T


@jax.jit
def jax_Tape_MT6(
    gamma: float,
    delta: float,
    kappa: float,
    h: float,
    sigma: float,
) -> jnp.ndarray:
    """
    Convert Tape parameters to MT six-vector.
    
    Parameters
    ----------
    gamma, delta, kappa, h, sigma : float
        Tape parameters (scalar)
    
    Returns
    -------
    jnp.ndarray
        Shape (6,) normalized six-vector
    """
    MT33 = jax_Tape_MT33(gamma, delta, kappa, h, sigma)
    return jax_MT33_MT6(MT33)


# Vectorized version for multiple samples
jax_Tape_MT6_batch = jax.vmap(jax_Tape_MT6, in_axes=(0, 0, 0, 0, 0))
