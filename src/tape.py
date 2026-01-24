"""
Tape & Tape parameterisation utilities for moment tensors.

Provides:
    - MT33_MT6
    - GD_E
    - Tape_MT33
    - Tape_MT6
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def MT33_MT6(MT33: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 moment tensor to a normalised six-vector.

    Six-vector ordering:
        [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz]
    """
    MT33 = np.asarray(MT33, dtype=float)
    if MT33.shape != (3, 3):
        raise ValueError(f"MT33 must be 3x3, got shape {MT33.shape}")

    mxx = MT33[0, 0]
    myy = MT33[1, 1]
    mzz = MT33[2, 2]
    mxy = MT33[0, 1]
    mxz = MT33[0, 2]
    myz = MT33[1, 2]

    mt6 = np.array(
        [mxx, myy, mzz, np.sqrt(2.0) * mxy, np.sqrt(2.0) * mxz, np.sqrt(2.0) * myz],
        dtype=float,
    )
    norm = np.sqrt(np.sum(mt6 * mt6))
    if norm == 0:
        return mt6
    return mt6 / norm


def GD_E(gamma: float, delta: float) -> np.ndarray:
    """
    Convert Tape parameters (gamma, delta) to eigenvalues (E1, E2, E3).
    """
    U = (1.0 / np.sqrt(6.0)) * np.array(
        [
            [np.sqrt(3.0), 0.0, -np.sqrt(3.0)],
            [-1.0, 2.0, -1.0],
            [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
        ],
        dtype=float,
    )
    gamma = float(gamma)
    delta = float(delta)
    phi = (np.pi / 2.0) - delta
    X = np.array(
        [
            np.cos(gamma) * np.sin(phi),
            np.sin(gamma) * np.sin(phi),
            np.cos(phi),
        ],
        dtype=float,
    )
    return U.T @ X


def FP_TNP(normal: np.ndarray, slip: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert fault normal and slip vectors to T, N, P axes.
    """
    normal = np.asarray(normal, dtype=float).reshape(3,)
    slip = np.asarray(slip, dtype=float).reshape(3,)

    T = normal + slip
    T = T / np.linalg.norm(T)
    P = normal - slip
    P = P / np.linalg.norm(P)
    N = -np.cross(T, P)
    return T, N, P


def SDR_TNP(strike: float, dip: float, rake: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert strike, dip, rake (radians, N-E-D system) to T, N, P axes.
    """
    strike = float(strike)
    dip = float(dip)
    rake = float(rake)

    N1 = np.array(
        [
            (np.cos(strike) * np.cos(rake))
            + (np.sin(strike) * np.cos(dip) * np.sin(rake)),
            (np.sin(strike) * np.cos(rake))
            - (np.cos(strike) * np.cos(dip) * np.sin(rake)),
            -np.sin(dip) * np.sin(rake),
        ],
        dtype=float,
    )
    N2 = np.array(
        [
            -np.sin(strike) * np.sin(dip),
            np.cos(strike) * np.sin(dip),
            -np.cos(dip),
        ],
        dtype=float,
    )
    return FP_TNP(N1, N2)


def Tape_MT33(
    gamma: float,
    delta: float,
    kappa: float,
    h: float,
    sigma: float,
) -> np.ndarray:
    """
    Convert Tape parameters to a 3x3 moment tensor (MT33).
    """
    E = GD_E(gamma, delta)
    D = np.diag(E)
    T, N, P = SDR_TNP(kappa, np.arccos(h), sigma)
    L = np.column_stack([T, N, P])
    return L @ D @ L.T


def Tape_MT6(
    gamma: ArrayLike,
    delta: ArrayLike,
    kappa: ArrayLike,
    h: ArrayLike,
    sigma: ArrayLike,
) -> np.ndarray:
    """
    Convert Tape parameters (gamma, delta, kappa, h=cos(dip), sigma) to MT six-vectors.

    Parameters
    ----------
    gamma, delta, kappa, h, sigma
        Scalars or 1D arrays of the same length.

    Returns
    -------
    np.ndarray
        MT 6-vectors. Shape (6,) for scalar inputs, or (6, N) for N samples.
    """
    g = np.asarray(gamma, dtype=float)
    d = np.asarray(delta, dtype=float)
    k = np.asarray(kappa, dtype=float)
    h = np.asarray(h, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if g.ndim == 0:
        MT33 = Tape_MT33(g, d, k, h, s)
        return MT33_MT6(MT33)

    g_flat = g.ravel()
    d_flat = d.ravel()
    k_flat = k.ravel()
    h_flat = h.ravel()
    s_flat = s.ravel()
    n = g_flat.size

    MT6 = np.empty((6, n), dtype=float)
    for i in range(n):
        MT33 = Tape_MT33(g_flat[i], d_flat[i], k_flat[i], h_flat[i], s_flat[i])
        MT6[:, i] = MT33_MT6(MT33)
    return MT6


