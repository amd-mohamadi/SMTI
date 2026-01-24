"""
PyTensor version of Tape & Tape parameterisation utilities for moment tensors.

This module provides PyTensor-based implementations of:
    - pt_MT33_MT6
    - pt_GD_E
    - pt_Tape_MT33
    - pt_Tape_MT6

These are designed to work with PyMC models and can be compiled by Nutpie/NumPyro.
The key difference from tape.py is that all operations use pytensor.tensor instead of numpy.
"""

from __future__ import annotations

from typing import Tuple, Union

import pytensor
import pytensor.tensor as pt
from pytensor import scan
import numpy as np


def pt_MT33_MT6(MT33: pt.TensorVariable) -> pt.TensorVariable:
    """
    Convert a 3x3 moment tensor to a normalised six-vector (PyTensor version).

    Parameters
    ----------
    MT33 : pt.TensorVariable
        3x3 moment tensor

    Returns
    -------
    pt.TensorVariable
        MT 6-vector [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz]
    """
    mxx = MT33[0, 0]
    myy = MT33[1, 1]
    mzz = MT33[2, 2]
    mxy = MT33[0, 1]
    mxz = MT33[0, 2]
    myz = MT33[1, 2]

    mt6 = pt.stack([
        mxx,
        myy,
        mzz,
        pt.sqrt(2.0) * mxy,
        pt.sqrt(2.0) * mxz,
        pt.sqrt(2.0) * myz,
    ])

    # Normalize
    norm = pt.sqrt(pt.sum(mt6 * mt6))
    # Avoid division by zero
    mt6_normalized = mt6 / pt.maximum(norm, 1e-10)

    return mt6_normalized


def pt_GD_E(gamma: pt.TensorVariable, delta: pt.TensorVariable) -> pt.TensorVariable:
    """
    Convert Tape parameters (gamma, delta) to eigenvalues (E1, E2, E3) (PyTensor version).

    Parameters
    ----------
    gamma : pt.TensorVariable
        Tape parameter gamma
    delta : pt.TensorVariable
        Tape parameter delta

    Returns
    -------
    pt.TensorVariable
        Shape (3,) eigenvalues
    """
    # Build U matrix using numpy (constant), then convert
    U_const = (1.0 / np.sqrt(6.0)) * np.array([
        [np.sqrt(3.0), 0.0, -np.sqrt(3.0)],
        [-1.0, 2.0, -1.0],
        [np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
    ], dtype='float64')
    U = pt.as_tensor_variable(U_const)

    phi = (pt.pi / 2.0) - delta
    X = pt.stack([
        pt.cos(gamma) * pt.sin(phi),
        pt.sin(gamma) * pt.sin(phi),
        pt.cos(phi),
    ])

    return pt.dot(U.T, X)


def pt_FP_TNP(
    normal: pt.TensorVariable, slip: pt.TensorVariable
) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """
    Convert fault normal and slip vectors to T, N, P axes (PyTensor version).

    Parameters
    ----------
    normal : pt.TensorVariable
        Shape (3,) normal vector
    slip : pt.TensorVariable
        Shape (3,) slip vector

    Returns
    -------
    tuple of pt.TensorVariable
        (T, N, P) axes, each shape (3,)
    """
    T = normal + slip
    T = T / pt.sqrt(pt.sum(T * T))

    P = normal - slip
    P = P / pt.sqrt(pt.sum(P * P))

    # Cross product: N = -T × P
    N = -pt.stack([
        T[1] * P[2] - T[2] * P[1],
        T[2] * P[0] - T[0] * P[2],
        T[0] * P[1] - T[1] * P[0],
    ])

    return T, N, P


def pt_SDR_TNP(
    strike: pt.TensorVariable, dip: pt.TensorVariable, rake: pt.TensorVariable
) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """
    Convert strike, dip, rake (radians, N-E-D system) to T, N, P axes (PyTensor version).

    Parameters
    ----------
    strike : pt.TensorVariable
        Strike angle in radians
    dip : pt.TensorVariable
        Dip angle in radians
    rake : pt.TensorVariable
        Rake angle in radians

    Returns
    -------
    tuple of pt.TensorVariable
        (T, N, P) axes, each shape (3,)
    """
    N1 = pt.stack([
        (pt.cos(strike) * pt.cos(rake))
        + (pt.sin(strike) * pt.cos(dip) * pt.sin(rake)),
        (pt.sin(strike) * pt.cos(rake))
        - (pt.cos(strike) * pt.cos(dip) * pt.sin(rake)),
        -pt.sin(dip) * pt.sin(rake),
    ])

    N2 = pt.stack([
        -pt.sin(strike) * pt.sin(dip),
        pt.cos(strike) * pt.sin(dip),
        -pt.cos(dip),
    ])

    return pt_FP_TNP(N1, N2)


def pt_Tape_MT33(
    gamma: pt.TensorVariable,
    delta: pt.TensorVariable,
    kappa: pt.TensorVariable,
    h: pt.TensorVariable,
    sigma: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Convert Tape parameters to a 3x3 moment tensor (PyTensor version).

    Parameters
    ----------
    gamma, delta, kappa, h, sigma : pt.TensorVariable
        Tape parameters

    Returns
    -------
    pt.TensorVariable
        Shape (3, 3) moment tensor
    """
    E = pt_GD_E(gamma, delta)
    D = pt.diag(E)
    T, N, P = pt_SDR_TNP(kappa, pt.arccos(h), sigma)

    # Stack T, N, P as columns: shape (3, 3)
    L = pt.stack([T, N, P], axis=1)

    return pt.dot(pt.dot(L, D), L.T)


def pt_Tape_MT6(
    gamma: pt.TensorVariable,
    delta: pt.TensorVariable,
    kappa: pt.TensorVariable,
    h: pt.TensorVariable,
    sigma: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Convert Tape parameters to MT six-vector (PyTensor version).

    This version works primarily with scalar inputs (standard for PyMC sampling).
    For multiple samples, use this function in a loop or with pytensor.scan.

    Parameters
    ----------
    gamma, delta, kappa, h, sigma : pt.TensorVariable
        Tape parameters (typically scalars in PyMC context)

    Returns
    -------
    pt.TensorVariable
        Shape (6,) six-vector
    """
    # Convert Tape parameters to 3x3 moment tensor
    MT33 = pt_Tape_MT33(gamma, delta, kappa, h, sigma)

    # Convert 3x3 to 6-vector
    mt6 = pt_MT33_MT6(MT33)

    return mt6
