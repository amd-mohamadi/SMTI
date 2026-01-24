"""
PyTensor version of forward modelling utilities for station ray-path coefficients.

Provides:
    - pt_station_angles
    - pt_forward_amplitude

These are designed to work with PyMC models and can be compiled by Nutpie/NumPyro.
The key difference from forward_model.py is that all operations use pytensor.tensor instead of numpy.
"""

from __future__ import annotations

from typing import Tuple, Union

import pytensor.tensor as pt
import numpy as np


def pt_station_angles(
    azimuth: pt.TensorVariable,
    takeoff_angle: pt.TensorVariable,
    phase: str,
    radians: bool = False,
) -> Union[pt.TensorVariable, Tuple[pt.TensorVariable, pt.TensorVariable]]:
    """
    Compute station MT coefficient vectors from azimuth and take-off angle (PyTensor version).

    Parameters
    ----------
    azimuth : pt.TensorVariable
        Station azimuths, shape (N,) in degrees (default) or radians if radians=True
    takeoff_angle : pt.TensorVariable
        Station takeoff angles, shape (N,) in degrees (default) or radians if radians=True
    phase : str
        'p', 'sh', 'sv', or ratio like 'p/sh'
    radians : bool
        Whether input angles are in radians (default: False, assumes degrees)

    Returns
    -------
    pt.TensorVariable or tuple of pt.TensorVariable
        Coefficients shape (N, 6) where columns are [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz]
        For ratio phases like 'p/sh', returns tuple of two arrays (numerator, denominator)

    Notes
    -----
    - TakeOffAngle is 0 down (positive z in NED system)
    - Input arrays must be 1D and same length
    """
    # Ensure inputs are PyTensor variables and convert from degrees to radians if needed
    az = pt.as_tensor_variable(azimuth, dtype='float64')
    to = pt.as_tensor_variable(takeoff_angle, dtype='float64')

    # Convert from degrees to radians if needed (only once)
    if not radians:
        az = az * pt.as_tensor(np.pi / 180.0, dtype='float64')
        to = to * pt.as_tensor(np.pi / 180.0, dtype='float64')

    phase_clean = phase.lower().rstrip("q")

    # Handle ratio phases (e.g., 'p/sh')
    # Note: This recursion pattern matches NumPy's behavior where the recursive
    # calls receive the ORIGINAL angles with radians=True, so they're treated
    # as radians without further conversion
    if "/" in phase_clean:
        num_phase, den_phase = phase_clean.split("/", 1)
        return (
            pt_station_angles(azimuth, takeoff_angle, num_phase, radians=True),
            pt_station_angles(azimuth, takeoff_angle, den_phase, radians=True),
        )

    # Single phase
    return _pt_station_angles_single(az, to, phase_clean)


def _pt_station_angles_single(
    az_radians: pt.TensorVariable,
    to_radians: pt.TensorVariable,
    phase: str,
) -> pt.TensorVariable:
    """
    Compute station coefficients for a single phase (assumes angles in radians).

    Parameters
    ----------
    az_radians : pt.TensorVariable
        Azimuths in radians
    to_radians : pt.TensorVariable
        Takeoff angles in radians
    phase : str
        'p', 'sh', or 'sv'

    Returns
    -------
    pt.TensorVariable
        Coefficients shape (N, 6)
    """
    az = az_radians
    to = to_radians
    phase_clean = phase.lower().rstrip("q")

    # Compute phase-specific coefficients
    if phase_clean == "p":
        mxx = pt.cos(az) * pt.cos(az) * pt.sin(to) * pt.sin(to)
        myy = pt.sin(az) * pt.sin(az) * pt.sin(to) * pt.sin(to)
        mzz = pt.cos(to) * pt.cos(to)
        mxy = pt.sqrt(2.0) * pt.sin(az) * pt.cos(az) * pt.sin(to) * pt.sin(to)
        mxz = pt.sqrt(2.0) * pt.cos(az) * pt.cos(to) * pt.sin(to)
        myz = pt.sqrt(2.0) * pt.sin(az) * pt.cos(to) * pt.sin(to)

    elif phase_clean == "sh":
        mxx = -pt.sin(az) * pt.cos(az) * pt.sin(to)
        myy = pt.sin(az) * pt.cos(az) * pt.sin(to)
        mzz = pt.zeros_like(az)
        mxy = (1.0 / pt.sqrt(2.0)) * pt.cos(2.0 * az) * pt.sin(to)
        mxz = -(1.0 / pt.sqrt(2.0)) * pt.sin(az) * pt.cos(to)
        myz = (1.0 / pt.sqrt(2.0)) * pt.cos(az) * pt.cos(to)

    elif phase_clean == "sv":
        mxx = pt.cos(az) * pt.cos(az) * pt.sin(to) * pt.cos(to)
        myy = pt.sin(az) * pt.sin(az) * pt.sin(to) * pt.cos(to)
        mzz = -pt.sin(to) * pt.cos(to)
        mxy = pt.sqrt(2.0) * pt.cos(az) * pt.sin(az) * pt.sin(to) * pt.cos(to)
        mxz = (1.0 / pt.sqrt(2.0)) * pt.cos(az) * pt.cos(2.0 * to)
        myz = (1.0 / pt.sqrt(2.0)) * pt.sin(az) * pt.cos(2.0 * to)

    else:
        raise ValueError(f"{phase} phase not recognised.")

    # Stack coefficients into (N, 6) array
    coeffs = pt.stack([mxx, myy, mzz, mxy, mxz, myz], axis=1)

    return coeffs


def pt_forward_amplitude(
    mt: pt.TensorVariable,
    azimuth: pt.TensorVariable,
    takeoff_angle: pt.TensorVariable,
    phase: str,
    radians: bool = False,
) -> pt.TensorVariable:
    """
    Predict forward-model amplitudes for a given MT and stations (PyTensor version).

    Parameters
    ----------
    mt : pt.TensorVariable
        Moment tensor six-vector, shape (6,)
    azimuth : pt.TensorVariable
        Station azimuths, shape (N,)
    takeoff_angle : pt.TensorVariable
        Station takeoff angles, shape (N,)
    phase : str
        'P', 'SH', 'SV' or ratio like 'P/SH'
    radians : bool
        Whether station angles are in radians

    Returns
    -------
    pt.TensorVariable
        Amplitudes shape (N,) for single phase, or ratio of amplitudes
    """
    A = pt_station_angles(azimuth, takeoff_angle, phase, radians=radians)

    mt_arr = pt.as_tensor_variable(mt, dtype='float64')
    # Ensure mt is shape (6,)
    if mt_arr.ndim != 1:
        raise ValueError(f"mt must be 1D, got shape {mt_arr.shape}")

    # Handle ratio phases
    if isinstance(A, tuple):
        A_num = A[0]  # shape (N, 6)
        A_den = A[1]  # shape (N, 6)
        num = pt.dot(A_num, mt_arr)  # shape (N,)
        den = pt.dot(A_den, mt_arr)  # shape (N,)
        # Compute ratio with safety
        out = num / pt.maximum(pt.abs(den), 1e-10) * pt.sign(den)
        return out

    # Single phase
    out = pt.dot(A, mt_arr)  # A is (N, 6), mt is (6,) -> result is (N,)
    return out
