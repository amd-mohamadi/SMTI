"""
Forward modelling utilities for station ray-path coefficients.

Provides:
    - station_angles
    - forward_amplitude

These mirror the behaviour of MTfit.forward_model but use NumPy ndarrays only.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def _as_column(a: Any) -> np.ndarray:
    """
    Convert input to a 2D float array of shape (N, 1).
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def station_angles(
    stations: Union[Dict[str, Any], Tuple[ArrayLike, ArrayLike]],
    phase: str,
    radians: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute station MT coefficient vectors from azimuth and take-off angle.

    Notes
    -----
    - TakeOffAngle is 0 down (positive z in NED system).
    - Returns an array of shape (N, 6) where N is number of stations and
      columns correspond to [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz].
    - If `phase` is a ratio (e.g., 'P/SH'), returns a tuple of two arrays
      (numerator, denominator), each of shape (N, 6).
    """
    # Get azimuth and takeoff angles
    if isinstance(stations, dict):
        azimuth = _as_column(stations["Azimuth"])
        takeoff_angle = _as_column(stations["TakeOffAngle"])
    else:
        azimuth = _as_column(stations[0])
        takeoff_angle = _as_column(stations[1])

    # Radian conversion if not radians
    if not radians:
        azimuth = azimuth * np.pi / 180.0
        takeoff_angle = takeoff_angle * np.pi / 180.0

    phase_clean = phase.lower().rstrip("q")

    # Handle ratio phases (e.g., 'p/sh')
    if "/" in phase_clean:
        num_phase, den_phase = phase_clean.split("/", 1)
        return (
            station_angles(stations, num_phase, radians=True),
            station_angles(stations, den_phase, radians=True),
        )

    az = azimuth
    to = takeoff_angle

    if phase_clean == "p":
        cols = [
            np.cos(az) * np.cos(az) * np.sin(to) * np.sin(to),  # Mxx
            np.sin(az) * np.sin(az) * np.sin(to) * np.sin(to),  # Myy
            np.cos(to) * np.cos(to),  # Mzz
            np.sqrt(2.0) * np.sin(az) * np.cos(az) * np.sin(to) * np.sin(to),  # sqrt(2)*Mxy
            np.sqrt(2.0) * np.cos(az) * np.cos(to) * np.sin(to),  # sqrt(2)*Mxz
            np.sqrt(2.0) * np.sin(az) * np.cos(to) * np.sin(to),  # sqrt(2)*Myz
        ]
    elif phase_clean == "sh":
        cols = [
            -np.sin(az) * np.cos(az) * np.sin(to),  # Mxx
            np.sin(az) * np.cos(az) * np.sin(to),  # Myy
            np.zeros_like(az),  # Mzz
            (1.0 / np.sqrt(2.0)) * np.cos(2.0 * az) * np.sin(to),  # sqrt(2)*Mxy
            -(1.0 / np.sqrt(2.0)) * np.sin(az) * np.cos(to),  # sqrt(2)*Mxz
            (1.0 / np.sqrt(2.0)) * np.cos(az) * np.cos(to),  # sqrt(2)*Myz
        ]
    elif phase_clean == "sv":
        cols = [
            np.cos(az) * np.cos(az) * np.sin(to) * np.cos(to),  # Mxx
            np.sin(az) * np.sin(az) * np.sin(to) * np.cos(to),  # Myy
            -np.sin(to) * np.cos(to),  # Mzz
            np.sqrt(2.0) * np.cos(az) * np.sin(az) * np.sin(to) * np.cos(to),  # sqrt(2)*Mxy
            (1.0 / np.sqrt(2.0)) * np.cos(az) * np.cos(2.0 * to),  # sqrt(2)*Mxz
            (1.0 / np.sqrt(2.0)) * np.sin(az) * np.cos(2.0 * to),  # sqrt(2)*Myz
        ]
    else:
        raise ValueError(f"{phase} phase not recognised.")

    return np.column_stack(cols).astype(float)


def forward_amplitude(
    mt: ArrayLike,
    stations: Union[Dict[str, Any], Tuple[ArrayLike, ArrayLike]],
    phase: str,
    radians: bool = False,
) -> np.ndarray:
    """
    Predict forward-model amplitudes for a given MT and stations.

    Parameters
    ----------
    mt
        Moment tensor six-vector(s) [Mxx, Myy, Mzz, sqrt(2)Mxy, sqrt(2)Mxz, sqrt(2)Myz].
        Shape (6,), (M, 6) or (6, M).
    stations
        Station dict or (azimuth, takeoff) tuple as for `station_angles`.
    phase
        'P', 'SH', 'SV' or ratio like 'P/SH'.
    radians
        Whether station angles are in radians.

    Returns
    -------
    np.ndarray
        If `mt` is (6,), returns shape (N,) for N stations.
        If `mt` is (M, 6) or (6, M), returns shape (N, M).
        For ratio phases, returns amplitude ratio with same shape.
    """
    A = station_angles(stations, phase, radians=radians)

    mt_arr = np.asarray(mt, dtype=float)
    if mt_arr.ndim == 1:
        mt_arr = mt_arr.reshape(6, 1)
    elif mt_arr.ndim == 2:
        if mt_arr.shape[1] == 6:
            mt_arr = mt_arr.T
        elif mt_arr.shape[0] == 6:
            pass
        else:
            raise ValueError(f"mt must have 6 components, got shape {mt_arr.shape}")
    else:
        raise ValueError("mt must be 1D or 2D array-like of six-vectors")

    if isinstance(A, tuple):
        A_num = np.asarray(A[0], dtype=float)
        A_den = np.asarray(A[1], dtype=float)
        num = A_num @ mt_arr
        den = A_den @ mt_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        return np.squeeze(out)

    A_arr = np.asarray(A, dtype=float)
    out = A_arr @ mt_arr
    return np.squeeze(out)


