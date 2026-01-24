"""
Data preparation utilities for SMC-based MT inversion.

Provides:
    - polarity_matrix
    - amplitude_matrix
    - amplitude_ratio_matrix

These mirror the corresponding helpers in MTfit.inversion but use ndarrays.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .forward_model import station_angles


def build_location_samples_from_errors(
    data: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    n_samples: int = 20,
    azimuth_error: Optional[float] = None,
    takeoff_error: Optional[float] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Construct MTfit-style ``location_samples`` from per-station azimuth and
    take-off angle errors, if present.

    The function scans all entries in an event dictionary for ``"Stations"``
    sub-dictionaries with ``"Azimuth"`` and ``"TakeOffAngle"`` (degrees).

    Uncertainties can be supplied in two ways:

    1. Globally, via the ``azimuth_error`` and ``takeoff_error`` keyword
       arguments (1σ in degrees). When provided, the same uncertainties
       are applied to all stations.
    2. Per station set, via optional ``"Azimuth_err"`` and ``"TakeOff_err"``
       arrays in each ``"Stations"`` dict (legacy behaviour).

    If no non-zero uncertainties are found (globally or per station), the
    function returns ``None`` and downstream code should treat locations as
    exact.

    If no non-zero errors are found in the event, ``None`` is returned and
    downstream code can behave as if locations are perfect.

    Parameters
    ----------
    data
        Event dictionary (typically already filtered by inversion options).
    rng
        NumPy random number generator used for sampling. If ``None``, a new
        :func:`numpy.random.default_rng` instance is created.
    n_samples
        Number of location samples to draw per station. If ``n_samples <= 0``,
        ``None`` is returned.

    Returns
    -------
    list of dict or None
        A ``location_samples`` list compatible with :func:`polarity_matrix`
        and :func:`amplitude_ratio_matrix`, or ``None`` if no usable errors
        are present.
    """
    if n_samples <= 0:
        return None

    if rng is None:
        rng = np.random.default_rng()

    station_info: Dict[str, Dict[str, float]] = {}
    any_nonzero = False

    for key, value in data.items():
        if not isinstance(value, dict):
            continue

        stations = value.get("Stations")
        if not isinstance(stations, dict):
            continue

        names = stations.get("Name")
        if names is None:
            continue

        az = np.asarray(stations["Azimuth"], dtype=float).reshape(-1)
        to = np.asarray(stations["TakeOffAngle"], dtype=float).reshape(-1)

        if len(names) != az.shape[0] or len(names) != to.shape[0]:
            # Skip malformed station sets rather than failing the whole event.
            continue

        # Prefer global errors if provided; otherwise fall back to per-station
        # Azimuth_err/TakeOff_err fields for backwards compatibility.
        if azimuth_error is not None or takeoff_error is not None:
            az_e = float(azimuth_error) if azimuth_error is not None else 0.0
            to_e = float(takeoff_error) if takeoff_error is not None else 0.0
            az_err = np.full_like(az, az_e, dtype=float)
            to_err = np.full_like(to, to_e, dtype=float)
            if np.any(az_err != 0.0) or np.any(to_err != 0.0):
                any_nonzero = True
        else:
            has_az_err = "Azimuth_err" in stations
            has_to_err = "TakeOff_err" in stations

            if has_az_err and has_to_err:
                az_err = np.asarray(stations["Azimuth_err"], dtype=float).reshape(-1)
                to_err = np.asarray(stations["TakeOff_err"], dtype=float).reshape(-1)

                if az_err.shape[0] != az.shape[0] or to_err.shape[0] != to.shape[0]:
                    # Mismatched error shapes: treat as no uncertainty for this set.
                    az_err = np.zeros_like(az)
                    to_err = np.zeros_like(to)
                else:
                    if np.any(az_err != 0.0) or np.any(to_err != 0.0):
                        any_nonzero = True
            else:
                # If either error array is missing, treat both angles as exact.
                az_err = np.zeros_like(az)
                to_err = np.zeros_like(to)

        for name, az0, to0, az_e, to_e in zip(names, az, to, az_err, to_err):
            if name not in station_info:
                station_info[name] = {
                    "az": float(az0),
                    "to": float(to0),
                    "az_err": float(az_e),
                    "to_err": float(to_e),
                }
            else:
                info = station_info[name]
                # Keep mean from first occurrence; accumulate maximum error.
                info["az_err"] = max(info["az_err"], float(az_e))
                info["to_err"] = max(info["to_err"], float(to_e))
                if az_e != 0.0 or to_e != 0.0:
                    any_nonzero = True

    if not station_info or not any_nonzero:
        return None

    # Build base arrays (sorted by station name for deterministic ordering)
    station_names = sorted(station_info.keys())
    n_sta = len(station_names)
    az_mu = np.empty(n_sta, dtype=float)
    to_mu = np.empty(n_sta, dtype=float)
    az_sig = np.empty(n_sta, dtype=float)
    to_sig = np.empty(n_sta, dtype=float)

    for i, name in enumerate(station_names):
        info = station_info[name]
        az_mu[i] = info["az"]
        to_mu[i] = info["to"]
        az_sig[i] = info["az_err"]
        to_sig[i] = info["to_err"]

    samples: List[Dict[str, Any]] = []
    for _ in range(n_samples):
        # Sample with Gaussian noise; zero sigma yields the exact mean.
        az_sample = rng.normal(az_mu, az_sig)
        to_sample = rng.normal(to_mu, to_sig)

        # Wrap angles to physical ranges
        az_sample = np.mod(az_sample, 360.0)
        to_sample = np.clip(to_sample, 0.0, 180.0)

        samples.append(
            {
                "Name": list(station_names),
                "Azimuth": az_sample.reshape(-1, 1),
                "TakeOffAngle": to_sample.reshape(-1, 1),
            }
        )

    return samples


def polarity_matrix(
    data: Dict[str, Any],
    location_samples: Any = False,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float]]:
    """
    Generate the polarity observation matrices from an event data dictionary.

    Parameters
    ----------
    data
        Event data dictionary with keys like 'PPolarity', 'SHPolarity', 'SVPolarity'.
    location_samples
        Optional list of location scatter samples (as in MTfit). If provided,
        the station sets are intersected and expanded over samples.

    Returns
    -------
    a : np.ndarray or bool
        Station-angle MT coefficients multiplied by the observed polarities.
        Shape (N_obs, N_loc_samples_or_1, 6).
    error : np.ndarray or bool
        Fractional uncertainty per observation, shape (N_obs,).
    incorrect_polarity_prob : np.ndarray or float
        Per-observation probability of polarity flip, or 0 if not provided.
    """
    a = False
    _a = False
    error = False
    incorrect_polarity_prob: Union[np.ndarray, float] = 0

    if location_samples:
        original_samples = [u.copy() for u in location_samples]

    # Find all polarity keys (excluding probability variants)
    for key in sorted(
        u for u in data.keys() if "polarity" in u.lower() and "prob" not in u.lower()
    ):
        mode = key.lower().split("polarity")[0]

        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(
                list(
                    set(location_samples[0]["Name"])
                    & set(data[key]["Stations"]["Name"])
                )
            )
            n_stations = len(selected_stations)
            indices = [location_samples[0]["Name"].index(u) for u in selected_stations]
            angles = np.zeros((n_stations, len(location_samples), 6), dtype=float)
            for i, sample in enumerate(location_samples):
                sample["Name"] = operator.itemgetter(*indices)(sample["Name"])
                sample["TakeOffAngle"] = sample["TakeOffAngle"][indices]
                sample["Azimuth"] = sample["Azimuth"][indices]
                angles[:, i, :] = station_angles(sample, mode)
            angles = np.asarray(angles, dtype=float)

            # Fix for station order change with location samples
            indices = [data[key]["Stations"]["Name"].index(u) for u in selected_stations]
            measured = data[key]["Measured"][indices]
            _error = np.asarray(data[key]["Error"][indices], dtype=float).flatten()

            if "IncorrectPolarityProbability" in data[key]:
                _incorrect_polarity_prob = np.asarray(
                    data[key]["IncorrectPolarityProbability"][indices], dtype=float
                ).flatten()
            else:
                _incorrect_polarity_prob = np.asarray([0.0], dtype=float)
        else:
            n_stations = int(np.prod(np.asarray(data[key]["Stations"]["TakeOffAngle"]).shape))
            angles = np.zeros((n_stations, 1, 6), dtype=float)
            angles[:, 0, :] = station_angles(data[key]["Stations"], mode)
            measured = data[key]["Measured"]
            _error = np.asarray(data[key]["Error"], dtype=float).flatten()
            if "IncorrectPolarityProbability" in data[key]:
                _incorrect_polarity_prob = np.asarray(
                    data[key]["IncorrectPolarityProbability"], dtype=float
                ).flatten()
            else:
                _incorrect_polarity_prob = np.asarray([0.0], dtype=float)

        # If angles have multiple dimensions (location samples), expand measurements
        if angles.ndim > 2:
            measured = np.asarray(measured, dtype=float)
            measured = np.expand_dims(measured, 1)

        if not _a:
            _a = True
            a = angles * measured
            error = _error
            incorrect_polarity_prob = _incorrect_polarity_prob
        else:
            a = np.append(a, angles * measured, axis=0)
            error = np.append(error, _error, axis=0)
            if len(_incorrect_polarity_prob) != n_stations:
                _incorrect_polarity_prob = np.kron(
                    _incorrect_polarity_prob, np.ones(n_stations)
                )
            incorrect_polarity_prob = np.append(
                incorrect_polarity_prob, _incorrect_polarity_prob, axis=0
            )

    if isinstance(incorrect_polarity_prob, np.ndarray) and np.sum(
        np.abs(incorrect_polarity_prob)
    ) == 0:
        incorrect_polarity_prob = 0.0

    return a, error, incorrect_polarity_prob


def amplitude_matrix(
    data: Dict[str, Any],
    location_samples: Any = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate absolute amplitude observation matrices from an event dictionary.

    This function supports keys like 'PAmplitude', 'SHAmplitude', 'SVAmplitude'
    (and potential RMS/Q variants), i.e. keys that contain 'Amplitude' but not
    'AmplitudeRatio'.

    Parameters
    ----------
    data
        Event data dictionary.
    location_samples
        Optional list of location scatter samples (as in MTfit).

    Returns
    -------
    a : np.ndarray
        Station MT coefficients for the given phase, shape
        (N_obs, N_loc_samples_or_1, 6).
    amplitude : np.ndarray
        Observed absolute amplitudes, shape (N_obs,).
    percentage_error : np.ndarray
        Fractional errors on amplitudes (1σ), shape (N_obs,).
    """
    angles_list: List[np.ndarray] = []
    amplitude_list: List[np.ndarray] = []
    perc_err_list: List[np.ndarray] = []

    if location_samples:
        original_samples = [u.copy() for u in location_samples]

    # Select amplitude keys that are not amplitude ratios
    for key in sorted(
        u
        for u in data.keys()
        if "amplitude" in u.lower() and "amplituderatio" not in u.lower()
    ):
        key_lower = key.lower()
        phase = key_lower.replace("_", "").split("amplitude")[0]
        phase = phase.rstrip("rms")
        phase = phase.rstrip("q")
        phase = phase.replace("_", "")
        if not phase:
            continue

        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(
                list(
                    set(location_samples[0]["Name"])
                    & set(data[key]["Stations"]["Name"])
                )
            )
            indices = [location_samples[0]["Name"].index(u) for u in selected_stations]
            angles = np.zeros(
                (len(selected_stations), len(location_samples), 6), dtype=float
            )
            for i, sample in enumerate(location_samples):
                sample["Name"] = operator.itemgetter(*indices)(sample["Name"])
                sample["TakeOffAngle"] = sample["TakeOffAngle"][indices]
                sample["Azimuth"] = sample["Azimuth"][indices]
                angles[:, i, :] = station_angles(sample, phase)
            angles = np.asarray(angles, dtype=float)

            data_indices = [
                data[key]["Stations"]["Name"].index(u) for u in selected_stations
            ]
            measured = np.asarray(data[key]["Measured"][data_indices], dtype=float)
            error = np.asarray(data[key]["Error"][data_indices], dtype=float)
        else:
            n_stations = int(
                np.prod(np.asarray(data[key]["Stations"]["TakeOffAngle"]).shape)
            )
            angles = np.zeros((n_stations, 1, 6), dtype=float)
            angles[:, 0, :] = station_angles(data[key]["Stations"], phase)
            angles = np.asarray(angles, dtype=float)
            measured = np.asarray(data[key]["Measured"], dtype=float)
            error = np.asarray(data[key]["Error"], dtype=float)

        measured = np.asarray(measured, dtype=float).reshape(-1)
        error = np.asarray(error, dtype=float).reshape(-1)

        amp_meas = np.abs(measured).astype(float)
        safe_amp = np.clip(amp_meas, 1e-12, np.inf)
        frac_err = (np.abs(error) / safe_amp).astype(float)

        angles_list.append(angles)
        amplitude_list.append(amp_meas)
        perc_err_list.append(frac_err)

    if not angles_list:
        raise ValueError("No absolute amplitude data found in event.")

    a = np.concatenate(angles_list, axis=0)
    amplitude = np.concatenate(amplitude_list, axis=0)
    percentage_error = np.concatenate(perc_err_list, axis=0)

    return a, amplitude, percentage_error


def amplitude_ratio_matrix(
    data: Dict[str, Any],
    location_samples: Any = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate amplitude ratio observation matrices from an event data dictionary.

    This function supports keys like 'P/SHAmplitudeRatio', 'P/SVAmplitudeRatio',
    'SH/SVAmplitudeRatio' (and their RMS/Q variants).

    Parameters
    ----------
    data
        Event data dictionary.
    location_samples
        Optional list of location scatter samples (as in MTfit).

    Returns
    -------
    a1 : np.ndarray
        Station MT coefficients for ratio numerator, shape (N_obs, N_loc_samples_or_1, 6).
    a2 : np.ndarray
        Station MT coefficients for ratio denominator, shape (N_obs, N_loc_samples_or_1, 6).
    amplitude_ratio : np.ndarray
        Observed |numerator/denominator| ratios, shape (N_obs,).
    percentage_error1 : np.ndarray
        Fractional errors on numerator amplitudes, shape (N_obs,).
    percentage_error2 : np.ndarray
        Fractional errors on denominator amplitudes, shape (N_obs,).
    """
    a = False
    a1 = False
    a2 = False
    percentage_error1 = False
    percentage_error2 = False
    amplitude_ratio = False

    if location_samples:
        original_samples = [u.copy() for u in location_samples]

    for key in sorted(
        u for u in data.keys() if "amplituderatio" in u.lower() or "amplitude_ratio" in u.lower()
    ):
        phase = key.replace("_", "").lower().split("amplituderatio")[0]
        phase = phase.rstrip("rms")
        phase = phase.rstrip("q")
        phase = phase.replace("_", "")

        if location_samples:
            location_samples = [u.copy() for u in original_samples]
            selected_stations = sorted(
                list(
                    set(location_samples[0]["Name"])
                    & set(data[key]["Stations"]["Name"])
                )
            )
            indices = [location_samples[0]["Name"].index(u) for u in selected_stations]
            angles1 = np.zeros(
                (len(selected_stations), len(location_samples), 6), dtype=float
            )
            angles2 = np.zeros_like(angles1)
            for i, sample in enumerate(location_samples):
                sample["Name"] = operator.itemgetter(*indices)(sample["Name"])
                sample["TakeOffAngle"] = sample["TakeOffAngle"][indices]
                sample["Azimuth"] = sample["Azimuth"][indices]
                num_phase, den_phase = phase.split("/")
                angles1[:, i, :] = station_angles(sample, num_phase)
                angles2[:, i, :] = station_angles(sample, den_phase)
            angles1 = np.asarray(angles1, dtype=float)
            angles2 = np.asarray(angles2, dtype=float)
            angles = [angles1, angles2]
            indices = [data[key]["Stations"]["Name"].index(u) for u in selected_stations]
            _measured = np.asarray(data[key]["Measured"][indices], dtype=float)
            error = np.asarray(data[key]["Error"][indices], dtype=float)
        else:
            n_stations = int(np.prod(np.asarray(data[key]["Stations"]["TakeOffAngle"]).shape))
            angles1 = np.zeros((n_stations, 1, 6), dtype=float)
            angles2 = np.zeros((n_stations, 1, 6), dtype=float)
            num_phase, den_phase = phase.split("/")
            angles1[:, 0, :] = station_angles(data[key]["Stations"], num_phase)
            angles2[:, 0, :] = station_angles(data[key]["Stations"], den_phase)
            angles1 = np.asarray(angles1, dtype=float)
            angles2 = np.asarray(angles2, dtype=float)
            angles = [angles1, angles2]
            _measured = np.asarray(data[key]["Measured"], dtype=float)
            error = np.asarray(data[key]["Error"], dtype=float)

        # Normalise measurement shapes:
        # - Standard case: _measured has two columns (numerator, denominator).
        # - Some data (e.g. Krafla SH/SVAmplitudeRatio) provide a single
        #   column which is already the ratio; in that case we treat the
        #   denominator amplitude as 1.0 and reuse the two error columns.
        if _measured.ndim == 1:
            _measured = _measured.reshape(-1, 1)
        if error.ndim == 1:
            error = error.reshape(-1, 1)

        if _measured.shape[1] >= 2:
            num_meas = _measured[:, 0]
            den_meas = _measured[:, 1]
            err_num = error[:, 0] if error.shape[1] >= 1 else error[:, 0]
            err_den = error[:, 1] if error.shape[1] >= 2 else error[:, 0]
        else:
            # Single-column "Measured" interpreted as a precomputed ratio.
            # Assume a notional denominator amplitude of 1.0 and reuse the
            # two error columns for numerator/denominator uncertainties.
            num_meas = _measured[:, 0]
            den_meas = np.ones_like(num_meas)
            err_num = error[:, 0] if error.shape[1] >= 1 else error[:, 0]
            err_den = error[:, 1] if error.shape[1] >= 2 else error[:, 0]

        this_amp_ratio = np.abs(num_meas / den_meas).astype(float)
        this_perc_err1 = (err_num / np.abs(num_meas)).astype(float)
        this_perc_err2 = (err_den / np.abs(den_meas)).astype(float)

        if not a:
            a = True
            a1 = angles[0]
            a2 = angles[1]
            amplitude_ratio = this_amp_ratio
            percentage_error1 = this_perc_err1
            percentage_error2 = this_perc_err2
        else:
            a1 = np.append(a1, angles[0], axis=0)
            a2 = np.append(a2, angles[1], axis=0)
            amplitude_ratio = np.append(amplitude_ratio, this_amp_ratio, axis=0)
            percentage_error1 = np.append(percentage_error1, this_perc_err1, axis=0)
            percentage_error2 = np.append(percentage_error2, this_perc_err2, axis=0)

    return a1, a2, amplitude_ratio, percentage_error1, percentage_error2
