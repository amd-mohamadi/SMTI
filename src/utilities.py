"""
Utility functions for analysing MT posterior samples from the SMCMTI
PyTensor inversion and for generating synthetic test data.

This module provides helpers to:

- Find a MAP-like MT6 sample using a multivariate Gaussian KDE.
- Approximate a MAP-like MT6 sample using a Gaussian mixture model.
- Check for multimodality in MT6 space using a Gaussian mixture model.
- Check for multimodality in parameter space (e.g. kappa, h, sigma) using HDBSCAN.
- Quantify angular uncertainty around a reference MT (e.g. MAP or median).
- Compute a scalar MT quality score from an InferenceData object.
- Compute block-wise Qdc (orientation / Kagan) and Qndc (source type / lune) scores.
- Generate MTfit-style synthetic events using the current forward model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import warnings
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.cluster import HDBSCAN
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

import os

from .moment_tesnor_conversion import MT33_MT6
from .forward_model import forward_amplitude


ESS_HEALTHY_BIC = 1000.0


def find_map_mt6_kde(mt6_samples: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Approximate a MAP MT6 sample using a multivariate Gaussian KDE.

    Parameters
    ----------
    mt6_samples : np.ndarray
        Moment tensor six-vectors with shape (6, n_samples).

    Returns
    -------
    mt6_map : np.ndarray
        MAP-like MT6 vector, shape (6,).
    index : int
        Index of the selected sample along the second axis of ``mt6_samples``.
    """
    mt6_samples = np.asarray(mt6_samples, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise ValueError(
            f"mt6_samples must have shape (6, n_samples), got {mt6_samples.shape}"
        )

    # Transpose for easier linear algebra: X is (n_samples, 6)
    X = mt6_samples.T

    # Centre data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Perform SVD for dimensionality reduction to handle cases where the
    # MTs live in a lower-dimensional subspace (e.g. DC constraint).
    # Keep only directions with non-negligible singular values.
    if X_centered.shape[0] > 1:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        if S.size == 0:
            # Degenerate case: all samples identical
            index = 0
            mt6_map = mt6_samples[:, index]
            return mt6_map, index

        tol = 1e-8 * S[0]
        keep = S > tol
        n_keep = int(np.sum(keep))
        if n_keep == 0:
            n_keep = 1
            keep[0] = True

        V_reduced = Vt[keep].T  # (6, n_keep)
        Y = X_centered @ V_reduced  # (n_samples, n_keep)
        data_for_kde = Y.T  # (n_keep, n_samples)
    else:
        # Only one sample: return it directly.
        index = 0
        mt6_map = mt6_samples[:, index]
        return mt6_map, index

    kde = gaussian_kde(data_for_kde)
    log_dens = kde.logpdf(data_for_kde)  # (n_samples,)

    index = int(np.argmax(log_dens))
    mt6_map = mt6_samples[:, index]
    return mt6_map, index


def find_map_mt6_gmm(
    mt6_samples: np.ndarray,
    max_components: int = 4,
    random_state: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Approximate a MAP MT6 sample using a Gaussian mixture model.

    This is typically faster than the full multivariate KDE-based
    ``find_map_mt6_kde`` for large numbers of samples and tends to pick
    the centre of the highest-weight posterior lobe.

    Parameters
    ----------
    mt6_samples : np.ndarray
        Moment tensor six-vectors with shape (6, n_samples).
    max_components : int, default 4
        Maximum number of mixture components to consider when selecting
        the model using BIC. The actual number is bounded above by the
        number of available samples.
    random_state : int, default 0
        Random state forwarded to :class:`sklearn.mixture.GaussianMixture`.

    Returns
    -------
    mt6_map : np.ndarray
        MAP-like MT6 vector, shape (6,).
    index : int
        Index of the selected sample along the second axis of
        ``mt6_samples``.
    """
    mt6_samples = np.asarray(mt6_samples, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise ValueError(
            f"mt6_samples must have shape (6, n_samples), got {mt6_samples.shape}"
        )

    n_samples = mt6_samples.shape[1]
    if n_samples == 0:
        raise ValueError("mt6_samples must contain at least one sample.")
    if n_samples == 1:
        return mt6_samples[:, 0], 0

    X = mt6_samples.T  # (n_samples, 6)

    # Limit the number of components so that we never ask for more
    # components than data points.
    max_components = int(max_components)
    if max_components < 1:
        max_components = 1
    max_components = min(max_components, n_samples)

    models: list[GaussianMixture] = []
    bics: list[float] = []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        ).fit(X)
        models.append(gm)
        bics.append(gm.bic(X))

    best_index = int(np.argmin(bics))
    best_model = models[best_index]

    # Take the mean of the highest-weight component as the dominant mode.
    main_comp = int(np.argmax(best_model.weights_))
    center = best_model.means_[main_comp]  # (6,)

    # Snap to the nearest actual posterior sample for consistency with
    # the rest of the workflow.
    diff = X - center
    dist2 = np.sum(diff * diff, axis=1)
    index = int(np.argmin(dist2))
    mt6_map = mt6_samples[:, index]
    return mt6_map, index


def cluster_mt6_posterior(
    mt6_samples: np.ndarray,
    max_components: int = 4,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Detect and characterise multimodality in MT6 posterior samples.

    Uses a Gaussian mixture model with BIC-based model selection.

    Parameters
    ----------
    mt6_samples : np.ndarray
        Moment tensor six-vectors with shape (6, n_samples).
    max_components : int, default 4
        Maximum number of mixture components (modes) to test.
    random_state : int, default 0
        Random state forwarded to :class:`sklearn.mixture.GaussianMixture`.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``n_components`` : int
            Selected number of mixture components.
        - ``weights`` : np.ndarray, shape (n_components,)
            Mixture weights (approximate posterior mass in each mode).
        - ``labels`` : np.ndarray, shape (n_samples,)
            Component assignments for each sample.
        - ``centers`` : np.ndarray, shape (n_components, 6)
            Mean MT6 vector for each component.
    """
    mt6_samples = np.asarray(mt6_samples, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise ValueError(
            f"mt6_samples must have shape (6, n_samples), got {mt6_samples.shape}"
        )

    X = mt6_samples.T  # (n_samples, 6)

    models = []
    bics = []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
        )
        gm.fit(X)
        models.append(gm)
        bics.append(gm.bic(X))

    best_k = int(np.argmin(bics)) + 1
    gm = models[best_k - 1]

    labels = gm.predict(X)
    weights = gm.weights_
    centers = gm.means_

    return {
        "n_components": np.asarray(best_k, dtype=int),
        "weights": np.asarray(weights, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "centers": np.asarray(centers, dtype=float),
    }


def cluster_mt6_posterior_hdbscan(
    mt6_samples: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Detect and characterise multimodality in MT6 posterior samples using HDBSCAN.

    Parameters
    ----------
    mt6_samples : np.ndarray
        Moment tensor six-vectors with shape (6, n_samples).
    min_cluster_size : int, default 50
        Minimum number of samples in a cluster.
    min_samples : int, optional
        HDBSCAN ``min_samples`` parameter. If ``None``, it defaults to
        ``min_cluster_size // 2``.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``n_components`` : int
            Number of clusters (excluding noise points).
        - ``weights`` : np.ndarray, shape (n_components,)
            Fraction of all samples assigned to each cluster.
        - ``labels`` : np.ndarray, shape (n_samples,)
            Cluster labels for each sample (``-1`` denotes noise).
        - ``cluster_ids`` : np.ndarray, shape (n_components,)
            Unique non-noise cluster labels.
        - ``centers`` : np.ndarray, shape (n_components, 6)
            Mean MT6 vector for each cluster.
        - ``noise_fraction`` : float
            Fraction of samples classified as noise.
    """
    mt6_samples = np.asarray(mt6_samples, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise ValueError(
            f"mt6_samples must have shape (6, n_samples), got {mt6_samples.shape}"
        )

    X = mt6_samples.T  # (n_samples, 6)
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X)

    n_samples = X.shape[0]
    noise_mask = labels < 0
    noise_fraction = float(noise_mask.sum()) / float(n_samples)

    cluster_mask = labels >= 0
    if not np.any(cluster_mask):
        return {
            "n_components": np.asarray(0, dtype=int),
            "weights": np.zeros(0, dtype=float),
            "labels": np.asarray(labels, dtype=int),
            "cluster_ids": np.zeros(0, dtype=int),
            "centers": np.zeros((0, 6), dtype=float),
            "noise_fraction": np.asarray(noise_fraction, dtype=float),
        }

    cluster_ids = np.unique(labels[cluster_mask])
    n_components = int(cluster_ids.size)

    weights = np.zeros(n_components, dtype=float)
    centers = np.zeros((n_components, 6), dtype=float)

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        weights[i] = float(mask.sum()) / float(n_samples)
        centers[i] = X[mask].mean(axis=0)

    return {
        "n_components": np.asarray(n_components, dtype=int),
        "weights": weights,
        "labels": np.asarray(labels, dtype=int),
        "cluster_ids": cluster_ids.astype(int),
        "centers": centers,
        "noise_fraction": np.asarray(noise_fraction, dtype=float),
    }


def angle_mode_analysis(
    mt6_ref: np.ndarray,
    mt6_samples: np.ndarray,
    prominence: float = 0.1,
    min_separation_deg: float = 15.0,
    grid_points: int = 512,
) -> Dict[str, np.ndarray]:
    """
    Detect modes in the angular distance distribution around a reference MT6.

    This uses a 1D Gaussian KDE on the angles between ``mt6_ref`` and each
    posterior sample, and ``scipy.signal.find_peaks`` to identify prominent
    modes. Each sample is then assigned to the nearest mode to estimate the
    weight (posterior probability mass) associated with each mode.

    Parameters
    ----------
    mt6_ref : np.ndarray
        Reference MT6 vector, shape (6,). Typically the MAP-like MT.
    mt6_samples : np.ndarray
        Posterior MT6 samples, shape (6, n_samples).
    prominence : float, default 0.1
        Minimum prominence for peaks in the *normalised* KDE density
        (0–1 scale). Larger values yield fewer modes.
    min_separation_deg : float, default 15.0
        Minimum separation (in degrees) between mode centres. Modes closer
        than this are merged by keeping only the higher-density one.
    grid_points : int, default 512
        Number of grid points for evaluating the KDE over [0, 180] degrees.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``angles_deg`` : np.ndarray, shape (n_samples,)
            Angular distance (degrees) between ``mt6_ref`` and each sample.
        - ``x_grid`` : np.ndarray, shape (grid_points,)
            Grid of angles where the KDE was evaluated.
        - ``density`` : np.ndarray, shape (grid_points,)
            KDE density values on ``x_grid``.
        - ``mode_centers_deg`` : np.ndarray, shape (n_modes,)
            Angle locations (degrees) of detected modes.
        - ``mode_weights`` : np.ndarray, shape (n_modes,)
            Fraction of samples associated with each mode (sums to 1).
        - ``labels`` : np.ndarray, shape (n_samples,)
            Index of the mode each sample is assigned to.
    """
    # Reuse angular distance computation
    ang_info = angular_uncertainty(mt6_ref, mt6_samples, quantiles=(50.0,))
    angles_deg = ang_info["angles_deg"]

    if angles_deg.size == 0:
        return {
            "angles_deg": angles_deg,
            "x_grid": np.linspace(0.0, 180.0, grid_points),
            "density": np.zeros(grid_points, dtype=float),
            "mode_centers_deg": np.zeros(0, dtype=float),
            "mode_weights": np.zeros(0, dtype=float),
            "labels": np.zeros(angles_deg.size, dtype=int),
        }

    # KDE on [0, 180] degrees
    kde = gaussian_kde(angles_deg)
    x_grid = np.linspace(0.0, 180.0, grid_points)
    density = kde(x_grid)

    if np.all(density <= 0.0):
        return {
            "angles_deg": angles_deg,
            "x_grid": x_grid,
            "density": density,
            "mode_centers_deg": np.zeros(0, dtype=float),
            "mode_weights": np.zeros(0, dtype=float),
            "labels": np.zeros(angles_deg.size, dtype=int),
        }

    # Normalise for peak detection
    density_norm = density / density.max()

    peak_indices, _ = find_peaks(density_norm, prominence=prominence)
    if peak_indices.size == 0:
        # Treat as effectively unimodal centred at the global maximum
        peak_indices = np.array([int(np.argmax(density_norm))])

    mode_centers = x_grid[peak_indices]

    # Enforce minimum separation between modes
    if mode_centers.size > 1:
        # Sort modes by decreasing density, then keep well-separated ones
        order = np.argsort(density_norm[peak_indices])[::-1]
        selected = []
        for idx in order:
            angle = mode_centers[idx]
            if not selected:
                selected.append(idx)
            else:
                if np.all(np.abs(mode_centers[selected] - angle) >= min_separation_deg):
                    selected.append(idx)
        selected = np.array(selected, dtype=int)
        peak_indices = peak_indices[selected]
        mode_centers = x_grid[peak_indices]

    n_modes = mode_centers.size

    # Assign each sample to nearest mode centre to estimate weights
    if n_modes == 1:
        labels = np.zeros(angles_deg.size, dtype=int)
        mode_weights = np.array([1.0], dtype=float)
    else:
        diffs = np.abs(
            angles_deg[:, None] - mode_centers[None, :]
        )  # (n_samples, n_modes)
        labels = np.argmin(diffs, axis=1)
        counts = np.bincount(labels, minlength=n_modes).astype(float)
        mode_weights = counts / counts.sum()

    return {
        "angles_deg": angles_deg,
        "x_grid": x_grid,
        "density": density,
        "mode_centers_deg": mode_centers,
        "mode_weights": mode_weights,
        "labels": labels,
    }


def cluster_kappa_h_sigma_hdbscan(
    kappa: np.ndarray,
    h: np.ndarray,
    sigma: np.ndarray,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Cluster posterior samples in (kappa, h, sigma) space using HDBSCAN.

    This focuses the clustering on source orientation and dip parameters,
    which are typically the most informative for multimodality, and keeps
    the dimensionality low (3D) for more robust density-based clustering.

    Parameters
    ----------
    kappa, h, sigma : np.ndarray
        1D arrays of posterior samples for each parameter, all with shape
        (n_samples,).
    min_cluster_size : int, optional
        Minimum cluster size. If ``None``, a fraction of the total number
        of samples is used (approximately 2%).
    min_samples : int, optional
        HDBSCAN ``min_samples`` parameter. If ``None``, it defaults to
        roughly one third of ``min_cluster_size``.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``n_components`` : int
            Number of clusters (excluding noise points).
        - ``weights`` : np.ndarray, shape (n_components,)
            Fraction of all samples assigned to each cluster.
        - ``labels`` : np.ndarray, shape (n_samples,)
            Cluster labels for each sample (``-1`` denotes noise).
        - ``cluster_ids`` : np.ndarray, shape (n_components,)
            Unique non-noise cluster labels.
        - ``centers`` : np.ndarray, shape (n_components, 3)
            Mean (kappa, h, sigma) for each cluster.
        - ``noise_fraction`` : float
            Fraction of samples classified as noise.
    """
    kappa = np.asarray(kappa, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    if not (kappa.shape == h.shape == sigma.shape):
        raise ValueError(
            f"kappa, h, sigma must have the same shape, got "
            f"{kappa.shape}, {h.shape}, {sigma.shape}"
        )

    n_samples = kappa.size
    if n_samples == 0:
        return {
            "n_components": np.asarray(0, dtype=int),
            "weights": np.zeros(0, dtype=float),
            "labels": np.zeros(0, dtype=int),
            "cluster_ids": np.zeros(0, dtype=int),
            "centers": np.zeros((0, 3), dtype=float),
            "noise_fraction": np.asarray(0.0, dtype=float),
        }

    X = np.column_stack([kappa, h, sigma])  # (n_samples, 3)

    if min_cluster_size is None:
        # Require clusters to contain at least ~2% of samples, but not fewer than 30.
        min_cluster_size = max(30, n_samples // 50)
    if min_samples is None:
        min_samples = max(5, min_cluster_size // 3)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(X)

    noise_mask = labels < 0
    noise_fraction = float(noise_mask.sum()) / float(n_samples)

    cluster_mask = labels >= 0
    if not np.any(cluster_mask):
        return {
            "n_components": np.asarray(0, dtype=int),
            "weights": np.zeros(0, dtype=float),
            "labels": np.asarray(labels, dtype=int),
            "cluster_ids": np.zeros(0, dtype=int),
            "centers": np.zeros((0, 3), dtype=float),
            "noise_fraction": np.asarray(noise_fraction, dtype=float),
        }

    cluster_ids = np.unique(labels[cluster_mask])
    n_components = int(cluster_ids.size)

    weights = np.zeros(n_components, dtype=float)
    centers = np.zeros((n_components, 3), dtype=float)

    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        weights[i] = float(mask.sum()) / float(n_samples)
        centers[i] = X[mask].mean(axis=0)

    return {
        "n_components": np.asarray(n_components, dtype=int),
        "weights": weights,
        "labels": np.asarray(labels, dtype=int),
        "cluster_ids": cluster_ids.astype(int),
        "centers": centers,
        "noise_fraction": np.asarray(noise_fraction, dtype=float),
    }


def angular_uncertainty(
    mt6_ref: np.ndarray,
    mt6_samples: np.ndarray,
    quantiles: Tuple[float, ...] = (50.0, 90.0),
) -> Dict[str, np.ndarray]:
    """
    Quantify angular uncertainty of MT6 samples around a reference MT6.

    Angles are computed in 6-D MT space between the reference vector and
    each posterior sample, and returned in degrees.

    Parameters
    ----------
    mt6_ref : np.ndarray
        Reference MT6 vector, shape (6,).
    mt6_samples : np.ndarray
        Posterior MT6 samples, shape (6, n_samples).
    quantiles : tuple of float, default (50.0, 90.0)
        Quantiles (in percent) to report for the angular distribution.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``angles_deg`` : np.ndarray, shape (n_samples,)
            Angular distance (degrees) between ``mt6_ref`` and each sample.
        - ``quantiles`` : np.ndarray, shape (len(quantiles),)
            Requested quantiles of ``angles_deg``.
    """
    mt6_ref = np.asarray(mt6_ref, dtype=float).reshape(6)
    mt6_samples = np.asarray(mt6_samples, dtype=float)
    if mt6_samples.ndim != 2 or mt6_samples.shape[0] != 6:
        raise ValueError(
            f"mt6_samples must have shape (6, n_samples), got {mt6_samples.shape}"
        )

    num = np.dot(mt6_ref, mt6_samples)  # (n_samples,)
    den = np.linalg.norm(mt6_ref) * np.linalg.norm(mt6_samples, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.clip(num / den, -1.0, 1.0)
    angles_rad = np.arccos(cosang)
    angles_deg = np.degrees(angles_rad)

    q = np.percentile(angles_deg, list(quantiles))

    return {
        "angles_deg": angles_deg,
        "quantiles": q,
    }


# ---------------------------------------------------------------------------
# Qdc / Qndc primitives (see smti_inversion/202607_q_factor_diagnosis.md)
# ---------------------------------------------------------------------------

# Four-fold DC principal-axis symmetries used by the Kagan angle (right-multiply
# on the orientation matrix U = [T N P]).
_DC_AXIS_SYMMETRIES = (
    np.diag([1.0, 1.0, 1.0]),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
)


def _as_1d(x: np.ndarray) -> np.ndarray:
    """Flatten parameter arrays that may be (n,), (chain, draw), etc."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def lune_unit_vectors(gamma: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    Map Tape (gamma, delta) to unit vectors on the eigenvalue lune sphere.

    Uses the same spherical coordinates as ``tape.GD_E``:
    ``phi = pi/2 - delta``, ``X = (cos g sin phi, sin g sin phi, cos phi)``.

    Parameters
    ----------
    gamma, delta : array-like
        Tape source-type parameters in radians (any broadcastable shape).

    Returns
    -------
    X : np.ndarray
        Unit vectors with shape ``(*broadcast_shape, 3)``.
    """
    g = np.asarray(gamma, dtype=float)
    d = np.asarray(delta, dtype=float)
    g, d = np.broadcast_arrays(g, d)
    phi = 0.5 * np.pi - d
    sin_phi = np.sin(phi)
    x0 = np.cos(g) * sin_phi
    x1 = np.sin(g) * sin_phi
    x2 = np.cos(phi)
    return np.stack([x0, x1, x2], axis=-1)


def lune_distance_deg(
    gamma_ref: float,
    delta_ref: float,
    gamma: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """
    Great-circle angular distance (degrees) on the Tape lune from a reference.

    Parameters
    ----------
    gamma_ref, delta_ref : float
        Reference source-type parameters (radians).
    gamma, delta : array-like
        Sample source-type parameters (radians).

    Returns
    -------
    angles_deg : np.ndarray
        Distance from the reference for each sample, in degrees.
    """
    x_ref = lune_unit_vectors(gamma_ref, delta_ref).reshape(3)
    x = lune_unit_vectors(gamma, delta)
    # Flatten trailing sample dims for the dot product, then reshape.
    flat = x.reshape(-1, 3)
    cosang = np.clip(flat @ x_ref, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosang))
    return angles.reshape(np.broadcast_shapes(np.shape(gamma), np.shape(delta)))


def orientation_matrices(
    kappa: np.ndarray,
    h: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Build orientation matrices ``U = [T N P]`` from Tape orientation params.

    Parameters
    ----------
    kappa, h, sigma : array-like
        Strike (rad), cos(dip), rake (rad). Broadcast together.

    Returns
    -------
    U : np.ndarray
        Shape ``(n, 3, 3)`` with columns T, N, P for each sample.
    """
    k = _as_1d(kappa)
    hh = np.clip(_as_1d(h), 0.0, 1.0)
    s = _as_1d(sigma)
    n = k.size
    if hh.size != n or s.size != n:
        k, hh, s = np.broadcast_arrays(k, hh, s)
        n = k.size

    # Vectorised SDR_TNP (same formulas as tape.SDR_TNP).
    dip = np.arccos(hh)
    cos_k = np.cos(k)
    sin_k = np.sin(k)
    cos_d = np.cos(dip)
    sin_d = np.sin(dip)
    cos_r = np.cos(s)
    sin_r = np.sin(s)

    # Fault normal N2 and slip N1 (Tape N-E-D convention).
    n1 = np.stack(
        [
            cos_k * cos_r + sin_k * cos_d * sin_r,
            sin_k * cos_r - cos_k * cos_d * sin_r,
            -sin_d * sin_r,
        ],
        axis=1,
    )
    n2 = np.stack(
        [
            -sin_k * sin_d,
            cos_k * sin_d,
            -cos_d,
        ],
        axis=1,
    )

    t = n1 + n2
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.maximum(t_norm, 1e-15)
    p = n1 - n2
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.maximum(p_norm, 1e-15)
    n_axis = -np.cross(t, p)
    n_norm = np.linalg.norm(n_axis, axis=1, keepdims=True)
    n_axis = n_axis / np.maximum(n_norm, 1e-15)

    # U columns: T, N, P  -> shape (n, 3, 3)
    return np.stack([t, n_axis, p], axis=2)


def kagan_angles_deg(
    kappa_ref: float,
    h_ref: float,
    sigma_ref: float,
    kappa: np.ndarray,
    h: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Kagan angles (degrees) between a reference orientation and many samples.

    The Kagan angle is the minimum rotation angle between two double-couple
    orientations, minimised over the four-fold principal-axis symmetry group.

    Parameters
    ----------
    kappa_ref, h_ref, sigma_ref : float
        Reference Tape orientation (radians / cos(dip) / radians).
    kappa, h, sigma : array-like
        Sample orientations, same units.

    Returns
    -------
    angles_deg : np.ndarray, shape (n_samples,)
    """
    u_ref = orientation_matrices(
        np.asarray(kappa_ref, dtype=float).reshape(1),
        np.asarray(h_ref, dtype=float).reshape(1),
        np.asarray(sigma_ref, dtype=float).reshape(1),
    )[0]  # (3, 3)
    u = orientation_matrices(kappa, h, sigma)  # (n, 3, 3)
    n = u.shape[0]
    min_ang = np.full(n, 180.0, dtype=float)
    u_ref_t = u_ref.T  # (3, 3)
    for S in _DC_AXIS_SYMMETRIES:
        # R_i = U_ref.T @ (U_i @ S)
        us = u @ S  # (n, 3, 3)
        r = np.einsum("ab,nbc->nac", u_ref_t, us)
        tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
        cosang = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        min_ang = np.minimum(min_ang, ang)
    return min_ang


def kagan_angle_deg(
    kappa1: float,
    h1: float,
    sigma1: float,
    kappa2: float,
    h2: float,
    sigma2: float,
) -> float:
    """Scalar Kagan angle (degrees) between two Tape orientations."""
    return float(
        kagan_angles_deg(kappa1, h1, sigma1, kappa2, h2, sigma2)[0]
    )


def _circular_mean(angles_rad: np.ndarray) -> float:
    """Circular mean of angles in radians."""
    a = np.asarray(angles_rad, dtype=float).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _angle_quantile_score(
    q_deg: float,
    angle_good: float,
    angle_bad: float,
) -> float:
    """Map an angular quantile to [0, 1] (good -> 1, bad or worse -> 0)."""
    span = max(float(angle_bad) - float(angle_good), 1e-9)
    return float(np.clip((float(angle_bad) - float(q_deg)) / span, 0.0, 1.0))


def _precision_from_quantiles(
    q50: float,
    q90: float,
    angle_good: float,
    angle_bad: float,
) -> float:
    return 0.5 * _angle_quantile_score(q50, angle_good, angle_bad) + 0.5 * _angle_quantile_score(
        q90, angle_good, angle_bad
    )


def _chain_shape(
    values: np.ndarray,
    n_chains: Optional[int],
) -> Optional[Tuple[int, int]]:
    if values.ndim == 2:
        return int(values.shape[0]), int(values.shape[1])
    if n_chains is not None and int(n_chains) > 1:
        n_total = _as_1d(values).size
        if n_total % int(n_chains) == 0:
            return int(n_chains), n_total // int(n_chains)
    return None


def _diag_gmm_bic(
    features: np.ndarray,
    k_max: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[int, float]]:
    n_samples = int(features.shape[0])
    k_hi = min(max(int(k_max), 1), n_samples)
    fits: Dict[int, Tuple[float, GaussianMixture]] = {}
    for n_components in range(1, k_hi + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    n_init=3,
                    random_state=int(seed),
                ).fit(features)
        except Exception:
            continue
        fits[n_components] = (float(model.bic(features)), model)
    if not fits:
        raise RuntimeError("all GaussianMixture fits failed")
    best_k = min(fits, key=lambda n_components: fits[n_components][0])
    responsibilities = fits[best_k][1].predict_proba(features)
    return np.asarray(responsibilities, dtype=float), {
        n_components: fit[0] for n_components, fit in fits.items()
    }


def _per_chain_mode_std(
    responsibilities: np.ndarray,
    components: Sequence[int],
    survivor_components: Sequence[int],
    chain_shape: Optional[Tuple[int, int]],
) -> float:
    if chain_shape is None or chain_shape[0] <= 1:
        return 0.0
    if responsibilities.shape[0] != chain_shape[0] * chain_shape[1]:
        return 0.0
    numerator = responsibilities[:, list(components)].sum(axis=1).reshape(chain_shape).mean(axis=1)
    denominator = (
        responsibilities[:, list(survivor_components)]
        .sum(axis=1)
        .reshape(chain_shape)
        .mean(axis=1)
    )
    per_chain = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(np.std(per_chain, ddof=0))


def cluster_orientation_params(
    kappa: np.ndarray,
    h: np.ndarray,
    sigma: np.ndarray,
    n_chains: Optional[int] = None,
    mode_min_weight: float = 0.10,
    mode_min_kagan_deg: float = 15.0,
    k_max: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Select orientation components by GMM+BIC and merge physical DC modes.

    Components closer than ``mode_min_kagan_deg`` are merged iteratively.
    Components below ``mode_min_weight`` are removed from the soft-weight
    normalization, while their hard-assigned samples are reassigned to the
    nearest surviving mode for strike/dip/rake summaries.
    """
    k_arr = np.asarray(kappa, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    s_arr = np.asarray(sigma, dtype=float)
    chain_shape = _chain_shape(k_arr, n_chains)
    k = _as_1d(k_arr)
    hh = _as_1d(h_arr)
    ss = _as_1d(s_arr)
    n = k.size
    nan = float("nan")
    if hh.size != n or ss.size != n:
        raise ValueError("kappa, h, and sigma must contain the same number of samples")
    if n == 0:
        return {
            "n_modes_dc": 0,
            "s_mode_dc": nan,
            "dc_top_mode_weight": nan,
            "dc_mode_weight_std": nan,
            "dc_mode1_weight": nan,
            "labels": np.zeros(0, dtype=int),
            "dc_mode0_strike_deg": nan,
            "dc_mode0_dip_deg": nan,
            "dc_mode0_rake_deg": nan,
            "dc_mode1_strike_deg": nan,
            "dc_mode1_dip_deg": nan,
            "dc_mode1_rake_deg": nan,
            "dc_mode_list": [],
            "dc_bic_by_k": {},
            "dc_q50_within_mode_deg": nan,
            "dc_q90_within_mode_deg": nan,
        }

    if n == 1:
        responsibilities = np.ones((1, 1), dtype=float)
        bic_by_k = {1: nan}
    else:
        features = np.column_stack([np.cos(k), np.sin(k), 2.0 * hh, ss])
        try:
            responsibilities, bic_by_k = _diag_gmm_bic(features, k_max=k_max, seed=seed)
        except Exception as exc:
            from scipy.cluster.vq import kmeans2

            warnings.warn(
                f"GMM+BIC orientation clustering failed ({exc}); using kmeans2 fallback",
                RuntimeWarning,
                stacklevel=2,
            )
            n_components = min(2, n)
            _, raw_labels = kmeans2(
                features,
                n_components,
                seed=int(seed),
                minit="++",
                iter=50,
            )
            responsibilities = np.eye(n_components, dtype=float)[np.asarray(raw_labels, dtype=int)]
            bic_by_k = {}

    raw_labels = responsibilities.argmax(axis=1)
    raw_weights = responsibilities.mean(axis=0)
    clusters = [
        {
            "components": [component],
            "members": np.flatnonzero(raw_labels == component),
            "weight": float(raw_weights[component]),
        }
        for component in range(responsibilities.shape[1])
        if np.any(raw_labels == component)
    ]

    def representative(cluster: Dict[str, Any]) -> Tuple[float, float, float]:
        members = cluster["members"]
        return (
            _circular_mean(k[members]),
            float(np.median(hh[members])),
            float(np.median(ss[members])),
        )

    while len(clusters) > 1:
        pairs = [
            (
                kagan_angle_deg(*representative(clusters[first]), *representative(clusters[second])),
                first,
                second,
            )
            for first in range(len(clusters))
            for second in range(first + 1, len(clusters))
        ]
        separation, first, second = min(pairs)
        if separation >= float(mode_min_kagan_deg):
            break
        clusters[first] = {
            "components": clusters[first]["components"] + clusters[second]["components"],
            "members": np.concatenate(
                [clusters[first]["members"], clusters[second]["members"]]
            ),
            "weight": clusters[first]["weight"] + clusters[second]["weight"],
        }
        del clusters[second]

    survivors = [
        cluster for cluster in clusters if cluster["weight"] >= float(mode_min_weight)
    ]
    if not survivors:
        survivors = [max(clusters, key=lambda cluster: cluster["weight"])]
    survivor_ids = {id(cluster) for cluster in survivors}
    dropped = [cluster for cluster in clusters if id(cluster) not in survivor_ids]
    for cluster in survivors:
        cluster["precision_members"] = cluster["members"].copy()

    survivor_representatives = [representative(cluster) for cluster in survivors]
    for cluster in dropped:
        members = cluster["members"]
        distances = np.column_stack(
            [
                kagan_angles_deg(*mode_ref, k[members], hh[members], ss[members])
                for mode_ref in survivor_representatives
            ]
        )
        nearest = np.argmin(distances, axis=1)
        for survivor_index, survivor in enumerate(survivors):
            assigned = members[nearest == survivor_index]
            survivor["members"] = np.concatenate([survivor["members"], assigned])

    weight_sum = float(sum(cluster["weight"] for cluster in survivors))
    for cluster in survivors:
        cluster["normalized_weight"] = cluster["weight"] / weight_sum
    survivors.sort(key=lambda cluster: cluster["normalized_weight"], reverse=True)
    survivor_components = [
        component for cluster in survivors for component in cluster["components"]
    ]

    labels = np.full(n, -1, dtype=int)
    mode_list = []
    for mode_index, cluster in enumerate(survivors):
        members = cluster["members"]
        precision_members = cluster["precision_members"]
        labels[members] = mode_index
        strike = float(np.degrees(_circular_mean(k[members])) % 360.0)
        dip = float(
            np.degrees(np.arccos(np.clip(np.mean(hh[members]), 0.0, 1.0)))
        )
        rake = float(np.degrees(np.mean(ss[members])))
        mode_ref = representative(
            {"members": precision_members}
        )
        angles = kagan_angles_deg(
            *mode_ref,
            k[precision_members],
            hh[precision_members],
            ss[precision_members],
        )
        mode_list.append(
            {
                "strike": strike,
                "dip": dip,
                "rake": rake,
                "weight": float(cluster["normalized_weight"]),
                "per_chain_weight_std": _per_chain_mode_std(
                    responsibilities,
                    cluster["components"],
                    survivor_components,
                    chain_shape,
                ),
                "q50_within_mode_deg": float(np.percentile(angles, 50.0)),
                "q90_within_mode_deg": float(np.percentile(angles, 90.0)),
            }
        )

    mode0 = mode_list[0]
    mode1 = mode_list[1] if len(mode_list) > 1 else None
    return {
        "n_modes_dc": len(mode_list),
        "s_mode_dc": mode0["weight"],
        "dc_top_mode_weight": mode0["weight"],
        "dc_mode_weight_std": mode0["per_chain_weight_std"],
        "dc_mode1_weight": mode1["weight"] if mode1 is not None else nan,
        "labels": labels,
        "dc_mode0_strike_deg": mode0["strike"],
        "dc_mode0_dip_deg": mode0["dip"],
        "dc_mode0_rake_deg": mode0["rake"],
        "dc_mode1_strike_deg": mode1["strike"] if mode1 is not None else nan,
        "dc_mode1_dip_deg": mode1["dip"] if mode1 is not None else nan,
        "dc_mode1_rake_deg": mode1["rake"] if mode1 is not None else nan,
        "dc_mode_list": mode_list,
        "dc_bic_by_k": bic_by_k,
        "dc_q50_within_mode_deg": mode0["q50_within_mode_deg"],
        "dc_q90_within_mode_deg": mode0["q90_within_mode_deg"],
    }


def cluster_lune_params(
    gamma: np.ndarray,
    delta: np.ndarray,
    n_chains: Optional[int] = None,
    mode_min_weight: float = 0.10,
    mode_min_lune_deg: float = 15.0,
    k_max: int = 6,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Select source-type components by GMM+BIC and merge them on the lune.

    Soft weights are renormalized after sub-threshold components are dropped.
    Dropped hard-assigned samples are retained in the nearest surviving mode's
    gamma/delta summary without transferring their probability mass.
    """
    g_arr = np.asarray(gamma, dtype=float)
    d_arr = np.asarray(delta, dtype=float)
    chain_shape = _chain_shape(g_arr, n_chains)
    g = _as_1d(g_arr)
    d = _as_1d(d_arr)
    n = g.size
    nan = float("nan")
    if d.size != n:
        raise ValueError("gamma and delta must contain the same number of samples")
    if n == 0:
        return {
            "n_modes_ndc": 0,
            "s_mode_ndc": nan,
            "ndc_top_mode_weight": nan,
            "ndc_mode_weight_std": nan,
            "labels": np.zeros(0, dtype=int),
            "ndc_mode_list": [],
            "ndc_bic_by_k": {},
            "ndc_q50_within_mode_deg": nan,
            "ndc_q90_within_mode_deg": nan,
        }

    if n == 1:
        responsibilities = np.ones((1, 1), dtype=float)
        bic_by_k = {1: nan}
    else:
        responsibilities, bic_by_k = _diag_gmm_bic(
            np.column_stack([g, d]),
            k_max=k_max,
            seed=seed,
        )
    raw_labels = responsibilities.argmax(axis=1)
    raw_weights = responsibilities.mean(axis=0)
    clusters = [
        {
            "components": [component],
            "members": np.flatnonzero(raw_labels == component),
            "weight": float(raw_weights[component]),
        }
        for component in range(responsibilities.shape[1])
        if np.any(raw_labels == component)
    ]

    def representative(cluster: Dict[str, Any]) -> Tuple[float, float]:
        members = cluster["members"]
        return float(np.median(g[members])), float(np.median(d[members]))

    while len(clusters) > 1:
        pairs = [
            (
                float(
                    lune_distance_deg(
                        *representative(clusters[first]),
                        *representative(clusters[second]),
                    )
                ),
                first,
                second,
            )
            for first in range(len(clusters))
            for second in range(first + 1, len(clusters))
        ]
        separation, first, second = min(pairs)
        if separation >= float(mode_min_lune_deg):
            break
        clusters[first] = {
            "components": clusters[first]["components"] + clusters[second]["components"],
            "members": np.concatenate(
                [clusters[first]["members"], clusters[second]["members"]]
            ),
            "weight": clusters[first]["weight"] + clusters[second]["weight"],
        }
        del clusters[second]

    survivors = [
        cluster for cluster in clusters if cluster["weight"] >= float(mode_min_weight)
    ]
    if not survivors:
        survivors = [max(clusters, key=lambda cluster: cluster["weight"])]
    survivor_ids = {id(cluster) for cluster in survivors}
    dropped = [cluster for cluster in clusters if id(cluster) not in survivor_ids]
    for cluster in survivors:
        cluster["precision_members"] = cluster["members"].copy()

    survivor_representatives = [representative(cluster) for cluster in survivors]
    for cluster in dropped:
        members = cluster["members"]
        distances = np.column_stack(
            [
                lune_distance_deg(mode_ref[0], mode_ref[1], g[members], d[members])
                for mode_ref in survivor_representatives
            ]
        )
        nearest = np.argmin(distances, axis=1)
        for survivor_index, survivor in enumerate(survivors):
            assigned = members[nearest == survivor_index]
            survivor["members"] = np.concatenate([survivor["members"], assigned])

    weight_sum = float(sum(cluster["weight"] for cluster in survivors))
    for cluster in survivors:
        cluster["normalized_weight"] = cluster["weight"] / weight_sum
    survivors.sort(key=lambda cluster: cluster["normalized_weight"], reverse=True)
    survivor_components = [
        component for cluster in survivors for component in cluster["components"]
    ]

    labels = np.full(n, -1, dtype=int)
    mode_list = []
    for mode_index, cluster in enumerate(survivors):
        members = cluster["members"]
        precision_members = cluster["precision_members"]
        labels[members] = mode_index
        mode_ref = representative({"members": precision_members})
        distances = lune_distance_deg(
            mode_ref[0],
            mode_ref[1],
            g[precision_members],
            d[precision_members],
        )
        mean_gamma = float(np.mean(g[members]))
        mean_delta = float(np.mean(d[members]))
        mode_list.append(
            {
                "gamma": mean_gamma,
                "delta": mean_delta,
                "gamma_deg": float(np.degrees(mean_gamma)),
                "delta_deg": float(np.degrees(mean_delta)),
                "weight": float(cluster["normalized_weight"]),
                "per_chain_weight_std": _per_chain_mode_std(
                    responsibilities,
                    cluster["components"],
                    survivor_components,
                    chain_shape,
                ),
                "q50_within_mode_deg": float(np.percentile(distances, 50.0)),
                "q90_within_mode_deg": float(np.percentile(distances, 90.0)),
            }
        )

    mode0 = mode_list[0]
    return {
        "n_modes_ndc": len(mode_list),
        "s_mode_ndc": mode0["weight"],
        "ndc_top_mode_weight": mode0["weight"],
        "ndc_mode_weight_std": mode0["per_chain_weight_std"],
        "labels": labels,
        "ndc_mode_list": mode_list,
        "ndc_bic_by_k": bic_by_k,
        "ndc_q50_within_mode_deg": mode0["q50_within_mode_deg"],
        "ndc_q90_within_mode_deg": mode0["q90_within_mode_deg"],
    }


def _block_ess_rhat(
    idata: Optional["az.InferenceData"],
    var_names: Sequence[str],
    ess_target: float,
    missing_conv: float = 0.5,
    apply_rhat_gate: bool = True,
) -> Tuple[float, float, float]:
    """
    Return ``(s_conv, ess_min, r_hat_max)`` for a parameter block.

    If ``idata`` is missing or ESS cannot be computed, ``s_conv`` falls back
    to ``missing_conv`` (neutral 0.5 by default).
    """
    if idata is None:
        return float(missing_conv), float("nan"), float("nan")

    try:
        summary = az.summary(idata, var_names=list(var_names), kind="all")
    except Exception:
        return float(missing_conv), float("nan"), float("nan")

    if summary is None or summary.empty:
        return float(missing_conv), float("nan"), float("nan")

    present = [v for v in var_names if v in summary.index]
    if not present:
        return float(missing_conv), float("nan"), float("nan")

    ess_vals = summary.loc[present, "ess_bulk"].to_numpy(dtype=float)
    rhat_vals = summary.loc[present, "r_hat"].to_numpy(dtype=float)
    finite_ess = ess_vals[np.isfinite(ess_vals)]
    finite_rhat = rhat_vals[np.isfinite(rhat_vals)]

    if finite_ess.size == 0:
        return float(missing_conv), float("nan"), float("nan")

    ess_min = float(np.min(finite_ess))
    s_ess = float(np.clip(ess_min / float(ess_target), 0.0, 1.0))
    r_hat_max = float(np.max(finite_rhat)) if finite_rhat.size else float("nan")

    s_conv = s_ess
    if apply_rhat_gate and np.isfinite(r_hat_max) and r_hat_max > 1.05:
        # Soft gate: full credit at r_hat<=1.05, zero by 1.10.
        s_conv *= float(np.clip((1.10 - r_hat_max) / 0.05, 0.0, 1.0))

    return s_conv, ess_min, r_hat_max


def mt_quality_scores_dc_ndc(
    gamma: np.ndarray,
    delta: np.ndarray,
    kappa: np.ndarray,
    h: np.ndarray,
    sigma: np.ndarray,
    ref_idx: Optional[int] = None,
    gamma_ref: Optional[float] = None,
    delta_ref: Optional[float] = None,
    kappa_ref: Optional[float] = None,
    h_ref: Optional[float] = None,
    sigma_ref: Optional[float] = None,
    idata: Optional["az.InferenceData"] = None,
    n_chains: Optional[int] = None,
    dc: bool = False,
    ess_target: float = 400.0,
    kagan_good: float = 10.0,
    kagan_bad: float = 45.0,
    lune_good: float = 5.0,
    lune_bad: float = 25.0,
    mode_min_weight: float = 0.10,
    mode_min_kagan_deg: float = 15.0,
    mode_min_lune_deg: float = 15.0,
    k_max: int = 6,
    ess_healthy_bic: float = ESS_HEALTHY_BIC,
    cluster_seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute separate orientation (Qdc) and source-type (Qndc) quality scores.

    The scores measure confidence in the dominant physical mode::

        Qdc  = s_mode_dc  * (s_prec_dc  + s_conv_dc)  / 2
        Qndc = s_mode_ndc * (s_prec_ndc + s_conv_ndc) / 2

    Precision is measured within the dominant merged mode. GMM components are
    merged with Kagan distance for DC orientation and lune distance for source
    type. Convergence uses blockwise bulk ESS and R-hat.

    Parameters
    ----------
    gamma, delta, kappa, h, sigma : array-like
        Posterior Tape parameters. Shapes may be ``(n,)`` or ``(n_chains, n_draws)``.
    ref_idx, gamma_ref, ... sigma_ref : optional
        Retained for API compatibility. Mode-conditioned precision uses each
        dominant mode's pooled median rather than an external reference.
    idata : arviz.InferenceData, optional
        Used for ESS / R-hat convergence sub-scores.
    n_chains : int, optional
        Chain count for per-chain mode-weight std (inferred from 2-D inputs).
    dc : bool
        If True, non-DC scores are set to NaN (gamma/delta fixed).
    ess_target : float
        Bulk ESS target for the convergence ramp (default 400).
    kagan_good, kagan_bad : float
        Kagan quantile anchors for ``s_prec_dc`` (degrees).
        Defaults ``(10, 45)`` match the legacy ramp
        ``clip((45 - q) / 35)`` used for 6-D MT angles.
    lune_good, lune_bad : float
        Lune quantile anchors for ``s_prec_ndc`` (degrees).
    mode_min_weight : float
        Soft-weight threshold for reporting a GMM component as a mode.
    mode_min_kagan_deg, mode_min_lune_deg : float
        Physical-mode merge thresholds in degrees.
    k_max : int
        Maximum GMM component count considered by BIC.
    ess_healthy_bic : float
        Warn when a fitted block has bulk ESS below this threshold.
    cluster_seed : int
        GMM random seed.

    Returns
    -------
    result : dict
        Flat dict of Qdc/Qndc, sub-scores, quantiles, ESS, and mode summary
        columns suitable for merging into ``mt_results.csv``.
    """
    g = np.asarray(gamma, dtype=float)
    d = np.asarray(delta, dtype=float)
    k = np.asarray(kappa, dtype=float)
    hh = np.asarray(h, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if n_chains is None and k.ndim == 2:
        n_chains = int(k.shape[0])

    g1 = _as_1d(g)
    d1 = _as_1d(d)
    k1 = _as_1d(k)
    h1 = _as_1d(hh)
    s1 = _as_1d(s)
    n = k1.size
    if any(values.size != n for values in (g1, d1, h1, s1)):
        raise ValueError(
            "gamma, delta, kappa, h, and sigma must contain the same number of samples"
        )

    nan = float("nan")
    out: Dict[str, Any] = {
        "Qdc": nan,
        "Qndc": nan,
        "s_prec_dc": nan,
        "s_mode_dc": nan,
        "s_conv_dc": nan,
        "s_prec_ndc": nan,
        "s_mode_ndc": nan,
        "s_conv_ndc": nan,
        "q50_kagan_deg": nan,
        "q90_kagan_deg": nan,
        "q50_lune_deg": nan,
        "q90_lune_deg": nan,
        "ess_min_dc": nan,
        "ess_min_ndc": nan,
        "rhat_max_dc": nan,
        "rhat_max_ndc": nan,
        "n_modes_dc": nan,
        "dc_top_mode_weight": nan,
        "dc_mode_weight_std": nan,
        "dc_mode1_weight": nan,
        "dc_mode0_strike_deg": nan,
        "dc_mode0_dip_deg": nan,
        "dc_mode0_rake_deg": nan,
        "dc_mode1_strike_deg": nan,
        "dc_mode1_dip_deg": nan,
        "dc_mode1_rake_deg": nan,
        "n_modes_ndc": nan,
        "ndc_top_mode_weight": nan,
        "ndc_mode_weight_std": nan,
        "dc_mode_list": [],
        "ndc_mode_list": [],
        "dc_bic_by_k": {},
        "ndc_bic_by_k": {},
    }

    if n == 0:
        return out

    cluster = cluster_orientation_params(
        k if k.ndim == 2 else k1,
        hh if hh.ndim == 2 else h1,
        s if s.ndim == 2 else s1,
        n_chains=n_chains,
        mode_min_weight=mode_min_weight,
        mode_min_kagan_deg=mode_min_kagan_deg,
        k_max=k_max,
        seed=cluster_seed,
    )
    q50_k = float(cluster["dc_q50_within_mode_deg"])
    q90_k = float(cluster["dc_q90_within_mode_deg"])
    s_prec_dc = _precision_from_quantiles(q50_k, q90_k, kagan_good, kagan_bad)
    s_mode_dc = float(cluster["s_mode_dc"])
    dc_mode_list = [dict(mode) for mode in cluster["dc_mode_list"]]
    for mode_index, mode in enumerate(dc_mode_list):
        mask = cluster["labels"] == mode_index
        mode["mean_gamma"] = float(np.mean(g1[mask]))
        mode["mean_delta"] = float(np.mean(d1[mask]))

    s_conv_dc, ess_min_dc, rhat_max_dc = _block_ess_rhat(
        idata, ("kappa", "h", "sigma"), ess_target=ess_target
    )
    if np.isfinite(ess_min_dc) and ess_min_dc < float(ess_healthy_bic):
        warnings.warn(
            f"orientation-block ess_min={ess_min_dc:.0f} < {ess_healthy_bic:.0f}: "
            f"BIC k-selection ran on correlated particles; "
            f"n_modes_dc={cluster['n_modes_dc']} may be inflated",
            RuntimeWarning,
            stacklevel=2,
        )
    qdc = s_mode_dc * (s_prec_dc + s_conv_dc) / 2.0

    out.update(
        {
            "Qdc": float(qdc),
            "s_prec_dc": float(s_prec_dc),
            "s_mode_dc": float(s_mode_dc),
            "s_conv_dc": float(s_conv_dc),
            "q50_kagan_deg": q50_k,
            "q90_kagan_deg": q90_k,
            "ess_min_dc": float(ess_min_dc),
            "rhat_max_dc": float(rhat_max_dc),
            "n_modes_dc": float(cluster["n_modes_dc"]),
            "dc_top_mode_weight": float(cluster["dc_top_mode_weight"]),
            "dc_mode_weight_std": float(cluster["dc_mode_weight_std"]),
            "dc_mode1_weight": float(cluster["dc_mode1_weight"]),
            "dc_mode0_strike_deg": float(cluster["dc_mode0_strike_deg"]),
            "dc_mode0_dip_deg": float(cluster["dc_mode0_dip_deg"]),
            "dc_mode0_rake_deg": float(cluster["dc_mode0_rake_deg"]),
            "dc_mode1_strike_deg": float(cluster["dc_mode1_strike_deg"]),
            "dc_mode1_dip_deg": float(cluster["dc_mode1_dip_deg"]),
            "dc_mode1_rake_deg": float(cluster["dc_mode1_rake_deg"]),
            "dc_mode_list": dc_mode_list,
            "dc_bic_by_k": dict(cluster["dc_bic_by_k"]),
        }
    )

    if dc:
        return out

    lune_cluster = cluster_lune_params(
        g if g.ndim == 2 else g1,
        d if d.ndim == 2 else d1,
        n_chains=n_chains,
        mode_min_weight=mode_min_weight,
        mode_min_lune_deg=mode_min_lune_deg,
        k_max=k_max,
        seed=cluster_seed,
    )
    q50_l = float(lune_cluster["ndc_q50_within_mode_deg"])
    q90_l = float(lune_cluster["ndc_q90_within_mode_deg"])
    s_prec_ndc = _precision_from_quantiles(q50_l, q90_l, lune_good, lune_bad)
    s_mode_ndc = float(lune_cluster["s_mode_ndc"])
    s_conv_ndc, ess_min_ndc, rhat_max_ndc = _block_ess_rhat(
        idata, ("gamma", "delta"), ess_target=ess_target
    )
    if np.isfinite(ess_min_ndc) and ess_min_ndc < float(ess_healthy_bic):
        warnings.warn(
            f"source-type block ess_min={ess_min_ndc:.0f} < {ess_healthy_bic:.0f}: "
            f"BIC k-selection ran on correlated particles; "
            f"n_modes_ndc={lune_cluster['n_modes_ndc']} may be inflated",
            RuntimeWarning,
            stacklevel=2,
        )
    qndc = s_mode_ndc * (s_prec_ndc + s_conv_ndc) / 2.0

    out.update(
        {
            "Qndc": float(qndc),
            "s_prec_ndc": float(s_prec_ndc),
            "s_mode_ndc": float(s_mode_ndc),
            "s_conv_ndc": float(s_conv_ndc),
            "q50_lune_deg": q50_l,
            "q90_lune_deg": q90_l,
            "ess_min_ndc": float(ess_min_ndc),
            "rhat_max_ndc": float(rhat_max_ndc),
            "n_modes_ndc": float(lune_cluster["n_modes_ndc"]),
            "ndc_top_mode_weight": float(lune_cluster["ndc_top_mode_weight"]),
            "ndc_mode_weight_std": float(lune_cluster["ndc_mode_weight_std"]),
            "ndc_mode_list": [dict(mode) for mode in lune_cluster["ndc_mode_list"]],
            "ndc_bic_by_k": dict(lune_cluster["ndc_bic_by_k"]),
        }
    )
    return out


def mt_quality_score(
    mt6_samples: np.ndarray,
    mt6_mode: np.ndarray,
    final_weights: Optional[np.ndarray] = None,
    n_particles: Optional[int] = None,
    idata: Optional["az.InferenceData"] = None,
    ess_target: float = 400.0,
    angle_good: float = 5.0,
    angle_bad: float = 30.0,
) -> Dict[str, float]:
    """
    Compute a scalar MT quality score ``Q`` in the range [0, 1].

    The score combines three components:

    - ``s_conv`` : sampler convergence from worst-case R-hat and bulk ESS.
    - ``s_angle`` : angular concentration of MT6 samples around the mode
      (more heavily weighted).
    - ``s_modes`` : degree of unimodality in angular space.

    Higher values indicate better-constrained, better-sampled solutions.

    Parameters
    ----------
    mt6_samples : np.ndarray
        Posterior MT6 samples, shape (6, n_samples).
    mt6_mode : np.ndarray
        Reference MT6 vector (typically the MAP-like solution), shape (6,).
    final_weights : np.ndarray, optional
        Normalized SMC posterior weights. Kept for compatibility and returned
        via ``ess_weights``/``ess_ratio`` diagnostics.
    n_particles : int, optional
        Total number of particles. If None, inferred from
        ``final_weights.size``.
    idata : arviz.InferenceData, optional
        InferenceData used to compute worst-case ``\hat{R}`` and bulk ESS for
        the convergence sub-score.
    ess_target : float, default 400.0
        Target bulk ESS used in the convergence ramp.
    angle_good : float, default 5.0
        Angular half-width (degrees) that is considered very good.
    angle_bad : float, default 30.0
        Angular half-width (degrees) that is considered very poor.

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``Q`` : overall quality score in [0, 1].
        - ``s_conv`` : convergence sub-score in [0, 1].
        - ``s_angle`` : angular concentration sub-score in [0, 1].
        - ``s_modes`` : unimodality sub-score in [0, 1].
        - ``ess_weights`` : effective sample size from importance weights.
        - ``ess_ratio`` : ESS / N_particles.
        - ``q50`` : 50% angular quantile (degrees).
        - ``q90`` : 90% angular quantile (degrees).
        - ``top_mode_weight`` : weight of the dominant angular mode.
        - ``quality_label`` : qualitative label ('high', 'moderate', 'low').
    """
    # ----- 1) Convergence score (worst-case R-hat and bulk ESS) -----
    if final_weights is not None:
        w = np.asarray(final_weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        n = n_particles if n_particles is not None else w.size
        ess_weights = float(1.0 / np.sum(w**2))
        ess_ratio = ess_weights / n
    else:
        ess_weights = 0.0
        ess_ratio = 0.0

    s_conv = 0.0
    if idata is not None:
        conv_summary = az.summary(
            idata,
            var_names=["gamma", "delta", "kappa", "h", "sigma"],
            kind="all",
        )
        if not conv_summary.empty:
            r_hat_vals = conv_summary["r_hat"].to_numpy(dtype=float)
            ess_vals = conv_summary["ess_bulk"].to_numpy(dtype=float)
            finite_rhat = r_hat_vals[np.isfinite(r_hat_vals)]
            finite_ess = ess_vals[np.isfinite(ess_vals)]
            s_rhat = 0.0
            s_ess = 0.0
            if finite_rhat.size:
                r_hat_max = float(finite_rhat.max())
                s_rhat = float(np.clip((1.10 - r_hat_max) / 0.10, 0.0, 1.0))
            if finite_ess.size:
                ess_min = float(finite_ess.min())
                s_ess = float(np.clip(ess_min / ess_target, 0.0, 1.0))
            s_conv = 0.5 * s_rhat + 0.5 * s_ess

    # ----- 2) Angular concentration score -----
    ang_info = angular_uncertainty(mt6_mode, mt6_samples, quantiles=(50.0, 90.0))
    q50, q90 = map(float, ang_info["quantiles"])

    def _angle_score(q_deg: float) -> float:
        # angle_good -> score ~1, angle_bad or worse -> score ~0
        return float(np.clip((angle_bad - q_deg) / (angle_bad - angle_good), 0.0, 1.0))

    s_angle = 0.5 * _angle_score(q50) + 0.5 * _angle_score(q90)

    # ----- 3) Unimodality score -----
    mode_info = angle_mode_analysis(mt6_mode, mt6_samples)
    if mode_info["mode_weights"].size:
        top_mode_weight = float(mode_info["mode_weights"].max())
    else:
        top_mode_weight = 1.0
    s_modes = top_mode_weight

    # ----- Combine -----
    # Heavier weight on angular concentration so that wide posteriors
    # receive clearly lower scores even when convergence diagnostics
    # look good.
    Q = 0.25 * s_conv + 0.6 * s_angle + 0.15 * s_modes

    if Q >= 0.8:
        quality_label = "high"
    elif Q >= 0.5:
        quality_label = "moderate"
    else:
        quality_label = "low"

    return {
        "Q": float(Q),
        "s_conv": float(s_conv),
        "s_angle": float(s_angle),
        "s_modes": float(s_modes),
        "ess_weights": float(ess_weights),
        "ess_ratio": float(ess_ratio),
        "q50": float(q50),
        "q90": float(q90),
        "top_mode_weight": float(top_mode_weight),
        "quality_label": quality_label,
    }


def generate_synthetic_event_data(
    inversion_options: Sequence[str],
    true_mt6: np.ndarray,
    azimuth: np.ndarray,
    takeoff: np.ndarray,
    azimuth_err: float = 0.0,
    takeoff_error: float = 0.0,
    polarity_error_range: Tuple[float, float] = (0.01, 0.05),
    amplitude_error_range: Tuple[float, float] = (0.05, 0.10),
    amplitude_error_mode: str = "amplitude",
    ratio_error_boost: float = 1.5,
    amplitude_bias_lognormal_sigma: Optional[float] = None,
    amplitude_outlier_fraction: float = 0.0,
    amplitude_outlier_scale: float = 3.0,
    amplitude_ratio_noise_model: str = "legacy",
    amplitude_ratio_station_sigma: float = 0.0,
    amplitude_ratio_observation_sigma: float = 0.0,
    amplitude_ratio_outlier_fraction: float = 0.0,
    amplitude_ratio_outlier_sigma: float = 0.0,
    amplitude_ratio_min_snr: Optional[float] = None,
    amplitude_ratio_low_snr_sigma: float = 0.0,
    amplitude_ratio_station_terms: Optional[Dict[str, Any]] = None,
    amplitude_bias_num: Optional[Any] = None,
    amplitude_bias_den: Optional[Any] = None,
    amplitude_bias_P: Optional[Any] = None,
    amplitude_bias_SH: Optional[Any] = None,
    amplitude_bias_SV: Optional[Any] = None,
    uid: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate a MTfit-style synthetic event dictionary for testing.

    This simplified generator takes station geometry (azimuths and takeoff
    angles) directly from the caller instead of sampling them internally.
    It is intended for controlled synthetic experiments with known moment
    tensor and user-defined station layouts.

    Parameters
    ----------
    inversion_options
        Sequence of data-type keys such as
        ``['PPolarity', 'PAmplitude', 'P/SHAmplitudeRatio']``.
        Polarity keys (``*Polarity``), absolute amplitudes (``*Amplitude``),
        and amplitude-ratio keys (``*AmplitudeRatio``) are supported.
    true_mt6
        True MT six-vector used to generate the synthetic data, shape (6,).
    azimuth
        Array of station azimuths in degrees, shape (N,) or (N, 1).
    takeoff
        Array of station takeoff angles in degrees, shape (N,) or (N, 1).
    azimuth_err
        Per-station 1σ azimuth error (degrees). Stored in the ``'Stations'``
        dict as ``'Azimuth_err'`` so it can be used by
        :func:`build_location_samples_from_errors`.
    takeoff_error
        Per-station 1σ takeoff-angle error (degrees). Stored as
        ``'TakeOff_err'`` in the generated station dictionaries.
    polarity_error_range
        Tuple ``(min, max)`` specifying a uniform range for the polarity
        measurement uncertainties. Values are stored in the ``'Error'`` field
        for polarity datasets.
    amplitude_error_range
        Tuple ``(min, max)`` specifying a uniform range for uncertainties.
        Interpretation depends on ``amplitude_error_mode``:

        - ``"amplitude"`` (default): fractional 1σ errors on numerator and
          denominator amplitudes (current behaviour).
        - ``"ratio"``: fractional 1σ error on the **ratio itself**. This makes
          values like ``(0.8, 1.0)`` mean “~80–100% scatter” in the observed
          ratio instead of inflating per-amplitude errors.
    amplitude_error_mode
        Either ``"amplitude"`` (default, legacy behaviour) or ``"ratio"``
        (noise specified directly on the ratio).

    ratio_error_boost
        Multiplicative inflation applied to ``amplitude_error_range`` **only
        in ratio mode** to conservatively down-weight amplitude ratios. The
        default (1.5) makes ranges like (0.8, 1.0) translate to roughly
        1.7–2.1 log-sigma on the ratio, forcing the inversion to lean on
        polarity data.
    amplitude_bias_lognormal_sigma
        If provided and > 0, apply per-station multiplicative biases drawn
        from ``LogNormal(mean=0, sigma=amplitude_bias_lognormal_sigma)`` to
        synthetic amplitudes when ``amplitude_ratio_noise_model="composite"``;
        in legacy mode it retains the existing ratio-only behaviour.
    amplitude_outlier_fraction
        Fraction of stations to flag as amplitude outliers. Those stations have
        their fractional errors inflated by ``amplitude_outlier_scale``.
    amplitude_outlier_scale
        Multiplicative inflation applied to the fractional error for outlier
        stations (see ``amplitude_outlier_fraction``).
    amplitude_ratio_noise_model
        ``"legacy"`` preserves the existing amplitude and ratio noise paths.
        ``"composite"`` builds noisy P, SH, and SV amplitudes once in log
        space using shared station terms, per-observation scatter, rare
        outliers, and weak-phase inflation, then reuses that same realization
        for absolute amplitudes and any requested amplitude ratios.
    amplitude_ratio_station_sigma
        Composite-mode station-term scatter expressed as a natural-log ratio
        sigma. It is split evenly across numerator and denominator phase
        amplitudes.
    amplitude_ratio_observation_sigma
        Composite-mode per-observation scatter expressed as a natural-log ratio
        sigma. It is split evenly across numerator and denominator phase
        amplitudes.
    amplitude_ratio_outlier_fraction
        Fraction of composite-mode observations assigned an additional gross
        error term.
    amplitude_ratio_outlier_sigma
        Natural-log ratio sigma of the additional composite-mode outlier term.
    amplitude_ratio_min_snr
        Optional SNR-proxy threshold used in composite mode. Stations whose
        phase amplitudes fall below the threshold receive additional noise
        inflation, which also covers nodal weak-phase behaviour.
    amplitude_ratio_low_snr_sigma
        Additional natural-log sigma scale used when
        ``amplitude_ratio_min_snr`` is triggered in composite mode.
    amplitude_ratio_station_terms
        Optional mapping of phase name to precomputed per-station log-amplitude
        residuals used in composite mode. Keys should be phase labels such as
        ``"P"``, ``"SH"``, or ``"SV"`` and values must be scalar or length-
        ``N_stations`` arrays.
    amplitude_bias_num
        Optional systematic multiplicative bias applied to the **true**
        numerator amplitudes when generating amplitude-ratio data. Can be
        a scalar (same factor for all stations) or an array of shape
        ``(N_stations,)``. This bias is *not* known to the inversion and
        is intended for testing the effect of mis-modelled amplitudes.
    amplitude_bias_den
        Optional systematic multiplicative bias applied to the **true**
        denominator amplitudes, with the same conventions as
        ``amplitude_bias_num``.
    amplitude_bias_P, amplitude_bias_SH, amplitude_bias_SV
        Optional multiplicative biases applied per phase (P, SH, SV) before
        building any ratios. Scalars apply the same factor to all stations;
        arrays of length ``N_stations`` allow station-specific biases. These
        are applied prior to ratio-specific ``amplitude_bias_num`` /
        ``amplitude_bias_den``.
    uid
        Optional UID string for the event. If ``None``, a default UID is used.
    rng
        Optional NumPy :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    dict
        Event dictionary with the same structure as :func:`example_data.synthetic_event`.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert rng is not None

    true_mt6 = np.asarray(true_mt6, dtype=float).reshape(6)

    azimuth = np.asarray(azimuth, dtype=float).reshape(-1, 1)
    takeoff = np.asarray(takeoff, dtype=float).reshape(-1, 1)
    if azimuth.shape[0] != takeoff.shape[0]:
        raise ValueError(
            f"azimuth and takeoff must have the same length, got "
            f"{azimuth.shape[0]} and {takeoff.shape[0]}"
        )

    n_stations = azimuth.shape[0]

    station_names = [f"S{idx:04d}" for idx in range(1, n_stations + 1)]

    # Base station dictionary; copied per data type.
    stations_base: Dict[str, Any] = {
        "Azimuth": azimuth,
        "TakeOffAngle": takeoff,
        "Name": station_names,
    }
    if azimuth_err > 0.0:
        stations_base["Azimuth_err"] = np.full_like(azimuth, float(azimuth_err))
    if takeoff_error > 0.0:
        stations_base["TakeOff_err"] = np.full_like(takeoff, float(takeoff_error))

    event: Dict[str, Any] = {}
    event["UID"] = uid or "synthetic_generated_event"

    pol_err_min, pol_err_max = polarity_error_range
    amp_err_min, amp_err_max = amplitude_error_range

    if amplitude_error_mode not in ("amplitude", "ratio"):
        raise ValueError(
            "amplitude_error_mode must be 'amplitude' or 'ratio', "
            f"got {amplitude_error_mode!r}"
        )
    if amplitude_ratio_noise_model not in ("legacy", "composite"):
        raise ValueError(
            "amplitude_ratio_noise_model must be 'legacy' or 'composite', "
            f"got {amplitude_ratio_noise_model!r}"
        )
    if ratio_error_boost <= 0:
        raise ValueError("ratio_error_boost must be positive.")
    if (
        amplitude_bias_lognormal_sigma is not None
        and amplitude_bias_lognormal_sigma < 0
    ):
        raise ValueError("amplitude_bias_lognormal_sigma must be non-negative.")
    if not (0.0 <= amplitude_outlier_fraction <= 1.0):
        raise ValueError("amplitude_outlier_fraction must be in [0, 1].")
    if amplitude_outlier_scale <= 0:
        raise ValueError("amplitude_outlier_scale must be positive.")
    if amplitude_ratio_station_sigma < 0:
        raise ValueError("amplitude_ratio_station_sigma must be non-negative.")
    if amplitude_ratio_observation_sigma < 0:
        raise ValueError("amplitude_ratio_observation_sigma must be non-negative.")
    if not (0.0 <= amplitude_ratio_outlier_fraction <= 1.0):
        raise ValueError("amplitude_ratio_outlier_fraction must be in [0, 1].")
    if amplitude_ratio_outlier_sigma < 0:
        raise ValueError("amplitude_ratio_outlier_sigma must be non-negative.")
    if amplitude_ratio_min_snr is not None and amplitude_ratio_min_snr <= 0:
        raise ValueError("amplitude_ratio_min_snr must be positive when provided.")
    if amplitude_ratio_low_snr_sigma < 0:
        raise ValueError("amplitude_ratio_low_snr_sigma must be non-negative.")

    def _apply_bias(arr: np.ndarray, bias: Optional[Any], label: str) -> np.ndarray:
        if bias is None:
            return arr
        bias_arr = np.asarray(bias, dtype=float).reshape(-1)
        if bias_arr.size == 1:
            return arr * bias_arr[0]
        if bias_arr.size == n_stations:
            return arr * bias_arr
        raise ValueError(f"{label} must be scalar or have length n_stations")

    phase_bias = {
        "p": amplitude_bias_P,
        "sh": amplitude_bias_SH,
        "sv": amplitude_bias_SV,
    }

    def _copy_stations() -> Dict[str, Any]:
        stations = {
            "Azimuth": azimuth.copy(),
            "TakeOffAngle": takeoff.copy(),
            "Name": list(station_names),
        }
        if "Azimuth_err" in stations_base:
            stations["Azimuth_err"] = stations_base["Azimuth_err"].copy()
        if "TakeOff_err" in stations_base:
            stations["TakeOff_err"] = stations_base["TakeOff_err"].copy()
        return stations

    def _normalise_phase_name(label: str) -> str:
        phase = label.replace("_", "").lower()
        phase = phase.rstrip("rms")
        phase = phase.rstrip("q")
        return phase

    composite_mode = amplitude_ratio_noise_model == "composite"
    composite_phase_cache: Dict[str, Dict[str, np.ndarray]] = {}

    if composite_mode:
        required_phases = set()
        for key in inversion_options:
            key_lower_clean = str(key).lower().replace("_", "")
            if "amplituderatio" in key_lower_clean:
                ratio_phase = key_lower_clean.split("amplituderatio")[0]
                ratio_phase = ratio_phase.rstrip("rms").rstrip("q")
                if "/" in ratio_phase:
                    num_phase, den_phase = ratio_phase.split("/", 1)
                    required_phases.add(num_phase)
                    required_phases.add(den_phase)
            elif "amplitude" in key_lower_clean:
                phase = key_lower_clean.split("amplitude")[0]
                phase = phase.rstrip("rms").rstrip("q")
                if phase:
                    required_phases.add(phase)

        station_terms_lookup: Dict[str, np.ndarray] = {}
        if amplitude_ratio_station_terms is not None:
            for phase_name, values in amplitude_ratio_station_terms.items():
                phase_key = _normalise_phase_name(str(phase_name))
                arr = np.asarray(values, dtype=float).reshape(-1)
                if arr.size == 1:
                    arr = np.full(n_stations, float(arr[0]), dtype=float)
                elif arr.size != n_stations:
                    raise ValueError(
                        "amplitude_ratio_station_terms[{!r}] must be scalar or have length n_stations".format(
                            phase_name
                        )
                    )
                station_terms_lookup[phase_key] = arr.astype(float)

        station_sigma_component = amplitude_ratio_station_sigma / np.sqrt(2.0)
        observation_sigma_component = amplitude_ratio_observation_sigma / np.sqrt(2.0)
        outlier_sigma_component = amplitude_ratio_outlier_sigma / np.sqrt(2.0)
        base_frac_scale = max(0.5 * (amp_err_min + amp_err_max), 1e-6)

        for phase in sorted(required_phases):
            amp_true = np.asarray(
                forward_amplitude(true_mt6, stations_base, phase.upper()),
                dtype=float,
            ).reshape(-1)
            amp_true = _apply_bias(
                amp_true,
                phase_bias.get(phase),
                f"amplitude_bias_{phase.upper()}",
            )
            amp_true = np.abs(amp_true)
            amp_safe = np.clip(amp_true, 1e-12, np.inf)
            log_amp_true = np.log(amp_safe)

            if amplitude_bias_lognormal_sigma and amplitude_bias_lognormal_sigma > 0:
                phase_bias_term = rng.normal(
                    0.0, amplitude_bias_lognormal_sigma, size=n_stations
                )
            else:
                phase_bias_term = np.zeros(n_stations, dtype=float)

            if phase in station_terms_lookup:
                station_term = station_terms_lookup[phase].copy()
            elif station_sigma_component > 0.0:
                station_term = rng.normal(0.0, station_sigma_component, size=n_stations)
            else:
                station_term = np.zeros(n_stations, dtype=float)

            base_sigma = rng.uniform(amp_err_min, amp_err_max, size=n_stations)
            if amplitude_outlier_fraction > 0.0:
                legacy_outlier_mask = (
                    rng.random(size=n_stations) < amplitude_outlier_fraction
                )
                base_sigma = np.where(
                    legacy_outlier_mask,
                    base_sigma * amplitude_outlier_scale,
                    base_sigma,
                )
            base_term = rng.normal(0.0, base_sigma)

            if observation_sigma_component > 0.0:
                observation_term = rng.normal(
                    0.0, observation_sigma_component, size=n_stations
                )
            else:
                observation_term = np.zeros(n_stations, dtype=float)

            outlier_sigma = np.zeros(n_stations, dtype=float)
            if amplitude_ratio_outlier_fraction > 0.0 and outlier_sigma_component > 0.0:
                ratio_outlier_mask = (
                    rng.random(size=n_stations) < amplitude_ratio_outlier_fraction
                )
                outlier_sigma = np.where(
                    ratio_outlier_mask,
                    outlier_sigma_component,
                    0.0,
                )
            outlier_term = rng.normal(0.0, outlier_sigma)

            low_snr_sigma = np.zeros(n_stations, dtype=float)
            if (
                amplitude_ratio_min_snr is not None
                and amplitude_ratio_low_snr_sigma > 0.0
            ):
                reference_amp = max(float(np.median(amp_safe)), 1e-12)
                snr_proxy = amp_safe / max(base_frac_scale * reference_amp, 1e-12)
                snr_shortfall = np.clip(
                    amplitude_ratio_min_snr / np.maximum(snr_proxy, 1e-12) - 1.0,
                    0.0,
                    5.0,
                )
                low_snr_sigma = amplitude_ratio_low_snr_sigma * snr_shortfall
            low_snr_term = rng.normal(0.0, low_snr_sigma)

            total_sigma = np.sqrt(
                base_sigma**2
                + station_sigma_component**2
                + observation_sigma_component**2
                + outlier_sigma**2
                + low_snr_sigma**2
            )
            measured = np.exp(
                log_amp_true
                + phase_bias_term
                + station_term
                + base_term
                + observation_term
                + outlier_term
                + low_snr_term
            )
            composite_phase_cache[phase] = {
                "measured": measured.astype(float),
                "error": (total_sigma * measured).astype(float),
                "sigma": total_sigma.astype(float),
            }

    # ---- Build each requested data type ----
    for key in inversion_options:
        key_str = str(key)
        key_lower = key_str.lower()

        # Polarity data (e.g. 'PPolarity', 'SHPolarity', 'SVPolarity')
        if "polarity" in key_lower and "prob" not in key_lower:
            phase = key_lower.split("polarity")[0]  # 'p', 'sh', 'sv'
            if phase == "":
                continue

            # Forward P/SH/SV amplitude for true MT to get polarity sign
            amp_pred = forward_amplitude(true_mt6, stations_base, phase.upper())
            amp_pred = np.asarray(amp_pred, dtype=float).reshape(-1)

            measured = np.where(amp_pred >= 0.0, 1, -1).astype(int).reshape(-1, 1)
            error_vals = rng.uniform(pol_err_min, pol_err_max, size=n_stations).reshape(
                -1, 1
            )

            event[key_str] = {
                "Measured": measured,
                "Error": error_vals,
                "Stations": _copy_stations(),
            }
        # Absolute amplitude data (e.g. 'PAmplitude', 'SHAmplitude', 'SVAmplitude')
        elif (
            "amplitude" in key_lower
            and "amplituderatio" not in key_lower
            and "amplitude_ratio" not in key_lower
        ):
            phase = key_lower.replace("_", "").split("amplitude")[0]  # 'p', 'sh', 'sv'
            phase = phase.rstrip("rms").rstrip("q")
            if phase == "":
                continue

            if composite_mode and phase in composite_phase_cache:
                event[key_str] = {
                    "Measured": composite_phase_cache[phase]["measured"].reshape(-1, 1),
                    "Error": composite_phase_cache[phase]["error"].reshape(-1, 1),
                    "Stations": _copy_stations(),
                }
                continue

            amp_true = np.asarray(
                forward_amplitude(true_mt6, stations_base, phase.upper()),
                dtype=float,
            ).reshape(-1)

            # Apply per-phase bias if requested
            amp_true = _apply_bias(
                amp_true, phase_bias.get(phase), f"amplitude_bias_{phase.upper()}"
            )

            amp_true = np.abs(amp_true)
            amp_safe = np.clip(amp_true, 1e-12, np.inf)

            # Draw fractional errors and convert to absolute 1σ errors
            frac = rng.uniform(amp_err_min, amp_err_max, size=n_stations)
            if amplitude_outlier_fraction > 0.0:
                mask = rng.random(size=n_stations) < amplitude_outlier_fraction
                frac = np.where(mask, frac * amplitude_outlier_scale, frac)
            err = frac * amp_safe

            # Add Gaussian noise to simulate measurement error
            noise = rng.normal(0.0, err)
            meas = amp_true + noise

            measured = meas.reshape(-1, 1)
            error = err.reshape(-1, 1)

            event[key_str] = {
                "Measured": measured,
                "Error": error,
                "Stations": _copy_stations(),
            }

        # Amplitude ratio data (e.g. 'P/SHAmplitudeRatio')
        elif "amplituderatio" in key_lower or "amplitude_ratio" in key_lower:
            phase = key_lower.replace("_", "").split("amplituderatio")[0]
            phase = phase.rstrip("rms").rstrip("q")
            if "/" not in phase:
                continue

            num_phase, den_phase = phase.split("/", 1)

            if composite_mode:
                if (
                    num_phase not in composite_phase_cache
                    or den_phase not in composite_phase_cache
                ):
                    raise KeyError(
                        "Composite amplitude cache missing phase pair {} / {}".format(
                            num_phase, den_phase
                        )
                    )

                num_meas = composite_phase_cache[num_phase]["measured"].copy()
                den_meas = composite_phase_cache[den_phase]["measured"].copy()
                num_sigma = composite_phase_cache[num_phase]["sigma"].copy()
                den_sigma = composite_phase_cache[den_phase]["sigma"].copy()

                if amplitude_bias_num is not None:
                    num_meas = _apply_bias(
                        num_meas, amplitude_bias_num, "amplitude_bias_num"
                    )
                if amplitude_bias_den is not None:
                    den_meas = _apply_bias(
                        den_meas, amplitude_bias_den, "amplitude_bias_den"
                    )

                ratio_meas = np.abs(
                    np.clip(num_meas, 1e-12, np.inf) / np.clip(den_meas, 1e-12, np.inf)
                )
                measured = ratio_meas.reshape(-1, 1)
                error = np.column_stack([ratio_meas * num_sigma, den_sigma])

                event[key_str] = {
                    "Measured": measured,
                    "Error": error,
                    "Stations": _copy_stations(),
                }
                continue

            num_amp = np.asarray(
                forward_amplitude(true_mt6, stations_base, num_phase.upper()),
                dtype=float,
            ).reshape(-1)
            den_amp = np.asarray(
                forward_amplitude(true_mt6, stations_base, den_phase.upper()),
                dtype=float,
            ).reshape(-1)

            # Apply per-phase biases first (P/SH/SV)
            num_amp = _apply_bias(
                num_amp,
                phase_bias.get(num_phase),
                f"amplitude_bias_{num_phase.upper()}",
            )
            den_amp = _apply_bias(
                den_amp,
                phase_bias.get(den_phase),
                f"amplitude_bias_{den_phase.upper()}",
            )

            # Optional per-station random biases (lognormal) to mimic site/path
            # effects in a realistic way.
            if amplitude_bias_lognormal_sigma and amplitude_bias_lognormal_sigma > 0:
                bias_num_ln = rng.lognormal(
                    mean=0.0, sigma=amplitude_bias_lognormal_sigma, size=n_stations
                )
                bias_den_ln = rng.lognormal(
                    mean=0.0, sigma=amplitude_bias_lognormal_sigma, size=n_stations
                )
                num_amp = num_amp * bias_num_ln
                den_amp = den_amp * bias_den_ln

            # Apply optional systematic multiplicative bias to the *true*
            # amplitudes before adding random noise. This simulates path /
            # site effects or mis-calibration in the data-generating
            # process that are not accounted for in the inversion model.
            if amplitude_bias_num is not None:
                bias_num = np.asarray(amplitude_bias_num, dtype=float).reshape(-1)
                if bias_num.size == 1:
                    num_amp = num_amp * bias_num[0]
                elif bias_num.size == n_stations:
                    num_amp = num_amp * bias_num
                else:
                    raise ValueError(
                        "amplitude_bias_num must be scalar or have length n_stations"
                    )
            if amplitude_bias_den is not None:
                bias_den = np.asarray(amplitude_bias_den, dtype=float).reshape(-1)
                if bias_den.size == 1:
                    den_amp = den_amp * bias_den[0]
                elif bias_den.size == n_stations:
                    den_amp = den_amp * bias_den
                else:
                    raise ValueError(
                        "amplitude_bias_den must be scalar or have length n_stations"
                    )

            num_amp = np.abs(num_amp)
            den_amp = np.abs(den_amp)

            # Avoid zero amplitudes in error computation
            num_safe = np.clip(num_amp, 1e-12, np.inf)
            den_safe = np.clip(den_amp, 1e-12, np.inf)

            if amplitude_error_mode == "ratio":
                # Noise is specified on the ratio directly. Draw fractional
                # scatter for the ratio and perturb with a lognormal factor.
                # Inflate by ratio_error_boost to make ratios more conservative.
                frac = rng.uniform(amp_err_min, amp_err_max, size=n_stations)
                frac = frac * ratio_error_boost
                if amplitude_outlier_fraction > 0.0:
                    mask = rng.random(size=n_stations) < amplitude_outlier_fraction
                    frac = np.where(mask, frac * amplitude_outlier_scale, frac)
                sigma_log = np.log1p(frac)

                ratio_true = np.abs(num_safe / den_safe)
                ratio_meas = ratio_true * rng.lognormal(mean=0.0, sigma=sigma_log)

                # Represent this as a single-column “Measured” ratio. Encode
                # the fractional error in the numerator column; leave the
                # denominator error at the same fractional level (unit
                # denominator) so perc_err1 ≈ perc_err2 ≈ frac and
                # log_sigma ≈ frac * sqrt(2).
                err_num = frac * np.abs(ratio_meas)
                err_den = frac * np.ones_like(frac)

                measured = ratio_meas.reshape(-1, 1)
                error = np.column_stack([err_num, err_den])

            else:
                # Fractional amplitude errors (legacy behaviour)
                frac1 = rng.uniform(amp_err_min, amp_err_max, size=n_stations)
                frac2 = rng.uniform(amp_err_min, amp_err_max, size=n_stations)
                if amplitude_outlier_fraction > 0.0:
                    mask = rng.random(size=n_stations) < amplitude_outlier_fraction
                    frac1 = np.where(mask, frac1 * amplitude_outlier_scale, frac1)
                    frac2 = np.where(mask, frac2 * amplitude_outlier_scale, frac2)
                err_num = frac1 * num_safe
                err_den = frac2 * den_safe

                # Add Gaussian noise to amplitudes to simulate measurement error
                noise_num = rng.normal(0.0, err_num)
                noise_den = rng.normal(0.0, err_den)
                meas_num = num_amp + noise_num
                meas_den = den_amp + noise_den

                measured = np.column_stack([meas_num, meas_den])
                error = np.column_stack([err_num, err_den])

            event[key_str] = {
                "Measured": measured,
                "Error": error,
                "Stations": _copy_stations(),
            }

    return event


# ------------------------------------------------------------------
# Helper functions moved from synthetic_event.py
# ------------------------------------------------------------------


def project_to_dc_mt6(mt33: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 MT onto the nearest double-couple (remove iso + set eigenvalues to [+m0,0,-m0]).
    Returns MT6 column vector with sqrt(2) scaling (MTfit convention).
    """
    mt = np.asarray(mt33, dtype=float)
    if mt.shape != (3, 3):
        raise ValueError(f"mt33 must be 3x3, got {mt.shape}")

    iso = np.trace(mt) / 3.0
    dev = mt - iso * np.eye(3)
    vals, vecs = np.linalg.eigh(dev)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    m0 = 0.5 * (abs(vals[0]) + abs(vals[2]))
    dc_vals = np.array([m0, 0.0, -m0], dtype=float)
    mt_dc = vecs @ np.diag(dc_vals) @ vecs.T
    return MT33_MT6(mt_dc)


def load_mtfit_reference_mt6(path: str = "synthetic_event.out") -> np.ndarray | None:
    """
    Load the MTfit "Sample Max Probability MT" six-vector from synthetic_event.out.

    Returns the last such block in the file (full MT inversion) as a 1D array of
    length 6, or None if it cannot be parsed.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        lines = f.readlines()

    indices = [i for i, line in enumerate(lines) if "Sample Max Probability MT" in line]
    if not indices:
        return None

    start = indices[-1] + 1
    values: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        # Expect lines like "[[-0.1579]" or "[ 0.1455]]"
        stripped = stripped.strip("[]")
        try:
            values.append(float(stripped))
        except ValueError:
            break

    if len(values) != 6:
        return None
    return np.asarray(values, dtype=float)
