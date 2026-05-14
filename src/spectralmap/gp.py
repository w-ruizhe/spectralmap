"""Gaussian-process covariance helpers for surface-map priors.

The functions in this module are mode independent: callers provide the pixel
grid and the basis matrix that maps coefficients to pixels.
"""

from __future__ import annotations

import numpy as np


def _as_1d_float(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square.")
    return 0.5 * (arr + arr.T)


def stabilize_covariance(covariance: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """Symmetrize a covariance matrix and add diagonal jitter."""
    cov = symmetrize(covariance)
    jitter = float(jitter)
    if jitter < 0.0 or not np.isfinite(jitter):
        raise ValueError("jitter must be finite and non-negative.")
    if jitter > 0.0:
        cov = cov + np.eye(cov.shape[0]) * jitter
    return cov


def great_circle_distance(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    radius: float = 1.0,
) -> np.ndarray:
    """Pairwise great-circle distances for latitude/longitude coordinates.

    Parameters
    ----------
    lat_deg, lon_deg
        Latitude and longitude in degrees. Inputs are flattened and must have
        the same length.
    radius
        Optional physical radius multiplier. The default returns angular
        distance in radians.
    """
    lat = np.deg2rad(_as_1d_float(lat_deg, "lat_deg"))
    lon = np.deg2rad(_as_1d_float(lon_deg, "lon_deg"))
    if lat.shape != lon.shape:
        raise ValueError("lat_deg and lon_deg must have the same shape.")

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    delta_lon = lon[:, np.newaxis] - lon[np.newaxis, :]
    cos_angle = (
        sin_lat[:, np.newaxis] * sin_lat[np.newaxis, :]
        + cos_lat[:, np.newaxis] * cos_lat[np.newaxis, :] * np.cos(delta_lon)
    )
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(radius) * angle


def squared_exponential_covariance_from_distance(
    distance: np.ndarray,
    *,
    amplitude: float = 1.0,
    length_scale: float = 1.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """Squared-exponential covariance from a precomputed distance matrix."""
    dist = np.asarray(distance, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distance must be a square matrix.")
    if np.any(~np.isfinite(dist)):
        raise ValueError("distance must contain only finite values.")

    amplitude = float(amplitude)
    length_scale = float(length_scale)
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite and strictly positive.")
    if length_scale <= 0.0 or not np.isfinite(length_scale):
        raise ValueError("length_scale must be finite and strictly positive.")

    cov = amplitude**2 * np.exp(-0.5 * (dist / length_scale) ** 2)
    return stabilize_covariance(cov, jitter=jitter)


def spherical_squared_exponential_covariance(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    amplitude: float = 1.0,
    length_scale: float = 1.0,
    radius: float = 1.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """Squared-exponential pixel covariance on a sphere.

    ``length_scale`` is measured in the same units as ``radius``. With the
    default ``radius=1``, use radians.
    """
    distance = great_circle_distance(lat_deg, lon_deg, radius=radius)
    return squared_exponential_covariance_from_distance(
        distance,
        amplitude=amplitude,
        length_scale=length_scale,
        jitter=jitter,
    )


def squared_exponential_1d_covariance(
    coordinates: np.ndarray,
    *,
    amplitude: float = 1.0,
    length_scale: float = 1.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """Squared-exponential covariance for one-dimensional coordinates."""
    x = _as_1d_float(coordinates, "coordinates")
    distance = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
    return squared_exponential_covariance_from_distance(
        distance,
        amplitude=amplitude,
        length_scale=length_scale,
        jitter=jitter,
    )


def pressure_squared_exponential_covariance(
    pressure: np.ndarray,
    *,
    amplitude: float = 1.0,
    length_scale: float = 1.0,
    jitter: float = 0.0,
) -> np.ndarray:
    """Squared-exponential covariance in natural log-pressure coordinates."""
    pressure = _as_1d_float(pressure, "pressure")
    if np.any(pressure <= 0.0):
        raise ValueError("pressure values must be strictly positive.")
    return squared_exponential_1d_covariance(
        np.log(pressure),
        amplitude=amplitude,
        length_scale=length_scale,
        jitter=jitter,
    )


def weighted_pseudoinverse(basis: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return the pseudoinverse of a pixel basis matrix.

    ``basis`` has shape ``(n_pixel, n_coeff)``. If ``weights`` are provided,
    the least-squares metric is weighted in pixel space.
    """
    B = np.asarray(basis, dtype=float)
    if B.ndim != 2:
        raise ValueError("basis must be a 2D array with shape (n_pixel, n_coeff).")
    if not np.all(np.isfinite(B)):
        raise ValueError("basis must contain only finite values.")

    if weights is None:
        return np.linalg.pinv(B)

    w = _as_1d_float(weights, "weights")
    if w.shape != (B.shape[0],):
        raise ValueError(f"weights must have length {B.shape[0]}.")
    if np.any(w <= 0.0):
        raise ValueError("weights must be strictly positive.")
    sqrt_w = np.sqrt(w)
    return np.linalg.pinv(B * sqrt_w[:, np.newaxis]) * sqrt_w[np.newaxis, :]


def project_pixel_covariance_to_coefficients(
    basis: np.ndarray,
    pixel_covariance: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    jitter: float = 0.0,
) -> np.ndarray:
    """Project a pixel-space covariance into coefficient space.

    This returns ``B_plus @ K_pixel @ B_plus.T``, where ``B_plus`` is the
    weighted pseudoinverse of ``basis``.
    """
    B = np.asarray(basis, dtype=float)
    K_pix = np.asarray(pixel_covariance, dtype=float)
    if K_pix.shape != (B.shape[0], B.shape[0]):
        raise ValueError(
            f"pixel_covariance must have shape ({B.shape[0]}, {B.shape[0]}), "
            f"got {K_pix.shape}."
        )
    B_plus = weighted_pseudoinverse(B, weights=weights)
    K_coeff = B_plus @ K_pix @ B_plus.T
    return stabilize_covariance(K_coeff, jitter=jitter)


def separable_covariance(depth_covariance: np.ndarray, coefficient_covariance: np.ndarray) -> np.ndarray:
    """Return the Kronecker product covariance for depth and coefficient axes."""
    return stabilize_covariance(
        np.kron(symmetrize(depth_covariance), symmetrize(coefficient_covariance))
    )
