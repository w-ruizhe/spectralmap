import numpy as np

from spectralmap.gp import (
    great_circle_distance,
    project_pixel_covariance_to_coefficients,
    separable_covariance,
    spherical_squared_exponential_covariance,
    squared_exponential_1d_covariance,
)


def test_spherical_squared_exponential_covariance_is_symmetric_psd():
    lat = np.array([0.0, 0.0, 45.0, -30.0])
    lon = np.array([0.0, 90.0, 180.0, -90.0])

    distance = great_circle_distance(lat, lon)
    cov = spherical_squared_exponential_covariance(
        lat,
        lon,
        amplitude=2.0,
        length_scale=1.0,
        jitter=1e-10,
    )

    assert distance.shape == (4, 4)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(cov, cov.T)
    assert np.all(np.isfinite(cov))
    assert np.min(np.linalg.eigvalsh(cov)) > 0.0


def test_project_pixel_covariance_to_coefficients_has_low_pixel_rank():
    rng = np.random.default_rng(5)
    n_pix, n_coeff = 30, 5
    basis = rng.normal(size=(n_pix, n_coeff))
    pixel_cov = spherical_squared_exponential_covariance(
        np.linspace(-60.0, 60.0, n_pix),
        np.linspace(-180.0, 180.0, n_pix),
        amplitude=1.0,
        length_scale=0.7,
        jitter=1e-9,
    )

    coeff_cov = project_pixel_covariance_to_coefficients(basis, pixel_cov, jitter=1e-10)
    approx_pixel_cov = basis @ coeff_cov @ basis.T

    assert coeff_cov.shape == (n_coeff, n_coeff)
    assert np.allclose(coeff_cov, coeff_cov.T)
    assert np.linalg.matrix_rank(approx_pixel_cov, tol=1e-8) <= n_coeff


def test_separable_covariance_shape():
    depth_cov = squared_exponential_1d_covariance([0.0, 0.5, 1.0], length_scale=0.4)
    coeff_cov = np.eye(4)
    joint = separable_covariance(depth_cov, coeff_cov)

    assert joint.shape == (12, 12)
    assert np.allclose(joint[:4, 4:8], depth_cov[0, 1] * coeff_cov)
