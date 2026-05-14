import numpy as np
import pytest

pytest.importorskip("starry")

from spectralmap.doppler import DopplerMap


def test_wind_broadened_design_matrix_zero_velocity():
    wavelength = np.linspace(1.0, 1.01, 6)
    spectrum = np.linspace(0.2, 0.8, wavelength.size)
    wind_velocity = np.array([0.0, 0.0])
    surface_design_matrix = np.eye(2)
    weights = np.array([0.25, 0.75])

    design = DopplerMap.wind_broadened_design_matrix(
        wavelength=wavelength,
        spectrum=spectrum,
        wind_velocity=wind_velocity,
        surface_design_matrix=surface_design_matrix,
        weights=weights,
    )

    expected = np.column_stack([weights[0] * spectrum, weights[1] * spectrum])
    assert design.shape == (wavelength.size, 2)
    assert np.allclose(design, expected)


def test_gaussian_linear_evidence_matches_dense_covariance():
    rng = np.random.default_rng(1)
    n_obs, n_coeff = 7, 3
    design = rng.normal(size=(n_obs, n_coeff))
    flux = rng.normal(size=n_obs)
    sigma_y = np.linspace(0.08, 0.15, n_obs)
    prior_mean = np.array([1.0, 0.1, -0.2])
    prior_precision = np.array([3.0, 5.0, 7.0])

    result = DopplerMap.gaussian_linear_evidence(
        design_matrix=design,
        flux=flux,
        sigma_y=sigma_y,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )

    prior_cov = np.diag(1.0 / prior_precision)
    predictive_cov = np.diag(sigma_y**2) + design @ prior_cov @ design.T
    resid = flux - design @ prior_mean
    sign, logdet = np.linalg.slogdet(predictive_cov)
    assert sign > 0
    dense_logp = -0.5 * (
        resid @ np.linalg.solve(predictive_cov, resid)
        + logdet
        + n_obs * np.log(2.0 * np.pi)
    )

    assert np.isclose(result["log_evidence"], dense_logp, rtol=0.0, atol=1e-8)
    assert result["posterior_mean"].shape == (n_coeff,)
    assert result["posterior_cov"].shape == (n_coeff, n_coeff)


def test_gaussian_linear_evidence_svd_rank_reduction_matches_dense_covariance():
    rng = np.random.default_rng(2)
    n_obs, n_coeff = 9, 5
    base = rng.normal(size=(n_obs, 3))
    design = np.column_stack(
        [
            base[:, 0],
            base[:, 1],
            base[:, 2],
            base[:, 0] + 2.0 * base[:, 1],
            np.zeros(n_obs),
        ]
    )
    flux = rng.normal(size=n_obs)
    sigma_y = np.linspace(0.05, 0.12, n_obs)
    prior_mean = np.array([1.0, 0.1, -0.2, 0.0, 0.3])
    prior_precision = 6.0

    result = DopplerMap.gaussian_linear_evidence(
        design_matrix=design,
        flux=flux,
        sigma_y=sigma_y,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
    )
    dense = DopplerMap.gaussian_linear_evidence(
        design_matrix=design,
        flux=flux,
        sigma_y=sigma_y,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        use_svd=False,
    )

    prior_cov = np.eye(n_coeff) / prior_precision
    predictive_cov = np.diag(sigma_y**2) + design @ prior_cov @ design.T
    resid = flux - design @ prior_mean
    sign, logdet = np.linalg.slogdet(predictive_cov)
    assert sign > 0
    dense_logp = -0.5 * (
        resid @ np.linalg.solve(predictive_cov, resid)
        + logdet
        + n_obs * np.log(2.0 * np.pi)
    )

    assert result["used_svd"] is True
    assert result["basis_rank"] == 3
    assert result["n_null"] == 2
    assert np.isclose(result["log_evidence"], dense_logp, rtol=0.0, atol=1e-8)
    assert np.isclose(result["log_evidence"], dense["log_evidence"], rtol=0.0, atol=1e-8)
    assert np.allclose(result["posterior_mean"], dense["posterior_mean"], atol=1e-8)


def test_surface_marginalized_wind_evidence_returns_design_matrix():
    wavelength = np.linspace(1.0, 1.01, 5)
    spectrum = 1.0 - 0.1 * np.cos(np.linspace(0.0, 2.0 * np.pi, wavelength.size))
    wind_velocity = np.array([[0.0, 1.0], [1.0, -1.0]])
    surface_design_matrix = np.array([[1.0, -0.5], [1.0, 0.5]])
    weights = np.array([[0.4, 0.6], [0.5, 0.5]])
    prior_mean = np.array([1.0, 0.0])

    dummy = DopplerMap.__new__(DopplerMap)
    design = DopplerMap.wind_broadened_design_matrix(
        wavelength=wavelength,
        spectrum=spectrum,
        wind_velocity=wind_velocity,
        surface_design_matrix=surface_design_matrix,
        weights=weights,
    )
    flux = design @ prior_mean
    result = dummy.surface_marginalized_wind_evidence(
        flux=flux,
        wavelength=wavelength,
        spectrum=spectrum,
        wind_velocity=wind_velocity,
        surface_design_matrix=surface_design_matrix,
        sigma_y=0.1,
        weights=weights,
        prior_mean=prior_mean,
        prior_precision=np.array([1.0, 25.0]),
        return_design_matrix=True,
    )

    assert np.isfinite(result["log_evidence"])
    assert result["design_matrix"].shape == design.shape
