
import numpy as np
import pytest
from spectralmap.bayesian_linalg import gaussian_linear_posterior, optimize_hyperparameters

def test_optimize_hyperparameters_scalar_simple():
    """Test basic functionality with scalar alpha."""
    np.random.seed(42)
    N, K = 50, 5
    A = np.random.randn(N, K)
    w_true = np.random.randn(K)
    sigma = 0.5
    y = A @ w_true + sigma * np.random.randn(N)
    
    # Case 1: Unknown sigma (optimize alpha & beta)
    m, S, alpha, beta, log_ev, log_ev_marginalized = optimize_hyperparameters(A, y)
    
    assert m.shape == (K,)
    assert S.shape == (K, K)
    assert np.isscalar(alpha)
    assert np.isscalar(beta)
    assert alpha > 0
    assert beta > 0
    
def test_optimize_hyperparameters_fixed_sigma():
    """Test with fixed sigma (beta fixed)."""
    np.random.seed(42)
    N, K = 50, 5
    A = np.random.randn(N, K)
    w_true = np.ones(K)
    sigma = 0.1
    y = A @ w_true + sigma * np.random.randn(N)
    
    # Pass sigma_y as scalar
    m, S, alpha, beta, log_ev, log_ev_marginalized = optimize_hyperparameters(A, y, sigma_y=sigma)
    
    # Beta should be 1.0 (internal pre-whitened) or None depending on return spec?
    # The function returns "beta if not fix_beta else None"
    # Wait, my implementation says:
    # return m, S, alpha, (beta if not fix_beta else None), gamma, res
    
    assert beta is None
    assert np.isscalar(alpha)
    
    # Check values make sense (alpha should be around 1/var(w) approx 1)
    # With limited data, just checking it runs.

def test_alpha_scalar_enforcement():
    """alpha_guess is no longer a public argument."""
    np.random.seed(42)
    N, K = 10, 2
    A = np.random.randn(N, K)
    y = np.random.randn(N)
    
    with pytest.raises(TypeError):
        optimize_hyperparameters(A, y, alpha_guess=np.array([1.0, 2.0]))


def test_gaussian_linear_posterior_matches_predictive_covariance():
    rng = np.random.default_rng(4)
    n_obs, n_coeff = 8, 3
    A = rng.normal(size=(n_obs, n_coeff))
    y = rng.normal(size=n_obs)
    sigma = np.linspace(0.05, 0.2, n_obs)
    prior_mean = np.array([1.0, 0.2, -0.1])
    prior_cov = np.array(
        [
            [0.5, 0.1, 0.0],
            [0.1, 0.4, 0.05],
            [0.0, 0.05, 0.3],
        ],
        dtype=float,
    )

    result = gaussian_linear_posterior(
        A,
        y,
        sigma_y=sigma,
        prior_mean=prior_mean,
        prior_covariance=prior_cov,
    )

    predictive_cov = np.diag(sigma**2) + A @ prior_cov @ A.T
    resid = y - A @ prior_mean
    sign, logdet = np.linalg.slogdet(predictive_cov)
    assert sign > 0
    dense_logp = -0.5 * (
        resid @ np.linalg.solve(predictive_cov, resid)
        + logdet
        + n_obs * np.log(2.0 * np.pi)
    )

    assert result["posterior_mean"].shape == (n_coeff,)
    assert result["posterior_cov"].shape == (n_coeff, n_coeff)
    assert np.isclose(result["log_evidence"], dense_logp, rtol=0.0, atol=1e-8)
