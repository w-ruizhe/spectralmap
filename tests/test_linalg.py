
import numpy as np
import pytest
from spectralmap.bayesian_linalg import optimize_hyperparameters

def test_optimize_hyperparameters_scalar_simple():
    """Test basic functionality with scalar alpha."""
    np.random.seed(42)
    N, K = 50, 5
    A = np.random.randn(N, K)
    w_true = np.random.randn(K)
    sigma = 0.5
    y = A @ w_true + sigma * np.random.randn(N)
    
    # Case 1: Unknown sigma (optimize alpha & beta)
    m, S, alpha, beta, gamma, log_ev, log_ev_marginalized = optimize_hyperparameters(A, y, alpha_guess=1.0)
    
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
    m, S, alpha, beta, gamma, log_ev, log_ev_marginalized = optimize_hyperparameters(A, y, sigma_y=sigma, alpha_guess=10.0)
    
    # Beta should be 1.0 (internal pre-whitened) or None depending on return spec?
    # The function returns "beta if not fix_beta else None"
    # Wait, my implementation says:
    # return m, S, alpha, (beta if not fix_beta else None), gamma, res
    
    assert beta is None
    assert np.isscalar(alpha)
    
    # Check values make sense (alpha should be around 1/var(w) approx 1)
    # With limited data, just checking it runs.

def test_alpha_scalar_enforcement():
    """Ensure vector input for alpha is converted or handled (if possible) or doc is respected."""
    np.random.seed(42)
    N, K = 10, 2
    A = np.random.randn(N, K)
    y = np.random.randn(N)
    
    # Passing a vector alpha_guess should be cast to scalar (first element or sum? code says float(guess))
    # My code: alpha = float(alpha_guess)
    # If a vector is passed, float([1,2]) raises TypeError in Python.
    
    with pytest.raises(TypeError):
        optimize_hyperparameters(A, y, alpha_guess=np.array([1.0, 2.0]))
