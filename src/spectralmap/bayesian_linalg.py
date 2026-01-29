import numpy as np

def solve_posterior(
    A, y, alpha, beta=1.0, mu0=None, ATA=None, jitter=1e-12
):
    """
    Computes posterior mean m and covariance S for the simple Bayesian Ridge model:
        y ~ N(A m, (1/beta) I)  
        m ~ N(mu0, (1/alpha) I)
    
    Returns
    -------
    m : posterior mean
    residuals : y - A @ m
    S : posterior covariance
    H : posterior precision
    logdetH : log determinant of H
    """
    N, k = A.shape
    
    if mu0 is None:
        mu0 = np.r_[1.0, np.zeros(k - 1)]
    
    if ATA is None:
        ATA = A.T @ A
        
    H = beta * ATA + alpha * np.eye(k)
    H_stabilized = H + np.eye(k) * jitter
    
    try:
        L = np.linalg.cholesky(H_stabilized)
        # S = L^-T L^-1
        S = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))
        
        rhs = beta * (A.T @ y) + alpha * mu0
        
        z = np.linalg.solve(L, rhs)
        m = np.linalg.solve(L.T, z)
        
        logdetH = 2.0 * np.sum(np.log(np.diag(L)))
        
    except np.linalg.LinAlgError:
        S = np.linalg.inv(H_stabilized)
        rhs = beta * (A.T @ y) + alpha * mu0
        m = S @ rhs
        sign, logdetH = np.linalg.slogdet(H_stabilized)
    
    residuals = y - A @ m
    
    return m, residuals, S, H, logdetH

def optimize_hyperparameters(
    A, y, 
    sigma_y=None, 
    alpha_guess=1.0, 
    beta_guess=None,
    mu0=None, 
    tol=1e-6, 
    maxit=500,
    jitter=1e-12,
):
    """
    Empirical Bayes (Type-II Maximum Likelihood) optimization of hyperparameters.
    
    Consolidates functionality for:
    1. Fixed noise structure (sigma_y provided) -> Optimize Alpha
    2. Unknown homoscedastic noise (sigma_y=None) -> Optimize Alpha & Beta
    
    Parameters
    ----------
    A : (N, K) Design matrix
    y : (N,) Target values
    sigma_y : (N,) or float, optional
        Known measurement uncertainties (1-sigma).
        If provided, `beta` is effectively fixed (pre-whitening applied).
    alpha_guess : float
        Initial guess for precision of weights (scalar only).
    beta_guess : float, optional
        Initial guess for noise precision (1/sigma^2) if sigma_y is None.
    
    Returns
    -------
    m : (k,) posterior mean
    S : (k, k) posterior covariance
    alpha : float, optimized prior precision
    beta : float, optimized noise precision (1.0 if sigma_y used)
    gamma : float effective degrees of freedom
    log_ev : float log evidence
    log_ev_marginalized : float log evidence marginalized over hyperparameters
    """
    
    A = np.asarray(A)
    y = np.asarray(y)
    N, k = A.shape
    
    # --- 1. Setup Noise Model ---
    if sigma_y is not None:
        # Case A: Fixed Heteroscedastic Noise
        # We pre-whiten the data so that eff_sigma = 1 (beta = 1)
        sigma_y = np.asarray(sigma_y)
        if sigma_y.ndim == 0:
            sigma_y = np.full(N, sigma_y)
        
        # Scale inputs
        A_eff = A / sigma_y[:, None]
        y_eff = y / sigma_y
        
        beta = 1.0
        fix_beta = True
        
        # Determine log|sigma| term for evidence
        log_prod_sigma = np.sum(np.log(sigma_y))
    else:
        # Case B: Unknown Homoscedastic Noise
        A_eff = A
        y_eff = y
        
        if beta_guess is None:
            # Heuristic init
            resid_init = y - np.mean(y)
            var_init = np.var(resid_init) if N > 1 else 1.0
            beta = 1.0 / (var_init + 1e-9)
        else:
            beta = float(beta_guess)
        
        fix_beta = False
        log_prod_sigma = 0.0 # Handled by beta term
        
    # --- 2. Setup Alpha Model (Scalar Only) ---
    alpha = float(alpha_guess)
        
    if mu0 is None:
        mu0 = np.r_[1.0, np.zeros(k - 1)]

    # Precompute A^T A (for efficiency inside loop)
    ATA = A_eff.T @ A_eff
    
    # --- 3. Optimization Loop (Fixed Point Updates) ---
    tiny = np.finfo(float).tiny
    
    for i in range(maxit):
        # E-step: Compute Posterior Statistics
        m, r_raw, S, H, logdetH = solve_posterior(
            A_eff, y_eff, alpha, beta=beta, mu0=mu0, ATA=ATA, jitter=jitter
        )
        
        # M-step: Update Hyperparameters
        diagS = np.diag(S)
        gamma = 1.0 - alpha * diagS
        gamma = np.sum(gamma)
        
        dw2 = (m - mu0)**2
        dw2_sum = np.sum(dw2)
        
        # -- Update Alpha (Scalar) --
        alpha_new = gamma / np.maximum(dw2_sum, tiny)
            
        # -- Update Beta (if not fixed) --
        # r_raw is (y_eff - A_eff m)
        chi2 = np.sum(r_raw**2)
        
        if not fix_beta:
            beta_new = np.maximum(N - gamma, tiny) / np.maximum(chi2, tiny)
        else:
            beta_new = 1.0
            
        # -- Convergence Check --
        max_change = np.abs(alpha - alpha_new) / np.maximum(alpha, 1e-9)
        
        if not fix_beta:
            beta_change = np.abs(beta - beta_new) / np.maximum(beta, 1e-9)
            max_change = max(max_change, beta_change)
            
        alpha = alpha_new
        beta = beta_new
        
        if max_change < tol:
            break

    # --- 4. Final Calculation & Evidence ---
    m, r_raw, S, H, logdetH = solve_posterior(
            A_eff, y_eff, alpha, beta=beta, mu0=mu0, ATA=ATA, jitter=jitter
    )
    # diagS = np.diag(S) # Unused in scalar final calc?
    
    # Re-calculate final chi2
    chi2 = np.sum(r_raw**2)
    
    # Log Likelihood P(y | params)
    log_ev_norm = -0.5 * N * np.log(2 * np.pi)
    
    if fix_beta:
        # N(Ax, sigma^2 I)
        log_ev_beta = -0.5 * chi2 - log_prod_sigma
    else:
        # N(Ax, beta^-1 I)
        log_ev_beta = 0.5 * N * np.log(beta) - 0.5 * beta * chi2
    
    # Evidence decomposition
    dw2 = (m - mu0)**2
    log_ev_alpha = 0.5 * k * np.log(alpha) - 0.5 * alpha * np.sum(dw2)
    
    # Occam factor (volume contraction)
    log_occam = -0.5 * logdetH
    
    log_ev = log_ev_beta + log_ev_norm + log_ev_alpha + log_occam

    # Laplace approximation correction (Approximate)
    # Variance of log(alpha) approx 2/gamma
    log_delta_log_alpha = 0.5 * np.log(2.0 / gamma) 
    
    num_regularizer = 1 
    if not fix_beta:
        num_regularizer += 1
        log_delta_log_beta = 0.5 * np.log(2.0 / (N - gamma))
    else:
        log_delta_log_beta = 0.0

    log_ev_marginalized = (
        log_ev 
        + log_delta_log_alpha 
        + log_delta_log_beta 
        + num_regularizer / 2.0 * np.log(2 * np.pi)
    )

    return m, S, alpha, (beta if not fix_beta else None), gamma, log_ev, log_ev_marginalized
    