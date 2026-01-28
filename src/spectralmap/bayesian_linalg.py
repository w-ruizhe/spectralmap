import numpy as np

def solve_posterior(A_scaled, y_scaled, Alpha, mu0, jitter=1e-12, ATA=None):
    """
    Helper to solve for posterior mean m and covariance S given
    scaled inputs (whitened by sigma_y).
    
    H = A_scaled.T @ A_scaled + diag(Alpha)
    m = H_inv @ (A_scaled.T @ y_scaled + Alpha * mu0)
    """
    k = len(Alpha)
    
    # Hessian (Precision) matrix
    if ATA is None:
        ATA = A_scaled.T @ A_scaled
    
    H = ATA + np.diag(Alpha)
    
    # Add jitter for stability if needed
    H_stabilized = H + np.eye(k) * jitter
    
    # Invert H to get Covariance S
    # Using cholesky is generally more stable than inv
    try:
        L = np.linalg.cholesky(H_stabilized)
        # S = inv(L.T @ L)
        S = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))
        
        # Posterior mean m
        # Right hand side: A^T Sigma^{-1} y + Alpha * mu0
        # Since A_scaled = A/sigma, A_scaled.T @ y_scaled is exactly A^T Sigma^{-1} y
        rhs = A_scaled.T @ y_scaled + Alpha * mu0
        m = np.linalg.solve(H_stabilized, rhs)
        
    except np.linalg.LinAlgError:
        # Fallback to standard solve/inv if Cholesky fails
        S = np.linalg.inv(H_stabilized)
        rhs = A_scaled.T @ y_scaled + Alpha * mu0
        m = S @ rhs

    # Residuals on the SCALED data: (y - Am)/sigma
    # This represents the "chi" vector, not raw residuals
    r_scaled = y_scaled - A_scaled @ m
    
    return m, r_scaled, S, H

def optimize_alpha_fixed_beta(
    A, y, sigma_y,
    alpha_guess=1.0,       # scalar or (k,) array
    mu0=None,
    tol=1e-8,
    maxit=10000,
    jitter=1e-12,
):
    
    """
    Bayesian linear regression / ARD with fixed measurement uncertainties (sigma_y).
    Optimizes alpha (hyperparameters) via Type II Maximum Likelihood.

    Model:
        y | w ~ N(A w, diag(sigma_y^2))
        w ~ N(mu0, Alpha^{-1})

    Args:
        A : (N, k) Design matrix
        y : (N,) Target vector
        sigma_y : (N,) 1-sigma uncertainties for y
        alpha_guess : Initial guess for precision of weights
    
    Returns
    -------
    m : (k,) posterior mean
    S : (k, k) posterior covariance
    alpha_out : scalar or (k,) depending on input alpha_guess
    gamma_i : (k,) effective degrees of freedom per weight
    log_ev_dict : dict with contributions to log evidence
    """
    A = np.asarray(A)
    y = np.asarray(y).reshape(-1)
    sigma_y = np.asarray(sigma_y).reshape(-1)
    N, k = A.shape

    if sigma_y.shape != (N,):
        raise ValueError(f"sigma_y must have shape ({N},), got {sigma_y.shape}")

    if mu0 is None:
        mu0 = np.r_[1, np.zeros(k - 1)]
    elif mu0.shape != (k,):
        raise ValueError(f"mu0 must have shape ({k},), got {mu0.shape}")

    # Detect scalar vs vector alpha
    alpha_guess_arr = np.asarray(alpha_guess)
    if alpha_guess_arr.ndim == 0:
        scalar_alpha = True
        Alpha = np.full(k, alpha_guess_arr)
        alpha_scalar = alpha_guess_arr
    else:
        if alpha_guess_arr.shape != (k,):
            raise ValueError(f"alpha_guess must be scalar or shape ({k},), got {alpha_guess_arr.shape}")
        scalar_alpha = False
        Alpha = alpha_guess_arr

    tiny = np.finfo(float).tiny

    # Pre-whiten data by sigma_y to simplify math
    # A_scaled corresponds to A / sigma
    # y_scaled corresponds to y / sigma
    A_scaled = A / sigma_y[:, None]
    y_scaled = y / sigma_y
    
    # Precompute ATA on scaled data (this is A^T Sigma^-1 A)
    ATA = A_scaled.T @ A_scaled

    # Fixed-point iterations
    for _ in range(maxit):
        m, r_scaled, S, H = solve_posterior(A_scaled, y_scaled, Alpha, mu0, jitter=jitter, ATA=ATA)
        diagS = np.diag(S)

        # gamma_j = 1 - alpha_j * S_jj
        gamma_i = 1.0 - Alpha * diagS
        gamma = np.sum(gamma_i)

        # Update Alpha
        dw = m - mu0
        dw2 = dw**2

        if scalar_alpha:
            # One shared alpha
            dw2_sum = dw2.sum()
            alpha_new_scalar = gamma / np.maximum(dw2_sum, tiny)
            Alpha_new = np.full(k, alpha_new_scalar)
        else:
            # ARD: one alpha per weight
            Alpha_new = gamma_i / np.maximum(dw2, tiny)
            Alpha_new = np.minimum(Alpha_new, 1 / jitter) # cap alpha

        # Convergence check
        if scalar_alpha:
            rel_alpha = np.abs(alpha_new_scalar - alpha_scalar) / np.maximum(1.0, alpha_scalar)
        else:
            rel_alpha = np.max(np.abs(Alpha_new - Alpha) / np.maximum(1.0, Alpha))

        Alpha = Alpha_new
        if scalar_alpha:
            alpha_scalar = alpha_new_scalar

        if rel_alpha <= tol:
            break

    # Final posterior calculation
    m, r_scaled, S, H = solve_posterior(A_scaled, y_scaled, Alpha, mu0, jitter=jitter, ATA=ATA)
    diagS = np.diag(S)
    gamma_i = 1.0 - Alpha * diagS
    gamma = np.sum(gamma_i)
    dw = m - mu0
    dw2 = dw**2

    # log|H|
    sign, logdetH = np.linalg.slogdet(H)
    if sign <= 0:
        # Try to recover if H is numerically unstable
        w, v = np.linalg.eigh(H)
        logdetH = np.sum(np.log(np.maximum(w, tiny)))

    # Evidence decomposition
    # log p(y | ...)
    # Term 1: Occam factor / Volume contraction -> -1/2 log|H|
    # Term 2: Prior volume -> +1/2 sum log(alpha)
    # Term 3: Likelihood normalization -> -sum log(sigma_y) - N/2 log(2pi)
    # Term 4: Best fit misfit -> -1/2 (chi2_likelihood + chi2_prior)
    
    chi2_prior = np.sum(Alpha * dw2)
    # r_scaled is (y - Am)/sigma, so r_scaled @ r_scaled is exactly chi-squared
    chi2_likelihood = r_scaled @ r_scaled 

    log_ev_alpha = -0.5 * chi2_prior + 0.5 * np.sum(np.log(Alpha))
    log_ev_sigma = -0.5 * chi2_likelihood - np.sum(np.log(sigma_y))
    log_ev_det = -0.5 * logdetH
    log_ev_normalization = -0.5 * N * np.log(2.0 * np.pi)
    
    log_ev = log_ev_alpha + log_ev_sigma + log_ev_det + log_ev_normalization

    # --- Marginalization corrections (Laplace approximation for Hyperparameters) ---
    # Since beta is fixed, we only compute correction for Alpha integration
    
    if scalar_alpha:
        # Variance of log(alpha) approx 2/gamma
        log_delta_log_alpha = 0.5 * np.log(2 / gamma)
        num_regularizer = 1
    else:
        constrained = Alpha < (1 / jitter - 10)
        # print(constrained)
        num_regularizer = np.sum(constrained) # No beta here
        
        # Curvature of evidence w.r.t log alpha_j is ~ gamma_j / 2  (or closely related forms)
        # Using MacKay's approximation for individual alphas:
        log_delta_log_alpha_vec = 0.5 * np.log(2 / (1 - Alpha**2 * diagS**2 + tiny))
        
        # Only include active hyperparameters
        log_delta_log_alpha = np.sum(log_delta_log_alpha_vec[constrained])
        
    # There is no log_delta_log_beta because beta is fixed (data).
    
    # print("num_regularizer:", num_regularizer)
    
    log_ev_marginalized = (
        log_ev + 
        log_delta_log_alpha + 
        num_regularizer / 2 * np.log(2 * np.pi)
    )

    log_ev_dict = dict(
        log_ev=log_ev,
        log_ev_alpha=log_ev_alpha,
        log_ev_sigma=log_ev_sigma, # Replaces log_ev_beta
        log_ev_det=log_ev_det,
        log_ev_normalization=log_ev_normalization,
        log_ev_marginalized=log_ev_marginalized,
    )

    # Return alpha in same "shape" as user gave
    if scalar_alpha:
        alpha_out = alpha_scalar
    else:
        alpha_out = Alpha

    return m, S, alpha_out, gamma_i, log_ev_dict

def optimize_alpha_beta(
    A, y,
    alpha_guess=1.0,       # scalar or (k,) array
    beta_guess=100.0,
    mu0=None,
    tol=1e-8,
    maxit=1000,
    jitter=1e-12,
):
    def solve_posterior(A, y, Alpha, beta, mu0=None, jitter=1e-12, ATA=None):
        """
        Solve for the posterior given diagonal prior precision Alpha (length k)
        and scalar noise precision beta.

        Model:
            y | w ~ N(A w, (1/beta) I)
            w ~ N(mu0, Alpha^{-1})

        Parameters
        ----------
        A : (N, k) design matrix
        y : (N,) data
        mu0 : (k,) prior mean
        Alpha : (k,) prior precisions (diagonal of prior precision matrix)
        beta : float, noise precision
        jitter : float, added to Hessian diagonal for numerical stability
        ATA : optional (k, k) precomputed A^T A

        Returns
        -------
        m : (k,) posterior mean
        r : (N,) residual y - A m
        S : (k, k) posterior covariance
        H : (k, k) posterior precision (Hessian)
        """
        N, k = A.shape

        if mu0 is None:
            mu0 = np.r_[1, np.zeros(k - 1)]
        elif mu0.shape != (k,):
            raise ValueError(f"mu0 must have shape ({k},), got {mu0.shape}")

        if ATA is None:
            ATA = A.T @ A

        I_k = np.eye(k)

        # Posterior precision in weight space:
        # H = Alpha_diag + beta A^T A
        H = np.diag(Alpha) + beta * ATA

        # Cholesky factorization of H
        try:
            L = np.linalg.cholesky(H + jitter * I_k)
        except np.linalg.LinAlgError as e:
            print("Alpha:", Alpha)
            print("beta:", beta)
            raise

        # Center data by prior mean
        y_centered = y - A @ mu0

        # RHS = beta A^T (y - A mu0)
        rhs = beta * (A.T @ y_centered)

        # Solve H u = rhs via Cholesky: H = L L^T
        z = np.linalg.solve(L, rhs)
        u = np.linalg.solve(L.T, z)

        # Posterior mean
        m = mu0 + u

        # Residual in data space
        r = y - A @ m

        # Posterior covariance S = H^{-1} = (L^T)^{-1} L^{-1}
        Z = np.linalg.solve(L, I_k)   # Z = L^{-1}
        S = Z.T @ Z                   # H^{-1}

        return m, r, S, H

    """
    Bayesian linear regression / ARD with evidence-maximizing hyperparameters.

    Model:
        y | w ~ N(A w, (1/beta) I)
        w ~ N(mu0, Alpha^{-1})

    If alpha_guess is:
        - scalar: one shared alpha (standard Bayesian ridge regression)
        - array-like of length k: ARD (one alpha_j per weight)

    Hyperparameter fixed-point updates (MacKay/Tipping):
        gamma_j   = 1 - alpha_j * S_jj
        alpha_j   <- gamma_j / (m_j - mu0_j)^2      (ARD)
        alpha     <- (sum_j gamma_j) / ||m - mu0||^2 (scalar case)
        1/beta    <- ||y - A m||^2 / (N - sum_j gamma_j)

    Evidence (log marginal likelihood):
        log p(y | A, Alpha, beta, mu0) =
            -1/2 [ N log(2π) + log|C| + (y - A mu0)^T C^{-1} (y - A mu0) ]

    but here we use the equivalent weight-space form with H.

    Returns
    -------
    m : (k,) posterior mean
    S : (k, k) posterior covariance
    alpha_out : scalar or (k,) depending on input alpha_guess
    beta : float, optimized noise precision
    gamma_i : (k,) effective degrees of freedom per weight
    log_ev_dict : dict with contributions to log evidence
    """
    A = np.asarray(A)
    y = np.asarray(y).reshape(-1)
    N, k = A.shape

    if mu0 is None:
        mu0 = np.r_[1, np.zeros(k - 1)]
    elif mu0.shape != (k,):
        raise ValueError(f"mu0 must have shape ({k},), got {mu0.shape}")

    # Detect scalar vs vector alpha
    alpha_guess_arr = np.asarray(alpha_guess)
    if alpha_guess_arr.ndim == 0:
        scalar_alpha = True
        Alpha = np.full(k, alpha_guess_arr)
        alpha_scalar = alpha_guess_arr
    else:
        if alpha_guess_arr.shape != (k,):
            raise ValueError(f"alpha_guess must be scalar or shape ({k},), got {alpha_guess_arr.shape}")
        scalar_alpha = False
        Alpha = alpha_guess_arr

    beta = beta_guess
    tiny = np.finfo(float).tiny

    ATA = A.T @ A

    # Fixed-point iterations
    for _ in range(maxit):
        m, r, S, H = solve_posterior(A, y, Alpha, beta, mu0=mu0, jitter=jitter, ATA=ATA)
        diagS = np.diag(S)

        # gamma_j = 1 - alpha_j * S_jj
        gamma_i = 1.0 - Alpha * diagS
        gamma = np.sum(gamma_i)

        # Update Alpha
        dw = m - mu0
        dw2 = dw**2

        if scalar_alpha:
            # One shared alpha
            dw2_sum = dw2.sum()
            alpha_new_scalar = gamma / np.maximum(dw2_sum, tiny)
            Alpha_new = np.full(k, alpha_new_scalar)
        else:
            # ARD: one alpha per weight
            Alpha_new = gamma_i / np.maximum(dw2, tiny)
            Alpha_new = np.minimum(Alpha_new, 1 / jitter) # cap alpha to prevent numerical issues

        # Update beta
        dof = np.maximum(N - gamma, tiny)
        r2 = r @ r
        beta_new = dof / np.maximum(r2, tiny)

        # Convergence check BEFORE overwriting
        if scalar_alpha:
            rel_alpha = np.abs(alpha_new_scalar - alpha_scalar) / np.maximum(1.0, alpha_scalar)
        else:
            rel_alpha = np.max(np.abs(Alpha_new - Alpha) / np.maximum(1.0, Alpha))

        rel_beta = np.abs(beta_new - beta) / np.maximum(1.0, beta)

        Alpha = Alpha_new
        beta = beta_new
        if scalar_alpha:
            alpha_scalar = alpha_new_scalar

        if max(rel_alpha, rel_beta) <= tol:
            break

    # Final posterior and evidence
    m, r, S, H = solve_posterior(A, y, Alpha, beta, mu0=mu0, jitter=jitter, ATA=ATA)
    diagS = np.diag(S)
    gamma_i = 1.0 - Alpha * diagS
    gamma = np.sum(gamma_i)


    # log|H|
    sign, logdetH = np.linalg.slogdet(H)
    if sign <= 0:
        raise np.linalg.LinAlgError("Posterior precision H not positive definite")

    # Evidence decomposition
    # log p(y | A, Alpha, beta, mu0) =
    #   -1/2 [ N log(2π) ]                  (normalization)
    #   -1/2 log|H|                         (Occam factor)
    #   + 1/2 sum_j log alpha_j             (from prior norm)
    #   + 1/2 N log beta                    (from likelihood norm)
    #   - (alpha Ew + beta Ed)
    log_ev_alpha = - 0.5 * np.sum(Alpha * dw2) + 0.5 * np.sum(np.log(Alpha))
    log_ev_beta = - 0.5 * beta * (r @ r)  + 0.5 * N * np.log(beta)
    log_ev_det = -0.5 * logdetH
    log_ev_normalization = -0.5 * N * np.log(2.0 * np.pi)
    log_ev = log_ev_alpha + log_ev_beta + log_ev_det + log_ev_normalization

    if scalar_alpha:
        log_delta_log_alpha = 0.5 * np.log(2 / gamma)
        num_regularizer = 2
    else:
        constrained = Alpha < (1 / jitter - 10)
        print(constrained)
        print(Alpha)
        num_regularizer = np.sum(constrained) + 1 # +1 for beta
        log_delta_log_alpha = 0.5 * np.log(2 / (1 - Alpha**2 * diagS**2))
        log_delta_log_alpha = log_delta_log_alpha[constrained]
        print(Alpha[constrained])
        print("log_delta_log_alpha:", log_delta_log_alpha)
        log_delta_log_alpha = np.sum(log_delta_log_alpha)
        
    log_delta_log_beta = 0.5 * np.log(2 / (N - gamma))
    print("num_regularizer:", num_regularizer)
    print("log_delta_log_alpha:", log_delta_log_alpha)
    print("log_delta_log_beta:", log_delta_log_beta)
    print("beta:", beta)
    print("Alpha:", Alpha)
    log_ev_marginalized_over_alpha_beta = log_ev + log_delta_log_alpha + log_delta_log_beta + num_regularizer / 2 * np.log(2 * np.pi)

    log_ev_dict = dict(
        log_ev=log_ev,
        log_ev_alpha=log_ev_alpha,
        log_ev_beta=log_ev_beta,
        log_ev_det=log_ev_det,
        log_ev_normalization=log_ev_normalization,
        log_ev_marginalized_over_alpha_beta=log_ev_marginalized_over_alpha_beta,
    )

    # Return alpha in same "shape" as user gave
    if scalar_alpha:
        alpha_out = alpha_scalar
    else:
        alpha_out = Alpha

    return m, S, alpha_out, beta, gamma_i, log_ev_dict