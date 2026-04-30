"""Bayesian linear algebra for regularized surface-map inference."""

import numpy as np


# Numerical constants for stability and convergence.
FLOOR = 1e-12
CONVERGENCE_TOL = 1e-6
ALPHA_BOUNDS = (1e-20, 1e20)


def _alpha_to_vec(alpha_in, k: int):
    """Expand scalar or grouped alpha precision into a coefficient vector."""
    if np.isscalar(alpha_in):
        return np.full(k, float(alpha_in), dtype=float), "scalar"

    arr = np.asarray(alpha_in, dtype=float).ravel()
    if arr.size == k:
        return arr.astype(float), "vector"
    if arr.size == 2:
        out = np.empty(k, dtype=float)
        out[0] = arr[0]
        if k > 1:
            out[1:] = arr[1]
        return out, "group2"

    raise ValueError(f"alpha must be a scalar or length-2 array; got length {arr.size}")

def linear_solve(
    A, y, alpha,
    beta=None,
    mu0=None,
    I=None,              # (n_pix, k) intensity operator
    lamda=None,          # lamda precision for intensity shrinkage (>0)
    w_pix=None,          # (n_pix,) nonnegative weights (e.g. solid angle per pixel)
    ATA=None,            # precomputed A^T A
    ITI=None,            # precomputed I^T I
):
    """Solve a Gaussian linear model with optional intensity shrinkage.

    The likelihood is ``y ~ N(A @ mu, beta^{-1} I)``. The coefficient prior is
    ``mu ~ N(mu0, alpha^{-1} I_k)`` and, when requested, an extra map-space
    shrinkage term applies ``Iw ~ N(0, lamda^{-1} I_npix)``.
    """

    N, k = A.shape

    if mu0 is None:
        raise ValueError("mu0 must be provided (even if just zeros)")

    alpha_vec, _ = _alpha_to_vec(alpha, k)

    if ATA is None:
        ATA = A.T @ A

    # Base precision
    H = beta * ATA + np.diag(alpha_vec)

    rhs = beta * (A.T @ y) + alpha_vec * mu0

    # Add intensity shrinkage if provided
    lamda_enabled = (type(I) is np.ndarray) and (lamda is not None) and (lamda > 0.0)
    if lamda_enabled:
        n_pix = I.shape[0]
        ws = np.sqrt(w_pix)
        Iw = I * ws[:, None]

        if ITI is None:
            ITI = Iw.T @ Iw

        H = H + lamda * ITI

    H_stabilized = H + np.eye(k) * FLOOR

    try:
        L = np.linalg.cholesky(H_stabilized)
        cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))

        z = np.linalg.solve(L, rhs)
        mu = np.linalg.solve(L.T, z)

        logdetH = 2.0 * np.sum(np.log(np.diag(L)))

    except np.linalg.LinAlgError:
        cov = np.linalg.inv(H_stabilized)
        mu = cov @ rhs
        sign, logdetH = np.linalg.slogdet(H_stabilized)

    r = y - A @ mu
    return mu, r, cov, H, logdetH

def optimize_hyperparameters(
    A: np.ndarray, y: np.ndarray,
    sigma_y=None,
    mu0=None,
    maxit=500,
    # --- new: intensity shrinkage ---
    I=None,              # (n_pix, k) intensity operator
    w_pix=None,          # (n_pix,) nonnegative weights (e.g. solid angle per pixel)
    lamda=None,        # fix lamda if provided
    damp=0.7,            # conservative default damping for stable updates
    verbose=True,
):
    """Optimize evidence hyperparameters and return the linear posterior."""
    N, k = A.shape

    # ---- noise handling (same as you had) ----
    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y)
        if sigma_y.ndim == 0:
            sigma_y = np.full(N, sigma_y)
        # Rescale A and y by noise std to get unit noise variance
        A = A / sigma_y[:, None]
        y = y / sigma_y
        beta = 1.0
        fix_beta = True
        log_prod_sigma = np.sum(np.log(sigma_y))
    else:
        resid_init = y - np.mean(y)
        var_init = np.var(resid_init) if N > 1 else 1.0
        beta = 1.0 / (var_init + FLOOR)

        fix_beta = False
        log_prod_sigma = 0.0

    alpha_init = 1.0
    alpha_vec, alpha_mode = _alpha_to_vec(alpha_init, k)
    if mu0 is None:
        mu0 = np.zeros(k)
    else:
        mu0 = np.asarray(mu0)

    ATA = A.T @ A
    
    # lamda init / fixed flags
    lamda_enabled = (I is not None) and (lamda is not None and lamda > 0.0)
    if lamda is not None and not lamda_enabled:
        raise ValueError("To regularize intensity, must provide valid I and lamda > 0")
    
    # ---- intensity pieces ----
    if lamda_enabled:
        I = np.asarray(I)
        ws = np.sqrt(w_pix)
        Iw = I * ws[:, None]
        n_pix_eff = np.sum(w_pix)
        ITI = Iw.T @ Iw
    else:
        n_pix_eff = 0.0
        ITI = None

    safe = float(FLOOR)


    if alpha_mode == "scalar":
        alpha_blocks = [np.arange(k, dtype=int)]
    else:
        alpha_blocks = [np.array([0], dtype=int)]
        if k > 1:
            alpha_blocks.append(np.arange(1, k, dtype=int))

    alpha_update_blocks = [True] * len(alpha_blocks)

    for i in range(maxit):
        mu, r, cov, H, logdetH = linear_solve(
            A, y, alpha_vec, beta=beta, mu0=mu0,
            I=I, lamda=lamda, w_pix=w_pix, ATA=ATA, ITI=ITI
        )
        # ---- gamma/update for alpha (block-wise) ----
        gamma_blocks = []
        alpha_new_vec = alpha_vec.copy()
        dw = mu - mu0
        for j, blk in enumerate(alpha_blocks):
            tr_blk = float(np.trace(cov[np.ix_(blk, blk)]))
            dw2_blk = float(dw[blk] @ dw[blk])
            a_blk = float(np.mean(alpha_vec[blk]))
            n_blk = blk.size

            gamma_blk = n_blk - a_blk * tr_blk
            gamma_blk = float(np.maximum(gamma_blk, safe))
            gamma_blocks.append(gamma_blk)

            den_blk = dw2_blk + tr_blk
            a_new_blk = gamma_blk / np.maximum(den_blk, safe)
            a_new_blk = float(np.clip(a_new_blk, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))

            if alpha_update_blocks[j]:
                alpha_new_vec[blk] = a_new_blk

        # ---- beta update ----
        # Effective number of data-constrained parameters should come from
        # the likelihood term directly: gamma_beta = beta * tr(A^T A cov).
        gamma_beta = float(beta * np.trace(ATA @ cov))
        gamma_beta = float(np.clip(gamma_beta, 0.0, N - safe))

        chi2 = np.sum(r**2)
        if not fix_beta:
            beta_new = (N - gamma_beta) / np.maximum(chi2, safe)
        else:
            beta_new = beta

        # ---- convergence check ----
        max_change = float(np.max(np.abs(alpha_vec - alpha_new_vec) / alpha_vec))
        if not fix_beta:
            max_change = max(max_change, np.abs(beta - beta_new) / beta)

        # ---- damping (recommended if oscillations) ----
        alpha_vec = (1 - damp) * alpha_vec + damp * alpha_new_vec
        beta  = (1 - damp) * beta  + damp * beta_new

        if max_change < CONVERGENCE_TOL:
            if verbose:
                print(f"Hyperparameter optimization converged after {i+1} iterations.")
            break

    # final posterior
    mu, r, cov, H, logdetH = linear_solve(
        A, y, alpha_vec, beta=beta, mu0=mu0, ATA=ATA,
        I=I, lamda=lamda, w_pix=w_pix, ITI=ITI
    )

    # ---- evidence (consistent with your existing style) ----
    chi2 = np.sum(r**2)
    log_ev_norm = -0.5 * N * np.log(2.0 * np.pi)

    if fix_beta:
        if sigma_y is not None:
            log_ev_beta = -0.5 * chi2 - log_prod_sigma
            beta_out = None
        else:
            log_ev_beta = 0.5 * N * np.log(beta) - 0.5 * beta * chi2
            beta_out = float(beta)
    else:
        log_ev_beta = 0.5 * N * np.log(beta) - 0.5 * beta * chi2
        beta_out = beta

    # alpha prior term (your style)
    dw2 = (mu - mu0)**2
    
    log_ev_alpha = 0.5 * np.sum(np.log(alpha_vec)) - 0.5 * np.sum(alpha_vec * dw2)
    
    # lamda shrinkage term (analogous; only if used)
    if lamda_enabled:
        dI = (Iw @ mu)
        log_ev_lam = 0.5 * n_pix_eff * np.log(lamda) - 0.5 * lamda * float(dI @ dI)
    else:
        log_ev_lam = 0.0

    log_occam = -0.5 * logdetH
    log_ev = log_ev_beta + log_ev_norm + log_ev_alpha + log_ev_lam + log_occam

    # ---- Laplace-ish marginalization correction (your style) ----
    log_delta_log_alpha = 0.0
    for j, blk in enumerate(alpha_blocks):
        if not alpha_update_blocks[j]:
            continue
        tr_blk = float(np.trace(cov[np.ix_(blk, blk)]))
        a_blk = float(np.mean(alpha_vec[blk]))
        gamma_blk = blk.size - a_blk * tr_blk
        log_delta_log_alpha += 0.5 * np.log(2.0 / np.maximum(gamma_blk, FLOOR))

    num_regularizer = int(np.sum(alpha_update_blocks))

    log_delta_log_beta = 0.0
    if not fix_beta:
        gamma_beta = float(beta * np.trace(ATA @ cov))
        gamma_beta = float(np.clip(gamma_beta, 0.0, N - FLOOR))
        num_regularizer += 1
        log_delta_log_beta = 0.5 * np.log(2.0 / (N - gamma_beta))

    log_ev_marginalized = (
        log_ev
        + log_delta_log_alpha
        + log_delta_log_beta
        + num_regularizer / 2.0 * np.log(2.0 * np.pi)
    )
    if alpha_mode == "scalar":
        alpha_out = float(alpha_vec[0])
    else:
        alpha_out = np.asarray([alpha_vec[0], alpha_vec[1] if k > 1 else alpha_vec[0]], dtype=float)

    if verbose:
        lam_print = lamda if lamda_enabled else 'disabled'
        print(f"Final hyperparameters: alpha={alpha_out}, beta={beta if not fix_beta else 'fixed'}, lamda={lam_print}")
    
    return mu, cov, alpha_out, beta_out, log_ev, log_ev_marginalized
