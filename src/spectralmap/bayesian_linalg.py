import numpy as np


def _alpha_to_vec(alpha_in, k: int):
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

def solve_posterior(
    A, y,
    alpha,
    beta=1.0,
    mu0=None,
    ATA=None,
    jitter=1e-12,
    # --- new: intensity shrinkage ---
    I=None,              # (n_pix, k) intensity operator
    lambda_fix=0.0,             # lambda precision for intensity shrinkage
    x0=None,             # (n_pix,) target intensity, default zeros
    w_pix=None,          # (n_pix,) nonnegative weights (e.g. solid angle per pixel)
    ITI=None,            # precompute I^T I if you want
    ITx0=None,           # precompute I^T x0 if you want
):
    """
    Gaussian linear model with optional intensity shrinkage term:
        y ~ N(A w, beta^{-1} I)
        w ~ N(mu0, alpha^{-1} I_k)
        (Iw) ~ N(x0, lambda_fix^{-1} I_{n_pix})   [pseudo-observation / shrinkage]
    """

    A = np.asarray(A)
    y = np.asarray(y)
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
    if I is not None and lambda_fix is not None and lambda_fix > 0.0:
        I = np.asarray(I)
        n_pix = I.shape[0]

        if x0 is None:
            x0 = np.zeros(n_pix)
        else:
            x0 = np.asarray(x0)

        if w_pix is not None:
            w_pix = np.asarray(w_pix, dtype=float)
            if w_pix.shape != (n_pix,):
                raise ValueError(f"w_pix must have shape ({n_pix},), got {w_pix.shape}")
            if np.any(w_pix < 0):
                raise ValueError("w_pix must be nonnegative")
            ws = np.sqrt(w_pix)
            Iw = I * ws[:, None]
            x0w = x0 * ws
        else:
            Iw = I
            x0w = x0

        if ITI is None:
            ITI = Iw.T @ Iw
        if ITx0 is None:
            ITx0 = Iw.T @ x0w

        H = H + lambda_fix * ITI
        rhs = rhs + lambda_fix * ITx0

    H_stabilized = H + np.eye(k) * jitter

    try:
        L = np.linalg.cholesky(H_stabilized)
        S = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(k)))

        z = np.linalg.solve(L, rhs)
        m = np.linalg.solve(L.T, z)

        logdetH = 2.0 * np.sum(np.log(np.diag(L)))

    except np.linalg.LinAlgError:
        S = np.linalg.inv(H_stabilized)
        m = S @ rhs
        sign, logdetH = np.linalg.slogdet(H_stabilized)

    residuals = y - A @ m
    return m, residuals, S, H, logdetH

def optimize_hyperparameters(
    A, y,
    sigma_y=None,
    mu0=None,
    tol=1e-6,
    maxit=500,
    jitter=1e-12,
    alpha_fix=None,
    # --- new: intensity shrinkage ---
    I=None,              # (n_pix, k)
    x0=None,             # (n_pix,) default zeros
    w_pix=None,          # (n_pix,) nonnegative weights (e.g. solid angle per pixel)
    lambda_fix=None,        # fix lambda if provided
    use_lambda=False,
    damp=0.7,            # conservative default damping for stable updates
    den_floor=1e-12,     # practical denominator floor (avoid tiny blowups)
    alpha_bounds=(1e-12, 1e12),
    lambda_bounds=(0.0, 1e12),
    verbose=True,
    print_every=1,
):
    A = np.asarray(A)
    y = np.asarray(y)
    N, k = A.shape

    # ---- noise handling (same as you had) ----
    if sigma_y is not None:
        sigma_y = np.asarray(sigma_y)
        if sigma_y.ndim == 0:
            sigma_y = np.full(N, sigma_y)

        A_eff = A / sigma_y[:, None]
        y_eff = y / sigma_y
        beta = 1.0
        fix_beta = True
        log_prod_sigma = np.sum(np.log(sigma_y))
    else:
        A_eff = A
        y_eff = y

        resid_init = y - np.mean(y)
        var_init = np.var(resid_init) if N > 1 else 1.0
        beta = 1.0 / (var_init + 1e-9)

        fix_beta = False
        log_prod_sigma = 0.0
    
    alpha_init = alpha_fix if alpha_fix is not None else 1.0
    alpha_vec, alpha_mode = _alpha_to_vec(alpha_init, k)
    if mu0 is None:
        mu0 = np.r_[1.0, np.zeros(k - 1)]
    else:
        mu0 = np.asarray(mu0)

    ATA = A_eff.T @ A_eff
    
    # lambda init / fixed flags
    lam_enabled = bool(use_lambda and (I is not None) and (lambda_fix is not None and lambda_fix > 0.0))
    # ---- intensity pieces ----
    if lam_enabled:
        I = np.asarray(I)
        n_pix = I.shape[0]
        if x0 is None:
            x0 = np.zeros(n_pix)
        else:
            x0 = np.asarray(x0)

        if w_pix.shape != (n_pix,):
            raise ValueError(f"w_pix must have shape ({n_pix},), got {w_pix.shape}")
        if np.any(w_pix < 0):
            raise ValueError("w_pix must be nonnegative")
        ws = np.sqrt(w_pix)
        Iw = I * ws[:, None]
        x0w = x0 * ws
        n_pix_eff = np.sum(w_pix)
        n_pix_eff = n_pix_eff
        ITI = Iw.T @ Iw
        ITx0 = Iw.T @ x0w
    else:
        n_pix = 0
        n_pix_eff = 0.0
        ITI = None
        ITx0 = None
    
    
    safe = float(den_floor)
    lambda_bounds_eff = (float(lambda_bounds[0]), float(lambda_bounds[1]))

    if alpha_mode == "scalar":
        alpha_blocks = [np.arange(k, dtype=int)]
    else:
        alpha_blocks = [np.array([0], dtype=int)]
        if k > 1:
            alpha_blocks.append(np.arange(1, k, dtype=int))

    alpha_update_blocks = [True] * len(alpha_blocks)

    for i in range(maxit):
        m, r_raw, S, H, logdetH = solve_posterior(
            A_eff, y_eff, alpha_vec, beta=beta, mu0=mu0, ATA=ATA, jitter=jitter,
            I=I, lambda_fix=lambda_fix, x0=x0, w_pix=w_pix, ITI=ITI, ITx0=ITx0
        )
        if verbose and (print_every is not None) and (print_every > 0):
            if (i % print_every) == 0:
                if alpha_mode == "scalar":
                    alpha_disp = float(alpha_vec[0])
                else:
                    alpha_disp = np.asarray([alpha_vec[0], alpha_vec[1] if k > 1 else alpha_vec[0]], dtype=float)
                print(f"Iter {i+1}: alpha={alpha_disp}, beta={beta if not fix_beta else 'fixed'}, lambda_fix={lambda_fix}, log_ev={logdetH:.3f}")

        # ---- gamma/update for alpha (block-wise) ----
        gamma_blocks = []
        alpha_new_vec = alpha_vec.copy()
        dw = m - mu0
        for j, blk in enumerate(alpha_blocks):
            tr_blk = float(np.trace(S[np.ix_(blk, blk)]))
            dw2_blk = float(dw[blk] @ dw[blk])
            a_blk = float(np.mean(alpha_vec[blk]))
            n_blk = blk.size

            gamma_blk = n_blk - a_blk * tr_blk
            gamma_blk = float(np.maximum(gamma_blk, safe))
            gamma_blocks.append(gamma_blk)

            den_blk = dw2_blk + tr_blk
            a_new_blk = gamma_blk / np.maximum(den_blk, safe)
            a_new_blk = float(np.clip(a_new_blk, alpha_bounds[0], alpha_bounds[1]))

            if alpha_update_blocks[j]:
                alpha_new_vec[blk] = a_new_blk

        gamma_total = float(np.sum(gamma_blocks))

        # ---- beta update ----
        # Effective number of data-constrained parameters should come from
        # the likelihood term directly: gamma_beta = beta * tr(A^T A S).
        gamma_beta = float(beta * np.trace(ATA @ S))
        gamma_beta = float(np.clip(gamma_beta, 0.0, N - safe))

        chi2 = np.sum(r_raw**2)
        if not fix_beta:
            beta_new = np.maximum(N - gamma_beta, safe) / np.maximum(chi2, safe)
        else:
            beta_new = beta

        # ---- convergence check ----
        max_change = float(np.max(np.abs(alpha_vec - alpha_new_vec) / np.maximum(alpha_vec, 1e-9)))
        if not fix_beta:
            max_change = max(max_change, np.abs(beta - beta_new) / np.maximum(beta, 1e-9))

        # ---- damping (recommended if oscillations) ----
        alpha_vec = (1 - damp) * alpha_vec + damp * alpha_new_vec
        beta  = (1 - damp) * beta  + damp * beta_new

        if max_change < tol:
            # print(f"Hyperparameter optimization converged after {i+1} iterations.")
            break

    # final posterior
    m, r_raw, S, H, logdetH = solve_posterior(
        A_eff, y_eff, alpha_vec, beta=beta, mu0=mu0, ATA=ATA, jitter=jitter,
        I=I, lambda_fix=lambda_fix, x0=x0, w_pix=w_pix, ITI=ITI, ITx0=ITx0
    )

    # ---- evidence (consistent with your existing style) ----
    chi2 = np.sum(r_raw**2)
    log_ev_norm = -0.5 * N * np.log(2 * np.pi)

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
    dw2 = (m - mu0)**2
    
    log_ev_alpha = 0.5 * np.sum(np.log(alpha_vec)) - 0.5 * np.sum(alpha_vec * dw2)
    
    # lambda shrinkage term (analogous; only if used)
    if lam_enabled and lambda_fix > 0:
        dI = (Iw @ m) - x0w
        log_ev_lam = 0.5 * n_pix_eff * np.log(lambda_fix) - 0.5 * lambda_fix * float(dI @ dI)
    else:
        log_ev_lam = 0.0

    log_occam = -0.5 * logdetH
    log_ev = log_ev_beta + log_ev_norm + log_ev_alpha + log_ev_lam + log_occam

    # ---- Laplace-ish marginalization correction (your style) ----
    log_delta_log_alpha = 0.0
    for j, blk in enumerate(alpha_blocks):
        if not alpha_update_blocks[j]:
            continue
        tr_blk = float(np.trace(S[np.ix_(blk, blk)]))
        a_blk = float(np.mean(alpha_vec[blk]))
        gamma_blk = blk.size - a_blk * tr_blk
        log_delta_log_alpha += 0.5 * np.log(2.0 / np.maximum(gamma_blk, 1e-12))

    num_regularizer = int(np.sum(alpha_update_blocks))
    log_delta_log_beta = 0.0
    if not fix_beta:
        gamma_beta = float(beta * np.trace(ATA @ S))
        gamma_beta = float(np.clip(gamma_beta, 0.0, N - 1e-12))
        num_regularizer += 1
        log_delta_log_beta = 0.5 * np.log(2.0 / np.maximum(N - gamma_beta, 1e-12))

    log_delta_log_lam = 0.0
    if lam_enabled and (lambda_fix is None):
        gamma_lam = n_pix_eff
        num_regularizer += 1
        log_delta_log_lam = 0.5 * np.log(2.0 / np.maximum(gamma_lam, 1e-12))

    log_ev_marginalized = (
        log_ev
        + log_delta_log_alpha
        + log_delta_log_beta
        + log_delta_log_lam
        + num_regularizer / 2.0 * np.log(2 * np.pi)
    )
    if verbose:
        
        if alpha_mode == "scalar":
            alpha_print = float(alpha_vec[0])
        else:
            alpha_print = np.asarray([alpha_vec[0], alpha_vec[1] if k > 1 else alpha_vec[0]], dtype=float)
    
        lam_print = lambda_fix if lam_enabled else 'disabled'
        print(f"Final hyperparameters: alpha={alpha_print}, beta={beta if not fix_beta else 'fixed'}, lambda_fix={lam_print}")
    if alpha_mode == "scalar":
        alpha_out = float(alpha_vec[0])
    else:
        alpha_out = np.asarray([alpha_vec[0], alpha_vec[1] if k > 1 else alpha_vec[0]], dtype=float)
    return m, S, alpha_out, beta_out, lambda_fix, log_ev, log_ev_marginalized
