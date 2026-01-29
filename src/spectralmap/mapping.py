"""Core mapping stubs.

Replace these with your real forward/inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import starry
# starry.config.lazy = False  # disable lazy evaluation
# starry.config.quiet = True  # disable warnings
from spectralmap.bayesian_linalg import optimize_hyperparameters

@dataclass
class LightCurveData:
    """Container for light curve (or spectrophotometric) time series."""
    def __init__(self, theta: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None, wavelength: np.ndarray | None = None, inc: int | None = None):
        self.flux = flux
        self.theta = theta
        self.flux_err = flux_err
        self.wavelength = wavelength
        self.inc = inc

class Map:
    """A helper mapping class wraps starry.Map."""
    def __init__(self, ydeg: int = 5, inc: int = 90, map_res: int = 30, map_type: str = 'Use starry.Map'):
        self.type = map_type
        self.inc = inc
        self.map = starry.Map(ydeg=ydeg, inc=inc)
        lats, lons = self.map.get_latlon_grid(res=map_res, projection='rect')
        self.lats, self.lons = lats, lons
        self.map_res = map_res
        self.ydeg = ydeg
        self.mean = None
        self.cov = None
        self.flux = None
        self.flux_err = None
        self.theta = None

        
    def design_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute design matrix for given observation angles theta."""
        A = self.map.design_matrix(theta=theta)
        
        # Evaluate to numpy if it's a theano/tensor variable
        if hasattr(A, 'eval'):
            A = A.eval()
            
        self.design_matrix_ = A
        self.theta = theta
        return A
    
    def intensity_design_matrix(self) -> np.ndarray:
        """Compute intensity design matrix for given lat/lon grid."""
        I = self.map.intensity_design_matrix(lat=self.lats.flatten(), lon=self.lons.flatten())
        if hasattr(I, 'eval'):
            I = I.eval()
        self.intensity_design_matrix_ = I
        return I
    
    def eigenmap_decomposition(self, theta: np.ndarray) -> np.ndarray:
        """Compute eigenmap image space"""
        A = self.design_matrix(theta)
        A = A[:, 1:]  # remove constant term
        A = A - A.mean(axis=0, keepdims=True) # center design matrix columns (only care about modulation)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        U = U * s[np.newaxis, :]  # scale U by singular values
        null_space = s <= 1e-8
        nul_U = U[:, null_space]
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]
        
        return U, Vt, nul_U, img_U, nul_Vt, img_Vt
    
    def solve_posterior(self, y: np.ndarray, sigma_y: np.ndarray | None = None, theta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve for posterior map coefficients given data."""
        U, Vt, nul_U, img_U, nul_Vt, img_Vt = self.eigenmap_decomposition(theta)

        mu0 = np.zeros(img_U.shape[1])
        mu0[0] = 0 # remember to remove the constant term from A before solving! and mean subtract data
        m, S, alpha, beta, gamma, log_ev, log_ev_marginalized = optimize_hyperparameters(
            img_U, y,
            sigma_y=sigma_y,
            alpha_guess=1.0,        # scalar or (d,) initial prior precisions
            mu0=mu0,
            maxit=5000,
        )
    
        n = Vt.shape[1]
        mean = np.zeros((n))
        cov = np.zeros((n, n))
        nul_dim = nul_Vt.shape[0]
        img_dim = img_Vt.shape[0]
        assert img_dim == m.shape[0] # sanity check
        mean[:] = img_Vt.T @ m  # back to original space
        cov[:] = img_Vt.T @ S @ img_Vt + nul_Vt.T @ nul_Vt / alpha # important: null space still carry uncertainty!
        
        self.mean = mean
        self.cov = cov
        self.flux = y
        self.flux_err = sigma_y if sigma_y is not None else 1/np.sqrt(beta)
        self.theta = theta

        return mean, cov, log_ev_marginalized
    
    def show(self, projection='ortho', **kwargs):
        self.map.y[0] = 1.0  # constant term
        self.map.y[1:] = self.mean
        self.map.show(projection=projection, **kwargs)
        return
    
    def draw(self, n_samples=10, plot=False, projection='ortho', **kwargs):
        """Draw random samples from the posterior map distribution."""
        samples = np.random.multivariate_normal(self.mean, self.cov, size=n_samples)
        
        for i in range(n_samples):
            self.map.y[0] = 1.0
            self.map.y[1:] = samples[i]
            self.map.show(projection=projection, **kwargs)
        return samples
    
    def plot_lightcurve(self):
        if self.flux is None or self.theta is None:
            print("No light curve data to plot. Run solve_posterior() first.")
        else:
            import matplotlib.pyplot as plt
            plt.errorbar(self.theta, self.flux, yerr=self.flux_err, label='Data')
            model_flux = self.design_matrix_[:, 1:] @ self.mean
            plt.plot(self.theta, model_flux, label='Model', color='C1')
            plt.xlabel('Phase Angle')
            plt.ylabel('Flux')
            plt.legend()


def fit_ydeg_range(data: LightCurveData, ydeg_min=2, ydeg_max=10):
    """Determine best spherical harmonic degree based on evidence."""
    ydeg_range = np.arange(ydeg_min, ydeg_max+1)
    ydeg_range = ydeg_range[ydeg_range%2 == 0]  # only even degrees
    n_wl = data.flux.shape[0]
    log_evs =  np.zeros((len(ydeg_range), n_wl))
    coeffs_means = []
    coeffs_covs = []
    inc = data.inc
    
    for i_ydeg, ydeg in enumerate(ydeg_range):
        map = Map(ydeg=ydeg)
        map.map.inc = inc
        mean_nwl = np.zeros((n_wl, map.map.Ny-1))
        cov_nwl = np.zeros((n_wl, map.map.Ny-1, map.map.Ny-1))

        for i_wl in range(n_wl):
            y = data.flux[i_wl]
            sigma_y = data.flux_err[i_wl] if data.flux_err is not None else None
            mean, cov, log_ev_marginalized = map.solve_posterior(y, sigma_y=sigma_y, theta=data.theta)
            log_evs[i_ydeg, i_wl] = log_ev_marginalized
            mean_nwl[i_wl] = mean
            cov_nwl[i_wl] = cov

        coeffs_means.append(np.array(mean_nwl))
        coeffs_covs.append(np.array(cov_nwl))
    
    return ydeg_range, log_evs, coeffs_means, coeffs_covs, log_evs


def best_ydeg_maps(data: LightCurveData, ydeg_min=2, ydeg_max=10, map_res=30):
    """Get best-fit maps for each wavelength based on evidence over ydeg range."""
    ydeg_range, log_evs, coeffs_means, coeffs_covs, log_evs = fit_ydeg_range(data, ydeg_min=ydeg_min, ydeg_max=ydeg_max)
    i_ydeg_best = np.argmax(log_evs, axis=0) # best i_ydeg for each wavelength
    I_cached = []
    for i, ydeg in enumerate(ydeg_range):
        map = Map(ydeg=ydeg, map_res=map_res)
        map.map.inc = data.inc
        I = map.intensity_design_matrix()
        I_cached.append(I)
    n_wl = data.flux.shape[0]
    I_all_wl = np.zeros((n_wl, map_res * map_res))
    I_cov_all_wl = np.zeros((n_wl, map_res * map_res, map_res * map_res))
    ydeg_all_wl = np.zeros(n_wl, dtype=int)
    for i_wl in range(n_wl):
        i_ydeg = i_ydeg_best[i_wl]
        ydeg = ydeg_range[i_ydeg]
        I = I_cached[i_ydeg]

        mean = coeffs_means[i_ydeg][i_wl]
        cov = coeffs_covs[i_ydeg][i_wl]
        I_all_wl[i_wl] = I[:, 1:] @ mean.T + I[:, 0]  # adding the constant term back separately
        I_cov_all_wl[i_wl] = I[:, 1:] @ cov @ I[:, 1:].T
        ydeg_all_wl[i_wl] = ydeg

    return ydeg_all_wl, I_all_wl, I_cov_all_wl
