"""Core mapping stubs.

Replace these with your real forward/inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import constants

import starry
# starry.config.lazy = False  # disable lazy evaluation
# starry.config.quiet = True  # disable warnings
from spectralmap.bayesian_linalg import optimize_hyperparameters

def intensity_to_temperature(intensity: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """
    Convert spectral intensity to brightness temperature using the inverse Planck function.
    
    Parameters
    ----------
    intensity : np.ndarray
        Spectral intensity in SI units (W/m^2/sr/m).
    wavelength : np.ndarray
        Wavelength in meters.
        
    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin.
    """
    h = constants.h
    c = constants.c
    k = constants.k
    
    # Avoid division by zero warnings by using a tiny lower bound if needed, 
    # though physically intensity must be > 0.
    
    val = (h * c) / (wavelength * k)
    # Planck: I = (2hc^2/lambda_fix^5) * 1/(exp(hc/lambda_fix k T) - 1)
    # exp(hc/lambda_fix k T) - 1 = 2hc^2 / (lambda_fix^5 * I)
    # T = (hc/lambda_fix k) / ln(1 + 2hc^2/(lambda_fix^5 * I))
    
    term = (2 * h * c**2) / (wavelength**5 * intensity)
    return val / np.log(1 + term)

@dataclass
class LightCurveData:
    """Container for light curve (or spectrophotometric) time series."""
    def __init__(self, theta: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None, wavelength: np.ndarray | None = None, inc: int | None = None):
        self.flux = flux
        self.theta = theta
        self.flux_err = flux_err
        self.wavelength = wavelength
        self.inc = inc


def expand_moll_values(values: np.ndarray, moll_mask: np.ndarray, fill_value=np.nan) -> np.ndarray:
    """Expand masked pixel values back to full map pixel length.

    Parameters
    ----------
    values : np.ndarray
        Array whose last dimension is the number of valid moll pixels.
    moll_mask : np.ndarray
        Boolean mask over full pixels (flat or map_res x map_res).
    fill_value : float
        Fill value for invalid pixels.
    """
    values = np.asarray(values)
    mask = np.asarray(moll_mask, dtype=bool).ravel()

    if values.shape[-1] != int(mask.sum()):
        raise ValueError(
            f"values last dimension ({values.shape[-1]}) does not match number of valid pixels ({int(mask.sum())})."
        )

    out_shape = values.shape[:-1] + (mask.size,)
    out = np.full(out_shape, fill_value, dtype=float)
    out[..., mask] = values
    return out

class Map:
    """Unified mapping interface for rotational and eclipse workflows.

    Parameters
    ----------
    mode : str
        Either ``'rotational'`` (default) or ``'eclipse'``.
    pri, sec : optional
        Required only for ``mode='eclipse'``.
    """
    def __init__(
        self,
        mode: str | None = "rotational",
        map_res: int | None = 30,
        ydeg: int | None = None,
        inc: int | None = None,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
    ):
        mode_norm = str(mode).lower()
        if mode_norm in {"rot", "rotational"}:
            self.mode = "rotational"
            self.inc = inc
            self.map = starry.Map(ydeg=ydeg, inc=inc)
            
        elif mode_norm in {"ecl", "eclipse"}:
            if pri is None or sec is None:
                raise ValueError("mode='eclipse' requires both pri and sec.")
            self.mode = "eclipse"
            self.inc = None
            self.pri = pri
            self.sec = sec
            self.sys = starry.System(pri, sec)
            self.map = self.sec.map
        else:
            raise ValueError("mode must be one of {'rotational', 'eclipse'}")

        self.map_res = map_res
        self.ydeg = ydeg
        self.mean = None
        self.cov = None
        self.flux = None
        self.flux_err = None
        self.theta = None
        self.hyper = None
        self.eclipse_depth = None
        
    def design_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute design matrix for given observation angles theta."""
        if self.mode == "eclipse":
            A_full = self.sys.design_matrix(theta)
            if hasattr(A_full, 'eval'):
                A_full = A_full.eval()
            A_full = np.asarray(A_full)
            A_star = A_full[:, :1]
            A_planet = A_full[:, 4:]
            A = np.column_stack((A_star, A_planet))
            self.A_star = A_star
            self.A_planet = A_planet
            self.design_matrix_ = A
            self.theta = theta
            return A
        elif self.mode == "rotational":
            A = self.map.design_matrix(theta=theta)
            if hasattr(A, 'eval'):
                A = A.eval()
            A = np.asarray(A)
            self.design_matrix_ = A
            self.theta = theta
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Currently only 'rotational' and 'eclipse' are supported.")
        return A
    
    def intensity_design_matrix(self, proj: str = "rect") -> np.ndarray:
        """Compute intensity design matrix for given lat/lon grid."""

        lats, lons = self.map.get_latlon_grid(res=self.map_res, projection=proj)
        if hasattr(lats, 'eval'):
            lats = lats.eval()
        if hasattr(lons, 'eval'):
            lons = lons.eval()
        self.proj = proj
        self.lats, self.lons = lats, lons
        self.moll_mask = np.isfinite(self.lats) & np.isfinite(self.lons)
        self.moll_mask_flat = self.moll_mask.flatten()
        lat_flat = self.lats.flatten()
        lon_flat = self.lons.flatten()
        mask = self.moll_mask_flat

        lat_safe = np.where(mask, lat_flat, 0.0)
        lon_safe = np.where(mask, lon_flat, 0.0)

        if self.mode == "eclipse":
            I_planet = self.sec.map.intensity_design_matrix(lat=lat_safe, lon=lon_safe)
            if hasattr(I_planet, 'eval'):
                I_planet = I_planet.eval()
            I_planet = np.asarray(I_planet)
            
            if I_planet.shape[0] == mask.shape[0]:
                I_planet[~mask, :] = 0.0
            self.I_planet = I_planet
            I = np.column_stack((np.zeros((I_planet.shape[0], 1)), I_planet))
            self.intensity_design_matrix_ = I
        
        elif self.mode == "rotational":
            I = self.map.intensity_design_matrix(lat=lat_safe, lon=lon_safe)
            if hasattr(I, 'eval'):
                I = I.eval()
            I = np.asarray(I)
            if I.shape[0] == mask.shape[0]:
                I[~mask, :] = 0.0
            self.intensity_design_matrix_ = I
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Only 'rotational' and 'eclipse' are supported.")
        
        return I
    
    def eigenmap_decomposition(self, theta: np.ndarray) -> np.ndarray:
        """Compute eigenmap image space"""
        A = self.design_matrix(theta)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        U = U * s[np.newaxis, :]  # scale U by singular values
        null_space = s <= 1e-8
        nul_U = U[:, null_space]
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]
        
        return U, Vt, nul_U, img_U, nul_Vt, img_Vt
    
    def solve_posterior(
        self,
        y: np.ndarray,
        sigma_y: np.ndarray | None = None,
        theta: np.ndarray | None = None,
        lambda_fix: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve posterior in reduced SVD/eigencurve space, then project back."""
        if kwargs:
            bad = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported keyword arguments: {bad}")

        if lambda_fix is None:
            use_lambda = False
        else:
            use_lambda = True
            if lambda_fix <= 0.0:
                raise ValueError("lambda is a regularization parameter and must be > 0.")
            
        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided the first time solve_posterior is called.")
            theta = self.theta

        A_full = self.design_matrix(theta)
        I_full = self.intensity_design_matrix()

        A_fit = A_full[:, 1:]
        I_fit = I_full[:, 1:]
        y_fit = y - A_full[:, 0]
        U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
        U = U * s[np.newaxis, :]
        null_space = s <= 1e-8
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]
        
        lats = self.lats.flatten()
        lons = self.lons.flatten()

        lat_u = np.unique(lats)
        lon_u = np.unique(lons)
        if lat_u.size > 1:
            dlat = np.deg2rad(np.median(np.diff(np.sort(lat_u))))
        else:
            dlat = np.deg2rad(180.0)
        if lon_u.size > 1:
            dlon = np.deg2rad(np.median(np.diff(np.sort(lon_u))))
        else:
            dlon = np.deg2rad(360.0)

        lat_r = np.deg2rad(lats)
        lat_lo = np.clip(lat_r - 0.5 * dlat, -0.5 * np.pi, 0.5 * np.pi)
        lat_hi = np.clip(lat_r + 0.5 * dlat, -0.5 * np.pi, 0.5 * np.pi)
        w_pix = dlon * (np.sin(lat_hi) - np.sin(lat_lo))
        w_pix = np.maximum(w_pix, 0.0)

        I_constraint = I_fit @ img_Vt.T
        mu0_fit = np.r_[0.0 if self.eclipse_depth is None else float(self.eclipse_depth), np.zeros(img_U.shape[1] - 1)]
        mean_img, cov_img, alpha, beta_out, lambda_fix, log_ev, log_ev_marginalized = optimize_hyperparameters(
            img_U,
            y_fit,
            sigma_y=sigma_y,
            mu0=mu0_fit,
            maxit=10000,
            lambda_fix=lambda_fix,
            use_lambda=use_lambda,
            I=I_constraint,
            w_pix=w_pix,
            verbose=verbose,
        )

        mean_fit = img_Vt.T @ mean_img
    
        alpha_arr = np.asarray(alpha, dtype=float).ravel()
        alpha_eff = float(alpha_arr[1]) if alpha_arr.size >= 2 else (float(alpha_arr[-1]) if alpha_arr.size > 0 else 1e-12)
        alpha_eff = max(alpha_eff, 1e-12)
        cov_fit = img_Vt.T @ cov_img @ img_Vt + nul_Vt.T @ nul_Vt / alpha_eff


        mean = np.zeros(A_full.shape[1])
        mean[0] = 1.0
        mean[1:] = mean_fit
        cov = np.zeros((A_full.shape[1], A_full.shape[1]))
        cov[1:, 1:] = cov_fit

        alpha_arr = np.asarray(alpha, dtype=float).ravel()
        alpha_h = alpha_arr.tolist() if alpha_arr.size > 1 else float(alpha_arr[0])

        self.hyper = {
            "alpha": alpha_h,
            "beta": None if beta_out is None else float(beta_out),
            "lambda_fix": lambda_fix,
            "log_ev": float(log_ev),
            "log_ev_marginalized": float(log_ev_marginalized),
            "planet_area_sr": float(np.sum(w_pix)),
        }

        if (self.mode == "eclipse") and (alpha_arr.size == 2):
            self.hyper["alpha_constant"] = float(alpha_arr[0])
            self.hyper["alpha_harmonic"] = float(alpha_arr[1])
        elif (self.mode == "rotational"):
            self.hyper["alpha_harmonic"] = float(alpha_arr[0]) if alpha_arr.size > 0 else np.nan

        if verbose:
            if (self.mode == "eclipse") and (alpha_arr.size == 2):
                alpha_print = (
                    f"alpha_constant={self.hyper['alpha_constant']}, "
                    f"alpha_harmonic={self.hyper['alpha_harmonic']}"
                )
            elif (self.mode == "rotational"):
                alpha_print = f"alpha_harmonic={self.hyper['alpha_harmonic']}"
            else:
                alpha_print = f"alpha={alpha_h}"
            print(
                f"Optimized hyperparameters: {alpha_print}, beta={beta_out if beta_out is not None else 'uncertainties provided'}, "
                f"lambda_fix={lambda_fix}, log_ev={log_ev}, log_ev_marginalized={log_ev_marginalized}"
            )   

        self.mean = mean
        self.cov = cov
        self.flux = y
        self.flux_err = sigma_y if sigma_y is not None else (np.nan if beta_out is None else 1 / np.sqrt(beta_out))
        self.theta = theta

        return mean, cov, log_ev_marginalized
    
    def show(self, projection='ortho', **kwargs):
        if self.mode == "eclipse":
            self.sec.map[:, :] = self.mean[1:]
            self.sec.map.show(projection=projection, **kwargs)
        elif self.mode == "rotational":
            self.map[:, :] = self.mean
            self.map.show(projection=projection, **kwargs)

        return
    
    def draw(self, n_samples=10, plot=False, projection='ortho', **kwargs):
        """Draw random samples from the posterior map distribution."""
        samples = np.random.multivariate_normal(self.mean, self.cov, size=n_samples)
        
        for i in range(n_samples):
            if self.mode == "eclipse":
                self.sec.map[:, :] = samples[i][1:]
                self.sec.map.show(projection=projection, **kwargs)
            else:
                self.map[:, :] = samples[i]
                self.map.show(projection=projection, **kwargs)
        return samples
    
    def plot_lightcurve(self):
        if self.flux is None or self.theta is None:
            print("No light curve data to plot. Run solve_posterior() first.")
        else:
            import matplotlib.pyplot as plt
            plt.errorbar(self.theta, self.flux, yerr=self.flux_err, label='Data')
            model_flux = self.design_matrix_ @ self.mean
            plt.plot(self.theta, model_flux, label='Model', color='C1', zorder=10)
            plt.xlabel('Phase Angle')
            plt.ylabel('Flux')
            plt.legend()




class Maps:
    '''Collection of mapping utilities for multi-wavelength data.'''
    
    def __init__(
        self,
        mode: str,
        ydegs: np.ndarray,
        map_res=30,
        verbose=True,
        n_eff_evidence: float | None = None,
    ):
        self.mode = mode
        self.ydegs = ydegs
        self.map_res = map_res
        self.lambda_fix = None
        self.verbose = verbose
        self.n_eff_evidence = n_eff_evidence
        self.moll_mask = None
        self.lats = None
        self.lons = None


    def fit_all_ydegs(self, data: LightCurveData):
        """Fit maps across all ydeg values, returning evidence and coefficients for each wavelength."""
        if self.mode == "rotational" and data.inc == 90:
            ydegs = self.ydegs[self.ydegs % 2 == 0] # null space
        else:
            ydegs = self.ydegs
        
        n_wl = data.flux.shape[0]
        n_ydeg = len(ydegs)
        log_evs = np.zeros((len(ydegs), n_wl))
        coeffs_means = []
        coeffs_covs = []
        wl_done_counts = np.zeros(n_wl, dtype=int)

        for i_ydeg, ydeg in enumerate(ydegs):
            map_obj = Map(ydeg=ydeg, inc=data.inc, mode=self.mode)
            mean_nwl = np.zeros((n_wl, map_obj.map.Ny))
            cov_nwl = np.zeros((n_wl, map_obj.map.Ny, map_obj.map.Ny))

            for i_wl in range(n_wl):
                y = data.flux[i_wl]
                sigma_y = data.flux_err[i_wl] if data.flux_err is not None else None
                mean, cov, log_ev_marginalized = map_obj.solve_posterior(
                    y,
                    sigma_y=sigma_y,
                    theta=data.theta,
                    lambda_fix=self.lambda_fix,
                    verbose=self.verbose,
                )
                log_evs[i_ydeg, i_wl] = log_ev_marginalized
                mean_nwl[i_wl] = mean
                cov_nwl[i_wl] = cov
                wl_done_counts[i_wl] += 1
                if self.verbose and wl_done_counts[i_wl] == n_ydeg:
                    ydeg_best_i = ydegs[np.argmax(log_evs[:, i_wl])]
                    print(f"Wavelength {i_wl+1}/{n_wl}: best ydeg={ydeg_best_i}")

            coeffs_means.append(np.array(mean_nwl))
            coeffs_covs.append(np.array(cov_nwl))

        return ydegs, log_evs, coeffs_means, coeffs_covs, log_evs


    def best_ydeg_maps(self, data: LightCurveData):
        """Get best-fit maps for each wavelength based on evidence over ydeg range."""
        ydegs, log_evs, coeffs_means, coeffs_covs, log_evs = self.fit_all_ydegs(data)

        i_ydeg_best = np.argmax(log_evs, axis=0)
        I_cached = []
        for ydeg in ydegs:
            map = Map(mode=self.mode, map_res=self.map_res, ydeg=ydeg, inc=data.inc)
            I = map.intensity_design_matrix(proj="moll")
            I_cached.append(I)
        

        moll_mask = map.moll_mask
        moll_mask_flat = moll_mask.flatten()
        self.moll_mask = moll_mask
        self.moll_mask_flat = moll_mask_flat
        self.lats = map.lats
        self.lons = map.lons
        n_wl = data.flux.shape[0]
        n_pix = int(moll_mask.sum())

        I_all_wl = np.zeros((n_wl, n_pix))
        I_cov_all_wl = np.zeros((n_wl, n_pix, n_pix))
        ydeg_all_wl = np.zeros(n_wl, dtype=int)
        for i_wl in range(n_wl):
            i_ydeg = i_ydeg_best[i_wl]
            ydeg = ydegs[i_ydeg]
            I = I_cached[i_ydeg]
            I_use = I[moll_mask_flat, :]

            mean = coeffs_means[i_ydeg][i_wl]
            cov = coeffs_covs[i_ydeg][i_wl]
            I_all_wl[i_wl] = I_use @ mean.T
            I_cov_all_wl[i_wl] = I_use @ cov @ I_use.T
            ydeg_all_wl[i_wl] = ydeg

        return ydeg_all_wl, I_all_wl, I_cov_all_wl

    @classmethod
    def plot(
        cls,
        values_by_wavelength: np.ndarray,
        moll_mask: np.ndarray | None = None,
        iw: int = 0,
        map_res: int | None = None,
        wl: np.ndarray | None = None,
        cbar_label: str = "Flux density",
        cmap: str = "inferno",
        levels: int = 200,
        smooth_boundary: bool = True,
        hide_ticks: bool = True,
        show_grid: bool = True,
        ax=None,
    ):
        """Plot a starry-like Mollweide map, handling masked/full inputs internally."""
        import matplotlib.pyplot as plt

        values = np.asarray(values_by_wavelength)
        if values.ndim == 1:
            values = values[None, :]

        n_pix_in = values.shape[-1]

        if moll_mask is not None:
            mask = np.asarray(moll_mask, dtype=bool).ravel()
            if map_res is None:
                map_res = int(np.sqrt(mask.size))

            if n_pix_in == mask.size:
                full = values
            elif n_pix_in == int(mask.sum()):
                full = expand_moll_values(values, mask)
            else:
                raise ValueError(
                    f"values last dimension ({n_pix_in}) does not match full ({mask.size}) or masked ({int(mask.sum())}) pixel count."
                )
        else:
            if map_res is None:
                root = int(np.sqrt(n_pix_in))
                if root * root == n_pix_in:
                    map_res = root
                    full = values
                else:
                    raise ValueError(
                        "Cannot infer map resolution from masked values. Provide map_res; mask handling is done internally."
                    )
            else:
                full_pix = int(map_res) * int(map_res)
                if n_pix_in == full_pix:
                    full = values
                else:
                    temp_map = cls(map_res=map_res, grid_projection="moll")
                    mask = temp_map.moll_mask_flat
                    if n_pix_in != int(mask.sum()):
                        raise ValueError(
                            f"values last dimension ({n_pix_in}) does not match expected masked pixel count ({int(mask.sum())}) for map_res={map_res}."
                        )
                    full = expand_moll_values(values, mask)

        img = full[iw].reshape(map_res, map_res)

        lon = np.linspace(-np.pi, np.pi, map_res)
        lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, map_res)
        lon2d, lat2d = np.meshgrid(lon, lat)

        if smooth_boundary:
            from scipy.ndimage import distance_transform_edt

            img_plot = img.copy()
            valid = np.isfinite(img_plot)
            nearest_idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            img_plot[~valid] = img_plot[tuple(nearest_idx[:, ~valid])]
        else:
            img_plot = np.ma.masked_invalid(img)

        if ax is None:
            fig = plt.figure(figsize=(8, 4.5))
            ax = fig.add_subplot(111, projection="mollweide")
        else:
            fig = ax.figure

        pcm = ax.contourf(lon2d, lat2d, img_plot, levels=levels, cmap=cmap)

        if show_grid:
            ax.grid(True, alpha=0.3)

        cb = fig.colorbar(pcm, ax=ax, pad=0.08)
        cb.set_label(cbar_label)

        if wl is None:
            ax.set_title(f"Mollweide map (wavelength bin {iw})")
        else:
            ax.set_title(f"{wl[iw]: .2f} $\\mu$m")

        if hide_ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        return fig, ax, pcm


def create_map(mode: str = "rotational", **kwargs) -> Map:
    """User-friendly factory for rotational/eclipsed mapping.

    Examples
    --------
    Rotational:
        create_map(mode="rotational", ydeg=5, inc=90, map_res=30)

    Eclipse:
        create_map(mode="eclipse", pri=pri, sec=sec, map_res=30)
    """
    return Map(mode=mode, **kwargs)
