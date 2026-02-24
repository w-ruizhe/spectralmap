"""Core mapping stubs.

Replace these with your real forward/inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
<<<<<<< HEAD
from tqdm.auto import tqdm
=======
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

import starry
# starry.config.lazy = False  # disable lazy evaluation
# starry.config.quiet = True  # disable warnings
from spectralmap.bayesian_linalg import optimize_hyperparameters
from spectralmap.utilities import expand_moll_values, gamma_log_prior_lambda, solid_angle_weights, log_delta_lambda

@dataclass
class LightCurveData:
    theta: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None
    wavelength: np.ndarray | None = None
    inc: int | None = None

def _normalize_mode(mode: str | None) -> str:
    mode_norm = str(mode or "rotational").lower()
    if mode_norm in {"rot", "rotational"}:
        return "rotational"
    if mode_norm in {"ecl", "eclipse"}:
        return "eclipse"
    raise ValueError("mode must be one of {'rotational', 'eclipse'}")


class Map:
    """Shared base class for rotational/eclipsed map inference."""

    mode: str = "base"

<<<<<<< HEAD
    def __init__(self, map_res: int = 30, ydeg: int = 2):
        self.map_res = map_res
        self.ydeg = ydeg
        self.map = None
=======
    def __init__(self, map_res: int | None = 30, ydeg: int | None = None):
        self.map_res = int(map_res) if map_res is not None else 30
        self.ydeg = ydeg
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        self.mu = None
        self.cov = None
        self.flux = None
        self.flux_err = None
        self.theta = None
        self.hyper = None
        self.eclipse_depth = None
        self.design_matrix_ = None
        self.intensity_design_matrix_ = None
        self.lat = None
        self.lon = None
<<<<<<< HEAD
        self.lat_flat = None
        self.lon_flat = None
        self.moll_mask = None
        self.moll_mask_flat = None
        self.observed_lon_range = None
        self.observed_mask = None
        self.projection = None
        
=======
        self.moll_mask = None
        self.moll_mask_flat = None
        self.observed_mask = np.ones(self.map_res**2, dtype=bool)  # default to all pixels observed

>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
    @property
    def n_coeff(self) -> int:
        return int(self.map.Ny)

    @staticmethod
    def _eval_to_numpy(value):
        if hasattr(value, "eval"):
            value = value.eval()
        return np.asarray(value)

    def _design_matrix_impl(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _intensity_design_matrix_impl(self, lat_safe: np.ndarray, lon_safe: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _apply_coefficients_to_map(self, coeffs: np.ndarray) -> None:
        raise NotImplementedError

    def design_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute design matrix for given observation angles theta."""
        A = self._design_matrix_impl(theta)
        A = self._eval_to_numpy(A)
        self.design_matrix_ = A
        self.theta = theta
        return A

    def get_latlon_grid(self, projection: str = "rect"):
<<<<<<< HEAD
        if projection != self.projection:
            lat, lon = self.map.get_latlon_grid(res=self.map_res, projection=projection)
            self.lat = self._eval_to_numpy(lat)
            self.lon = self._eval_to_numpy(lon)
            self.lat_flat = self.lat.flatten()
            self.lon_flat = self.lon.flatten()
            self.moll_mask = np.isfinite(self.lat) & np.isfinite(self.lon)
            self.moll_mask_flat = self.moll_mask.flatten()
            self.projection = projection
            self.observed_mask = (self.lon_flat > self.observed_lon_range[0]) & (self.lon_flat < self.observed_lon_range[1]) if self.observed_lon_range is not None else np.ones_like(self.lon_flat, dtype=bool)
=======
        lat, lon = self.map.get_latlon_grid(res=self.map_res, projection=projection)
        self.lat = self._eval_to_numpy(lat)
        self.lon = self._eval_to_numpy(lon)
        self.lat_flat = self.lat.flatten()
        self.lon_flat = self.lon.flatten()
        self.moll_mask = np.isfinite(self.lat) & np.isfinite(self.lon)
        self.moll_mask_flat = self.moll_mask.flatten()
        self.projection = projection
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        return self.lat, self.lon

    def intensity_design_matrix(self, projection: str = "rect") -> np.ndarray:
        """Compute intensity design matrix for given lat/lon grid."""
<<<<<<< HEAD
        self.get_latlon_grid(projection=projection)
        mask = self.moll_mask_flat
        I = self._intensity_design_matrix_impl(self.lat_flat[mask], self.lon_flat[mask])
        I = self._eval_to_numpy(I)    
        I = I[self.observed_mask[mask], :]
        self.intensity_design_matrix_ = I
        
=======
        lat, lon = self.get_latlon_grid(projection=projection)
        self.lat, self.lon = lat, lon
        mask = self.moll_mask_flat
        I = self._intensity_design_matrix_impl(self.lat_flat[mask], self.lon_flat[mask])
        I = self._eval_to_numpy(I)
        self.intensity_design_matrix_ = I
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        return I

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
        
        lambda_enabled = bool(lambda_fix is not None)
        if lambda_enabled:
            if lambda_fix <= 0.0:
                raise ValueError("lambda is a regularization parameter and must be > 0.")

        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided the first time solve_posterior is called.")
            theta = self.theta

        A_full = self.design_matrix(theta)
        A_fit = A_full[:, 1:]
        U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
        U = U * s[np.newaxis, :]
        null_space = s <= 1e-8
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]
        
        y_fit = y - A_full[:, 0]
        if lambda_enabled:
            if self.intensity_design_matrix_ is None:
<<<<<<< HEAD
                I_full = self.intensity_design_matrix(projection="rect")
            else:
                I_full = self.intensity_design_matrix_

            I_fit = I_full[:, 1:]
            I_constraint = I_fit @ img_Vt.T
            w_pix = solid_angle_weights(self.lat_flat, self.lon_flat)[self.observed_mask]
=======
                 I_full = self.intensity_design_matrix(projection="rect")
            else:
                I_full = self.intensity_design_matrix_
            I_fit = I_full[self.observed_mask, 1:]
            I_constraint = I_fit @ img_Vt.T
            w_pix = solid_angle_weights(self.lat_flat, self.lon_flat)
            w_pix = w_pix[self.observed_mask]
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        else:
            I_constraint = None
            w_pix = None

        
        mu0 = np.r_[0.0 if self.eclipse_depth is None else float(self.eclipse_depth), np.zeros(A_fit.shape[1] - 1)]
        mu0_img = img_Vt @ mu0          # because img_Vt is (r, k) so img_Vt @ mu0 gives (r,)

        mu_img, cov_img, alpha, beta_out, log_ev, log_ev_marginalized = optimize_hyperparameters(
            img_U,
            y_fit,
            sigma_y=sigma_y,
            mu0=mu0_img,
            lambda_fix=lambda_fix,
            I=I_constraint,
            w_pix=w_pix,
            verbose=False,
        )

        mu_fit = img_Vt.T @ mu_img
        alpha_arr = np.asarray(alpha, dtype=float).ravel()
        alpha_eff = float(alpha_arr[1]) if alpha_arr.size >= 2 else (float(alpha_arr[-1]) if alpha_arr.size > 0 else 1e-12)
        alpha_eff = max(alpha_eff, 1e-12)
        cov_fit = img_Vt.T @ cov_img @ img_Vt + nul_Vt.T @ nul_Vt / alpha_eff

        mu = np.zeros(A_full.shape[1])
        mu[0] = 1.0
        mu[1:] = mu_fit
        cov = np.zeros((A_full.shape[1], A_full.shape[1]))
        cov[1:, 1:] = cov_fit

        alpha_h = alpha_arr.tolist() if alpha_arr.size > 1 else float(alpha_arr[0])
        self.hyper = {
            "alpha": alpha_h,
            "beta": None if beta_out is None else float(beta_out),
            "lambda_fix": lambda_fix,
            "log_ev": float(log_ev),
            "log_ev_marginalized": float(log_ev_marginalized),
        }
<<<<<<< HEAD
=======
        if verbose:
            print(
                f"Optimized hyperparameters: {alpha_h}, beta={beta_out if beta_out is not None else 'uncertainties provided'}, "
                f"lambda_fix={lambda_fix if lambda_fix else 'disabled'}, log_ev={log_ev}, "
                f"log_ev_marginalized={log_ev_marginalized}"
            )
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

        self.mu = mu
        self.cov = cov
        self.flux = y
        self.flux_err = sigma_y if sigma_y is not None else (np.nan if beta_out is None else 1 / np.sqrt(beta_out))
        self.theta = theta
        return mu, cov, log_ev_marginalized

    def show(self, projection: str = "ortho", **kwargs):
        self._apply_coefficients_to_map(self.mu)
        self.map.show(projection=projection, **kwargs)
        return

    def draw(self, n_samples: int = 10, plot: bool = False, projection: str = "ortho", **kwargs):
        """Draw random samples from the posterior map distribution."""
        samples = np.random.multivariate_normal(self.mu, self.cov, size=n_samples)
        for i in range(n_samples):
            self._apply_coefficients_to_map(samples[i])
            self.map.show(projection=projection, **kwargs)
        return samples

    def plot_lightcurve(self):
        if self.flux is None or self.theta is None:
            print("No light curve data to plot. Run solve_posterior() first.")
            return
        import matplotlib.pyplot as plt
        plt.errorbar(self.theta, self.flux, yerr=self.flux_err, label="Data")
        model_flux = self.design_matrix_ @ self.mu
        plt.plot(self.theta, model_flux, label="Model", color="C1", zorder=10)
        plt.xlabel("Phase Angle")
        plt.ylabel("Flux")
        plt.legend()


class RotMap(Map):
    """Rotational mapping model."""

    mode = "rotational"

<<<<<<< HEAD
    def __init__(
        self,
        map_res: int | None = 30,
        ydeg: int | None = None,
        inc: int | None = None,
        observed_mask: np.ndarray | None = None,
    ):
=======
    def __init__(self, map_res: int | None = 30, ydeg: int | None = None, inc: int | None = None):
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        if ydeg is None:
            ydeg = 5
        super().__init__(map_res=map_res, ydeg=ydeg)
        self.inc = inc
        self.map = starry.Map(ydeg=ydeg, inc=inc)
<<<<<<< HEAD
        self.observed_mask = observed_mask
=======
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

    def _design_matrix_impl(self, theta: np.ndarray) -> np.ndarray:
        return self.map.design_matrix(theta=theta)

    def _intensity_design_matrix_impl(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        I = self.map.intensity_design_matrix(lat=lat, lon=lon)
        I = self._eval_to_numpy(I)
        return I

    def _apply_coefficients_to_map(self, coeffs: np.ndarray) -> None:
        self.map[:, :] = coeffs


class EclipseMap(Map):
    """Eclipse mapping model."""

    mode = "eclipse"

    def __init__(
        self,
        map_res: int | None = 30,
        ydeg: int | None = None,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        eclipse_depth: float | None = None,
<<<<<<< HEAD
        observed_lon_range: np.ndarray | None = None,
=======
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
    ):
        if pri is None or sec is None:
            raise ValueError("mode='eclipse' requires both pri and sec.")
        super().__init__(map_res=map_res, ydeg=ydeg)
<<<<<<< HEAD
=======
        self.inc = None
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        self.pri = pri
        self.sec = sec
        self.sec.map = starry.Map(ydeg=ydeg, map_res=map_res, inc=90) # currently default to edge-on for eclipse mapping
        self.sys = starry.System(pri, sec)
        self.map = self.sec.map
        self.eclipse_depth = eclipse_depth
<<<<<<< HEAD
        self.observed_lon_range = observed_lon_range
=======
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

    @property
    def n_coeff(self) -> int:
        return int(self.map.Ny) + 1

    def _design_matrix_impl(self, theta: np.ndarray) -> np.ndarray:
        A_full = self.sys.design_matrix(theta)
        A_full = self._eval_to_numpy(A_full)
        self.A_star = A_full[:, :1]
        self.A_planet = A_full[:, 4:]
        return np.column_stack((self.A_star, self.A_planet))

    def _intensity_design_matrix_impl(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        I_planet = self.sec.map.intensity_design_matrix(lat=lat, lon=lon)
        I_planet = self._eval_to_numpy(I_planet)
        self.I_planet = I_planet
        return np.column_stack((np.zeros((I_planet.shape[0], 1)), I_planet))

    def _apply_coefficients_to_map(self, coeffs: np.ndarray) -> None:
        self.sec.map[:, :] = coeffs[1:]


class Maps:
    """Shared multi-wavelength mapping utilities."""

    mode: str = "base"

    def __init__(
        self,
        ydegs: np.ndarray | float,
        lambdas: np.ndarray | float | None = None,
        mode: str | None = None,
        map_res: int = 30,
<<<<<<< HEAD
        observed_lon_range: np.ndarray | None = None,
=======
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        inc: int | None = None,
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        verbose=True,
    ):
        mode_norm = self.mode if mode is None else _normalize_mode(mode)
        if mode_norm not in {"rotational", "eclipse"}:
            raise ValueError("mode must be one of {'rotational', 'eclipse'}")
        self.mode = mode_norm
        self.ydegs = ydegs
        self.map_res = map_res
        self.lambdas = lambdas
        self.verbose = verbose
<<<<<<< HEAD
        self.observed_lon_range = observed_lon_range
        self.inc = None
        self.data = None
        self.pri = None
        self.sec = None
        self.eclipse_depth = None
        self.a_lambda = None
        self.b_lambda = None
        self.observed_mask = None
        self.maps = {}

    def _resolve_inc(self, data: LightCurveData) -> int | None:
        return data.inc if data.inc is not None else self.inc
    
    def get_map_for_ydeg(self, ydeg: int) -> Map:
        if ydeg in self.maps:
            return self.maps[ydeg]
        map = self._make_map_for_ydeg(ydeg)
        self.maps[ydeg] = map
        return map

    def _make_map_for_ydeg(self, ydeg: int) -> Map:
        raise NotImplementedError("Subclasses must implement _make_map_for_ydeg().")

    def _map_for_intensity_cache(self, ydeg: int, inc: int | None) -> Map:
        return self.get_map_for_ydeg(ydeg)

    def _cache_intensity_design_matrices(
        self,
        ydegs: np.ndarray,
        data: LightCurveData,
        projection: str = "moll",
    ) -> list[np.ndarray]:
        inc = self._resolve_inc(data)
        I_cached = []
        map_obj = None

        for ydeg in ydegs:
            map_obj = self._map_for_intensity_cache(ydeg, inc)
            I = map_obj.intensity_design_matrix(projection=projection)
            I_cached.append(I[:, :])

        if map_obj is None:
            raise ValueError("No ydegs provided for intensity cache.")

        self.moll_mask = map_obj.moll_mask
        self.moll_mask_flat = map_obj.moll_mask_flat
        self.lat = map_obj.lat
        self.lon = map_obj.lon
        self.lat_flat = map_obj.lat_flat
        self.lon_flat = map_obj.lon_flat
        self.observed_mask = map_obj.observed_mask
        return I_cached

    @staticmethod
    def _accumulate_bma_moments(weights: np.ndarray, n_pix: int, component_getter):
        mu_bma = np.zeros(n_pix)
        second_moment = np.zeros((n_pix, n_pix))

        for model_idx, weight in np.ndenumerate(weights):
            weight = float(weight)
            if weight == 0.0:
                continue

            I_use, mu, cov = component_getter(model_idx)
            I_mu = I_use @ mu
            I_cov = I_use @ cov @ I_use.T

            mu_bma += weight * I_mu
            second_moment += weight * (I_cov + np.outer(I_mu, I_mu))

        cov_bma = second_moment - np.outer(mu_bma, mu_bma)
        cov_bma = 0.5 * (cov_bma + cov_bma.T)
        return mu_bma, cov_bma
=======
        self.pri = pri
        self.sec = sec
        self.inc = inc
        self.mask = None
        self.mask_flat = None
        self.lat = None
        self.lon = None
        self.a_lambda = None
        self.b_lambda = None
        self.eclipse_depth = None
        self.maps = {}
        self.data

        if self.mode == "eclipse" and (self.pri is None or self.sec is None):
            raise ValueError("EclipseMaps requires both pri and sec.")

    def _resolve_inc(self, data: LightCurveData) -> int | None:
        return data.inc if data.inc is not None else self.inc
    
    def get_map_for_ydeg(self, ydeg: int) -> Map:
        if ydeg in self.maps:
            return self.maps[ydeg]
        if self.mode == "rotational":
            map = RotMap(map_res=self.map_res, ydeg=ydeg, inc=self._resolve_inc(self.data))
        else:
            map = EclipseMap(map_res=self.map_res, ydeg=ydeg, pri=self.pri, sec=self.sec, eclipse_depth=self.eclipse_depth)
        self.maps[ydeg] = map
        return map
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

    def fit_ydegs_fix_lambda(self, data: LightCurveData, lambda_fix=None):
        """Fit maps across all ydeg values, returning evidence and coefficients for each wavelength."""
        self.data = data
        inc = self._resolve_inc(data)
        if self.mode == "rotational" and inc == 90:
            ydegs = self.ydegs[self.ydegs % 2 == 0]
        else:
            ydegs = self.ydegs

        n_wl = data.flux.shape[0]
        n_ydeg = len(ydegs)
        log_evs = np.zeros((len(ydegs), n_wl))
        coeffs_mus = []
        coeffs_covs = []

<<<<<<< HEAD
        ydeg_iter = tqdm(ydegs, desc="ydeg", disable=not self.verbose, dynamic_ncols=True)
        for i_ydeg, ydeg in enumerate(ydeg_iter):
=======
        for i_ydeg, ydeg in enumerate(ydegs):
            print(f"Fitting ydeg={ydeg}...")
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
            map_obj = self.get_map_for_ydeg(ydeg)
            mu_nwl = np.zeros((n_wl, map_obj.n_coeff))
            cov_nwl = np.zeros((n_wl, map_obj.n_coeff, map_obj.n_coeff))

            for i_wl in range(n_wl):
<<<<<<< HEAD
                if self.verbose:
                    step = max(1, n_wl // 10)
                    if (i_wl == 0) or ((i_wl + 1) % step == 0) or (i_wl + 1 == n_wl):
                        ydeg_iter.set_postfix_str(f"wl {i_wl + 1}/{n_wl}")
=======
                print(f"  Wavelength {i_wl+1}/{n_wl}...")
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
                y = data.flux[i_wl]
                sigma_y = data.flux_err[i_wl] if data.flux_err is not None else None
                mu, cov, log_ev_marginalized = map_obj.solve_posterior(
                    y,
                    sigma_y=sigma_y,
                    theta=data.theta,
                    lambda_fix=lambda_fix,
                    verbose=self.verbose,
                )
                log_evs[i_ydeg, i_wl] = log_ev_marginalized
                mu_nwl[i_wl] = mu
                cov_nwl[i_wl] = cov

            coeffs_mus.append(np.array(mu_nwl))
            coeffs_covs.append(np.array(cov_nwl))

<<<<<<< HEAD
        if self.verbose:
            ydeg_iter.set_postfix_str("")

        if self.verbose:
            best_ydeg_per_wl = np.argmax(log_evs, axis=0)
            ydeg_counts = np.bincount(best_ydeg_per_wl, minlength=n_ydeg)
            ydeg_count_text = ", ".join(
                [f"ydeg={ydegs[i]}: {int(ydeg_counts[i])}" for i in range(n_ydeg)]
            )
            print(f"lambda={lambda_fix}: Best-evidence wavelength counts by ydeg: {ydeg_count_text}")

=======
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        return ydegs, log_evs, coeffs_mus, coeffs_covs
    



class RotMaps(Maps):
    """Rotational multi-wavelength mapping utilities."""
<<<<<<< HEAD

    mode = "rotational"

    def __init__(
        self,
        ydegs: np.ndarray,
        map_res=30,
        lambdas: float | None = None,
        verbose=True,
        inc: int | None = None,
        observed_lon_range: np.ndarray | None = None,
    ):
        super().__init__(
            ydegs=ydegs,
            map_res=map_res,
            lambdas=lambdas,
            verbose=verbose,
            observed_lon_range=observed_lon_range,
            mode="rotational",
        )
        self.inc = inc

    def _make_map_for_ydeg(self, ydeg: int) -> Map:
        return make_map(
            "rotational",
            map_res=self.map_res,
            ydeg=ydeg,
            inc=self._resolve_inc(self.data),
            observed_mask=self.observed_mask,
        )

    def marginalized_maps(self, data: LightCurveData):
        """Bayesian model average over ydeg for each wavelength.

        Returns
        -------
        w_all : np.ndarray
            BMA weights over (lambda, ydeg) for each wavelength.
        I_all_wl : np.ndarray
            BMA mu intensity map per wavelength.
        I_cov_all_wl : np.ndarray
            BMA covariance per wavelength (within-model + between-model variance).
        """
        ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all = self.fit_ydegs_fix_lambda(data)
        n_ydeg, n_wl = log_evs_all.shape

        log_prior = np.zeros(n_ydeg, dtype=float) # set uniform prior over ydeg for now

        I_cached = self._cache_intensity_design_matrices(ydegs, data, projection="moll")
=======

    mode = "rotational"

    def __init__(
        self,
        ydegs: np.ndarray,
        map_res=30,
        lambdas: float | None = None,
        verbose=True,
        inc: int | None = None,
    ):
        super().__init__(
            ydegs=ydegs,
            map_res=map_res,
            lambdas=lambdas,
            verbose=verbose,
            mode="rotational",
            inc=inc,
        )

    def marginalized_maps(self, data: LightCurveData):
        """Bayesian model average over ydeg for each wavelength.

        Returns
        -------
        w_all : np.ndarray
            BMA weights over (lambda, ydeg) for each wavelength.
        I_all_wl : np.ndarray
            BMA mu intensity map per wavelength.
        I_cov_all_wl : np.ndarray
            BMA covariance per wavelength (within-model + between-model variance).
        """
        ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all = self.fit_ydegs_fix_lambda(data)
        n_ydeg, n_wl = log_evs_all.shape

        log_prior = np.zeros(n_ydeg, dtype=float) # set uniform prior over ydeg for now

        inc = self._resolve_inc(data)
        I_cached = []
        
        for ydeg in ydegs:
            map_obj = self.get_map_for_ydeg(ydeg)

            I = map_obj.intensity_design_matrix(projection="moll")
            I_cached.append(I[:, :])
        self.moll_mask = map_obj.moll_mask
        self.moll_mask_flat = map_obj.moll_mask_flat
        self.lat = map_obj.lat
        self.lon = map_obj.lon
        self.lat_flat = map_obj.lat_flat
        self.lon_flat = map_obj.lon_flat
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc
        n_pix = I_cached[0].shape[0]

        I_all_wl = np.zeros((n_wl, n_pix))
        I_cov_all_wl = np.zeros((n_wl, n_pix, n_pix))


        log_post = log_evs_all + log_prior[:, np.newaxis] # shape (n_ydeg, n_wl)
        
        w_all = np.zeros_like(log_post)
        for i_wl in range(n_wl):
            # weights over ydeg FOR THIS WAVELENGTH
            logw = log_post[:, i_wl]              # (n_ydeg)
            m = np.max(logw)
            weights = np.exp(logw - m)
            w_sum = np.sum(weights)
            weights /= w_sum                           # normalize

            w_all[:, i_wl] = weights

<<<<<<< HEAD
            def get_components(model_idx):
                i_ydeg = model_idx[0]
                return (
                    I_cached[i_ydeg],
                    coeffs_mus_all[i_ydeg][i_wl],
                    coeffs_covs_all[i_ydeg][i_wl],
                )

            mu_bma, cov_bma = self._accumulate_bma_moments(weights, n_pix, get_components)

            I_all_wl[i_wl] = mu_bma
            I_cov_all_wl[i_wl] = cov_bma


        return w_all, I_all_wl, I_cov_all_wl


class EclipseMaps(Maps):
    """Eclipse multi-wavelength mapping utilities."""

    mode = "eclipse"

    def __init__(
        self,
        ydegs: np.ndarray,
        map_res = 30,
        lambdas: float | None = None,
        a_lambda: float | None = None,
        b_lambda: float | None = None,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        eclipse_depth: float | None = None,
        observed_lon_range: np.ndarray | None = None,
        verbose=True,
    ):
        super().__init__(
            ydegs=ydegs,
            map_res=map_res,
            lambdas=lambdas,
            verbose=verbose,
            mode="eclipse",
            observed_lon_range=observed_lon_range,
        )
        if (pri is None or sec is None):
            raise ValueError("EclipseMaps requires both primary and secondary objects to be passed in.")
        self.pri = pri
        self.sec = sec
        self.eclipse_depth = eclipse_depth
        self.a_lambda = a_lambda
        self.b_lambda = b_lambda

    def _make_map_for_ydeg(self, ydeg: int) -> Map:
        return make_map(
            "eclipse",
            map_res=self.map_res,
            ydeg=ydeg,
            pri=self.pri,
            sec=self.sec,
            eclipse_depth=self.eclipse_depth,
            observed_lon_range=self.observed_lon_range,
        )

    def _map_for_intensity_cache(self, ydeg: int, inc: int | None) -> Map:
        self.sec.map = starry.Map(ydeg=ydeg, map_res=self.map_res)
        return self._make_map_for_ydeg(ydeg)
        

    def fit_ydegs_lambda(self, data: LightCurveData):
        """Fit maps across all ydeg values, returning evidence and coefficients for each wavelength."""

        log_evs_all = []
        coeffs_mus_all = []
        coeffs_covs_all = []
        
        lambda_iter = tqdm(self.lambdas, desc="lambda", disable=not self.verbose, dynamic_ncols=True)
        for lambda_fix in lambda_iter:
            ydegs, log_evs, coeffs_mus, coeffs_covs = self.fit_ydegs_fix_lambda(data, lambda_fix=lambda_fix)
            log_evs_all.append(log_evs)
            coeffs_mus_all.append(coeffs_mus)
            coeffs_covs_all.append(coeffs_covs)
        log_evs_all = np.array(log_evs_all) # shape (n_lambda, n_ydeg, n_wl)
        coeffs_mus_all = coeffs_mus_all # shape (n_lambda, n_ydeg, n_wl, n_coeff)
        coeffs_covs_all = coeffs_covs_all # shape (n_lambda, n_ydeg, n_wl, n_coeff, n_coeff)

        if self.verbose:
            n_lambda, n_ydeg, n_wl = log_evs_all.shape
            best_flat_per_wl = np.argmax(log_evs_all.reshape(n_lambda * n_ydeg, n_wl), axis=0)
            best_ydeg_per_wl = best_flat_per_wl % n_ydeg
            ydeg_counts = np.bincount(best_ydeg_per_wl, minlength=n_ydeg)
            ydeg_count_text = ", ".join(
                [f"ydeg={ydegs[i]}: {int(ydeg_counts[i])}" for i in range(n_ydeg)]
            )
            print(f"Best-evidence wavelength counts by ydeg: {ydeg_count_text}")

        return ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all
    

    def marginalized_maps(self, data: LightCurveData):
        """Bayesian model average over ydeg and lambda for each wavelength.

        Returns
        -------
        ydeg_all_wl : np.ndarray
            MAP ydeg per wavelength (representative model).
        I_all_wl : np.ndarray
            BMA mu intensity map per wavelength.
        I_cov_all_wl : np.ndarray
            BMA covariance per wavelength (within-model + between-model variance).
        """
        ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all = self.fit_ydegs_lambda(data)
        n_lambda, n_ydeg, n_wl = log_evs_all.shape

        log_prior = np.zeros(n_ydeg, dtype=float) # set uniform prior over ydeg for now

        I_cached = self._cache_intensity_design_matrices(ydegs, data, projection="moll")
        n_pix = I_cached[0].shape[0]

        I_all_wl = np.zeros((n_wl, n_pix))
        I_cov_all_wl = np.zeros((n_wl, n_pix, n_pix))


        log_post = log_evs_all + log_prior[np.newaxis, :, np.newaxis] # shape (n_lambda, n_ydeg, n_wl)
        log_post = log_post + gamma_log_prior_lambda(self.lambdas[:, np.newaxis, np.newaxis], self.a_lambda, self.b_lambda) + log_delta_lambda(self.lambdas)[:, np.newaxis, np.newaxis] # add log prior over lambda
        
        rejected_mask = np.zeros_like(log_post, dtype=bool)
        for i_ydeg in range(n_ydeg):
            for i_lambda in range(n_lambda):
                for i_wl in range(n_wl):
                    I_mu = I_cached[i_ydeg] @ coeffs_mus_all[i_lambda][i_ydeg][i_wl] # shape (n_pix)
                    I_sigma = np.sqrt(np.diag(I_cached[i_ydeg] @ coeffs_covs_all[i_lambda][i_ydeg][i_wl] @ I_cached[i_ydeg].T)) # shape (n_pix)
                    if np.any(I_mu < -5e-5): # looser criterion to avoid rejecting models with small negatives
                        rejected_mask[i_lambda, i_ydeg, i_wl] = True
        
        w_all = np.zeros_like(log_post)
        for i_wl in range(n_wl):
            # weights over (lambda, ydeg) FOR THIS WAVELENGTH
            logw = log_post[:, :, i_wl]              # (n_lambda, n_ydeg)
            m = np.max(logw)
            weights = np.exp(logw - m)
            weights[rejected_mask[:, :, i_wl]] = 0.0
            if np.sum(weights) == 0:
                raise RuntimeError(f"Wavelength {i_wl}/{n_wl}: All models were rejected based on negative intensity. Consider adjusting the range of lambdas or inspecting the fitted maps for issues.")
            w_sum = np.sum(weights)
            weights /= w_sum # normalize

            w_all[:, :, i_wl] = weights

            def get_components(model_idx):
                i_lambda, i_ydeg = model_idx
                return (
                    I_cached[i_ydeg],
                    coeffs_mus_all[i_lambda][i_ydeg][i_wl],
                    coeffs_covs_all[i_lambda][i_ydeg][i_wl],
                )

            mu_bma, cov_bma = self._accumulate_bma_moments(weights, n_pix, get_components)
=======
            mu_bma = np.zeros(n_pix)
            second_moment = np.zeros((n_pix, n_pix))
            
            for i_ydeg in range(n_ydeg):
                weight = weights[i_ydeg]
                if weight == 0:
                    continue

                I_use = I_cached[i_ydeg]         # (n_pix, k)
                mu = coeffs_mus_all[i_ydeg][i_wl]   # (k,)
                cov  = coeffs_covs_all[i_ydeg][i_wl]    # (k,k)

                I_mu = I_use @ mu              # (n_pix,)
                I_cov = I_use @ cov @ I_use.T    # (n_pix,n_pix)

                mu_bma += weight * I_mu
                second_moment += weight * (I_cov + np.outer(I_mu, I_mu))

            cov_bma = second_moment - np.outer(mu_bma, mu_bma)
            cov_bma = 0.5 * (cov_bma + cov_bma.T)

            I_all_wl[i_wl] = mu_bma
            I_cov_all_wl[i_wl] = cov_bma


        return w_all, I_all_wl, I_cov_all_wl


class EclipseMaps(Maps):
    """Eclipse multi-wavelength mapping utilities."""

    mode = "eclipse"

    def __init__(
        self,
        ydegs: np.ndarray,
        map_res=30,
        lambdas: float | None = None,
        verbose=True,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        eclipse_depth: float | None = None,
        observed_mask: np.ndarray | None = None,
    ):
        super().__init__(
            ydegs=ydegs,
            map_res=map_res,
            lambdas=lambdas,
            verbose=verbose,
            pri=pri,
            sec=sec,
            mode="eclipse",
            inc=None,
        )
        self.eclipse_depth = eclipse_depth

    def fit_ydegs_lambda(self, data: LightCurveData):
        """Fit maps across all ydeg values, returning evidence and coefficients for each wavelength."""

        log_evs_all = []
        coeffs_mus_all = []
        coeffs_covs_all = []
        
        for lambda_fix in self.lambdas:
            if self.verbose:
                print(f"Fitting with lambda={lambda_fix}...")
            ydegs, log_evs, coeffs_mus, coeffs_covs = self.fit_ydegs_fix_lambda(data, lambda_fix=lambda_fix)
            if self.verbose:
                print(f"Completed fitting for lambda={lambda_fix}.")
            log_evs_all.append(log_evs)
            coeffs_mus_all.append(coeffs_mus)
            coeffs_covs_all.append(coeffs_covs)
        log_evs_all = np.array(log_evs_all) # shape (n_lambda, n_ydeg, n_wl)
        coeffs_mus_all = coeffs_mus_all # shape (n_lambda, n_ydeg, n_wl, n_coeff)
        coeffs_covs_all = coeffs_covs_all # shape (n_lambda, n_ydeg, n_wl, n_coeff, n_coeff)

        return ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all
    

    def marginalized_maps(self, data: LightCurveData):
        """Bayesian model average over ydeg and lambda for each wavelength.

        Returns
        -------
        ydeg_all_wl : np.ndarray
            MAP ydeg per wavelength (representative model).
        I_all_wl : np.ndarray
            BMA mu intensity map per wavelength.
        I_cov_all_wl : np.ndarray
            BMA covariance per wavelength (within-model + between-model variance).
        """
        ydegs, log_evs_all, coeffs_mus_all, coeffs_covs_all = self.fit_ydegs_lambda(data)
        n_lambda, n_ydeg, n_wl = log_evs_all.shape

        log_prior = np.zeros(n_ydeg, dtype=float) # set uniform prior over ydeg for now

        inc = self._resolve_inc(data)
        I_cached = []
        
        for ydeg in ydegs:
            if self.mode == "eclipse":
                self.sec.map = starry.Map(ydeg=ydeg, map_res=self.map_res)
            map_obj = make_map(
                mode=self.mode,
                map_res=self.map_res,
                ydeg=ydeg,
                inc=inc,
                pri=self.pri,
                sec=self.sec,
                eclipse_depth=self.eclipse_depth,
            )
            I = map_obj.intensity_design_matrix(projection="moll")
            I_cached.append(I[:, :])
        self.moll_mask = map_obj.moll_mask
        self.moll_mask_flat = map_obj.moll_mask_flat
        self.lat = map_obj.lat
        self.lon = map_obj.lon
        self.lat_flat = map_obj.lat_flat
        self.lon_flat = map_obj.lon_flat
        n_pix = I_cached[0].shape[0]

        I_all_wl = np.zeros((n_wl, n_pix))
        I_cov_all_wl = np.zeros((n_wl, n_pix, n_pix))


        log_post = log_evs_all + log_prior[np.newaxis, :, np.newaxis] # shape (n_lambda, n_ydeg, n_wl)
        log_post = log_post + gamma_log_prior_lambda(self.lambdas[:, np.newaxis, np.newaxis], self.a_lambda, self.b_lambda) + log_delta_lambda(self.lambdas)[:, np.newaxis, np.newaxis] # add log prior over lambda
        
        rejected_mask = np.zeros_like(log_post, dtype=bool)
        for i_ydeg in range(n_ydeg):
            for i_lambda in range(n_lambda):
                for i_wl in range(n_wl):
                    I_mu = I_cached[i_ydeg] @ coeffs_mus_all[i_lambda][i_ydeg][i_wl] # shape (n_pix)
                    I_sigma = np.sqrt(np.diag(I_cached[i_ydeg] @ coeffs_covs_all[i_lambda][i_ydeg][i_wl] @ I_cached[i_ydeg].T)) # shape (n_pix)
                    if np.any(I_mu - I_sigma < 0):
                        rejected_mask[i_lambda, i_ydeg, i_wl] = True

        if np.sum(~rejected_mask) == 0:
            raise RuntimeError("All models were rejected based on negative intensity. Consider adjusting the log prior over lambda or inspecting the fitted maps for issues.")
        
        w_all = np.zeros_like(log_post)
        for i_wl in range(n_wl):
            # weights over (lambda, ydeg) FOR THIS WAVELENGTH
            logw = log_post[:, :, i_wl]              # (n_lambda, n_ydeg)
            m = np.max(logw)
            weights = np.exp(logw - m)
            weights[rejected_mask[:, :, i_wl]] = 0.0
            w_sum = np.sum(weights)
            weights /= w_sum                           # normalize

            w_all[:, :, i_wl] = weights

            mu_bma = np.zeros(n_pix)
            second_moment = np.zeros((n_pix, n_pix))

            for i_lambda in range(n_lambda):
                for i_ydeg in range(n_ydeg):
                    weight = weights[i_lambda, i_ydeg]
                    if weight == 0:
                        continue

                    I_use = I_cached[i_ydeg]         # (n_pix, k)
                    mu = coeffs_mus_all[i_lambda][i_ydeg][i_wl]   # (k,)
                    cov  = coeffs_covs_all[i_lambda][i_ydeg][i_wl]    # (k,k)

                    mu_j = I_use @ mu              # (n_pix,)
                    cov_j = I_use @ cov @ I_use.T    # (n_pix,n_pix)

                    mu_bma += weight * mu_j
                    second_moment += weight * (cov_j + np.outer(mu_j, mu_j))

            cov_bma = second_moment - np.outer(mu_bma, mu_bma)
            cov_bma = 0.5 * (cov_bma + cov_bma.T)
>>>>>>> 2502033ec197f3286777e1035f40b198689b61fc

            I_all_wl[i_wl] = mu_bma
            I_cov_all_wl[i_wl] = cov_bma


        return w_all, I_all_wl, I_cov_all_wl


def make_map(mode: str = "rotational", **kwargs) -> Map:
    """Factory for rotational/eclipsed mapping classes."""
    mode_norm = _normalize_mode(mode)
    
    if mode_norm == "rotational":
        return RotMap(**kwargs)
    elif mode_norm == "eclipse":
        return EclipseMap(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def make_maps(mode: str = "rotational", **kwargs) -> Maps:
    """Factory for rotational/eclipsed multi-wavelength mapping classes."""
    mode_norm = _normalize_mode(mode)
    
    if mode_norm == "rotational":
        return RotMaps(**kwargs)
    elif mode_norm == "eclipse":
        return EclipseMaps(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")