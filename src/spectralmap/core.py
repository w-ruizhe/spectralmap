"""Core data containers and Bayesian surface-map inference classes.

This module contains the mode-independent pieces used by rotational and eclipse
mapping: validated light-curve inputs, common ``starry`` map construction, a
single-map posterior solver, and the multi-wavelength ``marginalize`` workflow.
Mode-specific geometry lives in :mod:`spectralmap.rotational`,
:mod:`spectralmap.eclipse`, and :mod:`spectralmap.doppler`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import starry

from spectralmap.cluster import get_best_polygon
starry.config.lazy = False  # disable lazy evaluation
starry.config.quiet = True  # disable warnings
from spectralmap.bayesian_linalg import optimize_hyperparameters
from spectralmap.utilities import solid_angle_weights
from spectralmap.plotting import COLOR_LIST

# Numerical tolerances used in posterior linear algebra.
SVD_NULLSPACE_TOL = 1e-8
ALPHA_EFF_FLOOR = 1e-12

@dataclass
class LightCurveData:
    """Validated light-curve container for one or more wavelength channels.

    Parameters
    ----------
    theta
        Rotational phase or eclipse angle in degrees, with length ``n_time``.
        Provide either ``theta`` or ``time``; provide both when later period
        marginalization needs the original time axis.
    time
        Observation times. When ``theta`` is absent, ``Maps.marginalize`` can
        derive phase angles from this axis and trial periods.
    flux
        Flux values with shape ``(n_wavelength, n_time)`` or ``(n_time,)``.
        The data are normalized by each wavelength channel mean.
    flux_err
        Optional flux uncertainties with the same shape as ``flux``.
    wl
        Optional wavelength grid with length ``n_wavelength``.
    """

    theta: np.ndarray | None = None
    time: np.ndarray | None = None
    flux: np.ndarray | None = None
    flux_err: np.ndarray | None = None
    wl: np.ndarray | None = None

    def __post_init__(self):
        """Normalize shapes and validate phase/time consistency."""
        if self.flux is None:
            raise ValueError("flux is required")

        has_theta = self.theta is not None
        has_time = self.time is not None

        if has_theta:
            self.theta = np.asarray(self.theta)
            if self.theta.ndim != 1:
                raise ValueError("theta must be a 1D array")
        else:
            self.theta = None

        if has_time:
            self.time = np.asarray(self.time)
            if self.time.ndim != 1:
                raise ValueError("time must be a 1D array")
        else:
            self.time = None

        if not has_theta and not has_time:
            raise ValueError("LightCurveData must provide at least one of theta or time.")

        if has_theta and has_time:
            assert self.theta.shape == self.time.shape, "theta and time must have the same shape"

        self.flux = np.asarray(self.flux)

        n_time = self.theta.shape[0] if has_theta else (self.time.shape[0] if has_time else None)

        if self.flux.ndim == 1:
            self.flux = self.flux[np.newaxis, :]
        elif self.flux.ndim != 2:
            raise ValueError("flux must be a 1D or 2D array")

        if self.flux.shape[1] != n_time:
            raise ValueError(
                "flux must have shape (n_wl, n_time) with n_time == len(theta)"
            )

        self.amplitude = np.nanmean(self.flux, axis=1)
        self.flux = self.flux / self.amplitude[:, np.newaxis]

        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err)
            if self.flux_err.ndim == 1:
                self.flux_err = self.flux_err[np.newaxis, :]
            elif self.flux_err.ndim != 2:
                raise ValueError("flux_err must be a 1D or 2D array")

            if self.flux_err.shape != self.flux.shape:
                raise ValueError("flux_err must have the same shape as flux")

            self.flux_err = self.flux_err / self.amplitude[:, np.newaxis]

        if self.wl is not None:
            self.wl = np.asarray(self.wl)
            if self.wl.ndim == 0:
                self.wl = self.wl[np.newaxis]
            elif self.wl.ndim != 1:
                raise ValueError("wl must be a 1D array")

            if self.wl.size != self.flux.shape[0]:
                raise ValueError("wl must have length n_wl")






def _resolve_limb_darkening(
    udeg: int | None,
    u: np.ndarray | list[float] | tuple[float, ...] | float | None,
) -> tuple[int | None, np.ndarray | None]:
    """Resolve ``starry`` limb-darkening degree and coefficient inputs."""
    if u is None:
        if udeg is None:
            return None, None
        udeg = int(udeg)
        if udeg < 0:
            raise ValueError("udeg must be >= 0.")
        return udeg, None

    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 0:
        u_arr = u_arr[np.newaxis]
    elif u_arr.ndim != 1:
        raise ValueError("u must be a scalar or 1D array-like.")

    if udeg is None:
        udeg = int(u_arr.size)
    else:
        udeg = int(udeg)

    if udeg < 0:
        raise ValueError("udeg must be >= 0.")
    if udeg == 0:
        raise ValueError("udeg must be > 0 when limb-darkening coefficients are provided.")
    if u_arr.size != udeg:
        raise ValueError(f"Expected {udeg} limb-darkening coefficients, got {u_arr.size}.")

    return udeg, u_arr


def _set_starry_limb_darkening_coeffs(
    starry_map,
    u: np.ndarray | list[float] | tuple[float, ...] | float | None,
) -> None:
    """Assign limb-darkening coefficients to a ``starry`` map."""
    if u is None:
        return

    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 0:
        u_arr = u_arr[np.newaxis]
    elif u_arr.ndim != 1:
        raise ValueError("u must be a scalar or 1D array-like.")

    if int(starry_map.udeg) != int(u_arr.size):
        raise ValueError(
            f"Expected {int(starry_map.udeg)} limb-darkening coefficients for this map, got {int(u_arr.size)}."
        )

    # In starry, map[1], map[2], ... index u1, u2, ... when udeg > 0.
    for i, coeff in enumerate(u_arr, start=1):
        starry_map[i] = float(coeff)


def _build_starry_map(
    ydeg: int,
    map_res: int | None = None,
    inc: int | None = None,
    udeg: int | None = None,
    u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
):
    """Build a ``starry.Map`` with optional inclination and limb darkening."""
    udeg_resolved, u_resolved = _resolve_limb_darkening(udeg=udeg, u=u)
    map_kwargs = {"ydeg": ydeg}
    if map_res is not None:
        map_kwargs["map_res"] = map_res
    if inc is not None:
        map_kwargs["inc"] = inc
    if udeg_resolved is not None:
        map_kwargs["udeg"] = udeg_resolved

    map_obj = starry.Map(**map_kwargs)

    _set_starry_limb_darkening_coeffs(map_obj, u_resolved)

    return map_obj


class Map:
    """Shared base class for single-channel map inference.

    Subclasses provide the geometry-specific design matrices and coefficient
    assignment. The base class handles Bayesian linear inference in eigencurve
    space and converts coefficient posteriors into map-space diagnostics.
    """

    def __init__(self, map_res: int = 30, ydeg: int = 2, projection: str = "rect"):
        self.map_res = map_res
        self.ydeg = ydeg
        self.map = None
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
        self.lat_flat = None
        self.lon_flat = None
        self.moll_mask = None
        self.moll_mask_flat = None
        self.observed_lon_range = None
        self.projection = None
        self.default_projection = projection
        self.null_uncertainty = True

    @property
    def n_coeff(self) -> int:
        """Number of coefficients solved by this map model."""
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
        """Cache and return the latitude/longitude grid for a projection."""
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
        return self.lat, self.lon

    def intensity_design_matrix(self, projection: str = "rect") -> np.ndarray:
        """Compute intensity design matrix for given lat/lon grid."""

        if self.intensity_design_matrix_ is not None and self.projection == projection:
            return self.intensity_design_matrix_

        self.get_latlon_grid(projection=projection)
        mask = self.moll_mask_flat
        I = self._intensity_design_matrix_impl(self.lat_flat[mask], self.lon_flat[mask])
        I = self._eval_to_numpy(I)
        I = I[self.observed_mask[mask], :]
        self.intensity_design_matrix_ = I
        self.projection = projection
        return I

    def solve_posterior(
        self,
        y: np.ndarray,
        sigma_y: np.ndarray | None = None,
        theta: np.ndarray | None = None,
        lamda: float | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve the linear-map posterior for one normalized light curve.

        Parameters
        ----------
        y
            Normalized flux vector with length ``n_time``.
        sigma_y
            Optional flux uncertainty vector or scalar.
        theta
            Observation phase angles in degrees. Required on first call.
        lamda
            Optional positive intensity regularization precision.
        verbose
            Passed through to hyperparameter optimization.

        Returns
        -------
        mu, cov, log_evidence
            Posterior coefficient mean, covariance, and marginalized log
            evidence for the fitted model.
        """

        lamda_enabled = bool(lamda is not None)
        if lamda_enabled:
            if lamda <= 0.0:
                raise ValueError("lamda is a regularization parameter and must be > 0.")

        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided the first time solve_posterior is called.")
            theta = self.theta

        A_full = self.design_matrix(theta)
        A_fit = A_full[:, 1:]
        U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
        U = U * s[np.newaxis, :]
        null_space = s <= SVD_NULLSPACE_TOL
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]

        y_fit = y - A_full[:, 0]
        if lamda_enabled:

            I_full = self.intensity_design_matrix(projection="moll")

            I_fit = I_full[:, 1:]
            I_constraint = I_fit @ img_Vt.T
            valid_observed = self.moll_mask_flat & self.observed_mask
            w_pix = solid_angle_weights(self.lat_flat[valid_observed], self.lon_flat[valid_observed])
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
            lamda=lamda,
            I=I_constraint,
            w_pix=w_pix,
            verbose=False,
        )

        mu_fit = img_Vt.T @ mu_img
        alpha_arr = np.asarray(alpha, dtype=float).ravel()
        alpha_eff = float(alpha_arr[1]) if alpha_arr.size >= 2 else (float(alpha_arr[-1]) if alpha_arr.size > 0 else ALPHA_EFF_FLOOR)
        alpha_eff = max(alpha_eff, ALPHA_EFF_FLOOR)

        if self.null_uncertainty:
            cov_fit = img_Vt.T @ cov_img @ img_Vt + nul_Vt.T @ nul_Vt / alpha_eff
        else:
            cov_fit = img_Vt.T @ cov_img @ img_Vt

        mu = np.zeros(A_full.shape[1])
        mu[0] = 1.0
        mu[1:] = mu_fit
        cov = np.zeros((A_full.shape[1], A_full.shape[1]))
        cov[1:, 1:] = cov_fit

        alpha_h = alpha_arr.tolist() if alpha_arr.size > 1 else float(alpha_arr[0])
        self.hyper = {
            "alpha": alpha_h,
            "beta": None if beta_out is None else float(beta_out),
            "lamda": lamda,
            "log_ev": float(log_ev),
            "log_ev_marginalized": float(log_ev_marginalized),
        }

        self.mu = mu
        self.cov = cov
        self.flux = y
        self.flux_err = sigma_y if sigma_y is not None else (np.nan if beta_out is None else 1 / np.sqrt(beta_out))
        self.theta = theta
        return mu, cov, log_ev_marginalized

    def show(self, projection: str = "ortho", **kwargs):
        """Render the posterior mean map with ``starry.Map.show``."""
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
        """Plot the observed light curve and posterior-mean model."""
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



class Maps:
    """Shared multi-wavelength model-marginalization utilities."""

    def __init__(
        self,
        map_res: int = 30,
        observed_lon_range: np.ndarray | None = None,
        verbose=True,
    ):
        self.map_res = map_res
        self.verbose = verbose
        self.observed_lon_range = observed_lon_range
        self.data = None
        self.pri = None
        self.sec = None
        self.eclipse_depth = None
        self.observed_mask = None
        self.maps = {}
        self.fixed_ydeg = None
        self.mixture_ = None
        self.null_uncertainty = True


    def marginalize(
        self,
        data: LightCurveData,
        *,
        inc: float | list[float] | np.ndarray,
        ydeg: int | list[int] | np.ndarray,
        prot: float | list[float] | np.ndarray | None = None,
        lamda: float | list[float] | np.ndarray | None = None,
        sigma_threshold: float | None = None,
    ):
        """Marginalize maps over discrete geometry and regularization axes.

        Each wavelength channel is fit for every support point in ``inc``,
        ``ydeg``, optional ``prot``, and optional ``lamda``. Scalars are treated
        as fixed values, list-like inputs are assigned equal prior weight, and
        dictionaries may provide ``values`` plus ``weights``, ``log_weights``,
        or a callable ``logpdf``.

        Returns
        -------
        mixture_weights, spatial_intensity, spatial_intensity_cov
            Component weights with shape ``(n_components, n_wavelength)``,
            posterior mean intensities with shape ``(n_wavelength, n_pixel)``,
            and per-wavelength covariance matrices.
        """
        if sigma_threshold is not None and sigma_threshold < 0:
            raise ValueError("sigma_threshold must be >= 0 when provided.")

        self.data = data
        n_wl = data.flux.shape[0]
        self._resolve_u_for_wavelength(i_wl=None, n_wl=n_wl)

        ydeg_values, ydeg_logw = self._resolve_axis(ydeg, "ydeg", dtype=int)
        inc_values, inc_logw = self._resolve_axis(inc, "inc", dtype=float)
        prot_values, prot_logw = self._resolve_axis(prot, "prot", dtype=float)
        lamda_values, lamda_logw = self._resolve_axis(lamda, "lamda", dtype=float)

        prot_enabled = bool(np.any(np.isfinite(prot_values)))
        lamda_enabled = bool(np.any(np.isfinite(lamda_values)))
        self._validate_phase_inputs_for_marginalize(data, prot_enabled=prot_enabled)

        n_components = int(
            len(inc_values)
            * len(prot_values)
            * len(ydeg_values)
            * len(lamda_values)
            * n_wl
        )

        mixture_logw = [[] for _ in range(n_wl)]
        mixture_mu = [[] for _ in range(n_wl)]
        mixture_cov = [[] for _ in range(n_wl)]
        mixture_meta = [[] for _ in range(n_wl)]

        map_cache = {}
        geometry_initialized = False

        with tqdm(
            total=n_components,
            desc=f"{self.__class__.__name__}.marginalize",
            disable=not bool(self.verbose),
            dynamic_ncols=True,
        ) as pbar:
            for i_inc, inc_value in enumerate(inc_values):
                lp_inc = float(inc_logw[i_inc])
                for i_prot, prot_value in enumerate(prot_values):
                    lp_prot = float(prot_logw[i_prot])

                    if prot_enabled:
                        theta_use = self._theta_from_period(data, prot_value)
                    else:
                        prot_value = None
                        theta_use = data.theta

                    for i_ydeg, ydeg_value in enumerate(ydeg_values):
                        lp_ydeg = float(ydeg_logw[i_ydeg])

                        key = (float(inc_value), int(ydeg_value))
                        if key not in map_cache:
                            map_cache[key] = self._make_map(ydeg=int(ydeg_value), inc=float(inc_value))
                        map_obj = map_cache[key]
                        map_obj.null_uncertainty = self.null_uncertainty

                        for i_lamda, lamda_value in enumerate(lamda_values):
                            lp_lamda = float(lamda_logw[i_lamda])
                            lp_total = lp_inc + lp_prot + lp_ydeg + lp_lamda
                            if not lamda_enabled:
                                lamda_value = None

                            for i_wl in range(n_wl):
                                self._set_map_limb_darkening_for_wavelength(map_obj, i_wl=i_wl, n_wl=n_wl)

                                y_wl = data.flux[i_wl]
                                sigma_wl = data.flux_err[i_wl] if data.flux_err is not None else None
                                mu, cov, log_ev = map_obj.solve_posterior(
                                    y_wl,
                                    sigma_y=sigma_wl,
                                    theta=theta_use,
                                    lamda=lamda_value,
                                    verbose=self.verbose,
                                )

                                # Limb-darkening updates can invalidate cached matrices.
                                # Always rebuild/use the current intensity design matrix.
                                I_use = map_obj.intensity_design_matrix(projection="moll")

                                if not geometry_initialized:
                                    self.moll_mask = map_obj.moll_mask
                                    self.moll_mask_flat = map_obj.moll_mask_flat
                                    self.lat = map_obj.lat
                                    self.lon = map_obj.lon
                                    self.lat_flat = map_obj.lat_flat
                                    self.lon_flat = map_obj.lon_flat
                                    self.observed_mask = map_obj.observed_mask
                                    geometry_initialized = True

                                I_mu = I_use @ mu
                                I_cov = I_use @ cov @ I_use.T

                                logw = float(log_ev + lp_total)
                                if sigma_threshold is not None:
                                    I_var = np.clip(np.diag(I_cov), a_min=0.0, a_max=None)
                                    I_sigma = np.sqrt(I_var)
                                    if np.any(I_mu + float(sigma_threshold) * I_sigma < 0.0):
                                        logw = float(-np.inf)

                                mixture_logw[i_wl].append(logw)
                                mixture_mu[i_wl].append(I_mu)
                                mixture_cov[i_wl].append(I_cov)
                                mixture_meta[i_wl].append(
                                    {
                                        "inc": inc_value,
                                        "prot": prot_value,
                                        "ydeg": ydeg_value,
                                        "lamda": lamda_value,
                                    }
                                )
                                pbar.update(1)

        if not geometry_initialized:
            raise RuntimeError("No mixture components were evaluated. Check priors and data.")

        if sigma_threshold is not None:
            for i_wl in range(n_wl):
                if not np.any(np.isfinite(np.asarray(mixture_logw[i_wl], dtype=float))):
                    raise RuntimeError(
                        f"Wavelength {i_wl}/{n_wl}: all models were rejected based on negative intensity. "
                        "Consider relaxing sigma_threshold or adjusting priors."
                    )

        mixture_weights, spatial_intensity, spatial_intensity_cov = self._finalize_mixture(
            mixture_logw,
            mixture_mu,
            mixture_cov,
            mixture_meta,
            axes={"inc": inc, "ydeg": ydeg, "prot": prot, "lamda": lamda},
        )

        spatial_spectra = spatial_intensity * data.amplitude[:, None] * np.pi
        spatial_spectra_cov = spatial_intensity_cov * (np.pi * data.amplitude[:, None, None])**2

        self.mixture_weights = mixture_weights
        self.spatial_intensity = spatial_intensity
        self.spatial_intensity_cov = spatial_intensity_cov
        self.spatial_spectra = spatial_spectra
        self.spatial_spectra_cov = spatial_spectra_cov
        return mixture_weights, spatial_intensity, spatial_intensity_cov


    def find_clusters(self, n_neighbors=100, n_corners=3, plot=True):
        """Cluster pixels by recovered spectra and compute regional spectra."""
        # --- 2. CONFIGURATION & SORTING ---
        X = self.spatial_spectra.T
        log_X = np.log10(X)
        pca = PCA(n_components=2)
        W = pca.fit_transform(log_X)
        self.pc_scores = W

        if n_corners == 2:
            W_centered = W - np.mean(W, axis=0)
            _, _, vh = np.linalg.svd(W_centered, full_matrices=False)
            max_var_dir = vh[0]
            proj = W_centered @ max_var_dir
            i_min = int(np.argmin(proj))
            i_max = int(np.argmax(proj))

            if i_min == i_max:
                raise ValueError("Could not determine two distinct anchors for n_corners=2")

            corner_indices = np.array([i_min, i_max], dtype=int)
            corner_coords = W[corner_indices]
        elif n_corners >= 3:
            corner_indices = get_best_polygon(W, n_corners=n_corners)
            corner_coords = W[corner_indices]

        else:
            raise ValueError("n_corners must be >= 2")

        # Sort by Angle
        centroid = np.mean(corner_coords, axis=0)
        angles = np.arctan2(corner_coords[:, 1] - centroid[1], corner_coords[:, 0] - centroid[0])
        sort_order = np.argsort(angles)
        corner_indices = corner_indices[sort_order]
        corner_coords = corner_coords[sort_order]
        anchor_points = W[corner_indices]
        K = len(anchor_points)
        centers = anchor_points[:, :2]

        # --- 3. K-NEAREST NEIGHBOR ASSIGNMENT ---
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")

        n_points = len(W)

        # Initialize labels as -1 (background / unassigned).
        labels = np.full(n_points, -1, dtype=int)

        # Distances from every point to every anchor, shape (N_points, K_anchors).
        all_dists = np.linalg.norm(W[:, None, :] - centers[None, :, :], axis=2)

        # Step 1: assign each point to its closest anchor.
        closest_anchor = np.argmin(all_dists, axis=1)

        # Step 2: for each anchor, keep at most n_neighbors closest points.
        # If an anchor has fewer than n_neighbors points, keep all of them.
        for k in range(K):
            candidates = np.where(closest_anchor == k)[0]
            if candidates.size == 0:
                continue
            if candidates.size <= n_neighbors:
                labels[candidates] = k
                continue

            # Deterministic tie-break: distance first, then point index.
            order = np.lexsort((candidates, all_dists[candidates, k]))
            keep = candidates[order[:n_neighbors]]
            labels[keep] = k

        # X: (N, D), C: (K, D)
        V = np.zeros((K+1, X.shape[0])) # (n_corners, n_spatial_points)

        for i in range(K+1):
            ind = (labels == i-1)
            print(i-1, np.sum(ind))
            N = np.sum(ind)
            if N == 0:
                continue
            weights = np.ones(N) / N
            V[i, ind] = weights

        regional_spectra = V @ X # (n_corners, n_wavelengths) use the original X (not log_X) for the mean

        regional_spectra_cov = np.einsum('ij,wjk,kl->wil', V, self.spatial_spectra_cov, V.T)
        regional_spectra_std = np.sqrt(np.diagonal(regional_spectra_cov, axis1=1, axis2=2)).T # (n_corners, n_wavelengths)


        self.regional_spectra = regional_spectra
        self.regional_spectra_std = regional_spectra_std
        self.regional_spectra_cov = regional_spectra_cov
        self.labels = labels

        if plot:
            fig, ax = plt.subplots(figsize=(7, 2.5), dpi=300)
            plot_colors = COLOR_LIST

            # 1. Plot UNASSIGNED points (grey, faint)
            mask_unassigned = labels == -1
            plt.scatter(W[mask_unassigned, 0], W[mask_unassigned, 1],
                        s=15, alpha=0.8, color=plot_colors[0], edgecolor='none', zorder=1)

            # 2. Plot ASSIGNED clusters
            for k in range(K):
                mask = labels == k
                color = plot_colors[(k + 1) % len(plot_colors)]
                plt.scatter(W[mask, 0], W[mask, 1],
                            s=15, alpha=0.8, color=color, edgecolor='none', zorder=2,
                            label=f'Cluster {k+1}')

            # 3. Plot Polygon
            poly_draw = np.vstack([corner_coords, corner_coords[0]])
            plt.plot(poly_draw[:, 0], poly_draw[:, 1], 'k-', lw=1.5, alpha=0.7, zorder=3)

            # 4. Labels
            for k in range(K):
                color = plot_colors[(k + 1) % len(plot_colors)]
                plt.text(centers[k, 0], centers[k, 1], f'{k+1}',
                        fontsize=10, fontname='Comic Sans MS', fontweight='bold', color='white',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, edgecolor='white', linewidth=1, alpha=0.9),
                        path_effects=[pe.Stroke(linewidth=1.5, foreground='gray'), pe.Normal()], zorder=10)

            plt.title(f'Classification in PC Space ({n_neighbors} neighbors)', fontsize=9)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.gca().set_aspect('equal')
            plt.tight_layout()

            return fig, ax



    @staticmethod
    def _parse_discrete_prior(prior, name: str):
        if prior is None:
            return None
        if not isinstance(prior, dict):
            raise TypeError(f"{name}_prior must be a dict with values/grid and weights/log_weights/logpdf.")

        values = prior.get("values", prior.get("grid", None))
        if values is None:
            raise ValueError(f"{name}_prior must provide 'values' (or 'grid').")

        values = np.asarray(values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name}_prior values must be a non-empty 1D array.")

        if "log_weights" in prior:
            log_weights = np.asarray(prior["log_weights"], dtype=float)
        elif "weights" in prior:
            weights = np.asarray(prior["weights"], dtype=float)
            if np.any(weights <= 0.0):
                raise ValueError(f"{name}_prior weights must be strictly positive.")
            log_weights = np.log(weights)
        elif "logpdf" in prior and callable(prior["logpdf"]):
            logpdf = prior["logpdf"]
            log_weights = np.asarray([logpdf(v) for v in values], dtype=float)
        else:
            raise ValueError(
                f"{name}_prior must provide one of: 'weights', 'log_weights', or callable 'logpdf'."
            )

        if log_weights.shape != values.shape:
            raise ValueError(f"{name}_prior weights must match values shape.")
        if not np.all(np.isfinite(log_weights) | np.isneginf(log_weights)):
            raise ValueError(f"{name}_prior log-weights must be finite or -inf.")

        return {"values": values, "log_weights": log_weights}

    def _theta_from_period(self, data: LightCurveData, prot: float) -> np.ndarray:
        if data.time is None:
            raise ValueError("Period marginalization requires LightCurveData.time to be provided.")
        if prot <= 0:
            raise ValueError("All period support points must be > 0.")
        t0 = float(data.time[0])
        theta = ((np.asarray(data.time, dtype=float) - t0) * (360.0 / float(prot)))
        return np.mod(theta, 360.0)


    def _validate_phase_inputs_for_marginalize(self, data: LightCurveData, prot_enabled: bool) -> None:
        has_theta = data.theta is not None
        has_time = data.time is not None

        if has_time and not has_theta and not prot_enabled:
            raise ValueError("When LightCurveData has time (without theta), pass prot as a value/list/prior to marginalize.")
        if has_theta and not has_time and prot_enabled:
            raise ValueError("When LightCurveData has theta (without time), prot must be None/NaN (or omitted).")

    def _resolve_axis(
        self,
        value,
        name: str,
        *,
        dtype
    ) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a scalar, non-empty 1D list-like, or prior dict.")

        if value is None:
            return np.array([np.nan], dtype=dtype), np.array([0.0], dtype=float)
        elif isinstance(value, dict):
            parsed = self._parse_discrete_prior(value, name)
            values = np.asarray(parsed["values"], dtype=dtype)
            logw = np.asarray(parsed["log_weights"], dtype=float)
        else:
            values_raw = np.asarray(value, dtype=dtype)
            if values_raw.ndim == 0:
                values = np.asarray([values_raw.item()], dtype=dtype)
                logw = np.asarray([0.0], dtype=float)
            elif values_raw.ndim == 1 and values_raw.size > 0:
                values = values_raw.astype(dtype)
                logw = np.full(values.size, -np.log(values.size), dtype=float)
            else:
                raise ValueError(f"{name} must be a scalar, non-empty 1D list-like, or prior dict.")

        return values, logw


    def _finalize_mixture(
        self,
        mixture_logw,
        mixture_mu,
        mixture_cov,
        mixture_meta,
        *,
        axes: dict,
    ):
        n_wl = len(mixture_logw)
        if n_wl == 0:
            raise RuntimeError("No mixture components were evaluated.")

        n_pix = np.asarray(mixture_mu[0][0]).size
        I_all_wl = np.zeros((n_wl, n_pix))
        I_cov_all_wl = np.zeros((n_wl, n_pix, n_pix))
        weights_by_wl = []

        for i_wl in range(n_wl):
            logw = np.asarray(mixture_logw[i_wl], dtype=float)
            m = np.max(logw)
            w = np.exp(logw - m)
            w_sum = np.sum(w)
            if w_sum <= 0.0:
                raise RuntimeError(f"Wavelength {i_wl}: all mixture component weights are zero.")
            w = w / w_sum
            weights_by_wl.append(w)

            mu_components = np.asarray(mixture_mu[i_wl], dtype=float)
            cov_components = np.asarray(mixture_cov[i_wl], dtype=float)

            mu_mix = np.tensordot(w, mu_components, axes=(0, 0))
            second = np.tensordot(w, cov_components, axes=(0, 0))
            second += np.einsum("k,ki,kj->ij", w, mu_components, mu_components)
            cov_mix = second - np.outer(mu_mix, mu_mix)
            cov_mix = 0.5 * (cov_mix + cov_mix.T)

            I_all_wl[i_wl] = mu_mix
            I_cov_all_wl[i_wl] = cov_mix

        n_components_set = {w.shape[0] for w in weights_by_wl}
        if len(n_components_set) != 1:
            raise RuntimeError("Inconsistent number of mixture components across wavelengths.")
        w_all = np.column_stack(weights_by_wl)

        self.mixture_ = {
            "axes": axes,
            "weights": [np.asarray(w, dtype=float) for w in weights_by_wl],
            "log_weights": [np.asarray(x, dtype=float) for x in mixture_logw],
            "components": mixture_meta,
            "mu_components": mixture_mu,
            "cov_components": mixture_cov,
        }

        return w_all, I_all_wl, I_cov_all_wl

    def _resolve_u_for_wavelength(
        self,
        i_wl: int | None = None,
        n_wl: int | None = None,
    ) -> np.ndarray | None:
        u = getattr(self, "u", None)
        if u is None:
            return None

        u_arr = np.asarray(u, dtype=float)
        if u_arr.ndim == 0:
            return u_arr[np.newaxis]
        if u_arr.ndim == 1:
            return u_arr
        if u_arr.ndim != 2:
            raise ValueError("u must be None, a scalar, a 1D array, or a 2D array with shape (n_wl, udeg).")

        if n_wl is not None and u_arr.shape[0] != n_wl:
            raise ValueError(
                f"Wavelength-dependent u must have shape (n_wl, udeg); expected first dimension {n_wl}, got {u_arr.shape[0]}."
            )

        if i_wl is None:
            return u_arr[0]
        return u_arr[i_wl]

    def _set_map_limb_darkening_for_wavelength(self, map_obj: Map, i_wl: int, n_wl: int) -> None:
        u_wl = self._resolve_u_for_wavelength(i_wl=i_wl, n_wl=n_wl)
        if u_wl is None:
            return

        current_u = np.asarray([map_obj.map[i] for i in range(1, int(map_obj.map.udeg) + 1)], dtype=float)
        u_wl_arr = np.asarray(u_wl, dtype=float).ravel()
        if current_u.shape == u_wl_arr.shape and np.allclose(current_u, u_wl_arr):
            return

        _set_starry_limb_darkening_coeffs(map_obj.map, u_wl_arr)
        map_obj.design_matrix_ = None
        map_obj.intensity_design_matrix_ = None

    def _make_map(self, ydeg: int, inc: float | None) -> Map:
        raise NotImplementedError("Subclasses must implement _make_map(...).")
