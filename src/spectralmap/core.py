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
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import starry

from spectralmap.cluster import get_best_polygon
starry.config.lazy = False  # disable lazy evaluation
starry.config.quiet = True  # disable warnings
from spectralmap.bayesian_linalg import gaussian_linear_posterior, optimize_hyperparameters
from spectralmap.utilities import solid_angle_weights
from spectralmap.plotting import COLOR_LIST

# Numerical tolerances used in posterior linear algebra.
SVD_NULLSPACE_TOL = 1e-8
ALPHA_EFF_FLOOR = 1e-12
SURFACE_LON_EXTENT = (-180.0, 180.0)
SURFACE_LAT_EXTENT = (-90.0, 90.0)
SURFACE_LON_TICKS = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
SURFACE_LAT_TICKS = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])

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
        Flux values are preserved by default to match the pre-refactor API.
        Set ``normalize=True`` to divide each wavelength channel by its mean.
    flux_err
        Optional flux uncertainties with the same shape as ``flux``.
    wl
        Optional wavelength grid with length ``n_wavelength``.
    normalize
        Whether to divide flux and flux errors by the per-channel mean.
    """

    theta: np.ndarray | None = None
    time: np.ndarray | None = None
    flux: np.ndarray | None = None
    flux_err: np.ndarray | None = None
    wl: np.ndarray | None = None
    normalize: bool = False

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

        flux_mean = np.nanmean(self.flux, axis=1)
        self.amplitude = flux_mean if self.normalize else np.ones_like(flux_mean)
        if self.normalize:
            self.flux = self.flux / self.amplitude[:, np.newaxis]

        if self.flux_err is not None:
            self.flux_err = np.asarray(self.flux_err)
            if self.flux_err.ndim == 1:
                self.flux_err = self.flux_err[np.newaxis, :]
            elif self.flux_err.ndim != 2:
                raise ValueError("flux_err must be a 1D or 2D array")

            if self.flux_err.shape != self.flux.shape:
                raise ValueError("flux_err must have the same shape as flux")

            if self.normalize:
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
        self.mask_2d = None
        self.mask_1d = None
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

    def _intensity_design_matrix_impl(self, lat_safe: np.ndarray, lon_safe: np.ndarray) -> np.ndarray:
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
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()
            finite_mask = np.isfinite(self.lat) & np.isfinite(self.lon)
            if self.observed_lon_range is None:
                self.mask_2d = finite_mask
            else:
                lon_mask = (self.lon > self.observed_lon_range[0]) & (self.lon < self.observed_lon_range[1])
                self.mask_2d = finite_mask & lon_mask
            self.mask_1d = self.mask_2d.ravel()
            self.projection = projection

        return self.lat, self.lon

    def intensity_design_matrix(self, projection: str = "rect") -> np.ndarray:
        """Compute intensity design matrix for given lat/lon grid."""

        if self.intensity_design_matrix_ is not None and self.projection == projection:
            return self.intensity_design_matrix_

        self.get_latlon_grid(projection=projection)
        I = self._intensity_design_matrix_impl(self.lat_flat[self.mask_1d], self.lon_flat[self.mask_1d])
        I = self._eval_to_numpy(I)
        self.intensity_design_matrix_ = I
        self.projection = projection
        return I

    def solve_posterior(
        self,
        y: np.ndarray,
        sigma_y: np.ndarray | None = None,
        theta: np.ndarray | None = None,
        lamda: float | None = None,
        prior_mean: np.ndarray | None = None,
        prior_covariance: np.ndarray | None = None,
        prior_precision: np.ndarray | float | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve the linear-map posterior for one light curve.

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
        prior_mean, prior_covariance, prior_precision
            Optional fixed Gaussian prior for the free coefficients. The first
            starry coefficient remains fixed to 1 in the current map model, so
            full-size priors are sliced to the free-coefficient block.
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
        fixed_prior_enabled = prior_covariance is not None or prior_precision is not None
        if fixed_prior_enabled and lamda_enabled:
            raise ValueError("Fixed dense coefficient priors cannot be combined with lamda.")

        default_mu0 = np.r_[0.0 if self.eclipse_depth is None else float(self.eclipse_depth), np.zeros(A_fit.shape[1] - 1)]
        if fixed_prior_enabled:
            if prior_mean is None:
                mu0_fit = default_mu0
            else:
                prior_mean = np.asarray(prior_mean, dtype=float).reshape(-1)
                if prior_mean.shape == (A_full.shape[1],):
                    mu0_fit = prior_mean[1:]
                elif prior_mean.shape == (A_fit.shape[1],):
                    mu0_fit = prior_mean
                else:
                    raise ValueError(
                        "prior_mean must have length equal to either the full coefficient "
                        f"count ({A_full.shape[1]}) or free coefficient count ({A_fit.shape[1]})."
                    )

            prior_cov_fit = None
            if prior_covariance is not None:
                prior_covariance = np.asarray(prior_covariance, dtype=float)
                if prior_covariance.shape == (A_full.shape[1], A_full.shape[1]):
                    prior_cov_fit = prior_covariance[1:, 1:]
                elif prior_covariance.shape == (A_fit.shape[1], A_fit.shape[1]):
                    prior_cov_fit = prior_covariance
                else:
                    raise ValueError(
                        "prior_covariance must have shape equal to either the full "
                        f"coefficient block ({A_full.shape[1]}, {A_full.shape[1]}) "
                        f"or free block ({A_fit.shape[1]}, {A_fit.shape[1]})."
                    )

            prior_prec_fit = None
            if prior_precision is not None:
                prior_precision = np.asarray(prior_precision, dtype=float)
                if prior_precision.ndim == 0:
                    prior_prec_fit = float(prior_precision)
                elif prior_precision.shape == (A_full.shape[1], A_full.shape[1]):
                    prior_prec_fit = prior_precision[1:, 1:]
                elif prior_precision.shape == (A_fit.shape[1], A_fit.shape[1]):
                    prior_prec_fit = prior_precision
                elif prior_precision.shape == (A_fit.shape[1],):
                    prior_prec_fit = prior_precision
                elif prior_precision.shape == (A_full.shape[1],):
                    prior_prec_fit = prior_precision[1:]
                else:
                    raise ValueError(
                        "prior_precision must be scalar, full-size, or free-size."
                    )

            result = gaussian_linear_posterior(
                A_fit,
                y - A_full[:, 0],
                sigma_y=sigma_y,
                prior_mean=mu0_fit,
                prior_covariance=prior_cov_fit,
                prior_precision=prior_prec_fit,
            )
            mu_fit = result["posterior_mean"]
            cov_fit = result["posterior_cov"]
            mu = np.zeros(A_full.shape[1])
            mu[0] = 1.0
            mu[1:] = mu_fit
            cov = np.zeros((A_full.shape[1], A_full.shape[1]))
            cov[1:, 1:] = cov_fit
            self.hyper = {
                "alpha": None,
                "beta": None,
                "lamda": None,
                "log_ev": float(result["log_evidence"]),
                "log_ev_marginalized": float(result["log_evidence"]),
                "fixed_prior": True,
            }
            self.mu = mu
            self.cov = cov
            self.flux = y
            self.flux_err = sigma_y
            self.theta = theta
            return mu, cov, float(result["log_evidence"])

        U, s, Vt = np.linalg.svd(A_fit, full_matrices=False)
        U = U * s[np.newaxis, :]
        null_space = s <= SVD_NULLSPACE_TOL
        img_U = U[:, ~null_space]
        nul_Vt = Vt[null_space, :]
        img_Vt = Vt[~null_space, :]

        y_fit = y - A_full[:, 0]
        if lamda_enabled:

            projection = self.default_projection if self.projection is None else self.projection
            I_full = self.intensity_design_matrix(projection=projection)

            I_fit = I_full[:, 1:]
            I_constraint = I_fit @ img_Vt.T
            w_pix = solid_angle_weights(self.lat_flat[self.mask_1d], self.lon_flat[self.mask_1d])
        else:
            I_constraint = None
            w_pix = None

        mu0_img = img_Vt @ default_mu0          # because img_Vt is (r, k) so img_Vt @ mu0 gives (r,)

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

    @staticmethod
    def _rng(random_state=None):
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    @staticmethod
    def _finite_yerr(yerr):
        if yerr is None:
            return None
        yerr_arr = np.asarray(yerr, dtype=float)
        if yerr_arr.ndim == 0 and not np.isfinite(float(yerr_arr)):
            return None
        if yerr_arr.size > 0 and not np.any(np.isfinite(yerr_arr)):
            return None
        return yerr

    @staticmethod
    def _sample_coefficients(mu, cov, n_samples: int, random_state=None):
        if n_samples <= 0:
            return np.empty((0, np.asarray(mu).size), dtype=float)
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        rng = Map._rng(random_state)
        return rng.multivariate_normal(np.asarray(mu, dtype=float), cov, size=int(n_samples))

    @staticmethod
    def _setup_lightcurve_axes(ax=None, residual_ax=None, plot_residuals: bool = False):
        if not plot_residuals:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure
            return fig, ax, None

        if residual_ax is None and ax is not None and not hasattr(ax, "plot"):
            axes = np.ravel(ax)
            if axes.size != 2:
                raise ValueError("ax must contain exactly two axes when plot_residuals=True.")
            ax, residual_ax = axes

        if ax is None and residual_ax is None:
            fig, (ax, residual_ax) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )
            return fig, ax, residual_ax

        if ax is None or residual_ax is None:
            raise ValueError("Pass both ax and residual_ax when supplying axes for plot_residuals=True.")
        if ax.figure is not residual_ax.figure:
            raise ValueError("ax and residual_ax must belong to the same figure.")
        return ax.figure, ax, residual_ax

    @staticmethod
    def _plot_residual_panel(
        residual_ax,
        x,
        residuals,
        yerr=None,
        x_label: str = "Phase Angle",
        residual_kwargs: dict | None = None,
    ):
        residual_style = {
            "fmt": ".",
            "label": "Residuals",
            "color": "C0",
            "alpha": 0.5,
            "markersize": 1.0,
        }
        if residual_kwargs is not None:
            residual_style.update(residual_kwargs)
        residual_ax.errorbar(x, residuals, yerr=yerr, **residual_style)
        residual_ax.axhline(0.0, color="0.3", linestyle="--", linewidth=0.8, zorder=1)
        residual_ax.set_xlabel(x_label)
        residual_ax.set_ylabel("Residual")

    def plot_lightcurve(
        self,
        n_samples: int = 0,
        *,
        ax=None,
        residual_ax=None,
        random_state=None,
        plot_mean: bool = True,
        plot_residuals: bool = False,
        data_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        residual_kwargs: dict | None = None,
    ):
        """Plot the observed light curve with optional posterior samples.

        Parameters
        ----------
        n_samples
            Number of coefficient posterior samples to draw and plot. The
            default of 0 preserves the previous behavior of plotting only the
            posterior-mean model.
        ax
            Optional Matplotlib axes.
        residual_ax
            Optional Matplotlib axes for residuals when ``plot_residuals=True``.
        random_state
            Seed or ``numpy.random.Generator`` used for posterior samples.
        plot_mean
            Whether to plot the posterior-mean model curve.
        plot_residuals
            Whether to add a lower panel showing ``data - posterior mean``.
        data_kwargs, model_kwargs, sample_kwargs, residual_kwargs
            Optional keyword overrides passed to Matplotlib.
        """
        if self.flux is None or self.theta is None:
            print("No light curve data to plot. Run solve_posterior() first.")
            return
        if self.mu is None or self.cov is None:
            raise RuntimeError("No posterior to plot. Run solve_posterior() first.")

        fig, ax, residual_ax = self._setup_lightcurve_axes(
            ax=ax,
            residual_ax=residual_ax,
            plot_residuals=plot_residuals,
        )

        data_style = {"fmt": ".", "label": "Data", "color": "C0", "alpha": 0.5, "markersize": 1.0}
        if data_kwargs is not None:
            data_style.update(data_kwargs)
        yerr = self._finite_yerr(self.flux_err)
        ax.errorbar(self.theta, self.flux, yerr=yerr, **data_style)

        A = self.design_matrix_ if self.design_matrix_ is not None else self.design_matrix(self.theta)
        mean_model = A @ self.mu

        if n_samples:
            sample_style = {"color": "C1", "alpha": 0.15, "lw": 1.0, "zorder": 5}
            if sample_kwargs is not None:
                sample_style.update(sample_kwargs)
            label = sample_style.pop("label", "Posterior samples")
            for i, coeffs in enumerate(self._sample_coefficients(self.mu, self.cov, n_samples, random_state=random_state)):
                ax.plot(self.theta, A @ coeffs, label=label if i == 0 else None, **sample_style)

        if plot_mean:
            model_style = {"label": "Posterior mean", "color": "C1", "zorder": 10}
            if model_kwargs is not None:
                model_style.update(model_kwargs)
            ax.plot(self.theta, mean_model, **model_style)

        if plot_residuals:
            self._plot_residual_panel(
                residual_ax,
                self.theta,
                np.asarray(self.flux, dtype=float) - mean_model,
                yerr=yerr,
                residual_kwargs=residual_kwargs,
            )
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Phase Angle")

        ax.set_ylabel("Flux")
        ax.legend()
        return (fig, (ax, residual_ax)) if plot_residuals else (fig, ax)



class Maps:
    """Shared multi-wavelength model-marginalization utilities."""

    def __init__(
        self,
        map_res: int = 30,
        observed_lon_range: np.ndarray | None = None,
        projection: str = "rect",
        verbose=True,
    ):
        self.map_res = map_res
        self.verbose = verbose
        self.observed_lon_range = observed_lon_range
        self.projection = projection
        self.data = None
        self.pri = None
        self.sec = None
        self.eclipse_depth = None
        self.mask_2d = None
        self.mask_1d = None
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
        mixtureI_I_mu = [[] for _ in range(n_wl)]
        mixtureI_I_cov = [[] for _ in range(n_wl)]
        mixture_coeff_mu = [[] for _ in range(n_wl)]
        mixture_coeff_cov = [[] for _ in range(n_wl)]
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
                                I_use = map_obj.intensity_design_matrix(projection=self.projection)

                                if not geometry_initialized:
                                    self.mask_2d = map_obj.mask_2d
                                    self.mask_1d = map_obj.mask_1d
                                    self.lat = map_obj.lat
                                    self.lon = map_obj.lon
                                    self.lat_flat = map_obj.lat_flat
                                    self.lon_flat = map_obj.lon_flat
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
                                mixtureI_I_mu[i_wl].append(I_mu)
                                mixtureI_I_cov[i_wl].append(I_cov)
                                mixture_coeff_mu[i_wl].append(mu.copy())
                                mixture_coeff_cov[i_wl].append(cov.copy())
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
            mixtureI_I_mu,
            mixtureI_I_cov,
            mixture_meta,
            mixture_coeff_mu,
            mixture_coeff_cov,
            axes={"inc": inc, "ydeg": ydeg, "prot": prot, "lamda": lamda},
        )

        spatial_spectra = spatial_intensity * data.amplitude[:, None] * np.pi
        spatial_spectra_cov = spatial_intensity_cov * (np.pi * data.amplitude[:, None, None])**2

        self.mixture_weights = mixture_weights
        self.spatial_intensity = spatial_intensity
        self.spatial_intensity_cov = spatial_intensity_cov
        self.spatial_spectra = spatial_spectra
        self.spatial_spectra_cov = spatial_spectra_cov
        return mixture_weights, spatial_spectra, spatial_spectra_cov


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

    def plot_pc_projection(self, upsample=4, extrapolate=False, ax=None):
        """Plot the PC1/PC2 surface projection.

        The implementation lives in :mod:`spectralmap.plotting`; this method is
        only a convenience wrapper so users can call it directly from a
        ``Maps`` instance.
        """
        from spectralmap.plotting import plot_pc_projection

        return plot_pc_projection(self, upsample=upsample, extrapolate=extrapolate, ax=ax)


    def plot_labels(
        self,
        ax=None,
        show_grid=True,
        hide_ticks=True,
        extrapolate=False,
        colorbar=True,
        cax=None,
    ):
        """Plot clustered region labels on the surface map."""
        from spectralmap.plotting import plot_labels

        return plot_labels(
            self,
            ax=ax,
            show_grid=show_grid,
            hide_ticks=hide_ticks,
            extrapolate=extrapolate,
            colorbar=colorbar,
            cax=cax,
        )

    def plot_spectra(self, axes=None, **kwargs):
        """Plot recovered regional spectra after find_clusters()."""
        from spectralmap.plotting import plot_spectra

        return plot_spectra(self, axes=axes, **kwargs)

    def show(
        self,
        i_wl: int = 0,
        n_samples: int = 0,
        *,
        random_state=None,
        projection: str | None = None,
        cmap: str = "inferno",
        colorbar: bool = True,
        figsize: tuple[float, float] | None = None,
        share_axes: bool = True,
        **imshow_kwargs,
    ):
        """Show the marginalized surface map mean or posterior samples.

        Parameters
        ----------
        i_wl
            Wavelength index to show.
        n_samples
            If 0, plot the marginalized posterior mean map. If positive, draw
            and plot that many map-intensity samples.
        random_state
            Seed or ``numpy.random.Generator`` used for posterior samples.
        projection
            Kept for API compatibility. The grid projection is the projection
            used during ``marginalize``.
        cmap, colorbar, figsize, share_axes, imshow_kwargs
            Matplotlib display controls.
        """
        if not hasattr(self, "spatial_intensity") or not hasattr(self, "spatial_intensity_cov"):
            raise RuntimeError("No marginalized surface posterior is available. Run marginalize() first.")

        if n_samples < 0:
            raise ValueError("n_samples must be >= 0.")

        if n_samples == 0:
            maps_to_plot = [self.spatial_intensity[i_wl]]
        else:
            rng = Map._rng(random_state)
            cov = np.asarray(self.spatial_intensity_cov[i_wl], dtype=float)
            cov = 0.5 * (cov + cov.T)
            samples = rng.multivariate_normal(np.asarray(self.spatial_intensity[i_wl], dtype=float), cov, size=int(n_samples))
            maps_to_plot = list(np.atleast_2d(samples))

        n_plot = len(maps_to_plot)
        grids_to_plot = [self._surface_map_grid(values) for values in maps_to_plot]
        if "norm" not in imshow_kwargs:
            finite_grids = [grid[np.isfinite(grid)].ravel() for grid in grids_to_plot]
            finite_grids = [values for values in finite_grids if values.size]
            if finite_grids:
                finite_values = np.concatenate(finite_grids)
                if "vmin" not in imshow_kwargs:
                    imshow_kwargs["vmin"] = float(np.nanmin(finite_values))
                if "vmax" not in imshow_kwargs:
                    imshow_kwargs["vmax"] = float(np.nanmax(finite_values))

        n_rows, n_cols = self._show_subplot_layout(n_plot)
        extent = imshow_kwargs.pop("extent", self._surface_map_extent())
        xticks, yticks = self._surface_axis_ticks(extent)
        if figsize is None:
            panel_width = 3.5
            panel_height = self._show_panel_height(extent, panel_width=panel_width)
            figsize = (min(panel_width * n_cols, 14.0), panel_height * n_rows)
        imshow_kwargs.setdefault("aspect", "auto")
        imshow_kwargs.setdefault("interpolation", "nearest")
        fig, axes_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            squeeze=False,
            sharex=share_axes,
            sharey=share_axes,
            gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        )
        fig.subplots_adjust(wspace=0.0, hspace=0.0)
        axes = axes_grid.ravel()

        images = []
        for ax, grid in zip(axes[:n_plot], grids_to_plot):
            im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, **imshow_kwargs)
            self._set_surface_axis_ticks(ax, xticks, yticks)
            ax.set_xlabel("Longitude (deg)")
            ax.set_ylabel("Latitude (deg)")
            images.append(im)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        if share_axes and n_plot > 1:
            xlim = axes[0].get_xlim()
            ylim = axes[0].get_ylim()
            for ax in axes[:n_plot]:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if ax.get_subplotspec().is_first_col():
                    ax.set_ylabel("Latitude (deg)")
                    ax.tick_params(axis="y", labelleft=True)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
                if ax.get_subplotspec().is_last_row():
                    ax.set_xlabel("Longitude (deg)")
                    ax.tick_params(axis="x", labelbottom=True)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)

        if colorbar and images:
            colorbar_axes = list(axes[:n_plot]) if n_plot > 1 else axes[0]
            fig.colorbar(images[0], ax=colorbar_axes, label="Intensity")

        return fig, axes if n_plot > 1 else axes[0]


    def plot_lightcurve(
        self,
        i_wl: int = 0,
        n_samples: int = 0,
        *,
        ax=None,
        residual_ax=None,
        random_state=None,
        plot_mean: bool = True,
        plot_data: bool = True,
        plot_residuals: bool = False,
        data_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        residual_kwargs: dict | None = None,
    ):
        """Plot a marginalized light-curve posterior after ``marginalize()``.

        The posterior is the full discrete/continuous mixture from
        ``marginalize``: samples first choose a model component according to
        its posterior weight, then draw coefficients from that component's
        Gaussian posterior. Set ``plot_residuals=True`` to add a lower panel
        showing ``data - posterior mean``.
        """
        if self.data is None or self.mixture_ is None:
            raise RuntimeError("No marginalized posterior is available. Run marginalize() first.")

        components_by_wl = self.mixture_.get("components")
        weights_by_wl = self.mixture_.get("weights")
        coeff_mu_by_wl = self.mixture_.get("coeff_mu_components")
        coeff_cov_by_wl = self.mixture_.get("coeff_cov_components")
        if (
            components_by_wl is None
            or weights_by_wl is None
            or coeff_mu_by_wl is None
            or coeff_cov_by_wl is None
        ):
            raise RuntimeError(
                "The marginalized posterior does not contain coefficient-space components. "
                "Run marginalize() with the current spectralmap version."
            )
        if i_wl >= len(components_by_wl):
            raise IndexError(f"i_wl={i_wl} is out of range for {len(components_by_wl)} wavelength channels.")

        components = components_by_wl[i_wl]
        weights = np.asarray(weights_by_wl[i_wl], dtype=float)
        coeff_mu = coeff_mu_by_wl[i_wl]
        coeff_cov = coeff_cov_by_wl[i_wl]
        if not components:
            raise RuntimeError("No mixture components are available for the selected wavelength.")
        if len(components) != weights.size or len(coeff_mu) != weights.size or len(coeff_cov) != weights.size:
            raise RuntimeError("Mixture component metadata, weights, and coefficient posteriors are inconsistent.")

        x = self.data.theta if self.data.theta is not None else self.data.time
        x_label = "Phase Angle" if self.data.theta is not None else "Time"
        if x is None:
            raise RuntimeError("No x-axis values are available on the marginalized LightCurveData.")

        fig, ax, residual_ax = Map._setup_lightcurve_axes(
            ax=ax,
            residual_ax=residual_ax,
            plot_residuals=plot_residuals,
        )
        yerr = None if self.data.flux_err is None else Map._finite_yerr(self.data.flux_err[i_wl])

        if plot_data:
            data_style = {"fmt": ".", "label": "Data", "color": "C0", "alpha": 0.5, "markersize": 1.0}
            if data_kwargs is not None:
                data_style.update(data_kwargs)
            ax.errorbar(x, self.data.flux[i_wl], yerr=yerr, **data_style)

        design_cache = {}

        def component_design(i_component):
            component = components[i_component]
            key = (
                int(component["ydeg"]),
                None if component.get("inc", None) is None else float(component["inc"]),
                None if component.get("prot", None) is None else float(component["prot"]),
            )
            if key not in design_cache:
                design_cache[key] = self._component_design_matrix(component, i_wl=i_wl)[0]
            return design_cache[key]

        if n_samples:
            rng = Map._rng(random_state)
            component_indices = rng.choice(np.arange(weights.size), size=int(n_samples), p=weights)
            sample_style = {"color": "C1", "alpha": 0.15, "lw": 1.0, "zorder": 5}
            if sample_kwargs is not None:
                sample_style.update(sample_kwargs)
            label = sample_style.pop("label", "Posterior samples")
            for i, i_component in enumerate(component_indices):
                coeffs = Map._sample_coefficients(
                    coeff_mu[i_component],
                    coeff_cov[i_component],
                    1,
                    random_state=rng,
                )[0]
                ax.plot(x, component_design(i_component) @ coeffs, label=label if i == 0 else None, **sample_style)

        if plot_mean or plot_residuals:
            mean_model = np.zeros_like(self.data.flux[i_wl], dtype=float)
            for i_component, weight in enumerate(weights):
                mean_model += float(weight) * (component_design(i_component) @ np.asarray(coeff_mu[i_component], dtype=float))

        if plot_mean:
            model_style = {"label": "Posterior mean", "color": "C1", "zorder": 10}
            if model_kwargs is not None:
                model_style.update(model_kwargs)
            ax.plot(x, mean_model, **model_style)

        if plot_residuals:
            Map._plot_residual_panel(
                residual_ax,
                x,
                np.asarray(self.data.flux[i_wl], dtype=float) - mean_model,
                yerr=yerr,
                x_label=x_label,
                residual_kwargs=residual_kwargs,
            )
            ax.set_xlabel("")
        else:
            ax.set_xlabel(x_label)

        ax.set_ylabel("Flux")
        ax.legend()
        return (fig, (ax, residual_ax)) if plot_residuals else (fig, ax)

    def plot_model_weights(
        self,
        x_axis: str,
        y_axis: str,
        i_wl: int = 0,
        ax=None,
        cmap: str = "viridis",
        colorbar: bool = True,
        annotate: bool = False,
        figsize: tuple[float, float] | None = None,
        log_scale: bool = False,
        **imshow_kwargs,
    ):
        """Plot marginalized model weights as a heatmap over two axes."""
        grid, x_values, y_values = self.model_weight_grid(x_axis=x_axis, y_axis=y_axis, i_wl=i_wl)

        if ax is None:
            if figsize is None:
                figsize = self._model_weight_figsize(len(x_values), len(y_values), colorbar=colorbar)
            fig, ax = plt.subplots(figsize=figsize, dpi=150)
        else:
            fig = ax.figure

        aspect = imshow_kwargs.pop("aspect", "equal")
        plot_grid = grid
        norm = imshow_kwargs.pop("norm", None)
        if log_scale:
            if norm is not None:
                raise ValueError("Pass either log_scale=True or norm, not both.")
            positive = grid[np.isfinite(grid) & (grid > 0)]
            if positive.size == 0:
                raise ValueError("Cannot use log_scale=True when all model weights are zero.")
            vmin = imshow_kwargs.pop("vmin", None)
            vmax = imshow_kwargs.pop("vmax", None)
            vmin = float(np.nanmin(positive)) if vmin is None else float(vmin)
            vmax = float(np.nanmax(positive)) if vmax is None else float(vmax)
            if vmin <= 0 or vmax <= 0:
                raise ValueError("Log-scaled model weights require positive vmin and vmax.")
            norm = LogNorm(vmin=vmin, vmax=vmax)
            plot_grid = np.ma.masked_less_equal(grid, 0)

        im = ax.imshow(plot_grid, origin="lower", aspect=aspect, cmap=cmap, norm=norm, **imshow_kwargs)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        x_sig = 2 if x_axis == "lamda" else None
        y_sig = 2 if y_axis == "lamda" else None
        ax.set_xticklabels([self._format_axis_value(v, significant_digits=x_sig) for v in x_values])
        ax.set_yticklabels([self._format_axis_value(v, significant_digits=y_sig) for v in y_values])

        if annotate:
            for j in range(grid.shape[0]):
                for i in range(grid.shape[1]):
                    ax.text(i, j, f"{grid[j, i]:.2g}", ha="center", va="center", color="white")

        if colorbar:
            fig.colorbar(im, ax=ax, label="Posterior weight")

        return fig, ax, im



    def model_weight_grid(self, x_axis: str, y_axis: str, i_wl: int = 0):
        """Aggregate model weights onto two selected marginalization axes.

        Parameters
        ----------
        x_axis, y_axis
            Names of two axes stored in ``self.mixture_["components"]`` such as
            ``"ydeg"``, ``"inc"``, ``"prot"``, or ``"lamda"``.
        i_wl
            Wavelength index whose model weights should be visualized.

        Returns
        -------
        grid, x_values, y_values
            ``grid[j, i]`` is the summed posterior weight for
            ``y_axis == y_values[j]`` and ``x_axis == x_values[i]``.
        """
        if self.mixture_ is None:
            raise RuntimeError("No marginalized model weights are available. Run marginalize() first.")
        if x_axis == y_axis:
            raise ValueError("x_axis and y_axis must be different.")

        components_by_wl = self.mixture_.get("components")
        weights_by_wl = self.mixture_.get("weights")
        if components_by_wl is None or weights_by_wl is None:
            raise RuntimeError("mixture_ does not contain component metadata and weights.")

        if i_wl < 0 or i_wl >= len(components_by_wl):
            raise IndexError(f"i_wl={i_wl} is out of range for {len(components_by_wl)} wavelength channels.")

        components = components_by_wl[i_wl]
        weights = np.asarray(weights_by_wl[i_wl], dtype=float)
        if len(components) != weights.size:
            raise RuntimeError("Component metadata and weight arrays have inconsistent lengths.")
        if not components:
            raise RuntimeError("No mixture components are available for the selected wavelength.")

        valid_axes = set(components[0].keys())
        missing = [axis for axis in (x_axis, y_axis) if axis not in valid_axes]
        if missing:
            raise ValueError(f"Unknown model axis/axes {missing}. Available axes are {sorted(valid_axes)}.")

        x_component_values = [self._canonical_axis_value(component[x_axis]) for component in components]
        y_component_values = [self._canonical_axis_value(component[y_axis]) for component in components]
        x_values = self._sorted_axis_values(x_component_values)
        y_values = self._sorted_axis_values(y_component_values)

        x_index = {value: i for i, value in enumerate(x_values)}
        y_index = {value: i for i, value in enumerate(y_values)}
        grid = np.zeros((len(y_values), len(x_values)), dtype=float)

        for weight, x_value, y_value in zip(weights, x_component_values, y_component_values):
            if not np.isfinite(weight):
                continue
            grid[y_index[y_value], x_index[x_value]] += float(weight)

        return grid, x_values, y_values

    def _surface_map_grid(self, values: np.ndarray) -> np.ndarray:
        if self.mask_2d is None or self.mask_1d is None:
            raise RuntimeError("No map grid is available. Run marginalize() first.")

        values = np.asarray(values, dtype=float)
        if values.size != int(np.sum(self.mask_1d)):
            raise ValueError(
                f"Expected {int(np.sum(self.mask_1d))} surface values for the observed grid, got {values.size}."
            )

        flat = np.full(self.mask_1d.shape, np.nan, dtype=float)
        flat[self.mask_1d] = values
        return flat.reshape(self.mask_2d.shape)

    def _surface_map_extent(self) -> tuple[float, float, float, float]:
        if self.lat is None or self.lon is None:
            raise RuntimeError("No latitude/longitude grid is available. Run marginalize() first.")

        return (*SURFACE_LON_EXTENT, *SURFACE_LAT_EXTENT)

    @staticmethod
    def _surface_axis_ticks(extent: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        lon_min, lon_max, lat_min, lat_max = extent
        xticks = SURFACE_LON_TICKS[
            (SURFACE_LON_TICKS >= min(lon_min, lon_max))
            & (SURFACE_LON_TICKS <= max(lon_min, lon_max))
        ]
        yticks = SURFACE_LAT_TICKS[
            (SURFACE_LAT_TICKS >= min(lat_min, lat_max))
            & (SURFACE_LAT_TICKS <= max(lat_min, lat_max))
        ]
        return xticks, yticks

    @staticmethod
    def _set_surface_axis_ticks(ax, xticks: np.ndarray, yticks: np.ndarray) -> None:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([Maps._format_geo_tick(tick) for tick in xticks])
        ax.set_yticklabels([Maps._format_geo_tick(tick) for tick in yticks])

    @staticmethod
    def _format_geo_tick(value: float) -> str:
        if np.isclose(value, np.round(value)):
            return str(int(np.round(value)))
        return f"{value:g}"

    @staticmethod
    def _show_panel_height(
        extent: tuple[float, float, float, float],
        *,
        panel_width: float,
    ) -> float:
        lon_min, lon_max, lat_min, lat_max = extent
        lon_span = abs(lon_max - lon_min)
        lat_span = abs(lat_max - lat_min)
        if lon_span <= 0.0 or lat_span <= 0.0:
            return 2.0
        return max(1.8, min(3.2, panel_width * lat_span / lon_span))

    @staticmethod
    def _show_subplot_layout(n_plot: int) -> tuple[int, int]:
        if n_plot <= 0:
            raise ValueError("n_plot must be positive.")
        if n_plot == 1:
            return 1, 1
        n_cols = min(4, int(np.ceil(np.sqrt(n_plot))))
        n_rows = int(np.ceil(n_plot / n_cols))
        return n_rows, n_cols

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
        mixtureI_I_mu,
        mixtureI_I_cov,
        mixture_meta,
        mixture_coeff_mu=None,
        mixture_coeff_cov=None,
        *,
        axes: dict,
    ):
        n_wl = len(mixture_logw)
        if n_wl == 0:
            raise RuntimeError("No mixture components were evaluated.")

        n_pix = np.asarray(mixtureI_I_mu[0][0]).size
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

            mu_components = np.asarray(mixtureI_I_mu[i_wl], dtype=float)
            cov_components = np.asarray(mixtureI_I_cov[i_wl], dtype=float)

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
            "mu_components": mixtureI_I_mu,
            "cov_components": mixtureI_I_cov,
        }
        if mixture_coeff_mu is not None and mixture_coeff_cov is not None:
            self.mixture_["coeff_mu_components"] = mixture_coeff_mu
            self.mixture_["coeff_cov_components"] = mixture_coeff_cov

        return w_all, I_all_wl, I_cov_all_wl

    @staticmethod
    def _canonical_axis_value(value):
        """Normalize mixture metadata values for grouping and display."""
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            value = arr.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, np.floating) and not np.isfinite(float(value)):
            return None
        return value

    @staticmethod
    def _sorted_axis_values(values):
        unique = []
        seen = set()
        for value in values:
            key = ("none", None) if value is None else ("value", value)
            if key in seen:
                continue
            seen.add(key)
            unique.append(value)

        non_none = [v for v in unique if v is not None]
        none_values = [v for v in unique if v is None]
        try:
            non_none = sorted(non_none, key=float)
        except (TypeError, ValueError):
            non_none = sorted(non_none, key=str)
        return non_none + none_values

    @staticmethod
    def _format_axis_value(value, significant_digits: int | None = None):
        if value is None:
            return "None"
        if isinstance(value, (float, np.floating)):
            if significant_digits is not None:
                return f"{float(value):.{int(significant_digits)}g}"
            return f"{float(value):g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    @staticmethod
    def _model_weight_figsize(n_x: int, n_y: int, colorbar: bool = True):
        """Choose a readable heatmap figure size from grid dimensions."""
        cell_size = 0.6
        width = 1.8 + cell_size * max(1, n_x)
        height = 1.6 + cell_size * max(1, n_y)
        if colorbar:
            width += 0.7
        return (float(np.clip(width, 4.0, 14.0)), float(np.clip(height, 3.0, 10.0)))

    def _component_theta(self, component: dict) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No light curve data are available. Run marginalize() first.")
        prot_value = component.get("prot", None)
        if prot_value is None:
            if self.data.theta is None:
                raise RuntimeError("Component does not specify a period and data.theta is unavailable.")
            return self.data.theta
        return self._theta_from_period(self.data, float(prot_value))

    def _component_design_matrix(self, component: dict, i_wl: int) -> tuple[np.ndarray, np.ndarray]:
        map_obj = self._make_map(ydeg=int(component["ydeg"]), inc=component.get("inc", None))
        if hasattr(map_obj, "null_uncertainty"):
            map_obj.null_uncertainty = self.null_uncertainty
        n_wl = self.data.flux.shape[0] if self.data is not None else None
        if n_wl is not None:
            self._set_map_limb_darkening_for_wavelength(map_obj, i_wl=i_wl, n_wl=n_wl)
        theta = self._component_theta(component)
        return map_obj.design_matrix(theta), theta

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
