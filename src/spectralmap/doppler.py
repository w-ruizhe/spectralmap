"""Doppler imaging model wrappers."""

from __future__ import annotations

import numpy as np
import starry

from spectralmap.bayesian_linalg import FLOOR, optimize_hyperparameters
from spectralmap.core import (
    Map,
    _resolve_limb_darkening,
    _set_starry_limb_darkening_coeffs,
)


class DopplerMap(Map):
    """Doppler imaging model based on ``starry.DopplerMap``.

    Notes
    -----
    This class follows the API style of ``RotMap``/``EclipseMap``, but unlike
    those photometric models it operates on a spectral time series with shape
    ``(nt, nw)``. For convenience, input arrays with shape ``(nw, nt)`` are
    detected and transposed automatically.
    """

    mode = "doppler"
    C_KMS = 299792.458

    def __init__(
        self,
        map_res: int | None = 30,
        ydeg: int | None = None,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        nt: int = 10,
        nc: int = 1,
        wav: np.ndarray | None = None,
        wav0: np.ndarray | None = None,
        wavc: float | None = None,
        inc: float = 90.0,
        veq: float = 0.0,
        vsini_max: float | None = None,
        solver: str = "bilinear",
        normalized: bool = True,
        projection: str = "rect",
    ):
        if ydeg is None:
            ydeg = 5
        if not hasattr(starry, "DopplerMap"):
            raise ImportError(
                "Your installed starry version does not provide DopplerMap. "
                "Please install a starry release with Doppler imaging support."
            )

        super().__init__(map_res=map_res, ydeg=ydeg, projection=projection)

        udeg_resolved, u_resolved = _resolve_limb_darkening(udeg=udeg, u=u)
        doppler_kwargs = {
            "ydeg": int(ydeg),
            "nt": int(nt),
            "nc": int(nc),
            "inc": float(inc),
            "veq": float(veq),
        }
        if udeg_resolved is not None:
            doppler_kwargs["udeg"] = int(udeg_resolved)
        if wav is not None:
            doppler_kwargs["wav"] = np.asarray(wav, dtype=float)
        if wav0 is not None:
            doppler_kwargs["wav0"] = np.asarray(wav0, dtype=float)
        if wavc is not None:
            doppler_kwargs["wavc"] = float(wavc)
        if vsini_max is not None:
            doppler_kwargs["vsini_max"] = float(vsini_max)

        self.map = starry.DopplerMap(**doppler_kwargs)
        _set_starry_limb_darkening_coeffs(self.map, u_resolved)

        self.udeg = udeg_resolved
        self.u = u_resolved
        self.nt = int(nt)
        self.nc = int(nc)
        self.wav = None if wav is None else np.asarray(wav, dtype=float)
        self.wav0 = None if wav0 is None else np.asarray(wav0, dtype=float)
        self.wavc = wavc
        self.inc = inc
        self.veq = veq
        self.vsini_max = vsini_max
        self.solver = solver
        self.normalized = bool(normalized)

    @property
    def n_coeff(self) -> int:
        return int(self.map.nc * self.map.Ny)

    @staticmethod
    def _coerce_flux_shape(
        flux: np.ndarray,
        nt: int,
        nw: int,
        label: str,
    ) -> np.ndarray:
        """Return flux as ``(nt, nw)``, accepting transposed input."""
        arr = np.asarray(flux, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{label} must be a 2D array with shape (nt, nw) or (nw, nt).")
        if arr.shape == (nt, nw):
            return arr
        if arr.shape == (nw, nt):
            return arr.T
        raise ValueError(
            f"{label} has shape {arr.shape}, but expected {(nt, nw)} or {(nw, nt)}."
        )

    def _design_matrix_impl(self, theta: np.ndarray) -> np.ndarray:
        return self.map.design_matrix(theta=theta, fix_spectrum=True)

    def _intensity_design_matrix_impl(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DopplerMap does not use a photometric intensity design matrix. "
            "Use map.render()/map.show() for spatial diagnostics."
        )

    def intensity_design_matrix(self, projection: str = "rect") -> np.ndarray:
        raise NotImplementedError(
            "DopplerMap does not expose the photometric intensity_design_matrix API."
        )

    def _apply_coefficients_to_map(self, coeffs: np.ndarray) -> None:
        coeffs = np.asarray(coeffs, dtype=float)
        if coeffs.size != self.n_coeff:
            raise ValueError(f"Expected {self.n_coeff} Doppler map coefficients, got {coeffs.size}.")

        coeffs = coeffs.reshape((int(self.map.nc), int(self.map.Ny)))
        if int(self.map.nc) == 1:
            self.map[:, :] = coeffs[0]
        else:
            self.map[:, :] = coeffs

    def _prepare_doppler_inputs(
        self,
        y: np.ndarray,
        sigma_y: np.ndarray | float | None,
        theta: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | float | None]:
        """Validate Doppler time-series inputs and orient flux as ``(nt, nw)``."""
        if theta is None:
            if self.theta is None:
                theta = np.linspace(0.0, 360.0, int(self.map.nt), endpoint=False)
            else:
                theta = self.theta

        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta.size != int(self.map.nt):
            raise ValueError(
                f"theta length ({theta.size}) must match DopplerMap.nt ({int(self.map.nt)})."
            )

        nt = int(self.map.nt)
        nw = int(self.map.nw)
        flux = self._coerce_flux_shape(y, nt=nt, nw=nw, label="y")

        flux_err = None
        if sigma_y is not None:
            sigma_arr = np.asarray(sigma_y, dtype=float)
            if sigma_arr.ndim == 0:
                flux_err = float(sigma_arr)
            elif sigma_arr.ndim == 1:
                if sigma_arr.size != nt:
                    raise ValueError(
                        f"sigma_y with ndim=1 must have length nt={nt}, got {sigma_arr.size}."
                    )
                flux_err = sigma_arr
            else:
                flux_err = self._coerce_flux_shape(sigma_arr, nt=nt, nw=nw, label="sigma_y")

        return theta, flux, flux_err

    @staticmethod
    def _compute_chi2(resid: np.ndarray, flux_err: np.ndarray | float | None) -> float:
        """Compute chi-square for scalar, time-only, or full uncertainty arrays."""
        if flux_err is None:
            return float(np.sum(resid ** 2))
        if np.isscalar(flux_err):
            return float(np.sum((resid / float(flux_err)) ** 2))
        if flux_err.ndim == 1:
            return float(np.sum((resid / flux_err[:, np.newaxis]) ** 2))
        return float(np.sum((resid / flux_err) ** 2))

    @staticmethod
    def _as_time_pixel_matrix(arr: np.ndarray | float | None, nt: int, npix: int, label: str):
        """Broadcast ``arr`` to ``(nt, npix)`` for surface-grid quantities."""
        if arr is None:
            return np.ones((nt, npix), dtype=float)

        out = np.asarray(arr, dtype=float)
        if out.ndim == 0:
            return np.full((nt, npix), float(out), dtype=float)
        if out.ndim == 1:
            if out.size != npix:
                raise ValueError(f"{label} with ndim=1 must have length {npix}, got {out.size}.")
            return np.tile(out[np.newaxis, :], (nt, 1))
        if out.ndim == 2:
            if out.shape == (nt, npix):
                return out
            if out.shape == (1, npix):
                return np.tile(out, (nt, 1))
        raise ValueError(f"{label} must be scalar, ({npix},), (1, {npix}), or ({nt}, {npix}).")

    @staticmethod
    def _as_time_design_cube(
        surface_design_matrix: np.ndarray,
        nt: int,
        npix: int,
    ) -> np.ndarray:
        """Broadcast a pixel-to-coefficient matrix to ``(nt, npix, n_coeff)``."""
        design = np.asarray(surface_design_matrix, dtype=float)
        if design.ndim == 2:
            if design.shape[0] != npix:
                raise ValueError(
                    "surface_design_matrix with ndim=2 must have shape "
                    f"({npix}, n_coeff), got {design.shape}."
                )
            return np.tile(design[np.newaxis, :, :], (nt, 1, 1))
        if design.ndim == 3:
            if design.shape[1] != npix:
                raise ValueError(
                    "surface_design_matrix with ndim=3 must have shape "
                    f"(nt, {npix}, n_coeff), got {design.shape}."
                )
            if design.shape[0] == nt:
                return design
            if design.shape[0] == 1:
                return np.tile(design, (nt, 1, 1))
        raise ValueError(
            "surface_design_matrix must have shape (n_pixel, n_coeff) or "
            "(nt, n_pixel, n_coeff)."
        )

    @staticmethod
    def wind_broadened_design_matrix(
        wavelength: np.ndarray,
        spectrum: np.ndarray,
        wind_velocity: np.ndarray,
        surface_design_matrix: np.ndarray,
        weights: np.ndarray | float | None = None,
        template_wavelength: np.ndarray | None = None,
        rv_kms: float = 0.0,
        fill_value: float | None = None,
        flatten: bool = True,
    ) -> np.ndarray:
        """Build the linear map from surface coefficients to Doppler-broadened spectra.

        Parameters
        ----------
        wavelength
            Output wavelength grid for each modeled spectrum.
        spectrum
            Rest-frame local template spectrum evaluated on ``template_wavelength``.
        wind_velocity
            Line-of-sight velocity at each predefined surface grid point in km/s.
            Accepted shapes are ``(n_pixel,)``, ``(1, n_pixel)``, and
            ``(nt, n_pixel)``.
        surface_design_matrix
            Matrix mapping spherical-harmonic coefficients to surface-grid
            amplitudes. Accepted shapes are ``(n_pixel, n_coeff)`` and
            ``(nt, n_pixel, n_coeff)``.
        weights
            Optional visibility, projected-area, and quadrature weights on the same
            surface grid as ``wind_velocity``.
        template_wavelength
            Wavelength grid for ``spectrum``. If omitted, ``wavelength`` is used.
        rv_kms
            Constant velocity added to all surface velocities.
        fill_value
            Value used outside the template wavelength range. By default, the edge
            template values are used.
        flatten
            If ``True``, return shape ``(nt * nw, n_coeff)``. Otherwise return
            shape ``(nt, nw, n_coeff)``.

        Notes
        -----
        For coefficient vector ``y``, the returned matrix ``A`` gives the model
        spectrum ``m = A @ y`` after summing velocity-shifted local spectra over
        the visible surface grid.
        """
        wavelength = np.asarray(wavelength, dtype=float).ravel()
        template_wavelength = wavelength if template_wavelength is None else template_wavelength
        template_wavelength = np.asarray(template_wavelength, dtype=float).ravel()
        spectrum = np.asarray(spectrum, dtype=float).ravel()

        if wavelength.size == 0:
            raise ValueError("wavelength must contain at least one point.")
        if template_wavelength.size != spectrum.size:
            raise ValueError(
                "template_wavelength and spectrum must have the same length; "
                f"got {template_wavelength.size} and {spectrum.size}."
            )
        if np.any(wavelength <= 0.0) or np.any(template_wavelength <= 0.0):
            raise ValueError("wavelength grids must be strictly positive.")

        velocity = np.asarray(wind_velocity, dtype=float)
        if velocity.ndim == 1:
            velocity = velocity[np.newaxis, :]
        if velocity.ndim != 2:
            raise ValueError("wind_velocity must have shape (n_pixel,) or (nt, n_pixel).")

        nt, npix = velocity.shape
        design_cube = DopplerMap._as_time_design_cube(surface_design_matrix, nt=nt, npix=npix)
        weights_matrix = DopplerMap._as_time_pixel_matrix(
            weights,
            nt=nt,
            npix=npix,
            label="weights",
        )

        if not np.all(np.isfinite(velocity)):
            raise ValueError("wind_velocity contains non-finite values.")
        if not np.all(np.isfinite(design_cube)):
            raise ValueError("surface_design_matrix contains non-finite values.")
        if not np.all(np.isfinite(weights_matrix)):
            raise ValueError("weights contains non-finite values.")

        sort_idx = np.argsort(template_wavelength)
        log_template_wavelength = np.log(template_wavelength[sort_idx])
        template_spectrum = spectrum[sort_idx]
        left = template_spectrum[0] if fill_value is None else float(fill_value)
        right = template_spectrum[-1] if fill_value is None else float(fill_value)

        nw = wavelength.size
        n_coeff = design_cube.shape[-1]
        out = np.empty((nt, nw, n_coeff), dtype=float)
        log_wavelength = np.log(wavelength)

        for t in range(nt):
            shifted_spectra = np.empty((nw, npix), dtype=float)
            total_velocity = velocity[t] + float(rv_kms)
            for p in range(npix):
                shifted_log_wavelength = log_wavelength - np.log1p(
                    total_velocity[p] / DopplerMap.C_KMS
                )
                shifted_spectra[:, p] = np.interp(
                    shifted_log_wavelength,
                    log_template_wavelength,
                    template_spectrum,
                    left=left,
                    right=right,
                )
            weighted_surface_basis = weights_matrix[t, :, np.newaxis] * design_cube[t]
            out[t] = shifted_spectra @ weighted_surface_basis

        if flatten:
            return out.reshape(nt * nw, n_coeff)
        return out

    @staticmethod
    def _coerce_observation_vector(
        flux: np.ndarray,
        sigma_y: np.ndarray | float | None,
        n_obs: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten observations and uncertainties for Gaussian evidence."""
        y = np.asarray(flux, dtype=float).reshape(-1)
        if y.size != n_obs:
            raise ValueError(f"flux has {y.size} elements, but design_matrix has {n_obs} rows.")

        if sigma_y is None:
            sigma = np.ones(n_obs, dtype=float)
        else:
            sigma_arr = np.asarray(sigma_y, dtype=float)
            if sigma_arr.ndim == 0:
                sigma = np.full(n_obs, float(sigma_arr), dtype=float)
            else:
                sigma = sigma_arr.reshape(-1)
                if sigma.size != n_obs:
                    raise ValueError(
                        f"sigma_y has {sigma.size} elements, but design_matrix has {n_obs} rows."
                    )
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma_y must be finite and strictly positive.")
        return y, sigma

    @staticmethod
    def _coerce_prior_precision(
        prior_precision: float | np.ndarray | None,
        prior_covariance: np.ndarray | None,
        n_coeff: int,
    ) -> tuple[np.ndarray, float]:
        """Return coefficient prior precision and log determinant of prior covariance."""
        if prior_precision is not None and prior_covariance is not None:
            raise ValueError("Provide either prior_precision or prior_covariance, not both.")

        if prior_covariance is not None:
            cov = np.asarray(prior_covariance, dtype=float)
            if cov.shape != (n_coeff, n_coeff):
                raise ValueError(
                    f"prior_covariance must have shape ({n_coeff}, {n_coeff}), got {cov.shape}."
                )
            sign, logdet_cov = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("prior_covariance must be positive definite.")
            precision = np.linalg.inv(cov)
            return precision, float(logdet_cov)

        precision_in = 1.0 if prior_precision is None else prior_precision
        precision_arr = np.asarray(precision_in, dtype=float)
        if precision_arr.ndim == 0:
            value = float(precision_arr)
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError("prior_precision must be finite and strictly positive.")
            return np.eye(n_coeff) * value, float(-n_coeff * np.log(value))
        if precision_arr.ndim == 1:
            if precision_arr.size != n_coeff:
                raise ValueError(
                    f"prior_precision vector must have length {n_coeff}, got {precision_arr.size}."
                )
            if np.any(~np.isfinite(precision_arr)) or np.any(precision_arr <= 0.0):
                raise ValueError("prior_precision entries must be finite and strictly positive.")
            return np.diag(precision_arr), float(-np.sum(np.log(precision_arr)))
        if precision_arr.shape != (n_coeff, n_coeff):
            raise ValueError(
                f"prior_precision matrix must have shape ({n_coeff}, {n_coeff}), "
                f"got {precision_arr.shape}."
            )
        sign, logdet_precision = np.linalg.slogdet(precision_arr)
        if sign <= 0:
            raise ValueError("prior_precision must be positive definite.")
        return precision_arr, float(-logdet_precision)

    @staticmethod
    def gaussian_linear_evidence(
        design_matrix: np.ndarray,
        flux: np.ndarray,
        sigma_y: np.ndarray | float | None = None,
        prior_mean: np.ndarray | None = None,
        prior_precision: float | np.ndarray | None = 1.0,
        prior_covariance: np.ndarray | None = None,
        use_svd: bool = True,
        svd_tol: float | None = None,
    ) -> dict:
        """Analytically marginalize Gaussian surface coefficients.

        This evaluates the evidence for ``d = A y + eps``,
        ``eps ~ N(0, C)``, and ``y ~ N(mu_y, Lambda_y)``. The marginalized
        likelihood is ``d ~ N(A mu_y, C + A Lambda_y A.T)``. The implementation
        uses the equivalent Woodbury form. When ``prior_precision`` is scalar,
        the design matrix is first rank-reduced with an SVD, so only the
        data-constrained coefficient subspace is factorized and null modes are
        analytically cancelled in the evidence.
        """
        A = np.asarray(design_matrix, dtype=float)
        if A.ndim == 3:
            A = A.reshape(A.shape[0] * A.shape[1], A.shape[2])
        if A.ndim != 2:
            raise ValueError("design_matrix must have shape (n_obs, n_coeff) or (nt, nw, n_coeff).")

        n_obs, n_coeff = A.shape
        y, sigma = DopplerMap._coerce_observation_vector(flux=flux, sigma_y=sigma_y, n_obs=n_obs)
        mu0 = (
            np.zeros(n_coeff, dtype=float)
            if prior_mean is None
            else np.asarray(prior_mean, dtype=float)
        )
        if mu0.shape != (n_coeff,):
            raise ValueError(f"prior_mean must have shape ({n_coeff},), got {mu0.shape}.")

        finite = np.isfinite(y) & np.isfinite(sigma) & np.all(np.isfinite(A), axis=1)
        if not np.all(finite):
            A = A[finite]
            y = y[finite]
            sigma = sigma[finite]
            n_obs = y.size

        precision, logdet_prior_cov = DopplerMap._coerce_prior_precision(
            prior_precision=prior_precision,
            prior_covariance=prior_covariance,
            n_coeff=n_coeff,
        )
        r0 = y - A @ mu0
        whitened_A = A / sigma[:, np.newaxis]
        whitened_r0 = r0 / sigma

        precision_input = 1.0 if prior_precision is None else prior_precision
        precision_arr = np.asarray(precision_input, dtype=float)
        scalar_precision = prior_covariance is None and precision_arr.ndim == 0
        can_reduce_basis = n_obs >= n_coeff
        if use_svd and scalar_precision and can_reduce_basis:
            alpha = float(precision_arr)
            if alpha <= 0.0 or not np.isfinite(alpha):
                raise ValueError("prior_precision must be finite and strictly positive.")

            U, singular_values, Vt = np.linalg.svd(whitened_A, full_matrices=False)
            if singular_values.size:
                tol = (
                    np.finfo(float).eps * max(whitened_A.shape) * singular_values[0]
                    if svd_tol is None
                    else float(svd_tol) * singular_values[0]
                )
                rank = int(np.sum(singular_values > tol))
            else:
                rank = 0

            if rank == 0:
                posterior_mean = mu0.copy()
                posterior_cov = np.eye(n_coeff) / alpha
                posterior_precision = np.eye(n_coeff) * alpha
                quad = float(whitened_r0 @ whitened_r0)
                logdet_predictive = float(np.sum(np.log(sigma**2)))
                log_evidence = -0.5 * (
                    quad + logdet_predictive + n_obs * np.log(2.0 * np.pi)
                )
                return {
                    "log_evidence": float(log_evidence),
                    "posterior_mean": posterior_mean,
                    "posterior_cov": posterior_cov,
                    "posterior_precision": posterior_precision,
                    "model_posterior_mean": A @ posterior_mean,
                    "model_prior_mean": A @ mu0,
                    "quad": quad,
                    "logdet_predictive": float(logdet_predictive),
                    "n_obs": int(n_obs),
                    "n_coeff": int(n_coeff),
                    "basis_rank": 0,
                    "n_null": int(n_coeff),
                    "used_svd": True,
                }

            U_rank = U[:, :rank]
            s_rank = singular_values[:rank]
            Vt_rank = Vt[:rank, :]
            reduced_design = U_rank * s_rank[np.newaxis, :]
            posterior_precision_rank = np.diag(alpha + s_rank**2)
            rhs_rank = reduced_design.T @ whitened_r0
            posterior_delta_rank = rhs_rank / (alpha + s_rank**2)
            posterior_cov_rank = np.diag(1.0 / (alpha + s_rank**2))
            posterior_delta = Vt_rank.T @ posterior_delta_rank
            posterior_mean = mu0 + posterior_delta

            posterior_cov = Vt_rank.T @ posterior_cov_rank @ Vt_rank
            if rank < n_coeff:
                null_vt = Vt[rank:, :]
                posterior_cov += null_vt.T @ null_vt / alpha
            posterior_precision = precision + whitened_A.T @ whitened_A
            posterior_precision = 0.5 * (posterior_precision + posterior_precision.T)

            quad = float(whitened_r0 @ whitened_r0 - rhs_rank @ posterior_delta_rank)
            logdet_noise = float(np.sum(np.log(sigma**2)))
            logdet_predictive = (
                logdet_noise
                - rank * np.log(alpha)
                + float(np.sum(np.log(alpha + s_rank**2)))
            )
            log_evidence = -0.5 * (
                quad + logdet_predictive + n_obs * np.log(2.0 * np.pi)
            )

            return {
                "log_evidence": float(log_evidence),
                "posterior_mean": posterior_mean,
                "posterior_cov": posterior_cov,
                "posterior_precision": posterior_precision,
                "model_posterior_mean": A @ posterior_mean,
                "model_prior_mean": A @ mu0,
                "quad": quad,
                "logdet_predictive": float(logdet_predictive),
                "n_obs": int(n_obs),
                "n_coeff": int(n_coeff),
                "basis_rank": int(rank),
                "n_null": int(n_coeff - rank),
                "used_svd": True,
            }

        posterior_precision = precision + whitened_A.T @ whitened_A
        posterior_precision = 0.5 * (posterior_precision + posterior_precision.T)
        rhs = whitened_A.T @ whitened_r0

        try:
            chol = np.linalg.cholesky(posterior_precision + FLOOR * np.eye(n_coeff))
            posterior_cov = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(n_coeff)))
            posterior_delta = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
            logdet_posterior_precision = 2.0 * np.sum(np.log(np.diag(chol)))
        except np.linalg.LinAlgError:
            posterior_cov = np.linalg.pinv(posterior_precision)
            posterior_delta = posterior_cov @ rhs
            sign, logdet_posterior_precision = np.linalg.slogdet(posterior_precision)
            if sign <= 0:
                raise ValueError("posterior precision is not positive definite.")

        posterior_mean = mu0 + posterior_delta
        quad = float(whitened_r0 @ whitened_r0 - rhs @ posterior_delta)
        logdet_noise = float(np.sum(np.log(sigma**2)))
        logdet_predictive = logdet_noise + logdet_prior_cov + float(logdet_posterior_precision)
        log_evidence = -0.5 * (quad + logdet_predictive + n_obs * np.log(2.0 * np.pi))

        return {
            "log_evidence": float(log_evidence),
            "posterior_mean": posterior_mean,
            "posterior_cov": posterior_cov,
            "posterior_precision": posterior_precision,
            "model_posterior_mean": A @ posterior_mean,
            "model_prior_mean": A @ mu0,
            "quad": quad,
            "logdet_predictive": float(logdet_predictive),
            "n_obs": int(n_obs),
            "n_coeff": int(n_coeff),
            "basis_rank": int(n_coeff),
            "n_null": 0,
            "used_svd": False,
        }

    def surface_marginalized_wind_evidence(
        self,
        flux: np.ndarray,
        wavelength: np.ndarray,
        spectrum: np.ndarray,
        wind_velocity: np.ndarray,
        surface_design_matrix: np.ndarray,
        sigma_y: np.ndarray | float | None = None,
        weights: np.ndarray | float | None = None,
        template_wavelength: np.ndarray | None = None,
        rv_kms: float = 0.0,
        prior_mean: np.ndarray | None = None,
        prior_precision: float | np.ndarray | None = 1.0,
        prior_covariance: np.ndarray | None = None,
        estimate_alpha: bool = False,
        verbose: bool = False,
        return_design_matrix: bool = False,
    ) -> dict:
        """Evidence for a fixed wind field after marginalizing surface coefficients.

        ``wind_velocity`` supplies the line-of-sight velocity on the surface grid.
        ``surface_design_matrix`` supplies the spherical-harmonic basis evaluated on
        that same grid. The method first constructs the wind-broadened linear design
        matrix, then integrates over Gaussian spherical-harmonic coefficients.

        If ``estimate_alpha`` is ``False``, ``prior_precision`` or
        ``prior_covariance`` defines the fixed Gaussian prior. If ``estimate_alpha``
        is ``True``, the existing ``spectralmap.bayesian_linalg`` iterative evidence
        optimizer is used to estimate a scalar coefficient prior precision.
        """
        design_matrix = self.wind_broadened_design_matrix(
            wavelength=wavelength,
            spectrum=spectrum,
            wind_velocity=wind_velocity,
            surface_design_matrix=surface_design_matrix,
            weights=weights,
            template_wavelength=template_wavelength,
            rv_kms=rv_kms,
            flatten=True,
        )

        if estimate_alpha:
            if prior_covariance is not None:
                raise ValueError("estimate_alpha=True is only supported without prior_covariance.")
            if prior_precision is not None:
                precision_arr = np.asarray(prior_precision, dtype=float)
                if precision_arr.ndim != 0 or not np.isclose(float(precision_arr), 1.0):
                    raise ValueError(
                        "estimate_alpha=True estimates alpha, so prior_precision is ignored."
                    )

            y, sigma = self._coerce_observation_vector(
                flux=flux,
                sigma_y=sigma_y,
                n_obs=design_matrix.shape[0],
            )
            finite = (
                np.isfinite(y)
                & np.isfinite(sigma)
                & np.all(np.isfinite(design_matrix), axis=1)
            )
            if not np.all(finite):
                design_matrix = design_matrix[finite]
                y = y[finite]
                sigma = sigma[finite]
            mu0 = (
                np.zeros(design_matrix.shape[1], dtype=float)
                if prior_mean is None
                else np.asarray(prior_mean, dtype=float)
            )
            if mu0.shape != (design_matrix.shape[1],):
                raise ValueError(
                    f"prior_mean must have shape ({design_matrix.shape[1]},), got {mu0.shape}."
                )
            mu, cov, alpha, beta, log_ev, log_ev_marginalized = optimize_hyperparameters(
                design_matrix,
                y,
                sigma_y=sigma,
                mu0=mu0,
                verbose=verbose,
            )
            result = {
                "log_evidence": float(log_ev),
                "log_evidence_marginalized": float(log_ev_marginalized),
                "posterior_mean": mu,
                "posterior_cov": cov,
                "alpha": alpha,
                "beta": beta,
                "model_posterior_mean": design_matrix @ mu,
                "model_prior_mean": design_matrix @ mu0,
                "n_obs": int(design_matrix.shape[0]),
                "n_coeff": int(design_matrix.shape[1]),
            }
        else:
            result = self.gaussian_linear_evidence(
                design_matrix=design_matrix,
                flux=flux,
                sigma_y=sigma_y,
                prior_mean=prior_mean,
                prior_precision=prior_precision,
                prior_covariance=prior_covariance,
            )

        if return_design_matrix:
            result["design_matrix"] = design_matrix
        return result

    def _finalize_doppler_solution(
        self,
        theta: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray | float | None,
        solve_result,
        solver_used: str,
        mode: str,
        extras: dict | None = None,
    ):
        """Store Doppler posterior metadata and return coefficients/log likelihood."""
        model_flux = self._eval_to_numpy(self.map.flux(theta=theta, normalize=self.normalized))
        resid = flux - model_flux
        chi2 = self._compute_chi2(resid=resid, flux_err=flux_err)
        log_like = -0.5 * chi2

        y_coeff = self._eval_to_numpy(self.map.y)
        self.mu = y_coeff.reshape(-1)
        self.cov = None
        self.flux = flux
        self.flux_err = flux_err
        self.theta = theta
        self.design_matrix_ = None
        self.intensity_design_matrix_ = None

        hyper = {
            "solver": solver_used,
            "normalized": self.normalized,
            "log_like": log_like,
            "chi2": chi2,
            "mode": mode,
            "solve_result": solve_result,
        }
        if extras:
            hyper.update(extras)
        self.hyper = hyper

        return self.mu, self.cov, log_like


    def solve_posterior(
        self,
        y: np.ndarray,
        sigma_y: np.ndarray | float | None = None,
        theta: np.ndarray | None = None,
        solver: str | None = None,
        spatial_cov: float | np.ndarray = 1e-4,
        spectral_cov: float | np.ndarray = 1e-3,
        baseline_var: float = 1e-2,
        spectral_guess: np.ndarray | None = None,
        initialize_spectrum: bool = True,
        spectral_lambda_init: float = 1e5,
        logT0: float = 12.0,
        logTf: float = 0.0,
        nlogT: int = 50,
        spectral_method: str = "L2",
        verbose: bool = False,
        **kwargs,
    ):
        """Section 7.3 workflow: unknown spectrum and unknown baseline.

        This follows the iterative strategy described in Luger et al. (2021):
        1) build a coarse spectral guess via aggressive regularized deconvolution,
        2) jointly solve for map and spectrum while marginalizing baseline,
        3) use a tempering schedule for stability.
        """
        theta, flux, flux_err = self._prepare_doppler_inputs(y=y, sigma_y=sigma_y, theta=theta)
        solver_use = self.solver if solver is None else solver

        if not self.normalized:
            raise ValueError(
                "Unknown-baseline inference requires normalized spectra (self.normalized=True)."
            )

        if spectral_guess is None and initialize_spectrum:
            init_kwargs = {
                "flux": flux,
                "solver": solver_use,
                "theta": theta,
                "normalized": True,
                "fix_map": True,
                "spectral_method": "L1",
                "spectral_lambda": float(spectral_lambda_init),
                "spectral_cov": spectral_cov,
                "baseline_var": float(baseline_var),
            }
            if flux_err is not None:
                init_kwargs["flux_err"] = flux_err
            if not verbose:
                init_kwargs["quiet"] = True

            try:
                self.map.solve(**init_kwargs)
                spectral_guess = self._eval_to_numpy(self.map.spectrum)
            except Exception:
                # Fallback: use the mean normalized spectrum as a robust initial guess.
                wav = self._eval_to_numpy(self.map.wav)
                wav0 = self._eval_to_numpy(self.map.wav0)
                mean_spec = np.nanmean(flux, axis=0)
                if wav0.size != mean_spec.size:
                    mean_spec = np.interp(wav0, wav, mean_spec, left=mean_spec[0], right=mean_spec[-1])

                if int(self.map.nc) == 1:
                    spectral_guess = np.asarray(mean_spec, dtype=float)
                else:
                    spectral_guess = np.tile(np.asarray(mean_spec, dtype=float)[np.newaxis, :], (int(self.map.nc), 1))

        solve_kwargs = {
            "flux": flux,
            "solver": solver_use,
            "theta": theta,
            "normalized": True,
            "fix_map": False,
            "fix_spectrum": False,
            "spatial_cov": spatial_cov,
            "spectral_cov": spectral_cov,
            "baseline_var": float(baseline_var),
            "logT0": float(logT0),
            "logTf": float(logTf),
            "nlogT": int(nlogT),
            "spectral_method": spectral_method,
        }
        if spectral_guess is not None:
            solve_kwargs["spectral_guess"] = spectral_guess
        if flux_err is not None:
            solve_kwargs["flux_err"] = flux_err
        if not verbose:
            solve_kwargs["quiet"] = True
        solve_kwargs.update(kwargs)

        solve_result = self.map.solve(**solve_kwargs)
        return self._finalize_doppler_solution(
            theta=theta,
            flux=flux,
            flux_err=flux_err,
            solve_result=solve_result,
            solver_used=solver_use,
            mode="unknown_spectrum_unknown_baseline",
            extras={
                "spatial_cov": spatial_cov,
                "spectral_cov": spectral_cov,
                "baseline_var": baseline_var,
                "logT0": logT0,
                "logTf": logTf,
                "nlogT": int(nlogT),
                "initialized_spectrum": bool(initialize_spectrum),
            },
        )

    def show(self, n: int = 0, projection: str = "moll", **kwargs):
        """Render a Doppler image component with ``starry.DopplerMap.show``."""
        self.map.show(n=n, projection=projection, **kwargs)
        return

    def solve_unknown_spectrum_baseline(self, *args, **kwargs):
        """Compatibility wrapper for the unknown-spectrum Doppler workflow."""
        return self.solve_posterior(*args, **kwargs)
