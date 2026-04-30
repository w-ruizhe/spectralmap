"""Doppler imaging model wrappers."""

from __future__ import annotations

import numpy as np
import starry

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
