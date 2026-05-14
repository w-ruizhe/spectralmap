"""Rotational phase-curve mapping models."""

from __future__ import annotations

import numpy as np

from spectralmap.bayesian_linalg import gaussian_linear_posterior
from spectralmap.core import (
    LightCurveData,
    Map,
    Maps,
    _build_starry_map,
)
from spectralmap.gp import (
    pressure_squared_exponential_covariance,
    project_pixel_covariance_to_coefficients,
    separable_covariance,
    spherical_squared_exponential_covariance,
    squared_exponential_1d_covariance,
)
from spectralmap.utilities import solid_angle_weights



class RotMap(Map):
    """Single-channel rotational map model built on ``starry.Map``."""

    def __init__(
        self,
        map_res: int | None = 30,
        ydeg: int | None = None,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        inc: int | None = None,
        projection: str = "rect",
    ):
        if ydeg is None:
            ydeg = 5
        super().__init__(map_res=map_res, ydeg=ydeg, projection=projection)
        self.inc = inc
        self.udeg = udeg
        self.u = u
        self.map = _build_starry_map(ydeg=ydeg, inc=inc, udeg=udeg, u=u)

    def _design_matrix_impl(self, theta: np.ndarray) -> np.ndarray:
        return self.map.design_matrix(theta=theta)

    def _intensity_design_matrix_impl(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        I = self.map.intensity_design_matrix(lat=lat, lon=lon)
        I = self._eval_to_numpy(I)
        return I

    def _apply_coefficients_to_map(self, coeffs: np.ndarray) -> None:
        self.map.y[:] = coeffs



class RotMaps(Maps):
    """Rotational multi-wavelength mapping utilities."""

    def __init__(
        self,
        map_res=30,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        projection: str = "rect",
        verbose=True,
    ):
        super().__init__(
            map_res=map_res,
            projection=projection,
            verbose=verbose,
        )
        self.udeg = udeg
        self.u = u
        self.projection = projection

    def _make_map(self, ydeg: int, inc: float | None) -> Map:
        return RotMap(
            map_res=self.map_res,
            ydeg=int(ydeg),
            udeg=self.udeg,
            u=self._resolve_u_for_wavelength(i_wl=None),
            inc=inc,
            projection=self.projection,
        )

    @staticmethod
    def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
        """Build a dense block diagonal matrix without requiring scipy."""
        n_rows = int(sum(block.shape[0] for block in blocks))
        n_cols = int(sum(block.shape[1] for block in blocks))
        out = np.zeros((n_rows, n_cols), dtype=float)
        row = 0
        col = 0
        for block in blocks:
            rr, cc = block.shape
            out[row:row + rr, col:col + cc] = block
            row += rr
            col += cc
        return out

    @staticmethod
    def _resolve_joint_prior_mean(prior_mean, n_wl: int, n_full: int) -> np.ndarray:
        n_free = n_full - 1
        if prior_mean is None:
            return np.zeros(n_wl * n_free, dtype=float)
        arr = np.asarray(prior_mean, dtype=float).reshape(-1)
        if arr.shape == (n_free,):
            return np.tile(arr, n_wl)
        if arr.shape == (n_full,):
            return np.tile(arr[1:], n_wl)
        if arr.shape == (n_wl * n_free,):
            return arr
        if arr.shape == (n_wl * n_full,):
            return arr.reshape(n_wl, n_full)[:, 1:].reshape(-1)
        raise ValueError(
            "prior_mean must be length n_free, n_full, n_wavelength*n_free, "
            "or n_wavelength*n_full."
        )

    @staticmethod
    def _resolve_depth_coordinates(
        data: LightCurveData,
        wavelength_indices: np.ndarray,
        *,
        depth_coordinates=None,
        pressure_coordinates=None,
    ) -> tuple[np.ndarray | None, str]:
        if pressure_coordinates is not None and depth_coordinates is not None:
            raise ValueError("Provide either pressure_coordinates or depth_coordinates, not both.")

        if pressure_coordinates is not None:
            coords = np.asarray(pressure_coordinates, dtype=float).reshape(-1)
            label = "pressure"
        elif depth_coordinates is not None:
            coords = np.asarray(depth_coordinates, dtype=float).reshape(-1)
            label = "depth"
        elif data.wl is not None:
            coords = np.asarray(data.wl, dtype=float).reshape(-1)
            label = "wavelength"
        else:
            return None, "identity"

        if coords.size == data.flux.shape[0]:
            coords = coords[wavelength_indices]
        elif coords.size != wavelength_indices.size:
            raise ValueError(
                "Depth/pressure coordinates must have length n_wavelength or "
                "len(wavelength_indices)."
            )
        return coords, label

    def solve_joint_wavelength_posterior(
        self,
        data: LightCurveData,
        *,
        inc: float,
        ydeg: int,
        prot: float | None = None,
        wavelength_indices: np.ndarray | list[int] | None = None,
        spatial_amplitude: float = 1.0,
        spatial_length_scale: float = 1.0,
        spatial_jitter: float = 1e-10,
        depth_coordinates=None,
        pressure_coordinates=None,
        depth_amplitude: float = 1.0,
        depth_length_scale: float | None = None,
        depth_jitter: float = 1e-10,
        prior_mean=None,
        return_design_matrix: bool = False,
    ) -> dict:
        """Jointly solve rotational maps for a wavelength group.

        The first starry coefficient is kept fixed at 1 for each wavelength,
        matching :meth:`Map.solve_posterior`. The GP prior is applied to the
        remaining free coefficients with a separable depth-by-surface covariance.
        """
        if not isinstance(data, LightCurveData):
            raise TypeError("data must be a LightCurveData instance.")
        n_data_wl = data.flux.shape[0]
        if wavelength_indices is None:
            wavelength_indices = np.arange(n_data_wl, dtype=int)
        else:
            wavelength_indices = np.asarray(wavelength_indices, dtype=int).reshape(-1)
        if wavelength_indices.size == 0:
            raise ValueError("wavelength_indices must contain at least one wavelength.")
        if np.any(wavelength_indices < 0) or np.any(wavelength_indices >= n_data_wl):
            raise ValueError("wavelength_indices contains out-of-range entries.")

        if prot is None:
            if data.theta is None:
                raise ValueError("data.theta is required when prot is None.")
            theta = data.theta
        else:
            theta = self._theta_from_period(data, float(prot))

        n_group = wavelength_indices.size
        map_obj = self._make_map(ydeg=int(ydeg), inc=float(inc))
        map_obj.null_uncertainty = self.null_uncertainty

        design_blocks = []
        intensity_blocks = []
        y_blocks = []
        sigma_blocks = []
        for group_pos, i_wl in enumerate(wavelength_indices):
            self._set_map_limb_darkening_for_wavelength(map_obj, i_wl=int(i_wl), n_wl=n_data_wl)
            A_full = map_obj.design_matrix(theta)
            design_blocks.append(np.asarray(A_full[:, 1:], dtype=float))
            y_blocks.append(np.asarray(data.flux[i_wl], dtype=float) - A_full[:, 0])
            if data.flux_err is None:
                sigma_blocks.append(np.ones(data.flux.shape[1], dtype=float))
            else:
                sigma_blocks.append(np.asarray(data.flux_err[i_wl], dtype=float))
            I_full = map_obj.intensity_design_matrix(projection=self.projection)
            intensity_blocks.append(np.asarray(I_full, dtype=float))

            if group_pos == 0:
                self.mask_2d = map_obj.mask_2d
                self.mask_1d = map_obj.mask_1d
                self.lat = map_obj.lat
                self.lon = map_obj.lon
                self.lat_flat = map_obj.lat_flat
                self.lon_flat = map_obj.lon_flat

        n_full = intensity_blocks[0].shape[1]
        n_free = n_full - 1
        pixel_cov = spherical_squared_exponential_covariance(
            map_obj.lat_flat[map_obj.mask_1d],
            map_obj.lon_flat[map_obj.mask_1d],
            amplitude=spatial_amplitude,
            length_scale=spatial_length_scale,
            jitter=spatial_jitter,
        )
        weights = solid_angle_weights(
            map_obj.lat_flat[map_obj.mask_1d],
            map_obj.lon_flat[map_obj.mask_1d],
        )
        spatial_coeff_cov = project_pixel_covariance_to_coefficients(
            intensity_blocks[0][:, 1:],
            pixel_cov,
            weights=weights,
            jitter=spatial_jitter,
        )
        if spatial_coeff_cov.shape != (n_free, n_free):
            raise RuntimeError("Projected spatial covariance has an unexpected shape.")

        coords, coord_label = self._resolve_depth_coordinates(
            data,
            wavelength_indices,
            depth_coordinates=depth_coordinates,
            pressure_coordinates=pressure_coordinates,
        )
        if depth_length_scale is None or n_group == 1:
            depth_cov = np.eye(n_group, dtype=float) * float(depth_amplitude) ** 2
            if depth_jitter > 0.0:
                depth_cov = depth_cov + np.eye(n_group) * float(depth_jitter)
            coord_label = "identity" if coords is None else coord_label
        elif coord_label == "pressure":
            depth_cov = pressure_squared_exponential_covariance(
                coords,
                amplitude=depth_amplitude,
                length_scale=depth_length_scale,
                jitter=depth_jitter,
            )
        else:
            depth_cov = squared_exponential_1d_covariance(
                coords,
                amplitude=depth_amplitude,
                length_scale=depth_length_scale,
                jitter=depth_jitter,
            )

        joint_prior_cov = separable_covariance(depth_cov, spatial_coeff_cov)
        design_matrix = self._block_diag(design_blocks)
        y_joint = np.concatenate(y_blocks)
        sigma_joint = np.concatenate(sigma_blocks)
        prior_mean_joint = self._resolve_joint_prior_mean(prior_mean, n_group, n_full)
        result = gaussian_linear_posterior(
            design_matrix,
            y_joint,
            sigma_y=sigma_joint,
            prior_mean=prior_mean_joint,
            prior_covariance=joint_prior_cov,
        )

        free_mu = result["posterior_mean"].reshape(n_group, n_free)
        free_cov = result["posterior_cov"]
        coeff_mu = np.zeros((n_group, n_full), dtype=float)
        coeff_mu[:, 0] = 1.0
        coeff_mu[:, 1:] = free_mu

        coeff_cov = np.zeros((n_group, n_full, n_full), dtype=float)
        spatial_intensity = np.zeros((n_group, intensity_blocks[0].shape[0]), dtype=float)
        spatial_intensity_cov = np.zeros(
            (n_group, intensity_blocks[0].shape[0], intensity_blocks[0].shape[0]),
            dtype=float,
        )
        for i in range(n_group):
            free_slice = slice(i * n_free, (i + 1) * n_free)
            cov_free = free_cov[free_slice, free_slice]
            coeff_cov[i, 1:, 1:] = cov_free
            I_use = intensity_blocks[i]
            spatial_intensity[i] = I_use @ coeff_mu[i]
            spatial_intensity_cov[i] = I_use[:, 1:] @ cov_free @ I_use[:, 1:].T

        out = {
            "wavelength_indices": wavelength_indices.copy(),
            "theta": np.asarray(theta, dtype=float),
            "coeff_mu": coeff_mu,
            "coeff_cov": coeff_cov,
            "joint_free_coeff_mu": result["posterior_mean"],
            "joint_free_coeff_cov": result["posterior_cov"],
            "spatial_intensity": spatial_intensity,
            "spatial_intensity_cov": spatial_intensity_cov,
            "log_evidence": float(result["log_evidence"]),
            "spatial_coeff_cov_prior": spatial_coeff_cov,
            "depth_cov_prior": depth_cov,
            "joint_prior_cov": joint_prior_cov,
            "depth_coordinate_label": coord_label,
        }
        if coords is not None:
            out["depth_coordinates"] = np.asarray(coords, dtype=float)
        if return_design_matrix:
            out["design_matrix"] = design_matrix

        self.joint_wavelength_posterior_ = out
        return out


def make_map(map_res: int | None = 30,
        ydeg: int | None = None,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        inc: int | None = None,
        projection: str = "rect") -> RotMap:
        """Create a single rotational map model."""

        return RotMap(map_res=map_res, ydeg=ydeg, udeg=udeg, u=u, inc=inc, projection=projection)

def make_maps(
        map_res=30,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        projection: str = "rect",
        verbose=True) -> RotMaps:
    """Create a multi-wavelength rotational mapping driver."""

    return RotMaps(map_res=map_res, udeg=udeg, u=u, projection=projection, verbose=verbose)
