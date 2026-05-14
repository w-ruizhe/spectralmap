"""Secondary-eclipse surface mapping models."""

from __future__ import annotations

import numpy as np
import starry

from spectralmap.core import (
    Map,
    Maps,
    _build_starry_map,
)



class EclipseMap(Map):
    """Eclipse mapping model."""

    def __init__(
        self,
        map_res: int | None = 30,
        ydeg: int | None = None,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        eclipse_depth: float | None = None,
        observed_lon_range: np.ndarray | None = None,
        projection: str = "rect",
    ):
        if pri is None or sec is None:
            raise ValueError("EclipseMap requires both pri and sec.")
        super().__init__(map_res=map_res, ydeg=ydeg, projection=projection)
        self.pri = pri
        self.sec = sec
        self.udeg = udeg
        self.u = u
        # Eclipse mapping currently assumes an edge-on emitting body map.
        self.sec.map = _build_starry_map(ydeg=ydeg, map_res=map_res, inc=90, udeg=udeg, u=u)
        self.sys = starry.System(pri, sec)
        self.map = self.sec.map
        self.eclipse_depth = eclipse_depth
        self.observed_lon_range = observed_lon_range

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
        self.sec.map.y[:] = coeffs[1:]




class EclipseMaps(Maps):
    """Eclipse multi-wavelength mapping utilities."""

    def __init__(
        self,
        map_res = 30,
        udeg: int | None = None,
        u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
        pri: starry.Primary | None = None,
        sec: starry.Secondary | None = None,
        eclipse_depth: float | None = None,
        observed_lon_range: np.ndarray | None = None,
        projection: str = "rect",
        verbose=True,
    ):
        super().__init__(
            map_res=map_res,
            observed_lon_range=observed_lon_range,
            projection=projection,
            verbose=verbose,
        )
        if (pri is None or sec is None):
            raise ValueError("EclipseMaps requires both primary and secondary objects to be passed in.")
        self.pri = pri
        self.sec = sec
        self.eclipse_depth = eclipse_depth
        self.udeg = udeg
        self.u = u
        self.projection = projection

    def _make_map(self, ydeg: int, inc: float | None) -> Map:
        # For eclipse mode, inclination is encoded on the secondary object.
        if inc is not None and hasattr(self.sec.map, "inc"):
            self.sec.map.inc = float(inc)
        return EclipseMap(
            map_res=self.map_res,
            ydeg=int(ydeg),
            udeg=self.udeg,
            u=self._resolve_u_for_wavelength(i_wl=None),
            pri=self.pri,
            sec=self.sec,
            eclipse_depth=self.eclipse_depth,
            observed_lon_range=self.observed_lon_range,
            projection=self.projection,
        )


def make_map(
    map_res: int | None = 30,
    ydeg: int | None = None,
    udeg: int | None = None,
    u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
    pri: starry.Primary | None = None,
    sec: starry.Secondary | None = None,
    eclipse_depth: float | None = None,
    observed_lon_range: np.ndarray | None = None,
    projection: str = "rect",
) -> EclipseMap:
    """Create a single eclipse map model."""
    return EclipseMap(
        map_res=map_res,
        ydeg=ydeg,
        udeg=udeg,
        u=u,
        pri=pri,
        sec=sec,
        eclipse_depth=eclipse_depth,
        observed_lon_range=observed_lon_range,
        projection=projection
    )


def make_maps(
    map_res=30,
    udeg: int | None = None,
    u: np.ndarray | list[float] | tuple[float, ...] | float | None = None,
    pri: starry.Primary | None = None,
    sec: starry.Secondary | None = None,
    eclipse_depth: float | None = None,
    observed_lon_range: np.ndarray | None = None,
    projection: str = "rect",
    verbose=True,
) -> EclipseMaps:
    """Create a multi-wavelength eclipse mapping driver."""
    return EclipseMaps(
        map_res=map_res,
        udeg=udeg,
        u=u,
        pri=pri,
        sec=sec,
        eclipse_depth=eclipse_depth,
        observed_lon_range=observed_lon_range,
        projection=projection,
        verbose=verbose,
    )
