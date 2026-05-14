"""Rotational phase-curve mapping models."""

from __future__ import annotations

import numpy as np

from spectralmap.core import (
    Map,
    Maps,
    _build_starry_map,
)



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
