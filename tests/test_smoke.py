import numpy as np
import pytest


def _starry_has_doppler_map() -> bool:
    try:
        import starry
    except ModuleNotFoundError:
        return False

    return hasattr(starry, "DopplerMap")


def _starry_available() -> bool:
    try:
        import starry  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_import():
    import spectralmap  # noqa: F401
    import starry

    map = starry.Map(ydeg=4)
    map.design_matrix(theta=np.linspace(0, 360, 10))


def test_core_has_no_mode_factory_helpers():
    import spectralmap.core as core

    assert not hasattr(core, "make_map")
    assert not hasattr(core, "make_maps")


def test_make_map_accepts_limb_darkening_args():
    from spectralmap.rotational import make_map

    map_obj = make_map(ydeg=2, udeg=2, u=[0.3, 0.1], inc=70)

    assert map_obj.map.udeg == 2
    assert np.isclose(map_obj.map[1], 0.3)
    assert np.isclose(map_obj.map[2], 0.1)


def test_make_maps_passes_limb_darkening_args():
    from spectralmap.rotational import make_maps

    maps = make_maps(udeg=1, u=[0.25])
    map_obj = maps._make_map(ydeg=2, inc=85)

    assert map_obj.map.udeg == 1
    assert np.isclose(map_obj.map[1], 0.25)


@pytest.mark.skipif(not _starry_has_doppler_map(), reason="starry.DopplerMap is unavailable")
def test_make_map_accepts_doppler_args():
    from spectralmap.doppler import DopplerMap

    wav = np.linspace(643.0, 643.2, 8)
    wav0 = np.linspace(642.9, 643.3, 16)
    map_obj = DopplerMap(
        ydeg=2,
        udeg=1,
        u=[0.2],
        nt=5,
        wav=wav,
        wav0=wav0,
        inc=70.0,
        veq=2.0e4,
    )

    assert map_obj.mode == "doppler"
    assert int(map_obj.map.nt) == 5
    assert int(map_obj.map.nw) == wav.size
    assert int(map_obj.map.udeg) == 1
    assert hasattr(map_obj, "solve_unknown_spectrum_baseline")


@pytest.mark.skipif(not _starry_has_doppler_map(), reason="starry.DopplerMap is unavailable")
def test_doppler_module_has_no_make_maps_factory():
    import spectralmap.doppler as doppler

    assert not hasattr(doppler, "make_maps")


@pytest.mark.skipif(not _starry_has_doppler_map(), reason="starry.DopplerMap is unavailable")
def test_doppler_flux_shape_helper_accepts_transpose():
    from spectralmap.doppler import DopplerMap

    flux = np.arange(15, dtype=float).reshape(3, 5)

    assert np.allclose(DopplerMap._coerce_flux_shape(flux, nt=3, nw=5, label="flux"), flux)
    assert np.allclose(DopplerMap._coerce_flux_shape(flux.T, nt=3, nw=5, label="flux"), flux)


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_rotmaps_marginalize_over_ydeg_inc_prot():
    from spectralmap.core import LightCurveData
    from spectralmap.rotational import RotMaps

    class _FakeMap:
        def __init__(self, ydeg: int, inc: float):
            self.ydeg = int(ydeg)
            self.inc = float(inc)
            self.moll_mask = np.ones((1, 3), dtype=bool)
            self.moll_mask_flat = self.moll_mask.ravel()
            self.lat = np.zeros((1, 3), dtype=float)
            self.lon = np.array([[0.0, 30.0, 60.0]], dtype=float)
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()
            self.observed_mask = np.ones(3, dtype=bool)

        def solve_posterior(self, y, sigma_y=None, theta=None, lamda=None, verbose=False):
            theta = np.asarray(theta, dtype=float)
            theta_signal = float(np.mean(theta))
            base = 0.1 * self.ydeg + 0.01 * self.inc + 1e-3 * theta_signal
            mu = np.array([1.0, base], dtype=float)
            cov = np.eye(2, dtype=float) * 0.01
            log_ev = -base * base
            return mu, cov, log_ev

        def intensity_design_matrix(self, projection="rect"):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=float,
            )

    maps = RotMaps(map_res=3, verbose=False)

    maps._make_map = lambda ydeg, inc: _FakeMap(ydeg=ydeg, inc=inc)

    data = LightCurveData(
        time=np.array([0.0, 1.0, 2.0, 3.0]),
        flux=np.ones((2, 4), dtype=float),
        flux_err=np.full((2, 4), 0.1, dtype=float),
    )

    w_all, I_all_wl, I_cov_all_wl = maps.marginalize(
        data,
        ydeg={"values": [2, 4], "weights": [0.6, 0.4]},
        inc={"values": [60.0, 80.0], "weights": [0.5, 0.5]},
        prot={"values": [2.0, 4.0], "weights": [0.7, 0.3]},
    )

    n_components = 2 * 2 * 2
    assert w_all.shape == (n_components, 2)
    assert I_all_wl.shape == (2, 3)
    assert I_cov_all_wl.shape == (2, 3, 3)
    assert np.allclose(np.sum(w_all, axis=0), 1.0)
    assert maps.mixture_ is not None
    assert len(maps.mixture_["components"][0]) == n_components


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_rotmaps_marginalize_over_lambda_axis():
    from spectralmap.core import LightCurveData
    from spectralmap.rotational import RotMaps

    class _FakeMap:
        def __init__(self, ydeg: int, inc: float):
            self.ydeg = int(ydeg)
            self.inc = float(inc)
            self.moll_mask = np.ones((1, 3), dtype=bool)
            self.moll_mask_flat = self.moll_mask.ravel()
            self.lat = np.zeros((1, 3), dtype=float)
            self.lon = np.array([[0.0, 30.0, 60.0]], dtype=float)
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()
            self.observed_mask = np.ones(3, dtype=bool)

        def solve_posterior(self, y, sigma_y=None, theta=None, lamda=None, verbose=False):
            theta = np.asarray(theta, dtype=float)
            theta_signal = float(np.mean(theta))
            lamda_term = 0.0 if lamda is None else float(lamda)
            base = 0.1 * self.ydeg + 0.01 * self.inc + 1e-3 * theta_signal + 1e-4 * lamda_term
            mu = np.array([1.0, base], dtype=float)
            cov = np.eye(2, dtype=float) * 0.01
            log_ev = -base * base
            return mu, cov, log_ev

        def intensity_design_matrix(self, projection="rect"):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=float,
            )

    maps = RotMaps(map_res=3, verbose=False)
    maps._make_map = lambda ydeg, inc: _FakeMap(ydeg=ydeg, inc=inc)

    data = LightCurveData(
        theta=np.array([0.0, 90.0, 180.0, 270.0]),
        flux=np.ones((2, 4), dtype=float),
        flux_err=np.full((2, 4), 0.1, dtype=float),
    )

    w_all, I_all_wl, I_cov_all_wl = maps.marginalize(
        data,
        ydeg=[2, 4],
        inc=[70.0, 80.0],
        prot=None,
        lamda=[1e2, 1e3],
    )

    n_components = 2 * 2 * 1 * 2
    assert w_all.shape == (n_components, 2)
    assert I_all_wl.shape == (2, 3)
    assert I_cov_all_wl.shape == (2, 3, 3)
    assert np.allclose(np.sum(w_all, axis=0), 1.0)

    lamda_values = [c["lamda"] for c in maps.mixture_["components"][0]]
    assert set(np.round(lamda_values, 6)) == {100.0, 1000.0}


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_lightcurvedata_requires_at_least_one_phase_axis():
    from spectralmap.core import LightCurveData

    # Both axes are allowed if they are consistent.
    data = LightCurveData(
        theta=np.array([0.0, 90.0]),
        time=np.array([0.0, 1.0]),
        flux=np.ones((1, 2), dtype=float),
    )
    assert data.theta is not None
    assert data.time is not None

    with pytest.raises(ValueError):
        LightCurveData(
            flux=np.ones((1, 2), dtype=float),
        )


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_eclipsemaps_replaced_marginalized_maps_api():
    from spectralmap.eclipse import EclipseMaps

    assert hasattr(EclipseMaps, "marginalize")
    assert not hasattr(EclipseMaps, "marginalized_maps")
