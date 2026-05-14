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


def test_plot_pc_projection_accepts_observed_lon_subset():
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from spectralmap.plotting import plot_pc_projection

    mask_1d = np.array([True, False, True, False, True, False])
    maps = SimpleNamespace(
        mask_2d=mask_1d.reshape(2, 3),
        mask_1d=mask_1d,
        pc_scores=np.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
    )

    fig, ax = plot_pc_projection(maps, upsample=1, extrapolate=False)
    assert ax.get_title() == "PC1 and PC2 Overlay"
    plt.close(fig)


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_rotmaps_marginalize_over_ydeg_inc_prot():
    from spectralmap.core import LightCurveData
    from spectralmap.rotational import RotMaps

    class _FakeMap:
        def __init__(self, ydeg: int, inc: float):
            self.ydeg = int(ydeg)
            self.inc = float(inc)
            self.mask_2d = np.ones((1, 3), dtype=bool)
            self.mask_1d = self.mask_2d.ravel()
            self.lat = np.zeros((1, 3), dtype=float)
            self.lon = np.array([[0.0, 30.0, 60.0]], dtype=float)
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()

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
            self.mask_2d = np.ones((1, 3), dtype=bool)
            self.mask_1d = self.mask_2d.ravel()
            self.lat = np.zeros((1, 3), dtype=float)
            self.lon = np.array([[0.0, 30.0, 60.0]], dtype=float)
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()

        def solve_posterior(self, y, sigma_y=None, theta=None, lamda=None, verbose=False):
            theta = np.asarray(theta, dtype=float)
            theta_signal = float(np.mean(theta))
            lamda_term = 0.0 if lamda is None else float(lamda)
            base = 0.1 * self.ydeg + 0.01 * self.inc + 1e-3 * theta_signal + 1e-4 * lamda_term
            mu = np.array([1.0, base], dtype=float)
            cov = np.eye(2, dtype=float) * 0.01
            log_ev = -base * base
            return mu, cov, log_ev

        def design_matrix(self, theta):
            theta = np.asarray(theta, dtype=float)
            return np.column_stack((np.ones_like(theta), np.cos(np.deg2rad(theta))))

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
        wl=np.array([1.0, 2.0], dtype=float),
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
    assert len(maps.mixture_["coeff_mu_components"][0]) == n_components
    assert len(maps.mixture_["coeff_cov_components"][0]) == n_components

    grid, x_values, y_values = maps.model_weight_grid("lamda", "ydeg", i_wl=0)
    assert grid.shape == (2, 2)
    assert np.allclose(x_values, [100.0, 1000.0])
    assert np.allclose(y_values, [2, 4])
    assert np.isclose(np.sum(grid), 1.0)

    fig, ax, im = maps.plot_model_weights("lamda", "ydeg", i_wl=0, colorbar=False)
    assert ax.get_xlabel() == "lamda"
    assert ax.get_ylabel() == "ydeg"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["1e+02", "1e+03"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["2", "4"]
    assert im.get_array().shape == (2, 2)
    import matplotlib.pyplot as plt
    plt.close(fig)

    original_mixture = maps.mixture_

    components = [{"lamda": float(x), "ydeg": y} for y in [2, 4] for x in range(6)]
    maps.mixture_ = {"components": [components], "weights": [np.full(len(components), 1.0 / len(components))]}
    fig, ax, im = maps.plot_model_weights("lamda", "ydeg", i_wl=0, colorbar=False)
    width, height = fig.get_size_inches()
    assert width > height
    plt.close(fig)

    components = [{"lamda": float(x), "ydeg": y} for y in range(6) for x in [1, 2]]
    maps.mixture_ = {"components": [components], "weights": [np.full(len(components), 1.0 / len(components))]}
    fig, ax, im = maps.plot_model_weights("lamda", "ydeg", i_wl=0, colorbar=False)
    width, height = fig.get_size_inches()
    assert height > width
    plt.close(fig)

    components = [{"lamda": 1.0, "ydeg": 2}, {"lamda": 10.0, "ydeg": 4}]
    maps.mixture_ = {"components": [components], "weights": [np.array([0.1, 0.9])]}
    fig, ax, im = maps.plot_model_weights("lamda", "ydeg", i_wl=0, colorbar=True, log_scale=True)
    assert im.norm.__class__.__name__ == "LogNorm"
    assert np.ma.is_masked(im.get_array()[0, 1])
    plt.close(fig)

    maps.mixture_ = original_mixture
    fig, ax = maps.plot_lightcurve(i_wl=0, n_samples=3, random_state=0)
    labels = [line.get_label() for line in ax.lines]
    assert "Posterior samples" in labels
    assert "Posterior mean" in labels
    assert ax.containers[0].lines[0].get_markersize() == 1.0
    assert maps.mixture_ is original_mixture
    plt.close(fig)

    fig, axes = maps.plot_lightcurve(i_wl=0, plot_residuals=True)
    ax, residual_ax = axes
    residual_labels = [container.get_label() for container in residual_ax.containers]
    assert ax.get_xlabel() == ""
    assert residual_ax.get_xlabel() == "Phase Angle"
    assert residual_ax.get_ylabel() == "Residual"
    assert "Residuals" in residual_labels
    assert maps.mixture_ is original_mixture
    plt.close(fig)

    fig, ax = maps.show(i_wl=0, n_samples=0, colorbar=False)
    assert ax.get_title() == ""
    assert ax.images[0].get_array().shape == (1, 3)
    assert ax.images[0].get_extent() == [-180.0, 180.0, -90.0, 90.0]
    assert ax.get_xlabel() == "Longitude (deg)"
    assert ax.get_ylabel() == "Latitude (deg)"
    assert list(ax.get_xticks()) == [-180.0, -90.0, 0.0, 90.0, 180.0]
    assert list(ax.get_yticks()) == [-90.0, -45.0, 0.0, 45.0, 90.0]
    plt.close(fig)

    fig, axes = maps.show(i_wl=0, n_samples=2, random_state=0, colorbar=False)
    assert len(axes) == 2
    assert axes[0].get_title() == ""
    assert axes[1].get_title() == ""
    assert fig.subplotpars.wspace == 0.0
    assert fig.subplotpars.hspace == 0.0
    plt.close(fig)

    fig, axes = maps.show(i_wl=0, n_samples=5, random_state=0, colorbar=False)
    assert len(axes) == 6
    assert sum(ax.get_visible() for ax in axes) == 5
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[1])
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[3])
    assert all(ax.get_xlim() == axes[0].get_xlim() for ax in axes[:5])
    assert all(ax.get_ylim() == axes[0].get_ylim() for ax in axes[:5])
    assert len({ax.images[0].get_clim() for ax in axes[:5]}) == 1
    assert np.isclose(axes[0].get_position().y0, axes[3].get_position().y1)
    assert axes[4].get_title() == ""
    plt.close(fig)

    maps.pc_scores = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    fig, ax = maps.plot_pc_projection(upsample=1, extrapolate=False)
    assert ax.get_title() == "PC1 and PC2 Overlay"
    plt.close(fig)

    maps.regional_spectra = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
        ],
        dtype=float,
    )
    maps.regional_spectra_std = np.full((2, 2), 0.01, dtype=float)
    maps.regional_spectra_cov = np.zeros((2, 2, 2), dtype=float)
    maps.labels = np.array([-1, 0, 0], dtype=int)

    fig, ax, pcm, cb = maps.plot_labels(colorbar=False)
    assert cb is None
    assert ax.get_xlabel() == "Longitude (deg)"
    plt.close(fig)

    maps.regional_spectra = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
        ],
        dtype=float,
    )
    maps.regional_spectra_std = np.full((3, 2), 0.01, dtype=float)
    maps.regional_spectra_cov = np.zeros((3, 2, 2), dtype=float)
    maps.labels = np.array([-1, 1, 1], dtype=int)
    fig, ax, pcm, cb = maps.plot_labels(colorbar=True)
    assert [tick.get_text() for tick in cb.ax.get_yticklabels()] == [
        "Background",
        "Region 1",
        "Region 2",
    ]
    plt.close(fig)

    axes = maps.plot_spectra()
    assert axes[0].get_title() == "Recovered Regional Spectra"
    plt.close(axes[0].figure)


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_rotmaps_joint_wavelength_posterior_matches_independent_diagonal_depth_prior():
    from spectralmap.bayesian_linalg import gaussian_linear_posterior
    from spectralmap.core import LightCurveData
    from spectralmap.rotational import RotMaps

    class _FakeMap:
        def __init__(self):
            self.mask_2d = np.ones((1, 3), dtype=bool)
            self.mask_1d = self.mask_2d.ravel()
            self.lat = np.zeros((1, 3), dtype=float)
            self.lon = np.array([[-60.0, 0.0, 60.0]], dtype=float)
            self.lat_flat = self.lat.ravel()
            self.lon_flat = self.lon.ravel()

        def design_matrix(self, theta):
            x = np.cos(np.deg2rad(theta))
            return np.column_stack((np.ones_like(x), x))

        def intensity_design_matrix(self, projection="rect"):
            return np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=float,
            )

    theta = np.array([0.0, 90.0, 180.0, 270.0])
    x = np.cos(np.deg2rad(theta))
    flux = np.vstack([1.0 + 0.25 * x, 1.0 - 0.15 * x])
    flux_err = np.full_like(flux, 0.05)
    data = LightCurveData(theta=theta, flux=flux, flux_err=flux_err, wl=np.array([1.0, 2.0]))

    maps = RotMaps(map_res=3, verbose=False)
    fake = _FakeMap()
    maps._make_map = lambda ydeg, inc: fake

    joint = maps.solve_joint_wavelength_posterior(
        data,
        inc=70.0,
        ydeg=1,
        spatial_length_scale=1.0,
        depth_length_scale=None,
        depth_jitter=0.0,
    )

    A_fit = fake.design_matrix(theta)[:, 1:]
    for i_wl in range(2):
        prior_cov = joint["joint_prior_cov"][i_wl:i_wl + 1, i_wl:i_wl + 1]
        independent = gaussian_linear_posterior(
            A_fit,
            flux[i_wl] - 1.0,
            sigma_y=flux_err[i_wl],
            prior_covariance=prior_cov,
        )
        assert np.allclose(joint["coeff_mu"][i_wl, 1:], independent["posterior_mean"])
        assert np.allclose(joint["coeff_cov"][i_wl, 1:, 1:], independent["posterior_cov"])


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_rotmaps_joint_wavelength_posterior_couples_nearby_depths():
    from spectralmap.core import LightCurveData
    from spectralmap.rotational import RotMaps

    class _FakeMap:
        mask_2d = np.ones((1, 3), dtype=bool)
        mask_1d = mask_2d.ravel()
        lat = np.zeros((1, 3), dtype=float)
        lon = np.array([[-60.0, 0.0, 60.0]], dtype=float)
        lat_flat = lat.ravel()
        lon_flat = lon.ravel()

        def design_matrix(self, theta):
            x = np.cos(np.deg2rad(theta))
            return np.column_stack((np.ones_like(x), x))

        def intensity_design_matrix(self, projection="rect"):
            return np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)

    theta = np.array([0.0, 90.0, 180.0, 270.0])
    x = np.cos(np.deg2rad(theta))
    flux = np.vstack([1.0 + 0.4 * x, np.ones_like(x)])
    flux_err = np.vstack([np.full(theta.size, 0.04), np.full(theta.size, 10.0)])
    data = LightCurveData(theta=theta, flux=flux, flux_err=flux_err, wl=np.array([1.0, 1.05]))

    maps = RotMaps(map_res=3, verbose=False)
    maps._make_map = lambda ydeg, inc: _FakeMap()
    coupled = maps.solve_joint_wavelength_posterior(
        data,
        inc=70.0,
        ydeg=1,
        spatial_length_scale=1.0,
        depth_length_scale=10.0,
    )

    assert coupled["depth_cov_prior"][0, 1] > 0.9 * coupled["depth_cov_prior"][0, 0]
    assert abs(coupled["joint_free_coeff_cov"][0, 1]) > 0.0
    assert abs(coupled["coeff_mu"][1, 1]) > 1e-4


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
def test_lightcurvedata_preserves_flux_units_by_default():
    from spectralmap.core import LightCurveData

    flux = np.array([[2.0, 4.0, 6.0]], dtype=float)
    flux_err = np.array([[0.2, 0.4, 0.6]], dtype=float)

    data = LightCurveData(
        theta=np.array([0.0, 90.0, 180.0]),
        flux=flux,
        flux_err=flux_err,
    )

    assert np.allclose(data.flux, flux)
    assert np.allclose(data.flux_err, flux_err)
    assert np.allclose(data.amplitude, np.ones(1))


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_lightcurvedata_normalize_is_opt_in():
    from spectralmap.core import LightCurveData

    flux = np.array([[2.0, 4.0, 6.0]], dtype=float)
    flux_err = np.array([[0.2, 0.4, 0.6]], dtype=float)

    data = LightCurveData(
        theta=np.array([0.0, 90.0, 180.0]),
        flux=flux,
        flux_err=flux_err,
        normalize=True,
    )

    assert np.allclose(data.amplitude, np.array([4.0]))
    assert np.allclose(data.flux, flux / 4.0)
    assert np.allclose(data.flux_err, flux_err / 4.0)


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_make_maps_defaults_to_rect_projection():
    from spectralmap.rotational import make_maps

    maps = make_maps(verbose=False)
    map_obj = maps._make_map(ydeg=2, inc=85)

    assert maps.projection == "rect"
    assert map_obj.default_projection == "rect"


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_make_maps_accepts_projection_override():
    from spectralmap.rotational import make_maps

    maps = make_maps(projection="moll", verbose=False)
    map_obj = maps._make_map(ydeg=2, inc=85)

    assert maps.projection == "moll"
    assert map_obj.default_projection == "moll"


@pytest.mark.skipif(not _starry_available(), reason="starry is unavailable")
def test_eclipsemaps_replaced_marginalized_maps_api():
    from spectralmap.eclipse import EclipseMaps

    assert hasattr(EclipseMaps, "marginalize")
    assert not hasattr(EclipseMaps, "marginalized_maps")
