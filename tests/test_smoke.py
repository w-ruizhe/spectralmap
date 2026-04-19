import numpy as np
import pytest


def _starry_has_doppler_map() -> bool:
    try:
        import starry
    except ModuleNotFoundError:
        return False

    return hasattr(starry, "DopplerMap")


def test_import():
    import spectralmap  # noqa: F401
    import starry

    map = starry.Map(ydeg=4)
    map.design_matrix(theta=np.linspace(0, 360, 10))


def test_make_map_accepts_limb_darkening_args():
    from spectralmap.maps import make_map

    map_obj = make_map(mode="rotational", ydeg=2, udeg=2, u=[0.3, 0.1], inc=70)

    assert map_obj.map.udeg == 2
    assert np.isclose(map_obj.map[1], 0.3)
    assert np.isclose(map_obj.map[2], 0.1)


def test_make_maps_passes_limb_darkening_args():
    from spectralmap.maps import make_maps

    maps = make_maps(mode="rotational", ydegs=np.array([2, 3]), udeg=1, u=[0.25], inc=85)
    map_obj = maps.get_map_for_ydeg(2)

    assert map_obj.map.udeg == 1
    assert np.isclose(map_obj.map[1], 0.25)


@pytest.mark.skipif(not _starry_has_doppler_map(), reason="starry.DopplerMap is unavailable")
def test_make_map_accepts_doppler_args():
    from spectralmap.maps import make_map

    wav = np.linspace(643.0, 643.2, 8)
    wav0 = np.linspace(642.9, 643.3, 16)
    map_obj = make_map(
        mode="doppler",
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
def test_make_maps_doppler_not_supported():
    from spectralmap.maps import make_maps

    wav = np.linspace(643.0, 643.2, 8)
    with pytest.raises(NotImplementedError):
        make_maps(mode="doppler", ydegs=np.array([1, 2]), nt=4, wav=wav, veq=1.5e4)