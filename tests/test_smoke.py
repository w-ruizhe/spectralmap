def test_import():
    import spectralmap  # noqa: F401
    import starry
    import numpy as np

    map = starry.Map(ydeg=4)
    map.design_matrix(theta=np.linspace(0, 360, 10))


def test_make_map_accepts_limb_darkening_args():
    from spectralmap.mapping import make_map

    map_obj = make_map(mode="rotational", ydeg=2, udeg=2, u=[0.3, 0.1], inc=70)

    assert map_obj.map.udeg == 2
    assert np.isclose(map_obj.map[1], 0.3)
    assert np.isclose(map_obj.map[2], 0.1)


def test_make_maps_passes_limb_darkening_args():
    from spectralmap.mapping import make_maps

    maps = make_maps(mode="rotational", ydegs=np.array([2, 3]), udeg=1, u=[0.25], inc=85)
    map_obj = maps.get_map_for_ydeg(2)

    assert map_obj.map.udeg == 1
    assert np.isclose(map_obj.map[1], 0.25)