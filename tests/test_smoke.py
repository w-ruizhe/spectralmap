def test_import():
    import spectralmap  # noqa: F401
    import starry
    import numpy as np
    map = starry.map(ydeg=4)
    map = starry.Map(ydeg=4)
    map.design_matrix(theta=np.linspace(0, 360, 10))