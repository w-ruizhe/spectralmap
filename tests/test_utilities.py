import numpy as np
import pytest

try:
    from spectralmap.utilities import bin_flux
except ModuleNotFoundError as exc:
    if exc.name in {"spectralmap", "sklearn"}:
        bin_flux = None
    else:
        raise


pytestmark = pytest.mark.skipif(
    bin_flux is None,
    reason="spectralmap utilities import requires optional dependencies not installed in this environment",
)


def test_bin_flux_shapes_and_counts():
    time = np.linspace(0.0, 9.0, 10)
    flux = np.vstack([time, 2.0 * time])

    centers, flux_binned, flux_err_binned, counts = bin_flux(time, flux, n_bins=5)

    assert centers.shape == (5,)
    assert flux_binned.shape == (2, 5)
    assert flux_err_binned is None
    assert counts.shape == (5,)
    assert np.all(counts == 2)


def test_bin_flux_weighted_mean_and_error():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    flux = np.array([[1.0, 3.0, 10.0, 14.0]])
    flux_err = np.array([[1.0, 1.0, 2.0, 2.0]])

    _, flux_binned, flux_err_binned, counts = bin_flux(
        time,
        flux,
        n_bins=2,
        flux_err=flux_err,
    )

    # First bin: weighted mean of [1, 3] with equal weights -> 2
    assert np.isclose(flux_binned[0, 0], 2.0)
    # Error: sqrt(1 / (1 + 1))
    assert np.isclose(flux_err_binned[0, 0], np.sqrt(0.5))

    # Second bin: weighted mean of [10, 14] with equal weights -> 12
    assert np.isclose(flux_binned[0, 1], 12.0)
    # Error: sqrt(1 / (1/4 + 1/4)) = sqrt(2)
    assert np.isclose(flux_err_binned[0, 1], np.sqrt(2.0))

    assert np.all(counts == np.array([2, 2]))


def test_bin_flux_no_wraparound_behavior():
    # Around a 360-degree-style boundary, this should be treated as linear time,
    # not wrapped phase.
    time = np.array([350.0, 355.0, 360.0, 365.0])
    flux = np.array([[1.0, 2.0, 3.0, 4.0]])

    _, flux_binned, _, counts = bin_flux(time, flux, n_bins=2)

    # Bin 1: [350, 355] -> mean 1.5
    # Bin 2: [360, 365] -> mean 3.5
    assert np.isclose(flux_binned[0, 0], 1.5)
    assert np.isclose(flux_binned[0, 1], 3.5)
    assert np.all(counts == np.array([2, 2]))
