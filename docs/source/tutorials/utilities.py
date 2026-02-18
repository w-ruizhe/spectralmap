import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from spectralmap.mapping import expand_moll_values


def plot_mollweide_projection(
    values_by_wavelength,
    moll_mask,
    iw=0,
    map_res=None,
    wl=None,
    cbar_label="Flux density",
    cmap="inferno",
    levels=200,
    smooth_boundary=True,
    hide_ticks=True,
    show_grid=True,
    ax=None,
):
    """Plot a starry-like Mollweide map from masked pixel values.

    Parameters
    ----------
    values_by_wavelength : ndarray
        Array shaped (n_wavelength, n_valid_pixels).
    moll_mask : ndarray
        Boolean mask of full map footprint (flat or map_res x map_res).
    iw : int, optional
        Wavelength index to plot.
    map_res : int, optional
        Map resolution. If None, inferred from moll_mask.
    wl : ndarray, optional
        Wavelength array for title in microns.
    cbar_label : str, optional
        Colorbar label.
    cmap : str, optional
        Matplotlib colormap name.
    levels : int, optional
        Number of contour levels.
    smooth_boundary : bool, optional
        If True, fills NaNs near footprint edge with nearest valid value.
    hide_ticks : bool, optional
        If True, hides x/y tick labels.
    show_grid : bool, optional
        If True, draws projected grid.
    ax : matplotlib axis, optional
        Existing Mollweide axis.

    Returns
    -------
    fig, ax, pcm
        Figure, axis, and contour set.
    """
    if map_res is None:
        if np.asarray(moll_mask).ndim == 2:
            map_res = int(np.asarray(moll_mask).shape[0])
        else:
            map_res = int(np.sqrt(np.asarray(moll_mask).size))

    full = expand_moll_values(values_by_wavelength, moll_mask)
    img = full[iw].reshape(map_res, map_res)

    lon = np.linspace(-np.pi, np.pi, map_res)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, map_res)
    lon2d, lat2d = np.meshgrid(lon, lat)

    if smooth_boundary:
        img_plot = img.copy()
        valid = np.isfinite(img_plot)
        nearest_idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        img_plot[~valid] = img_plot[tuple(nearest_idx[:, ~valid])]
    else:
        img_plot = np.ma.masked_invalid(img)

    if ax is None:
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")
    else:
        fig = ax.figure

    pcm = ax.contourf(lon2d, lat2d, img_plot, levels=levels, cmap=cmap)

    if show_grid:
        ax.grid(True, alpha=0.3)

    cb = fig.colorbar(pcm, ax=ax, pad=0.08)
    cb.set_label(cbar_label)

    if wl is None:
        ax.set_title(f"Mollweide map (wavelength bin {iw})")
    else:
        ax.set_title(f"{wl[iw]: .2f} $\\mu$m")

    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return fig, ax, pcm
