from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy.ndimage import distance_transform_edt, zoom, gaussian_filter1d
from spectralmap.utilities import expand_values_with_mask

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.linestyle":'--',
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "figure.figsize": (3.5, 2.5),  # ApJ column width ~3.5 in; double column ~7 in
})

PAPER_FIGURES_DIR = "/Users/rworzger/Library/Mobile Documents/com~apple~CloudDocs/Spatially_Resolved_Atmospheric_Retrievals_of_Luhman_16B_with_JWST/figures/"
ECLIPSE_PAPER_FIGURES_DIR = '/Users/rworzger/Library/Mobile Documents/com~apple~CloudDocs/EclipseMapping_Paper/figures'
BG_COLOR = "#E8E8E8"
tab10 = plt.get_cmap("tab10").colors
COLOR_LIST = [BG_COLOR] + [tab10[i] for i in [0, 3, 4, 1, 2, 5, 6, 7, 8, 9]]


def get_cmap(n_colors: int):
    if n_colors <= len(COLOR_LIST):
        return mcolors.ListedColormap(COLOR_LIST[:n_colors])
    raise ValueError(
        f"Requested {n_colors} colors, but only {len(COLOR_LIST)} are available in the custom colormap."
    )


def plot_pc_projection(maps, upsample=4, extrapolate=False, ax=None):
    mask_1d = maps.mask_1d
    mask_2d = maps.mask_2d
    pc_full = expand_values_with_mask(maps.pc_scores.T, mask_1d).T
    pc_grid = pc_full.reshape(mask_2d.shape + (maps.pc_scores.shape[1],))
    h, w = mask_2d.shape

    def norm_01(x):
        finite = np.isfinite(x)
        if not np.any(finite):
            return np.zeros_like(x, dtype=float)
        vmin, vmax = x[finite].min(), x[finite].max()
        out = np.zeros_like(x, dtype=float)
        if vmin == vmax:
            return out
        out[finite] = (x[finite] - vmin) / (vmax - vmin)
        return out

    if not np.any(mask_2d):
        raise ValueError("No finite PC scores to plot.")

    r_val = norm_01(pc_grid[..., 0][mask_2d])
    b_val = norm_01(pc_grid[..., 1][mask_2d])

    r_chan = np.zeros((h, w))
    g_chan = np.full((h, w), 0.15)
    b_chan = np.zeros((h, w))

    r_chan[mask_2d] = r_val
    b_chan[mask_2d] = b_val

    if extrapolate:
        invalid = ~mask_2d
        if np.any(invalid):
            nearest_idx = distance_transform_edt(
                invalid, return_distances=False, return_indices=True
            )
            r_chan[invalid] = r_chan[nearest_idx[0, invalid], nearest_idx[1, invalid]]
            g_chan[invalid] = g_chan[nearest_idx[0, invalid], nearest_idx[1, invalid]]
            b_chan[invalid] = b_chan[nearest_idx[0, invalid], nearest_idx[1, invalid]]

    r_sm = np.clip(zoom(r_chan, upsample, order=3), 0.0, 1.0)
    g_sm = np.clip(zoom(g_chan, upsample, order=3), 0.0, 1.0)
    b_sm = np.clip(zoom(b_chan, upsample, order=3), 0.0, 1.0)
    a_sm = np.ones_like(r_sm)

    h_new, w_new = r_sm.shape
    lon_edges = np.linspace(-np.pi, np.pi, w_new + 1)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, h_new + 1)
    lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

    rgba_colors = np.dstack((r_sm, g_sm, b_sm, a_sm)).reshape(-1, 4)

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection="mollweide")
    else:
        fig = ax.figure

    dummy_data = np.zeros((h_new, w_new))
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        dummy_data,
        shading="flat",
        zorder=2,
        antialiased=False,
        linewidth=0,
        edgecolors="none",
        rasterized=True,
    )

    mesh.set_array(None)
    mesh.set_facecolors(rgba_colors)
    mesh.set_edgecolors("none")
    mesh.set_linewidth(0)

    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    legend_elements = [
        Patch(facecolor=(1.0, 0.15, 0.0), label="High PC1"),
        Patch(facecolor=(0.0, 0.15, 1.0), label="High PC2"),
        Patch(facecolor=(1.0, 0.15, 1.0), label="High Both"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    ax.set_title("PC1 and PC2 Overlay", pad=15)

    return fig, ax


def plot_labels(
    maps,
    ax=None,
    show_grid=True,
    hide_ticks=True,
    extrapolate=False,
    colorbar=True,
    cax=None,
):
    N = maps.regional_spectra.shape[0]
    cluster_names = ["Background"] + [f"Region {i+1}" for i in range(N - 1)]

    mask_1d = maps.mask_1d
    mask_2d = maps.mask_2d
    labels_full = expand_values_with_mask(maps.labels, mask_1d)
    img = labels_full.reshape(mask_2d.shape)

    lon_edges = np.linspace(-np.pi, np.pi, mask_2d.shape[1] + 1)
    lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, mask_2d.shape[0] + 1)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    if ax is None:
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")
    else:
        fig = ax.figure

    finite = np.isfinite(img)
    if not np.any(finite):
        raise ValueError("No finite labels to plot.")
    labels_int = np.asarray(img[finite], dtype=int)
    min_label = -1
    max_label = N - 2
    if np.any((labels_int < min_label) | (labels_int > max_label)):
        raise ValueError(
            f"Labels must be in the range [{min_label}, {max_label}] for {N} regional spectra."
        )
    tick_vals = np.arange(min_label, max_label + 1)
    bounds = np.arange(min_label - 0.5, max_label + 1.5, 1.0)

    cmap = get_cmap(N)
    cmap_size = int(getattr(cmap, "N", len(getattr(cmap, "colors", []))))
    norm = mcolors.BoundaryNorm(bounds, cmap_size)

    nearest_idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
    filled = img.copy()
    if extrapolate:
        filled[~finite] = filled[tuple(nearest_idx[:, ~finite])]

    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        filled,
        cmap=cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
        linewidth=0,
        edgecolors="none",
        rasterized=True,
    )

    if show_grid:
        ax.grid(True, alpha=0.3)

    cb = None
    if colorbar:
        if cax is None:
            cb = fig.colorbar(pcm, ax=ax, pad=0.08)
        else:
            cb = fig.colorbar(pcm, cax=cax)
        cb.set_label("Cluster label")
        cb.set_ticks(tick_vals)
        if cluster_names is not None:
            cb.set_ticklabels(cluster_names)

    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(True, linestyle="--", color="gray", alpha=0.6)
    ax.set_ylabel("Latitude (deg)", fontsize=8)
    ax.set_xlabel("Longitude (deg)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    if cb is not None:
        if cluster_names is not None:
            cb.ax.set_yticklabels(cluster_names, fontsize=7)
        cb.ax.yaxis.set_tick_params(length=0)
        cb.outline.set_edgecolor("black")

    return fig, ax, pcm, cb


def plot_spectra(maps, axes=None):

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    regional_spectra = maps.regional_spectra
    regional_spectra_std = maps.regional_spectra_std
    N = maps.regional_spectra.shape[0]

    color_list = COLOR_LIST[:N]
    cluster_names = ["Background"] + [f"Region {i+1}" for i in range(N - 1)]
    wl = maps.data.wl


    background_flux = regional_spectra[0]
    # Smooth spectra with a Gaussian filter before plotting.
    gaussian_sigma = 1

    # Plot the recovered spectra for each region
    for i in range(1, N):
        mean_flux = regional_spectra[i]
        error_flux = regional_spectra_std[i] # Fixed variable name
        color = color_list[i]
        label = cluster_names[i]

        axes[0].plot(wl, mean_flux, label=f"{label}", color=color, linewidth=1.2, linestyle='--')
        axes[0].fill_between(wl,
                            mean_flux - error_flux,
                            mean_flux + error_flux,
                            alpha=0.25, color=color)
        axes[1].plot(wl, gaussian_filter1d(mean_flux/background_flux, sigma=gaussian_sigma, mode='nearest'), label=f"{label}", color=color,
                    linewidth=1.2, linestyle='--')
        axes[1].fill_between(wl,
                            gaussian_filter1d((mean_flux - error_flux)/background_flux, sigma=gaussian_sigma, mode='nearest'),
                            gaussian_filter1d((mean_flux + error_flux)/background_flux, sigma=gaussian_sigma, mode='nearest'),
                            alpha=0.25, color=color)

    # Overlay the range of the observed time-series variability
    time_series = np.sort(maps.data.flux.T, axis=0) * maps.data.amplitude
    axes[0].fill_between(wl, time_series[0, :],
                time_series[-1, :], color='black', alpha=0.10, zorder=0, label="Observed Range")

    axes[1].fill_between(wl, gaussian_filter1d(time_series[0, :]/background_flux, sigma=gaussian_sigma, mode='nearest'),
                gaussian_filter1d(time_series[-1, :]/background_flux, sigma=gaussian_sigma, mode='nearest'), color='black', alpha=0.10, zorder=0)

    # Formatting
    axes[1].set_xlabel(r"Wavelength ($\mu$m)", fontsize=9)
    axes[0].set_ylabel(r"Flux (W/m$^2$/$\mu$m)", fontsize=9)
    axes[0].set_title("Recovered Regional Spectra")
    axes[0].legend(loc='upper right', ncol=2)
    # axes[0].set_xscale("log")
    # axes[1].set_ylim(0.8, 1.2)
    axes[1].set_ylabel(r"F/$F_{\rm{mean}}$", fontsize=9)

    return axes
