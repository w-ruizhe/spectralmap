from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, zoom

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


def plot_mollweide_rgb(pc1_scores, pc2_scores, mask_2d, upsample=4, extrapolate=True, ax=None):
    mask_2d = np.asarray(mask_2d, dtype=bool)
    h, w = mask_2d.shape

    def norm_01(x):
        x = np.nan_to_num(x)
        vmin, vmax = x.min(), x.max()
        if vmin == vmax:
            return np.zeros_like(x)
        return (x - vmin) / (vmax - vmin)

    r_val = norm_01(pc1_scores)
    b_val = norm_01(pc2_scores)

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
    ax.set_title("PC1 and PC2 Overlay (Seamless Edge)", pad=15)

    return fig, ax


def plot_mollweide_labels(
    labels_masked,
    moll_mask,
    map_res=None,
    cmap=None,
    names=None,
    ax=None,
    show_grid=True,
    hide_ticks=True,
    extrapolate=True,
    add_colorbar=True,
    cax=None,
):
    moll_mask = np.asarray(moll_mask, dtype=bool)
    if map_res is None:
        map_res = moll_mask.shape[0] if moll_mask.ndim == 2 else int(np.sqrt(moll_mask.size))
    mask_flat = moll_mask.ravel()

    full = np.full(mask_flat.size, np.nan)
    labels_masked = np.asarray(labels_masked)
    if labels_masked.size == mask_flat.sum():
        full[mask_flat] = labels_masked
    elif labels_masked.size == mask_flat.size:
        full[:] = labels_masked
    else:
        raise ValueError("labels size must match mask.sum() or mask.size")

    img = full.reshape(map_res, map_res)

    lon_edges = np.linspace(-np.pi, np.pi, map_res + 1)
    lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, map_res + 1)
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
    min_label = int(labels_int.min())
    max_label = int(labels_int.max())
    tick_vals = np.arange(min_label, max_label + 1)
    bounds = np.arange(min_label - 0.5, max_label + 1.5, 1.0)
    n_bins = len(bounds) - 1

    if cmap is None:
        cmap = get_cmap(n_bins)
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
    if add_colorbar:
        if cax is None:
            cb = fig.colorbar(pcm, ax=ax, pad=0.08)
        else:
            cb = fig.colorbar(pcm, cax=cax)
        cb.set_label("Cluster label")
        cb.set_ticks(tick_vals)
        if names is not None:
            cb.set_ticklabels(names)

    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(True, linestyle="--", color="gray", alpha=0.6)
    ax.set_ylabel("Latitude (deg)", fontsize=8)
    ax.set_xlabel("Longitude (deg)", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    if cb is not None:
        if names is not None:
            cb.ax.set_yticklabels(names, fontsize=7)
        cb.ax.yaxis.set_tick_params(length=0)
        cb.outline.set_edgecolor("black")

    return fig, ax, pcm, cb