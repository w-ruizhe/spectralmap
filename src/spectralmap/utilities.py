from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from matplotlib.patches import Patch
from scipy.ndimage import zoom, distance_transform_edt
from matplotlib import colors as mcolors

def plot_mollweide_rgb_perfect_edges(pc1_scores, pc2_scores, mask_2d, upsample=4, extrapolate=True):
    mask_2d = np.asarray(mask_2d, dtype=bool)
    H, W = mask_2d.shape
    
    # 1. Safely normalize to [0, 1]
    def norm_01(x):
        x = np.nan_to_num(x)
        vmin, vmax = x.min(), x.max()
        if vmin == vmax: 
            return np.zeros_like(x)
        return (x - vmin) / (vmax - vmin)
    
    r_val = norm_01(pc1_scores)
    b_val = norm_01(pc2_scores)
    
    # 2. Fill baseline channels
    R = np.zeros((H, W))
    G = np.full((H, W), 0.15) # Baseline green so it's never pure black
    B = np.zeros((H, W))
    
    R[mask_2d] = r_val
    B[mask_2d] = b_val
    
    # --- 3. Extrapolate colors to fill the ENTIRE rectangle ---
    # We flood the empty space with the nearest valid colors.
    if extrapolate:
        invalid = ~mask_2d
        if np.any(invalid):
            # Finds the row/col indices of the nearest valid pixel (where invalid == 0)
            nearest_idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
            R[invalid] = R[nearest_idx[0, invalid], nearest_idx[1, invalid]]
            G[invalid] = G[nearest_idx[0, invalid], nearest_idx[1, invalid]]
            B[invalid] = B[nearest_idx[0, invalid], nearest_idx[1, invalid]]

    # --- 4. Smooth the filled color channels ---
    # order=3 is bicubic (smoother than bilinear). Clip to keep RGB in bounds.
    R_sm = np.clip(zoom(R, upsample, order=3), 0.0, 1.0)
    G_sm = np.clip(zoom(G, upsample, order=3), 0.0, 1.0)
    B_sm = np.clip(zoom(B, upsample, order=3), 0.0, 1.0)
    
    # Set Alpha to 1.0 EVERYWHERE. Matplotlib's axis will naturally clip the map into an ellipse.
    A_sm = np.ones_like(R_sm)
    
    # 5. Coordinate Grid
    H_new, W_new = R_sm.shape
    lon_edges = np.linspace(-np.pi, np.pi, W_new + 1)
    lat_edges = np.linspace(-np.pi/2, np.pi/2, H_new + 1)
    Lon, Lat = np.meshgrid(lon_edges, lat_edges)
    
    # Flatten colors
    rgba_colors = np.dstack((R_sm, G_sm, B_sm, A_sm)).reshape(-1, 4)
    
    # 6. Plotting
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='mollweide')
    
    dummy_data = np.zeros((H_new, W_new))
    mesh = ax.pcolormesh(
        Lon, Lat, dummy_data, 
        shading='flat', 
        zorder=2
    )
    
    mesh.set_array(None)              
    mesh.set_facecolors(rgba_colors)  
    mesh.set_edgecolors('none')       
    
    ax.grid(True, alpha=0.3, linewidth=0.5, zorder=3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    legend_elements = [
        Patch(facecolor=(1.0, 0.15, 0.0), label='High PC1 (Red)'),
        Patch(facecolor=(0.0, 0.15, 1.0), label='High PC2 (Blue)'),
        Patch(facecolor=(1.0, 0.15, 1.0), label='High Both (Magenta)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    ax.set_title("PC1 and PC2 Overlay (Seamless Edge)", pad=15)

    return fig, ax


def bin_flux_by_theta(
    theta: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 50,
    flux_err: np.ndarray | None = None,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Bin flux in rotational phase degrees (theta) from 0 to 360.

    Parameters
    ----------
    theta : np.ndarray
        1D phase array of length n_time.
    flux : np.ndarray
        Flux array with shape (..., n_time), e.g. (n_wavelength, n_time).
    n_bins : int, optional
        Number of phase bins.
    flux_err : np.ndarray, optional
        Uncertainty array with same shape as ``flux``. If provided, uses
        inverse-variance weighted means per bin.
    min_count : int, optional
        Minimum samples required in a bin to report a value.

    Returns
    -------
    theta_centers : np.ndarray
        Bin centers, shape (n_bins,).
    flux_binned : np.ndarray
        Binned flux, shape (..., n_bins).
    flux_err_binned : np.ndarray or None
        Binned uncertainty, shape (..., n_bins), or None if ``flux_err`` is None.
    counts : np.ndarray
        Number of samples per bin, shape (n_bins,).
    """
    theta = np.asarray(theta, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float)

    if theta.ndim != 1:
        raise ValueError("theta must be 1D.")
    if flux.shape[-1] != theta.size:
        raise ValueError(
            f"flux last dimension ({flux.shape[-1]}) must match theta length ({theta.size})."
        )
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")

    phase = np.mod(theta, 360.0)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.digitize(phase, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)

    lead_shape = flux.shape[:-1]
    flux_flat = flux.reshape(-1, theta.size)
    flux_binned_flat = np.full((flux_flat.shape[0], n_bins), np.nan, dtype=float)

    flux_err_binned_flat = None
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        if flux_err.shape != flux.shape:
            raise ValueError("flux_err must have the same shape as flux.")
        flux_err_flat = flux_err.reshape(-1, theta.size)
        flux_err_binned_flat = np.full((flux_flat.shape[0], n_bins), np.nan, dtype=float)

    for j in range(n_bins):
        mask = bin_idx == j
        if np.sum(mask) < min_count:
            continue

        y = flux_flat[:, mask]
        if flux_err is None:
            flux_binned_flat[:, j] = np.nanmean(y, axis=1)
        else:
            ye = flux_err_flat[:, mask]
            w = 1.0 / np.maximum(ye, 1e-30) ** 2
            valid = np.isfinite(y) & np.isfinite(w) & (w > 0)
            w_sum = np.sum(np.where(valid, w, 0.0), axis=1)

            num = np.sum(np.where(valid, w * y, 0.0), axis=1)
            good = w_sum > 0
            flux_binned_flat[good, j] = num[good] / w_sum[good]
            flux_err_binned_flat[good, j] = np.sqrt(1.0 / w_sum[good])

    flux_binned = flux_binned_flat.reshape(lead_shape + (n_bins,))
    if flux_err_binned_flat is None:
        flux_err_binned = None
    else:
        flux_err_binned = flux_err_binned_flat.reshape(lead_shape + (n_bins,))

    return centers, flux_binned, flux_err_binned, counts

def intensity_to_temperature(intensity: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """
    Convert spectral intensity to brightness temperature using the inverse Planck function.
    
    Parameters
    ----------
    intensity : np.ndarray
        Spectral intensity in SI units (W/m^2/sr/m).
    wavelength : np.ndarray
        Wavelength in meters.
        
    Returns
    -------
    np.ndarray
        Brightness temperature in Kelvin.
    """
    h = constants.h
    c = constants.c
    k = constants.k
    
    # Avoid division by zero warnings by using a tiny lower bound if needed, 
    # though physically intensity must be > 0.
    
    val = (h * c) / (wavelength * k)
    # Planck: I = (2hc^2/lambda_fix^5) * 1/(exp(hc/lambda_fix k T) - 1)
    # exp(hc/lambda_fix k T) - 1 = 2hc^2 / (lambda_fix^5 * I)
    # T = (hc/lambda_fix k) / ln(1 + 2hc^2/(lambda_fix^5 * I))
    
    term = (2 * h * c**2) / (wavelength**5 * intensity)
    return val / np.log(1 + term)



def expand_moll_values(values: np.ndarray, moll_mask: np.ndarray, fill_value=np.nan) -> np.ndarray:
    """Expand masked pixel values back to full map pixel length.

    Parameters
    ----------
    values : np.ndarray
        Array whose last dimension is the number of valid moll pixels.
    moll_mask : np.ndarray
        Boolean mask over full pixels (flat or map_res x map_res).
    fill_value : float
        Fill value for invalid pixels.
    """
    values = np.asarray(values)
    mask = np.asarray(moll_mask, dtype=bool).ravel()

    if values.shape[-1] != int(mask.sum()):
        raise ValueError(
            f"values last dimension ({values.shape[-1]}) does not match number of valid pixels ({int(mask.sum())})."
        )

    out_shape = values.shape[:-1] + (mask.size,)
    out = np.full(out_shape, fill_value, dtype=float)
    out[..., mask] = values
    return out


def logsumexp(logw):
    m = np.max(logw)
    return m + np.log(np.sum(np.exp(logw - m)))


def gamma_log_prior_lambda(lam, a, b):
    """
    Log of an (unnormalized) Gamma(a, b) prior over lambda, vectorized.

    Parameters
    ----------
    lam : float or array-like
        Lambda values (>0).
    a, b : float
        Shape and rate parameters (>0).

    Returns
    -------
    out : float or np.ndarray
        Log prior values (same shape as lam). Invalid entries get -inf.
        If lam is scalar, returns a scalar float.
    """
    lam_arr = np.asarray(lam, dtype=float)

    # Invalid hyperparameters -> all -inf (match original behavior)
    if (a <= 0) or (b <= 0):
        out = np.full(lam_arr.shape, -np.inf, dtype=float)
        return float(out) if lam_arr.ndim == 0 else out

    out = np.full(lam_arr.shape, -np.inf, dtype=float)
    ok = lam_arr > 0
    out[ok] = (a - 1.0) * np.log(lam_arr[ok]) - b * lam_arr[ok]

    return float(out) if lam_arr.ndim == 0 else out

def solid_angle_weights(lat, lon):
    """Compute pixel solid angle weights for a lat/lon grid."""
    lat_u = np.unique(lat)
    lon_u = np.unique(lon)
    dlat = np.deg2rad(np.median(np.diff(np.sort(lat_u)))) if lat_u.size > 1 else np.deg2rad(180.0)
    dlon = np.deg2rad(np.median(np.diff(np.sort(lon_u)))) if lon_u.size > 1 else np.deg2rad(360.0)

    lat_r = np.deg2rad(lat)
    lat_lo = np.clip(lat_r - 0.5 * dlat, -0.5 * np.pi, 0.5 * np.pi)
    lat_hi = np.clip(lat_r + 0.5 * dlat, -0.5 * np.pi, 0.5 * np.pi)
    w_pix = dlon * (np.sin(lat_hi) - np.sin(lat_lo))
    w_pix = np.maximum(w_pix, 0.0)
    return w_pix

def log_delta_lambda(lambdas):
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1 or lambdas.size < 2:
        raise ValueError("lambdas must be 1D with at least 2 points.")
    if not np.all(np.diff(lambdas) > 0):
        raise ValueError("lambdas must be strictly increasing.")

    d = np.empty_like(lambdas)
    d[1:-1] = 0.5 * (lambdas[2:] - lambdas[:-2])   # centered widths
    d[0]    = lambdas[1] - lambdas[0]              # edge widths
    d[-1]   = lambdas[-1] - lambdas[-2]
    return np.log(d)


def plot_mollweide_labels(
    labels_masked,          # (n_valid_pix,) or (map_res*map_res,) if already full
    moll_mask,              # full-grid mask, shape (map_res,map_res) or flat
    map_res=None,
    cmap=None,
    names=None,
    ax=None,
    show_grid=True,
    hide_ticks=True,
    extrapolate=True
):
    moll_mask = np.asarray(moll_mask, dtype=bool)
    if map_res is None:
        map_res = moll_mask.shape[0] if moll_mask.ndim == 2 else int(np.sqrt(moll_mask.size))
    mask_flat = moll_mask.ravel()

    # Expand masked labels back to full grid
    full = np.full(mask_flat.size, np.nan)
    labels_masked = np.asarray(labels_masked)
    if labels_masked.size == mask_flat.sum():
        full[mask_flat] = labels_masked
    elif labels_masked.size == mask_flat.size:
        full[:] = labels_masked
    else:
        raise ValueError("labels size must match mask.sum() or mask.size")

    img = full.reshape(map_res, map_res)

    # lon/lat edges for pcolormesh (edges, not centers)
    lon_edges = np.linspace(-np.pi, np.pi, map_res + 1)
    lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, map_res + 1)
    lon2d, lat2d = np.meshgrid(lon_edges, lat_edges)

    if ax is None:
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")
    else:
        fig = ax.figure

    # Discrete norm: bins centered on integers
    # valid labels are assumed to be integers -1..K-1
    finite = np.isfinite(img)
    if not np.any(finite):
        raise ValueError("No finite labels to plot.")
    K = int(np.nanmax(img)) + 1
    N = K + 1
    bounds = np.arange(-1.5, K + 0.5, 1.0)
    if cmap is None:
        cmap = plt.get_cmap("tab20", K)
        norm = mcolors.BoundaryNorm(bounds, cmap.K)
    else:
        norm = mcolors.BoundaryNorm(bounds, len(cmap.colors))

    # Mask outside footprint
    nearest_idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
    filled = img.copy()
    if extrapolate:
        filled[~finite] = filled[tuple(nearest_idx[:, ~finite])]
    
    pcm = ax.pcolormesh(
        lon2d, lat2d, filled,
        cmap=cmap, norm=norm,
        shading="flat",   # crisp pixels
        antialiased=False # reduces edge halos
    )

    if show_grid:
        ax.grid(True, alpha=0.3)

    cb = fig.colorbar(pcm, ax=ax, pad=0.08)
    cb.set_label("Cluster label")

    # Put ticks at integer centers
    cb.set_ticks(np.arange(N)-1)
    if names is not None:
        cb.set_ticklabels(names)

    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.grid(True, linestyle='--', color='gray', alpha=0.6)
    ax.set_ylabel("Latitude (deg)", fontsize=8)
    ax.set_xlabel("Longitude (deg)", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)

    cb.ax.set_yticklabels(names, fontsize=7)
    cb.ax.yaxis.set_tick_params(length=0)
    cb.outline.set_edgecolor('black')

    return fig, ax, pcm, cb
