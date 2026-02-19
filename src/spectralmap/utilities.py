import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from scipy import constants

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


def plot_mollweide_projection(
    values_by_wavelength,
    moll_mask,
    map_res=None,
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
    map_res : int, optional
        Map resolution. If None, inferred from moll_mask.
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
    fig, ax, pcm, cb
        Figure, axis, contour set, and colorbar.
    """
    if map_res is None:
        if np.asarray(moll_mask).ndim == 2:
            map_res = int(np.asarray(moll_mask).shape[0])
        else:
            map_res = int(np.sqrt(np.asarray(moll_mask).size))

    full = expand_moll_values(values_by_wavelength, moll_mask)
    img = full.reshape(map_res, map_res)

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

    if hide_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    return fig, ax, pcm, cb

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