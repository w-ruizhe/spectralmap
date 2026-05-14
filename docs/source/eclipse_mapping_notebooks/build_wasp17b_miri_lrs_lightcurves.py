from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "mast_downloads" / "jw01353-o006_t001_miri_p750l-slitlessprism_x1dints.fits"
OUTPUT = ROOT / "data" / "WASP17b_MIRI_LRS_x1dints_lightcurves.npz"

PERIOD_DAYS = 3.73548546
T0_MJD_TDB = 60016.226452
ECLIPSE_CENTER_MJD_TDB = 60018.09562
ECLIPSE_DURATION_DAYS = 4.42 / 24.0

M_STAR_MSUN = 1.370
R_STAR_RSUN = 1.583
M_PLANET_MJUP = 0.477
R_PLANET_RJUP = 1.921
RP_RS = 0.12472
A_OVER_RSTAR = 7.110
INCLINATION_DEG = 87.217
STELLAR_TEFF_K = 6550.0


def read_x1dints(path):
    times = []
    waves = []
    fluxes = []
    errors = []
    dqs = []
    segments = []
    exposures = []
    int_nums = []

    with fits.open(path, memmap=True) as hdul:
        extract_hdus = [h for h in hdul[1:] if h.name == "EXTRACT1D"]
        for i, hdu in enumerate(extract_hdus):
            data = hdu.data
            n = len(data)
            exposure = 1 if i < 6 else 2
            global_segment = i + 1
            times.append(np.asarray(data["TDB-MID"], dtype=float))
            waves.append(np.asarray(data["WAVELENGTH"], dtype=float))
            fluxes.append(np.asarray(data["FLUX"], dtype=float))
            errors.append(np.asarray(data["FLUX_ERROR"], dtype=float))
            dqs.append(np.asarray(data["DQ"], dtype=np.uint32))
            segments.append(np.full(n, global_segment, dtype=int))
            exposures.append(np.full(n, exposure, dtype=int))
            int_nums.append(np.asarray(data["INT_NUM"], dtype=int))

    time = np.concatenate(times)
    order = np.argsort(time)

    return {
        "time": time[order],
        "wavelength": np.concatenate(waves)[order],
        "flux": np.concatenate(fluxes)[order],
        "error": np.concatenate(errors)[order],
        "dq": np.concatenate(dqs)[order],
        "segment": np.concatenate(segments)[order],
        "exposure": np.concatenate(exposures)[order],
        "int_num": np.concatenate(int_nums)[order],
    }


def weighted_bin(wavelength, flux, error, dq, lo, hi):
    in_bin = (wavelength >= lo) & (wavelength < hi)
    good = in_bin & np.isfinite(flux) & np.isfinite(error) & (error > 0) & (dq == 0)
    if np.count_nonzero(good) < 3:
        good = in_bin & np.isfinite(flux)
    if np.count_nonzero(good) == 0:
        return np.nan, np.nan

    y = flux[good]
    e = error[good]
    if np.all(np.isfinite(e)) and np.all(e > 0):
        w = 1.0 / np.square(e)
        return np.sum(w * y) / np.sum(w), np.sqrt(1.0 / np.sum(w))

    return np.nanmean(y), np.nanstd(y) / np.sqrt(max(1, y.size))


def bin_light_curves(data, centers, half_width):
    n_time = data["time"].size
    n_wl = centers.size
    flux = np.full((n_wl, n_time), np.nan)
    err = np.full((n_wl, n_time), np.nan)

    for i, center in enumerate(centers):
        lo = center - half_width
        hi = center + half_width
        for j in range(n_time):
            flux[i, j], err[i, j] = weighted_bin(
                data["wavelength"][j],
                data["flux"][j],
                data["error"][j],
                data["dq"][j],
                lo,
                hi,
            )

    white = np.full(n_time, np.nan)
    white_err = np.full(n_time, np.nan)
    for j in range(n_time):
        white[j], white_err[j] = weighted_bin(
            data["wavelength"][j],
            data["flux"][j],
            data["error"][j],
            data["dq"][j],
            5.0,
            12.0,
        )

    return white, white_err, flux, err


def ramp_model(time, c0, c1, c2, tau):
    t = time - np.nanmin(time)
    return c0 + c1 * (time - np.nanmedian(time)) + c2 * np.exp(-t / tau)


def fit_white_tau(time, white, mask):
    y = white / np.nanmedian(white[mask])
    p0 = [1.0, 0.0, 0.01, 0.04]
    bounds = ([0.5, -10.0, -1.0, 0.002], [1.5, 10.0, 1.0, 1.0])
    popt, _ = curve_fit(ramp_model, time[mask], y[mask], p0=p0, bounds=bounds, maxfev=20000)
    return float(popt[3])


def detrend_with_fixed_tau(time, values, errors, mask, tau):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    corrected = np.full_like(values, np.nan, dtype=float)
    corrected_err = np.full_like(errors, np.nan, dtype=float)
    normalized = np.full_like(values, np.nan, dtype=float)
    normalized_err = np.full_like(errors, np.nan, dtype=float)
    baselines = np.full_like(values, np.nan, dtype=float)

    x_all = np.column_stack(
        [
            np.ones_like(time),
            time - np.nanmedian(time),
            np.exp(-(time - np.nanmin(time)) / tau),
        ]
    )

    for i in range(values.shape[0]):
        good = mask & np.isfinite(values[i]) & np.isfinite(errors[i]) & (errors[i] > 0)
        if np.count_nonzero(good) < 10:
            continue

        scale = np.nanmedian(values[i, good])
        y = values[i] / scale
        sigma = errors[i] / scale
        w = 1.0 / np.square(sigma[good])
        x = x_all[good]
        beta = np.linalg.solve((x.T * w) @ x, (x.T * w) @ y[good])
        baseline = x_all @ beta
        baselines[i] = baseline * scale
        normalized[i] = values[i] / scale
        normalized_err[i] = errors[i] / scale
        corrected[i] = values[i] / baselines[i]
        corrected_err[i] = errors[i] / baselines[i]

    return corrected, corrected_err, normalized, normalized_err, baselines


def channel_outlier_mask(flux):
    bad = np.zeros_like(flux, dtype=bool)
    for i in range(flux.shape[0]):
        local = median_filter(flux[i], size=31, mode="nearest")
        resid = flux[i] - local
        mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        threshold = max(8.0 * mad, 0.008)
        bad[i] = np.abs(resid) > threshold
    return bad


def main():
    data = read_x1dints(INPUT)
    centers = np.arange(5.25, 12.0, 0.5)
    half_width = 0.25

    white, white_err, flux, flux_err = bin_light_curves(data, centers, half_width)

    n_time = data["time"].size
    global_int = np.arange(1, n_time + 1)
    first_in_segment = np.r_[True, data["segment"][1:] != data["segment"][:-1]]
    out_of_eclipse = np.abs(data["time"] - ECLIPSE_CENTER_MJD_TDB) > (0.5 * ECLIPSE_DURATION_DAYS + 0.02)
    paper_like_mask = (global_int > 65) & ~first_in_segment & out_of_eclipse
    finite_white = np.isfinite(white) & np.isfinite(white_err) & (white_err > 0)
    baseline_mask = paper_like_mask & finite_white

    tau = fit_white_tau(data["time"], white, baseline_mask)
    all_values = np.vstack([white[None, :], flux])
    all_errors = np.vstack([white_err[None, :], flux_err])
    corr, corr_err, raw_norm, raw_norm_err, baseline = detrend_with_fixed_tau(
        data["time"], all_values, all_errors, baseline_mask, tau
    )

    finite_corrected_white = np.isfinite(corr[0]) & np.isfinite(corr_err[0]) & (corr_err[0] > 0)
    spectral_outlier = channel_outlier_mask(corr[1:])
    spectral_outlier_any = np.any(spectral_outlier, axis=0)
    science_mask = (
        (global_int > 65)
        & ~first_in_segment
        & finite_corrected_white
        & (corr[0] > 0.985)
        & (corr[0] < 1.01)
        & ~spectral_outlier_any
    )

    phase_full = ((data["time"] - T0_MJD_TDB) / PERIOD_DAYS) % 1.0
    phase_from_eclipse_full = (data["time"] - ECLIPSE_CENTER_MJD_TDB) / PERIOD_DAYS

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        time=data["time"][science_mask],
        phase=phase_full[science_mask],
        phase_from_eclipse=phase_from_eclipse_full[science_mask],
        flux_white=corr[0, science_mask],
        flux_err_white=corr_err[0, science_mask],
        flux=corr[1:, science_mask],
        flux_err=corr_err[1:, science_mask],
        wl=centers,
        bin_half_width=np.full_like(centers, half_width, dtype=float),
        flux_white_raw=raw_norm[0, science_mask],
        flux_err_white_raw=raw_norm_err[0, science_mask],
        flux_raw=raw_norm[1:, science_mask],
        flux_err_raw=raw_norm_err[1:, science_mask],
        time_full=data["time"],
        phase_full=phase_full,
        phase_from_eclipse_full=phase_from_eclipse_full,
        flux_white_full=corr[0],
        flux_err_white_full=corr_err[0],
        flux_full=corr[1:],
        flux_err_full=corr_err[1:],
        flux_white_raw_full=raw_norm[0],
        flux_err_white_raw_full=raw_norm_err[0],
        flux_raw_full=raw_norm[1:],
        flux_err_raw_full=raw_norm_err[1:],
        raw_white_jy=white,
        raw_white_jy_err=white_err,
        raw_spectral_jy=flux,
        raw_spectral_jy_err=flux_err,
        baseline_white_jy=baseline[0],
        baseline_spectral_jy=baseline[1:],
        wavelength_grid=np.nanmedian(data["wavelength"], axis=0),
        median_raw_spectrum_jy=np.nanmedian(data["flux"], axis=0),
        segment=data["segment"][science_mask],
        exposure=data["exposure"][science_mask],
        int_num=data["int_num"][science_mask],
        global_int=global_int[science_mask],
        segment_full=data["segment"],
        exposure_full=data["exposure"],
        int_num_full=data["int_num"],
        global_int_full=global_int,
        first_in_segment_full=first_in_segment,
        out_of_eclipse_full=out_of_eclipse,
        baseline_mask_full=baseline_mask,
        science_mask_full=science_mask,
        spectral_outlier_mask_full=spectral_outlier,
        spectral_outlier_any_full=spectral_outlier_any,
        fitted_ramp_tau_days=np.array(tau),
        period_days=np.array(PERIOD_DAYS),
        t0_mjd_tdb=np.array(T0_MJD_TDB),
        eclipse_center_mjd_tdb=np.array(ECLIPSE_CENTER_MJD_TDB),
        eclipse_duration_days=np.array(ECLIPSE_DURATION_DAYS),
        m_star_msun=np.array(M_STAR_MSUN),
        r_star_rsun=np.array(R_STAR_RSUN),
        m_planet_mjup=np.array(M_PLANET_MJUP),
        r_planet_rjup=np.array(R_PLANET_RJUP),
        rp_rs=np.array(RP_RS),
        a_over_rstar=np.array(A_OVER_RSTAR),
        inclination_deg=np.array(INCLINATION_DEG),
        stellar_teff_k=np.array(STELLAR_TEFF_K),
        source_product=np.array(INPUT.name),
        source_note=np.array(
            "Observed MAST CALJWST x1dints product. Light curves are binned from per-integration "
            "extracted spectra and divided by a simple out-of-eclipse ramp+linear baseline. "
            "They are not synthetic and are not the authors' ExoTiC-MIRI corrected light curves."
        ),
        systematics_model=np.array("fixed-tau exponential ramp plus linear time baseline, fit out of eclipse"),
    )

    print(f"Wrote {OUTPUT}")
    print(f"Integrations: {n_time}")
    print(f"Wavelength bins: {centers.size}")
    print(f"Kept integrations: {np.count_nonzero(science_mask)}")
    print(f"Spectral outlier integrations removed: {np.count_nonzero(spectral_outlier_any & finite_corrected_white)}")
    print(f"White flux range, kept: {np.nanmin(corr[0, science_mask]):.6f} to {np.nanmax(corr[0, science_mask]):.6f}")
    print(f"Fitted white-light ramp tau: {tau:.6f} d")
    print(f"Baseline points: {np.count_nonzero(baseline_mask)}")


if __name__ == "__main__":
    main()
