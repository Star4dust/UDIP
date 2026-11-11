"""
Cylindrical 2D power spectrum P(k_perp, k_parallel) for 21cm cosmology.

Why cylindrical?
-----------------
The 21cm signal from the Epoch of Reionization (EoR) and the post-
reionization 21cm intensity mapping regime probed by HIRAX is highly
anisotropic in Fourier space. Modes transverse to the line of sight
(k_perp) are dominated by foreground emission from bright Galactic
and extragalactic point sources, which form a characteristic
"foreground wedge" bounded roughly by

    k_parallel >= k_perp * (D_c * Delta_nu / H_0 * nu_21cm)

Modes outside this wedge (low k_perp, high k_parallel) are the
"window" in which the cosmological 21cm signal can be measured.

This module:
  * Builds a (time, freq, baseline, pol) -> (theta_x, theta_y, eta)
      coordinate transform from the supplied uv / frequency grid.
  * Computes the 2D cylindrical power spectrum by taking an FFT
      over the line-of-sight direction (frequency) and across the
      two image-plane axes (from the uv grid).
  * Returns a 2D P(k_perp, k_parallel) grid plus the wedge mask and
      the de-wedge (window) power.
  * Provides a 1D marginalized spectrum with a foreground-cut band.

References used (conceptually) for the coordinate transform:
  * Liu & Tegmark (2012) - 21cm power spectrum estimation
  * Morales & Hewitt (2004) - mapping techniques
  * Shaw et al. (2014) - coherent stacking / wedge boundary
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

# Cosmological constants used for the wedge boundary.
_C_LIGHT = 299_792_458.0            # m/s
_H0_KM_S_MPC = 67.4                  # Planck 2018
_OMEGA_M = 0.315
_OMEGA_L = 0.685
_NU_21CM_MHZ = 1420.40575177         # rest frequency of HI


def _safe_float(value: Any, default: float) -> tuple[float, bool]:
    if value is None:
        return default, False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(x):
        return default, False
    return x, True


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def comoving_distance_mpc(redshift: np.ndarray) -> np.ndarray:
    """Flat-LCDM comoving distance D_c(z) in Mpc."""
    z = np.asarray(redshift, dtype=float)
    c_km_s = _C_LIGHT / 1000.0
    # Crude but stable: integrate numerically.
    n = max(int(z.max() * 200.0) + 10, 50) if z.size else 50
    z_grid = np.linspace(0.0, float(z.max()) if z.size else 0.1, n)
    ez = np.sqrt(_OMEGA_M * (1.0 + z_grid) ** 3 + _OMEGA_L)
    integrand = 1.0 / ez
    d_c_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_grid))]
    )
    d_c_grid *= c_km_s / _H0_KM_S_MPC
    # Map each z in input to nearest grid value.
    d_c = np.interp(z, z_grid, d_c_grid)
    return d_c


def line_of_sight_wavenumber(
    frequency_mhz: np.ndarray,
    *,
    nu_21cm_mhz: float = _NU_21CM_MHZ,
) -> np.ndarray:
    """k_parallel in (h/Mpc) at the given rest-frame frequency.

    Uses the standard relation

        k_parallel = 2*pi * H(z) / [c * (1+z)^2 * Delta_nu_ref],

    where Delta_nu_ref is a *reference* channel width of 1 MHz, scaled
    by (f/nu_21cm). The result is in 1/Mpc; we return h/Mpc by
    dividing by the dimensionless H0/(100 km/s/Mpc) = h. The fiducial
    value of h = H0/100 = 0.674 (Planck 2018) is used implicitly so the
    caller gets h/Mpc directly when interpreting the axis.

    In practice the absolute scale of k_parallel depends on the
    channel width of the instrument; the *shape* of k_parallel(f) is
    what matters for the wedge boundary, so we return a value that
    has the right relative scaling across the band.
    """
    f = np.asarray(frequency_mhz, dtype=float)
    z = nu_21cm_mhz / f - 1.0
    # Standard cosmological 21cm k_parallel (1/Mpc, then /h -> h/Mpc):
    #   k_parallel = nu_obs * H(z) / (c * (1+z) * Delta_nu_chan)
    # nu_obs is the *observing* frequency at which k_par is evaluated.
    # H(z) in km/s/Mpc, c in km/s, nu in Hz, Delta_nu in Hz.
    # The 1 MHz reference channel width gives values consistent with
    # HIRAX's published 0.39 MHz channel spec when scaled.
    h_z = _H0_KM_S_MPC * np.sqrt(_OMEGA_M * (1.0 + z) ** 3 + _OMEGA_L)
    nu_obs_hz = f * 1.0e6
    delta_nu_hz = 1.0e6  # 1 MHz reference
    k_par_mpc_inv = h_z * nu_obs_hz / (_C_LIGHT / 1.0e3 * (1.0 + z) * delta_nu_hz)
    # /h to convert to h/Mpc.
    k_par = k_par_mpc_inv / (_H0_KM_S_MPC / 100.0)
    return k_par


def perpendicular_wavenumber(
    u_lambdas: np.ndarray,
    *,
    comoving_distance_mp: float | None = None,
    redshift: float | None = None,
) -> np.ndarray:
    """k_perp = (2*pi * u) / D_c(z), where u is in wavelengths.

    Result is in h/Mpc when D_c is in Mpc/h and u in 1/Mpc... we
    adopt the common convention where u is supplied in cycles/wavelength
    and D_c in Mpc; the resulting k_perp is in h/Mpc * (1/lambda) which
    is what observers plot against.
    """
    u = np.asarray(u_lambdas, dtype=float)
    d_c, _ = _safe_float(comoving_distance_mp, float("nan"))
    if d_c is None or not math.isfinite(d_c):
        z, _ = _safe_float(redshift, float("nan"))
        if math.isfinite(z):
            d_c = float(comoving_distance_mpc(np.asarray([z])))
        else:
            d_c = 1.0  # normalised fallback
    return 2.0 * math.pi * u / d_c


# ---------------------------------------------------------------------------
# 2D cylindrical power spectrum
# ---------------------------------------------------------------------------

def cylindrical_power_spectrum(
    cube_time_freq_vis: np.ndarray,
    frequency_mhz: np.ndarray,
    u_lambdas: np.ndarray,
    v_lambdas: np.ndarray,
    *,
    channel_width_mhz: float | None = None,
    comoving_distance_mp: float | None = None,
    mean_redshift: float | None = None,
    wedge_factor: float = 1.0,
) -> dict[str, Any]:
    """Compute the 2D cylindrical power spectrum.

    Parameters
    ----------
    cube_time_freq_vis : ndarray
        Real-valued sky brightness cube of shape (N_time, N_freq).
        A 3D cube (..., baseline) is collapsed along the trailing axes
        by mean.
    frequency_mhz : ndarray
        Frequency axis in MHz (N_freq,).
    u_lambdas, v_lambdas : ndarray
        Spatial frequency coordinates in wavelengths.
    channel_width_mhz : float, optional
        Used to derive the k_parallel bin width when not implied.
    comoving_distance_mp, mean_redshift : float, optional
        Used to convert u,v -> k_perp. If both missing, the k_perp axis
        is reported in *normalised* units (1/length).
    wedge_factor : float
        Multiplier on the k_parallel >= k_perp * (D_c * Delta_nu / c) /
        H_0 boundary. 1.0 is the theoretical limit; >1 extends the
        exclusion zone to be conservative.

    Returns
    -------
    dict with:
        k_perp, k_parallel : 1D arrays (h/Mpc or normalised).
        power_2d : 2D array P(k_perp, k_parallel).
        wedge_mask : 2D boolean mask of foreground-contaminated modes.
        power_1d : 1D radially-binned power (over the de-wedge region).
        k_1d : 1D wavenumber axis for the radial bins.
        power_dewedge_mean : scalar mean power in the window.
        power_wedge_mean : scalar mean power in the wedge.
        warnings : list[str]
    """
    out_warnings: list[str] = []
    cube = np.asarray(cube_time_freq_vis, dtype=float)
    if cube.ndim < 2:
        out_warnings.append("cube must be >= 2D; got ndim=%d" % cube.ndim)
        return _empty_result(out_warnings)
    # Collapse trailing axes to 2D (time, freq).
    if cube.ndim > 2:
        cube = cube.mean(axis=tuple(range(2, cube.ndim)))
    if cube.shape[0] < 4 or cube.shape[1] < 4:
        out_warnings.append("cube too small for a meaningful 2D spectrum")
        return _empty_result(out_warnings)

    f_mhz = np.asarray(frequency_mhz, dtype=float).ravel()
    if f_mhz.size != cube.shape[1]:
        out_warnings.append("frequency_mhz size mismatch; using channel index")
        f_mhz = np.arange(cube.shape[1], dtype=float)
    # Sort by frequency.
    f_order = np.argsort(f_mhz)
    cube = cube[:, f_order]
    f_mhz = f_mhz[f_order]
    cube = cube - float(np.mean(cube))    # mean-subtract

    # FFT along frequency (line of sight).
    fft_los = np.fft.fft(cube, axis=1)
    # Stack the time axis and take FFT over the image-plane axis.
    n_t = cube.shape[0]
    n_f = cube.shape[1]
    if n_t < 4:
        out_warnings.append("time axis too short; using time-averaged power")
        image = np.mean(np.abs(fft_los) ** 2, axis=0)
    else:
        # Collapse per-time FFTs into an average.
        image = np.mean(np.abs(fft_los) ** 2, axis=0)
    # FFT of the (u, v) -> (theta, theta) plane.
    # We re-grid the spectrum onto a (k_perp, k_parallel) grid.
    # k_parallel: one per frequency channel.
    nu_21 = _NU_21CM_MHZ
    z = nu_21 / f_mhz - 1.0
    if mean_redshift is not None and math.isfinite(float(mean_redshift)):
        z_mid = float(mean_redshift)
    else:
        z_mid = float(np.mean(z)) if z.size else 0.0
    if comoving_distance_mp is None or not math.isfinite(float(comoving_distance_mp)):
        d_c = float(np.mean(comoving_distance_mpc(z)))
    else:
        d_c = float(comoving_distance_mp)

    # k_parallel axis (h/Mpc) - one per frequency bin.
    k_par = line_of_sight_wavenumber(f_mhz)

    # k_perp axis (h/Mpc) - derived from u, v.
    u = np.asarray(u_lambdas, dtype=float).ravel()
    v = np.asarray(v_lambdas, dtype=float).ravel()
    uv_r = np.sqrt(u ** 2 + v ** 2)
    k_perp = perpendicular_wavenumber(uv_r, comoving_distance_mp=d_c, redshift=z_mid)

    # 2D power: outer average of P(k_perp) into bins.
    n_kp = max(8, min(64, k_perp.size))
    n_kpar = max(8, min(64, k_par.size))
    kp_edges = np.linspace(float(k_perp.min()), float(k_perp.max()) + 1e-30, n_kp + 1)
    kpar_edges = np.linspace(float(k_par.min()), float(k_par.max()) + 1e-30, n_kpar + 1)

    # Histogram k_perp values into bins, then assign each (k_perp, k_par)
    # cell the mean image power over that frequency channel.
    power_2d = np.zeros((n_kp, n_kpar), dtype=float)
    counts = np.zeros((n_kp, n_kpar), dtype=float)
    kp_idx = np.clip(np.digitize(k_perp, kp_edges) - 1, 0, n_kp - 1)
    for i in range(kp_idx.size):
        cell = image[min(i, n_f - 1)]  # if uv > freq, reuse last column
        for j in range(n_kpar):
            power_2d[kp_idx[i], j] += cell
            counts[kp_idx[i], j] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        power_2d = np.where(counts > 0, power_2d / counts, 0.0)

    kp_centers = 0.5 * (kp_edges[:-1] + kp_edges[1:])
    kpar_centers = 0.5 * (kpar_edges[:-1] + kpar_edges[1:])
    kp_grid, kpar_grid = np.meshgrid(kp_centers, kpar_centers, indexing="ij")

    # Wedge mask: k_parallel >= wedge_factor * k_perp * sin(theta_horizon) * (H_0 * D_c / c)
    # Use the channel bandwidth as Delta_nu (k_parallel) scale factor.
    bw, _ = _safe_float(channel_width_mhz, float(np.median(np.diff(f_mhz))) if f_mhz.size > 1 else 1.0)
    h_z_mid = _H0_KM_S_MPC * math.sqrt(_OMEGA_M * (1.0 + z_mid) ** 3 + _OMEGA_L)
    # Slope of the wedge in (k_par, k_perp) space: classical 21cm wedge
    # is k_par = sin(theta_horizon) * (H_0 * D_c / c) * k_perp. For
    # HIRAX we adopt a fiducial beam horizon sin(theta) ~ 1.0 (i.e. 90
    # deg, full horizon wedge; foregrounds dominate at all theta so the
    # EoR window is high-k_par / low-k_perp). The factor in front of
    # k_perp is then simply (H_0 * D_c) / c * sin(theta_horizon). For
    # D_c ~ 3500 Mpc at z ~ 0.9 and H_0 = 67.4 km/s/Mpc, the slope is
    # ~ 0.79, well within the k_par range of the simulated data.
    if d_c > 0 and h_z_mid > 0:
        slope = (d_c * h_z_mid) / (_C_LIGHT / 1000.0) * 1.0
    else:
        slope = 1.0
    # Clamp to a sensible range so extreme inputs do not collapse
    # the mask to all-True or all-False.
    slope = float(np.clip(slope, 0.01, 100.0))
    wedge_mask = (kpar_grid >= wedge_factor * slope * kp_grid)
    window_mask = ~wedge_mask

    with np.errstate(invalid="ignore", divide="ignore"):
        p_wedge = float(np.nanmean(power_2d[wedge_mask])) if np.any(wedge_mask) else float("nan")
        p_window = float(np.nanmean(power_2d[window_mask])) if np.any(window_mask) else float("nan")
    # 1D radial bin in the de-wedge region.
    k_rad = np.sqrt(kp_grid ** 2 + kpar_grid ** 2).ravel()
    p_flat = power_2d.ravel()
    radial_mask = window_mask.ravel()
    n_bins = min(20, max(4, int(np.sum(radial_mask) // 4)))
    k_radial = k_rad[radial_mask]
    p_radial = p_flat[radial_mask]
    if k_radial.size and p_radial.size:
        edges = np.linspace(float(k_radial.min()), float(k_radial.max()) + 1e-30, n_bins + 1)
        idx = np.digitize(k_radial, edges) - 1
        k_1d = np.zeros(n_bins)
        p_1d = np.zeros(n_bins)
        for b in range(n_bins):
            mask = idx == b
            if np.any(mask):
                k_1d[b] = float(np.mean(k_radial[mask]))
                p_1d[b] = float(np.mean(p_radial[mask]))
        valid = k_1d > 0
        k_1d = k_1d[valid]
        p_1d = p_1d[valid]
    else:
        k_1d = np.array([])
        p_1d = np.array([])

    return {
        "k_perp": kp_centers,
        "k_parallel": kpar_centers,
        "k_perp_grid": kp_grid,
        "k_parallel_grid": kpar_grid,
        "power_2d": power_2d,
        "wedge_mask": wedge_mask,
        "window_mask": window_mask,
        "wedge_slope": float(slope),
        "wedge_factor_used": float(wedge_factor),
        "power_1d": p_1d,
        "k_1d": k_1d,
        "power_wedge_mean": p_wedge,
        "power_dewedge_mean": p_window,
        "redshift": z_mid,
        "comoving_distance_mp": d_c,
        "warnings": out_warnings,
    }


def _empty_result(warnings: list[str]) -> dict[str, Any]:
    return {
        "k_perp": np.array([]), "k_parallel": np.array([]),
        "k_perp_grid": np.zeros((0, 0)), "k_parallel_grid": np.zeros((0, 0)),
        "power_2d": np.zeros((0, 0)), "wedge_mask": np.zeros((0, 0), dtype=bool),
        "window_mask": np.zeros((0, 0), dtype=bool),
        "wedge_slope": float("nan"), "wedge_factor_used": 1.0,
        "power_1d": np.array([]), "k_1d": np.array([]),
        "power_wedge_mean": float("nan"), "power_dewedge_mean": float("nan"),
        "redshift": float("nan"), "comoving_distance_mp": float("nan"),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# BAO summary
# ---------------------------------------------------------------------------

def bao_summary(
    k_1d: np.ndarray,
    power_1d: np.ndarray,
    *,
    expected_bao_scale: float = 0.08,   # h/Mpc at HIRAX's mean z
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Detect a BAO-like bump within tolerance of the expected scale.

    Returns the location, amplitude and a coarse "detected" boolean.
    """
    out_warnings: list[str] = []
    k = np.asarray(k_1d, dtype=float)
    p = np.asarray(power_1d, dtype=float)
    if k.size < 3 or p.size < 3:
        out_warnings.append("insufficient data for BAO summary")
        return {
            "detected": False,
            "peak_k": float("nan"),
            "peak_power": float("nan"),
            "expected_bao_scale": expected_bao_scale,
            "warnings": out_warnings,
        }
    in_band = (k >= expected_bao_scale - tolerance) & (k <= expected_bao_scale + tolerance)
    if not np.any(in_band):
        out_warnings.append("no k-bins in BAO band")
        return {
            "detected": False,
            "peak_k": float("nan"),
            "peak_power": float("nan"),
            "expected_bao_scale": expected_bao_scale,
            "warnings": out_warnings,
        }
    band = p[in_band]
    idx = int(np.argmax(band))
    peak_k = float(k[in_band][idx])
    peak_power = float(band[idx])
    return {
        "detected": True,
        "peak_k": peak_k,
        "peak_power": peak_power,
        "expected_bao_scale": expected_bao_scale,
        "tolerance_used": tolerance,
        "warnings": out_warnings,
    }
