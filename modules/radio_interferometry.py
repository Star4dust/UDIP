"""
UDIP v3 — modules/radio_interferometry.py
Radio Interferometry Pipeline for HIRAX and Radio Astronomy Data
Core scientific capabilities for interferometric data processing
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# VISIBILITY PROCESSING
# ============================================================================

def compute_visibility_statistics(visibilities: np.ndarray) -> Dict:
    """
    Compute basic statistics for visibility data.
    
    Args:
        visibilities: Complex visibility array (time, freq, baseline, pol)
    
    Returns:
        dict: Statistics including amplitude, phase, real, imag stats
    """
    arr = np.asarray(visibilities)
    amp = np.abs(arr)
    phase = np.angle(arr)
    real = np.real(arr)
    imag = np.imag(arr)
    finite_amp = amp[np.isfinite(amp)]
    finite_phase = phase[np.isfinite(phase)]
    finite_real = real[np.isfinite(real)]
    finite_imag = imag[np.isfinite(imag)]
    stats = {
        "amplitude": {
            "mean": float(np.mean(finite_amp)) if finite_amp.size else 0.0,
            "std": float(np.std(finite_amp)) if finite_amp.size else 0.0,
            "min": float(np.min(finite_amp)) if finite_amp.size else 0.0,
            "max": float(np.max(finite_amp)) if finite_amp.size else 0.0,
            "median": float(np.median(finite_amp)) if finite_amp.size else 0.0,
        },
        "phase": {
            "mean": float(np.mean(finite_phase)) if finite_phase.size else 0.0,
            "std": float(np.std(finite_phase)) if finite_phase.size else 0.0,
            "min": float(np.min(finite_phase)) if finite_phase.size else 0.0,
            "max": float(np.max(finite_phase)) if finite_phase.size else 0.0,
        },
        "real": {
            "mean": float(np.mean(finite_real)) if finite_real.size else 0.0,
            "std": float(np.std(finite_real)) if finite_real.size else 0.0,
        },
        "imaginary": {
            "mean": float(np.mean(finite_imag)) if finite_imag.size else 0.0,
            "std": float(np.std(finite_imag)) if finite_imag.size else 0.0,
        },
        "n_samples": int(arr.size),
    }
    return stats


def flag_bad_visibilities(visibilities: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Flag bad visibilities based on robust amplitude sigma clipping.
    
    Args:
        visibilities: Complex visibility array
        threshold: Sigma threshold for flagging
    
    Returns:
        np.ndarray: Boolean mask of flagged visibilities
    """
    amp = np.asarray(np.abs(visibilities), dtype=float)
    finite = np.isfinite(amp)
    if not np.any(finite):
        return np.zeros_like(amp, dtype=bool)

    clean_amp = amp[finite]
    center = float(np.nanmedian(clean_amp))
    mad = float(np.nanmedian(np.abs(clean_amp - center)))
    robust_sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(clean_amp))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.nanstd(clean_amp))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return np.zeros_like(amp, dtype=bool)

    limit = center + threshold * robust_sigma
    mask = finite & (amp > limit)
    return mask


def apply_visibility_flags(visibilities: np.ndarray, mask: np.ndarray, fill_value: complex | float | None = None) -> np.ndarray:
    """
    Apply flags to visibilities by setting flagged values to NaN.
    
    Args:
        visibilities: Complex visibility array
        mask: Boolean mask of flagged visibilities
        fill_value: Optional fill value for masked samples.
    
    Returns:
        np.ndarray: Flagged visibility array with NaNs
    """
    flagged = np.asarray(visibilities).copy()
    mask = np.asarray(mask, dtype=bool)
    if fill_value is None:
        fill_value = (np.nan + 1j * np.nan) if np.iscomplexobj(flagged) else np.nan
    flagged[mask] = fill_value
    return flagged


def summarize_visibility_flags(mask: np.ndarray) -> Dict:
    """Summarize a boolean flag mask for UI feedback and regression checks."""
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.size)
    flagged = int(np.count_nonzero(mask))
    return {
        "n_total": total,
        "n_flagged": flagged,
        "flagged_fraction": float(flagged / total) if total else 0.0,
    }


# ============================================================================
# BASELINE ANALYSIS
# ============================================================================

def compute_baseline_lengths(baseline_coords: np.ndarray) -> np.ndarray:
    """
    Compute baseline lengths from antenna coordinates.
    
    Args:
        baseline_coords: Array of baseline coordinates (n_baselines, 3) or (n_baselines, 2)
    
    Returns:
        np.ndarray: Baseline lengths in wavelengths
    """
    if baseline_coords.shape[1] == 3:
        # 3D coordinates (u, v, w)
        lengths = np.sqrt(np.sum(baseline_coords**2, axis=1))
    elif baseline_coords.shape[1] == 2:
        # 2D coordinates (u, v)
        lengths = np.sqrt(np.sum(baseline_coords**2, axis=1))
    else:
        raise ValueError("Baseline coordinates must be 2D or 3D")
    
    return lengths


def bin_baselines_by_length(baseline_lengths: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin baselines by length.
    
    Args:
        baseline_lengths: Array of baseline lengths
        n_bins: Number of bins
    
    Returns:
        tuple: (bin_edges, bin_indices)
    """
    max_length = np.max(baseline_lengths)
    min_length = np.min(baseline_lengths)
    
    bin_edges = np.linspace(min_length, max_length, n_bins + 1)
    bin_indices = np.digitize(baseline_lengths, bin_edges) - 1
    
    return bin_edges, bin_indices


def compute_baseline_correlation(visibilities: np.ndarray, baseline_indices: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute correlation matrix between baselines.
    
    Args:
        visibilities: Visibility array (time, freq, baseline, ...)
        baseline_indices: Baseline bin indices
        n_bins: Number of baseline bins
    
    Returns:
        np.ndarray: Correlation matrix
    """
    # Average visibilities by baseline bin
    correlations = np.zeros((n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            vis_i = visibilities[:, :, baseline_indices == i]
            vis_j = visibilities[:, :, baseline_indices == j]
            
            # Flatten and compute correlation on real+imag concatenation for complex preservation
            flat_i = np.concatenate([vis_i.real.ravel(), vis_i.imag.ravel()])
            flat_j = np.concatenate([vis_j.real.ravel(), vis_j.imag.ravel()])
            
            if flat_i.size == 0 or flat_j.size == 0:
                correlations[i, j] = 0.0
                continue

            mask = np.isfinite(flat_i) & np.isfinite(flat_j)
            if np.sum(mask) > 1:
                correlations[i, j] = np.corrcoef(flat_i[mask], flat_j[mask])[0, 1]
            else:
                correlations[i, j] = 0.0
    
    return correlations


def summarize_baseline_amplitude_profile(visibilities: np.ndarray, baseline_coords: np.ndarray, n_bins: int = 8) -> Dict:
    """Summarize mean visibility amplitude as a function of baseline length bin."""
    vis_arr = np.asarray(visibilities, dtype=np.complex128).reshape(-1)
    coords = np.asarray(baseline_coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    if coords.shape[0] != vis_arr.size and coords.shape[0] > 0:
        n = min(coords.shape[0], vis_arr.size)
        coords = coords[:n]
        vis_arr = vis_arr[:n]

    if vis_arr.size == 0 or coords.size == 0:
        empty = np.array([], dtype=float)
        return {"bin_edges": empty, "bin_centers": empty, "mean_amplitude": empty, "median_amplitude": empty, "counts": empty, "n_bins": 0}

    lengths = compute_baseline_lengths(coords)
    finite = np.isfinite(lengths) & np.isfinite(vis_arr.real) & np.isfinite(vis_arr.imag)
    lengths = lengths[finite]
    vis_arr = vis_arr[finite]
    if vis_arr.size == 0:
        empty = np.array([], dtype=float)
        return {"bin_edges": empty, "bin_centers": empty, "mean_amplitude": empty, "median_amplitude": empty, "counts": empty, "n_bins": 0}

    n_bins = max(int(n_bins), 1)
    lo = float(np.min(lengths))
    hi = float(np.max(lengths))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_indices = np.clip(np.digitize(lengths, bin_edges, right=False) - 1, 0, n_bins - 1)
    amp = np.abs(vis_arr)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_amp = np.full(n_bins, np.nan, dtype=float)
    median_amp = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for idx in range(n_bins):
        samples = amp[bin_indices == idx]
        counts[idx] = int(samples.size)
        if samples.size:
            mean_amp[idx] = float(np.nanmean(samples))
            median_amp[idx] = float(np.nanmedian(samples))

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "mean_amplitude": mean_amp,
        "median_amplitude": median_amp,
        "counts": counts,
        "n_bins": n_bins,
    }


# ============================================================================
# UV-PLANE ANALYSIS
# ============================================================================

def generate_uv_coverage(u_coords: np.ndarray, v_coords: np.ndarray, times: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze UV-plane coverage.
    
    Args:
        u_coords: U coordinates (wavelengths)
        v_coords: V coordinates (wavelengths)
        times: Optional time array for time-dependent coverage
    
    Returns:
        dict: UV coverage statistics and coverage density
    """
    u_coords = np.asarray(u_coords).reshape(-1)
    v_coords = np.asarray(v_coords).reshape(-1)
    if u_coords.size == 0 or v_coords.size == 0:
        return {
            "u_range": (0.0, 0.0),
            "v_range": (0.0, 0.0),
            "max_uv_distance": 0.0,
            "min_uv_distance": 0.0,
            "mean_uv_distance": 0.0,
            "n_uv_points": 0,
            "coverage_density": np.zeros((50, 50)),
            "density_max": 0.0,
            "density_mean": 0.0,
        }

    uv_dist = np.sqrt(u_coords**2 + v_coords**2)
    uv_data = {
        "u_range": (float(np.min(u_coords)), float(np.max(u_coords))),
        "v_range": (float(np.min(v_coords)), float(np.max(v_coords))),
        "max_uv_distance": float(np.max(uv_dist)),
        "min_uv_distance": float(np.min(uv_dist)),
        "mean_uv_distance": float(np.mean(uv_dist)),
        "n_uv_points": int(len(u_coords))
    }
    
    # Compute coverage density (2D histogram)
    hist, xedges, yedges = np.histogram2d(u_coords, v_coords, bins=50)
    uv_data["coverage_density"] = hist
    uv_data["density_max"] = float(np.max(hist))
    uv_data["density_mean"] = float(np.mean(hist))
    
    return uv_data


def compute_uv_weights(u_coords: np.ndarray, v_coords: np.ndarray, weighting: str = "uniform") -> np.ndarray:
    """
    Compute UV-plane weights for imaging.
    
    Args:
        u_coords: U coordinates
        v_coords: V coordinates
        weighting: Weighting scheme ('uniform', 'natural', 'robust')
    
    Returns:
        np.ndarray: Weights for each UV point
    """
    u_coords = np.asarray(u_coords, dtype=float)
    v_coords = np.asarray(v_coords, dtype=float)
    uv_radius = np.sqrt(u_coords**2 + v_coords**2)
    
    if weighting == "uniform":
        weights = np.ones_like(uv_radius)
    elif weighting == "natural":
        weights = 1.0 / np.maximum(uv_radius, 1e-10)
    elif weighting == "robust":
        # Robust weighting (Briggs 1995)
        median_radius = np.nanmedian(uv_radius)
        if not np.isfinite(median_radius) or median_radius <= 0:
            median_radius = 1.0
        weights = 1.0 / (1.0 + (uv_radius / median_radius)**2)
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")
    
    # Normalize weights
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(uv_radius) / max(len(uv_radius), 1)
    return weights / total


def grid_visibilities_to_uv_plane(
    visibilities: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    image_size: int = 256,
    cell_size: float = 1.0,
    weights: Optional[np.ndarray] = None,
    hermitian: bool = True,
) -> np.ndarray:
    """Grid sparse visibilities onto a centered UV plane using nearest-neighbour bins."""
    visibilities = np.asarray(visibilities, dtype=np.complex128).reshape(-1)
    u_coords = np.asarray(u_coords, dtype=float).reshape(-1)
    v_coords = np.asarray(v_coords, dtype=float).reshape(-1)

    n = min(visibilities.size, u_coords.size, v_coords.size)
    if n == 0:
        return np.zeros((image_size, image_size), dtype=np.complex128)

    visibilities = visibilities[:n]
    u_coords = u_coords[:n]
    v_coords = v_coords[:n]

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)[:n]
        if weights.size < n:
            weights = np.pad(weights, (0, n - weights.size), constant_values=1.0)

    uv_grid = np.zeros((image_size, image_size), dtype=np.complex128)
    weight_grid = np.zeros((image_size, image_size), dtype=float)
    center = image_size // 2

    u_idx = np.rint(u_coords / cell_size).astype(int) + center
    v_idx = np.rint(v_coords / cell_size).astype(int) + center
    valid = np.isfinite(u_coords) & np.isfinite(v_coords) & np.isfinite(weights) & np.isfinite(visibilities.real) & np.isfinite(visibilities.imag)
    valid &= (u_idx >= 0) & (u_idx < image_size) & (v_idx >= 0) & (v_idx < image_size)

    for i in np.where(valid)[0]:
        w = float(weights[i])
        if w <= 0:
            continue
        uv_grid[v_idx[i], u_idx[i]] += visibilities[i] * w
        weight_grid[v_idx[i], u_idx[i]] += w

        if hermitian:
            mu = int(-u_idx[i] + 2 * center)
            mv = int(-v_idx[i] + 2 * center)
            if 0 <= mu < image_size and 0 <= mv < image_size and (mu != u_idx[i] or mv != v_idx[i]):
                uv_grid[mv, mu] += np.conj(visibilities[i]) * w
                weight_grid[mv, mu] += w

    with np.errstate(divide="ignore", invalid="ignore"):
        uv_grid = np.where(weight_grid > 0, uv_grid / weight_grid, 0.0)
    return uv_grid


# ============================================================================
# FFT IMAGING
# ============================================================================

def dirty_image_from_visibilities(visibilities: np.ndarray, 
                                  u_coords: np.ndarray, 
                                  v_coords: np.ndarray,
                                  image_size: int = 256,
                                  cell_size: float = 1.0,
                                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute dirty image from visibilities using inverse FFT.
    
    Args:
        visibilities: Complex visibility array (n_uv_points,)
        u_coords: U coordinates (wavelengths)
        v_coords: V coordinates (wavelengths)
        image_size: Size of output image (pixels)
        cell_size: Size of each pixel (wavelengths)
        weights: Optional imaging weights
    
    Returns:
        np.ndarray: Dirty image
    """
    uv_grid = grid_visibilities_to_uv_plane(
        visibilities=visibilities,
        u_coords=u_coords,
        v_coords=v_coords,
        image_size=image_size,
        cell_size=cell_size,
        weights=weights,
        hermitian=True,
    )
    dirty_image = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid)))
    dirty_image = np.real(dirty_image)
    dirty_image = np.nan_to_num(dirty_image, nan=0.0, posinf=0.0, neginf=0.0)
    return dirty_image


def compute_point_spread_function(u_coords: np.ndarray,
                                  v_coords: np.ndarray,
                                  image_size: int = 256,
                                  cell_size: float = 1.0) -> np.ndarray:
    """
    Compute Point Spread Function (PSF) from UV coverage.
    """
    u_coords = np.asarray(u_coords, dtype=float).reshape(-1)
    v_coords = np.asarray(v_coords, dtype=float).reshape(-1)
    n = min(u_coords.size, v_coords.size)
    if n == 0:
        return np.zeros((image_size, image_size), dtype=float)
    uv_grid = grid_visibilities_to_uv_plane(
        visibilities=np.ones(n, dtype=np.complex128),
        u_coords=u_coords[:n],
        v_coords=v_coords[:n],
        image_size=image_size,
        cell_size=cell_size,
        weights=None,
        hermitian=True,
    )
    psf = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(uv_grid))))
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(psf))) if psf.size else 0.0
    if peak <= 0:
        return psf
    return psf / peak


def compute_beam_parameters(psf: np.ndarray) -> Dict:
    """
    Compute beam parameters from PSF.
    """
    psf = np.asarray(psf, dtype=float)
    if psf.size == 0 or not np.any(np.isfinite(psf)):
        return {"peak": 0.0, "peak_location": (0, 0), "fwhm_x": 0.0, "fwhm_y": 0.0, "fwhm_avg": 0.0, "beam_area": 0.0}

    peak = float(np.nanmax(psf))
    peak_loc = np.unravel_index(int(np.nanargmax(psf)), psf.shape)
    half_max = peak / 2.0
    above_half = np.isfinite(psf) & (psf > half_max) if peak > 0 else np.zeros_like(psf, dtype=bool)
    if np.any(above_half):
        rows, cols = np.where(above_half)
        fwhm_x = float(np.max(cols) - np.min(cols) + 1)
        fwhm_y = float(np.max(rows) - np.min(rows) + 1)
        fwhm_avg = 0.5 * (fwhm_x + fwhm_y)
    else:
        fwhm_x = fwhm_y = fwhm_avg = 0.0

    beam_area = float(np.nansum(psf)) if np.any(np.isfinite(psf)) else 0.0
    return {
        "peak": peak,
        "peak_location": peak_loc,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "fwhm_avg": fwhm_avg,
        "beam_area": beam_area,
    }


def simulate_point_source_visibilities(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    sources: list[dict[str, float]],
) -> np.ndarray:
    """Simulate a flat-sky point-source model with simple Fourier phases.

    Each source uses direction cosines l,m (radians) and flux density in arbitrary units.
    """
    u_coords = np.asarray(u_coords, dtype=float).reshape(-1)
    v_coords = np.asarray(v_coords, dtype=float).reshape(-1)
    vis = np.zeros_like(u_coords, dtype=np.complex128)
    for source in sources:
        flux = float(source.get("flux", 1.0))
        l = float(source.get("l", 0.0))
        m = float(source.get("m", 0.0))
        vis += flux * np.exp(-2j * np.pi * (u_coords * l + v_coords * m))
    return vis


def compute_closure_phase(vis_ab: np.ndarray, vis_bc: np.ndarray, vis_ca: np.ndarray) -> Dict:
    """Compute closure phase from three baseline visibilities."""
    a = np.asarray(vis_ab, dtype=np.complex128).reshape(-1)
    b = np.asarray(vis_bc, dtype=np.complex128).reshape(-1)
    c = np.asarray(vis_ca, dtype=np.complex128).reshape(-1)
    n = min(a.size, b.size, c.size)
    if n == 0:
        empty = np.array([], dtype=float)
        return {"closure_phase": empty, "mean_closure_phase": 0.0, "rms_closure_phase": 0.0, "n_samples": 0}

    closure_phase = np.angle(a[:n] * b[:n] * c[:n])
    closure_phase = np.angle(np.exp(1j * closure_phase))
    mean_phase = float(np.angle(np.mean(np.exp(1j * closure_phase)))) if closure_phase.size else 0.0
    rms_phase = float(np.sqrt(np.mean(np.square(closure_phase)))) if closure_phase.size else 0.0
    return {
        "closure_phase": closure_phase,
        "mean_closure_phase": mean_phase,
        "rms_closure_phase": rms_phase,
        "n_samples": int(n),
    }


def compute_closure_amplitude(vis_ab: np.ndarray, vis_cd: np.ndarray, vis_ac: np.ndarray, vis_bd: np.ndarray) -> Dict:
    """Compute closure amplitude from four baseline visibilities."""
    ab = np.asarray(vis_ab, dtype=np.complex128).reshape(-1)
    cd = np.asarray(vis_cd, dtype=np.complex128).reshape(-1)
    ac = np.asarray(vis_ac, dtype=np.complex128).reshape(-1)
    bd = np.asarray(vis_bd, dtype=np.complex128).reshape(-1)
    n = min(ab.size, cd.size, ac.size, bd.size)
    if n == 0:
        empty = np.array([], dtype=float)
        return {"closure_amplitude": empty, "mean_closure_amplitude": 0.0, "std_closure_amplitude": 0.0, "n_samples": 0}

    numerator = np.abs(ab[:n]) * np.abs(cd[:n])
    denominator = np.abs(ac[:n]) * np.abs(bd[:n])
    with np.errstate(divide="ignore", invalid="ignore"):
        closure_amplitude = np.where(denominator > 0, numerator / denominator, np.nan)
    closure_amplitude = np.asarray(closure_amplitude, dtype=float)
    closure_amplitude = closure_amplitude[np.isfinite(closure_amplitude)]
    return {
        "closure_amplitude": closure_amplitude,
        "mean_closure_amplitude": float(np.mean(closure_amplitude)) if closure_amplitude.size else 0.0,
        "std_closure_amplitude": float(np.std(closure_amplitude)) if closure_amplitude.size else 0.0,
        "n_samples": int(n),
    }


def compute_delay_spectrum(visibilities: np.ndarray, sample_spacing_hz: float = 1.0, axis: int = -1) -> Dict:
    """Compute a delay spectrum via Fourier transform over the frequency axis."""
    arr = np.asarray(visibilities, dtype=np.complex128)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    axis = axis if axis >= 0 else arr.ndim + axis
    moved = np.moveaxis(arr, axis, -1)
    n = moved.shape[-1]
    if n == 0:
        return {"delay_axis": np.array([]), "spectrum": np.array([]), "power_spectrum": np.array([]), "average_power": np.array([]), "peak_delay": 0.0, "peak_power": 0.0}

    spectrum = np.fft.fftshift(np.fft.fft(moved, axis=-1), axes=-1)
    delay_axis = np.fft.fftshift(np.fft.fftfreq(n, d=sample_spacing_hz))
    power = np.abs(spectrum) ** 2
    average_power = power if power.ndim == 1 else power.mean(axis=tuple(range(power.ndim - 1)))
    peak_idx = int(np.argmax(average_power)) if average_power.size else 0
    return {
        "delay_axis": delay_axis,
        "spectrum": spectrum,
        "power_spectrum": power,
        "average_power": average_power,
        "peak_delay": float(delay_axis[peak_idx]) if delay_axis.size else 0.0,
        "peak_power": float(average_power[peak_idx]) if average_power.size else 0.0,
    }


def compute_fringe_rate_spectrum(visibilities: np.ndarray, sample_spacing_s: float = 1.0, axis: int = -1) -> Dict:
    """Compute a fringe-rate spectrum via Fourier transform over the time axis."""
    arr = np.asarray(visibilities, dtype=np.complex128)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    axis = axis if axis >= 0 else arr.ndim + axis
    moved = np.moveaxis(arr, axis, -1)
    n = moved.shape[-1]
    if n == 0:
        return {"fringe_rate_axis": np.array([]), "spectrum": np.array([]), "power_spectrum": np.array([]), "average_power": np.array([]), "peak_fringe_rate": 0.0, "peak_power": 0.0}

    spectrum = np.fft.fftshift(np.fft.fft(moved, axis=-1), axes=-1)
    fringe_rate_axis = np.fft.fftshift(np.fft.fftfreq(n, d=sample_spacing_s))
    power = np.abs(spectrum) ** 2
    average_power = power if power.ndim == 1 else power.mean(axis=tuple(range(power.ndim - 1)))
    peak_idx = int(np.argmax(average_power)) if average_power.size else 0
    return {
        "fringe_rate_axis": fringe_rate_axis,
        "spectrum": spectrum,
        "power_spectrum": power,
        "average_power": average_power,
        "peak_fringe_rate": float(fringe_rate_axis[peak_idx]) if fringe_rate_axis.size else 0.0,
        "peak_power": float(average_power[peak_idx]) if average_power.size else 0.0,
    }


def compute_stokes_parameters(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray | None = None, yx: np.ndarray | None = None) -> Dict:
    """Compute Stokes-like polarization products from correlation visibilities."""
    xx_arr = np.asarray(xx, dtype=np.complex128).reshape(-1)
    yy_arr = np.asarray(yy, dtype=np.complex128).reshape(-1)
    n = min(xx_arr.size, yy_arr.size)
    if n == 0:
        empty = np.array([], dtype=np.complex128)
        return {"I": empty, "Q": empty, "U": empty, "V": empty, "mean_linear_fraction": 0.0, "mean_circular_fraction": 0.0, "n_samples": 0}

    xx_arr = xx_arr[:n]
    yy_arr = yy_arr[:n]
    if xy is None:
        xy_arr = np.zeros(n, dtype=np.complex128)
    else:
        xy_arr = np.asarray(xy, dtype=np.complex128).reshape(-1)[:n]
    if yx is None:
        yx_arr = np.zeros(n, dtype=np.complex128)
    else:
        yx_arr = np.asarray(yx, dtype=np.complex128).reshape(-1)[:n]

    stokes_i = 0.5 * (xx_arr + yy_arr)
    stokes_q = 0.5 * (xx_arr - yy_arr)
    stokes_u = 0.5 * (xy_arr + yx_arr)
    stokes_v = -0.5j * (xy_arr - yx_arr)

    i_amp = np.abs(stokes_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        linear_fraction = np.where(i_amp > 0, np.sqrt(np.abs(stokes_q) ** 2 + np.abs(stokes_u) ** 2) / i_amp, 0.0)
        circular_fraction = np.where(i_amp > 0, np.abs(stokes_v) / i_amp, 0.0)

    return {
        "I": stokes_i,
        "Q": stokes_q,
        "U": stokes_u,
        "V": stokes_v,
        "linear_fraction": linear_fraction,
        "circular_fraction": circular_fraction,
        "mean_linear_fraction": float(np.mean(linear_fraction)) if linear_fraction.size else 0.0,
        "mean_circular_fraction": float(np.mean(circular_fraction)) if circular_fraction.size else 0.0,
        "n_samples": int(n),
    }


def compute_delay_rate_map(vis_matrix: np.ndarray, time_spacing_s: float = 1.0, frequency_spacing_hz: float = 1.0) -> Dict:
    """Compute a 2D delay-rate power map from a time-frequency visibility matrix.

    The first axis is treated as time and the second axis as frequency.
    """
    arr = np.asarray(vis_matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("Delay-rate map expects a 2D time-frequency matrix")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    power_map = np.abs(spectrum) ** 2
    fringe_rate_axis = np.fft.fftshift(np.fft.fftfreq(arr.shape[0], d=time_spacing_s))
    delay_axis = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], d=frequency_spacing_hz))
    peak_index = np.unravel_index(np.argmax(power_map), power_map.shape) if power_map.size else (0, 0)
    return {
        "delay_axis": delay_axis,
        "fringe_rate_axis": fringe_rate_axis,
        "power_map": power_map,
        "peak_delay": float(delay_axis[peak_index[1]]) if delay_axis.size else 0.0,
        "peak_fringe_rate": float(fringe_rate_axis[peak_index[0]]) if fringe_rate_axis.size else 0.0,
        "peak_power": float(power_map[peak_index]) if power_map.size else 0.0,
    }


def compute_polarization_matrix(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray | None = None, yx: np.ndarray | None = None) -> Dict:
    """Summarize a 2x2 polarization matrix using mean absolute correlations."""
    xx_arr = np.asarray(xx, dtype=np.complex128).reshape(-1)
    yy_arr = np.asarray(yy, dtype=np.complex128).reshape(-1)
    n = min(xx_arr.size, yy_arr.size)
    if n == 0:
        matrix = np.zeros((2, 2), dtype=float)
        return {"matrix": matrix, "labels": ["X", "Y"], "n_samples": 0}

    xx_arr = xx_arr[:n]
    yy_arr = yy_arr[:n]
    xy_arr = np.asarray(xy, dtype=np.complex128).reshape(-1)[:n] if xy is not None else np.zeros(n, dtype=np.complex128)
    yx_arr = np.asarray(yx, dtype=np.complex128).reshape(-1)[:n] if yx is not None else np.zeros(n, dtype=np.complex128)
    matrix = np.array(
        [
            [float(np.mean(np.abs(xx_arr))) if xx_arr.size else 0.0, float(np.mean(np.abs(xy_arr))) if xy_arr.size else 0.0],
            [float(np.mean(np.abs(yx_arr))) if yx_arr.size else 0.0, float(np.mean(np.abs(yy_arr))) if yy_arr.size else 0.0],
        ],
        dtype=float,
    )
    return {"matrix": matrix, "labels": ["X", "Y"], "n_samples": int(n)}


def _gaussian_beam_kernel(shape: tuple[int, int], fwhm_x: float, fwhm_y: float) -> np.ndarray:
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        return np.zeros(shape, dtype=float)
    sigma_x = max(float(fwhm_x), 1.0) / 2.355
    sigma_y = max(float(fwhm_y), 1.0) / 2.355
    yy, xx = np.indices(shape, dtype=float)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    kernel = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return kernel


def restore_clean_image(components: np.ndarray, residual: np.ndarray, psf: np.ndarray) -> Dict:
    """Restore CLEAN components with a simple Gaussian approximation to the synthesized beam."""
    components = np.asarray(components, dtype=float)
    residual = np.asarray(residual, dtype=float)
    psf = np.asarray(psf, dtype=float)
    if components.shape != residual.shape:
        raise ValueError(f"Components shape {components.shape} must match residual shape {residual.shape}")
    beam = compute_beam_parameters(psf)
    kernel = _gaussian_beam_kernel(components.shape, beam.get("fwhm_x", 0.0), beam.get("fwhm_y", 0.0))
    if np.any(kernel):
        kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
        restored_components = np.real(np.fft.ifft2(np.fft.fft2(components) * kernel_ft))
    else:
        restored_components = components.copy()
    restored_image = np.nan_to_num(restored_components + residual, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "restored_image": restored_image,
        "restored_components": restored_components,
        "restoring_beam": kernel,
        "beam_parameters": beam,
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "restored_rms": float(np.sqrt(np.mean(np.square(restored_image)))) if restored_image.size else 0.0,
    }


def compare_dirty_and_restored_images(dirty_image: np.ndarray, restored_image: np.ndarray) -> Dict:
    dirty = np.asarray(dirty_image, dtype=float)
    restored = np.asarray(restored_image, dtype=float)
    diff = np.nan_to_num(restored - dirty, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "dirty_rms": float(np.sqrt(np.mean(np.square(dirty)))) if dirty.size else 0.0,
        "restored_rms": float(np.sqrt(np.mean(np.square(restored)))) if restored.size else 0.0,
        "difference_rms": float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0,
        "difference_image": diff,
    }


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def run_interferometry_pipeline(visibilities: np.ndarray,
                                 u_coords: np.ndarray,
                                 v_coords: np.ndarray,
                                 compute_image: bool = True,
                                 compute_psf: bool = True) -> Dict:
    """
    Run complete interferometry pipeline.
    
    Args:
        visibilities: Complex visibility array
        u_coords: U coordinates
        v_coords: V coordinates
        compute_image: Whether to compute dirty image
        compute_psf: Whether to compute PSF
    
    Returns:
        dict: Pipeline results including statistics, coverage, images
    """
    results = {}
    
    # Visibility statistics
    results["visibility_stats"] = compute_visibility_statistics(visibilities)
    
    # UV coverage analysis
    results["uv_coverage"] = generate_uv_coverage(u_coords, v_coords)
    
    # Baseline analysis
    baseline_lengths = compute_baseline_lengths(np.column_stack([u_coords, v_coords]))
    results["baseline_lengths"] = baseline_lengths
    
    # Dirty image
    if compute_image:
        results["dirty_image"] = dirty_image_from_visibilities(visibilities, u_coords, v_coords)
    
    # PSF
    if compute_psf:
        psf = compute_point_spread_function(u_coords, v_coords)
        results["psf"] = psf
        results["beam_parameters"] = compute_beam_parameters(psf)
    
    return results


# ============================================================================
# CALIBRATION UTILITIES
# ============================================================================

def compute_gain_solution(visibilities: np.ndarray, model_visibilities: np.ndarray) -> np.ndarray:
    """
    Compute gain solution using least-squares calibration.
    
    Args:
        visibilities: Measured visibilities
        model_visibilities: Model visibilities
    
    Returns:
        np.ndarray: Gain solution
    """
    # Simple gain calibration: g = V_measured / V_model
    with np.errstate(divide='ignore', invalid='ignore'):
        gains = visibilities / model_visibilities
    
    # Handle division by zero
    gains[np.isinf(gains)] = 1.0
    gains[np.isnan(gains)] = 1.0
    
    return gains


def apply_gains(visibilities: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    Apply gain solution to visibilities.
    """
    visibilities = np.asarray(visibilities)
    gains = np.asarray(gains)
    if gains.shape != visibilities.shape:
        raise ValueError(f"Gain shape {gains.shape} does not match visibilities shape {visibilities.shape}")
    with np.errstate(divide='ignore', invalid='ignore'):
        calibrated = visibilities / gains
    calibrated[~np.isfinite(calibrated)] = 0.0
    return calibrated


def solve_phase_calibration(visibilities: np.ndarray, model_visibilities: np.ndarray) -> np.ndarray:
    """Estimate phase-only correction in radians from measured/model visibilities."""
    measured_phase = np.angle(visibilities)
    model_phase = np.angle(model_visibilities)
    return np.angle(np.exp(1j * (measured_phase - model_phase)))


def apply_phase_calibration(visibilities: np.ndarray, phase_solution: np.ndarray) -> np.ndarray:
    """Apply a phase-only calibration solution while preserving amplitudes."""
    return visibilities * np.exp(-1j * phase_solution)


def solve_bandpass(visibilities: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Estimate a complex bandpass response along a frequency axis.

    The result is normalized by the median amplitude so correction preserves the
    overall flux scale as much as possible for a first-pass calibration.
    """
    visibilities = np.asarray(visibilities)
    if axis < 0:
        axis += visibilities.ndim
    if visibilities.ndim == 0 or visibilities.shape[axis] == 0:
        return np.ones(1, dtype=np.complex128)
    response = np.asarray(
        np.nanmean(visibilities, axis=tuple(i for i in range(visibilities.ndim) if i != axis)),
        dtype=np.complex128,
    )
    amp = np.abs(response)
    scale = np.nanmedian(amp[amp > 0]) if np.any(amp > 0) else 1.0
    response = response / (scale or 1.0)
    response = np.where(np.isfinite(response), response, 1.0 + 0j)
    return response


def apply_bandpass(visibilities: np.ndarray, bandpass: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply a complex bandpass correction along the selected frequency axis."""
    visibilities = np.asarray(visibilities)
    bandpass = np.asarray(bandpass)
    if axis < 0:
        axis += visibilities.ndim
    if bandpass.ndim != 1:
        bandpass = bandpass.reshape(-1)
    if visibilities.shape[axis] != bandpass.shape[0]:
        raise ValueError(f"Bandpass length {bandpass.shape[0]} does not match axis length {visibilities.shape[axis]}")
    shape = [1] * visibilities.ndim
    shape[axis] = bandpass.shape[0]
    denom = bandpass.reshape(shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = visibilities / denom
    corrected = np.asarray(corrected)
    corrected[~np.isfinite(corrected)] = 0.0
    return corrected


def estimate_delay_phase_slope(visibilities: np.ndarray, frequencies_hz: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Estimate delay from phase slope versus frequency.

    Delay is returned in seconds using phase = 2*pi*frequency*delay.
    """
    phase = np.unwrap(np.angle(visibilities), axis=axis)
    moved = np.moveaxis(phase, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    delays = []
    for row in flat:
        if np.all(np.isfinite(row)):
            slope, _ = np.polyfit(frequencies_hz, row, 1)
            delays.append(slope / (2 * np.pi))
        else:
            delays.append(0.0)
    return np.asarray(delays).reshape(moved.shape[:-1])


def apply_delay_correction(visibilities: np.ndarray, frequencies_hz: np.ndarray, delays_s: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply delay correction derived from phase slope."""
    moved = np.moveaxis(visibilities, axis, -1)
    delay_flat = delays_s.reshape(-1)
    vis_flat = moved.reshape(-1, moved.shape[-1])
    phase = np.exp(-2j * np.pi * frequencies_hz[np.newaxis, :] * delay_flat[:, np.newaxis])
    corrected = (vis_flat * phase).reshape(moved.shape)
    return np.moveaxis(corrected, -1, axis)


def compute_calibration_residuals(visibilities: np.ndarray, model_visibilities: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Return residual visibilities plus amplitude/phase residual summaries."""
    measured = np.asarray(visibilities)
    model = np.asarray(model_visibilities)
    if measured.shape != model.shape:
        raise ValueError(f"Measured shape {measured.shape} must match model shape {model.shape}")
    residual = measured - model
    amplitude_residual = np.abs(residual)
    phase_residual = np.angle(measured) - np.angle(model)
    phase_residual = np.angle(np.exp(1j * phase_residual))
    return {
        "residual_visibilities": residual,
        "amplitude_residual": amplitude_residual,
        "phase_residual": phase_residual,
        "mean_amplitude_residual": float(np.mean(amplitude_residual)) if amplitude_residual.size else 0.0,
        "rms_amplitude_residual": float(np.sqrt(np.mean(np.square(amplitude_residual)))) if amplitude_residual.size else 0.0,
        "phase_rms": float(np.sqrt(np.mean(np.square(phase_residual)))) if phase_residual.size else 0.0,
    }


def summarize_gain_stability(gains: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray | float]:
    """Summarize gain amplitudes and phases for drift inspection."""
    arr = np.asarray(gains)
    amp = np.abs(arr)
    phase = np.angle(arr)
    amp_mean = np.mean(amp, axis=axis)
    amp_std = np.std(amp, axis=axis)
    phase_mean = np.mean(phase, axis=axis)
    phase_std = np.std(phase, axis=axis)
    return {
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        "mean_amplitude": float(np.mean(amp)) if amp.size else 0.0,
        "std_amplitude": float(np.std(amp)) if amp.size else 0.0,
    }


def summarize_phase_drift(visibilities: np.ndarray, axis: int = 1) -> Dict[str, np.ndarray | float]:
    """Inspect phase drift across the selected axis with an unwrapped phase trend."""
    phase = np.unwrap(np.angle(np.asarray(visibilities)), axis=axis)
    moved = np.moveaxis(phase, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    slopes = []
    for row in flat:
        if row.size < 2 or not np.all(np.isfinite(row)):
            slopes.append(0.0)
            continue
        x = np.arange(row.size, dtype=float)
        slope, _ = np.polyfit(x, row, 1)
        slopes.append(float(slope))
    slopes_arr = np.asarray(slopes).reshape(moved.shape[:-1])
    return {
        "unwrapped_phase": phase,
        "phase_slope": slopes_arr,
        "mean_slope": float(np.mean(slopes_arr)) if slopes_arr.size else 0.0,
        "rms_slope": float(np.sqrt(np.mean(np.square(slopes_arr)))) if slopes_arr.size else 0.0,
    }


def hogbom_clean(dirty_image: np.ndarray,
                 psf: np.ndarray,
                 gain: float = 0.1,
                 threshold: float | None = None,
                 max_iterations: int = 1000) -> Dict:
    """
    Minimal Hogbom CLEAN implementation.

    Returns component and residual images. Restoring beam convolution is left for
    future CASA/WSClean integration rather than pretending a beam model exists.
    """
    residual = dirty_image.astype(float, copy=True)
    components = np.zeros_like(residual)
    psf_peak = np.unravel_index(np.argmax(np.abs(psf)), psf.shape)
    threshold = float(threshold if threshold is not None else 0.01 * np.nanmax(np.abs(dirty_image)))

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak_val = residual[peak_idx]
        if abs(peak_val) < threshold:
            break

        component = gain * peak_val
        components[peak_idx] += component

        shift = (peak_idx[0] - psf_peak[0], peak_idx[1] - psf_peak[1])
        shifted_psf = np.roll(np.roll(psf, shift[0], axis=0), shift[1], axis=1)
        residual -= component * shifted_psf

    return {
        "components": components,
        "residual": residual,
        "iterations": iterations,
        "threshold": threshold,
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_interferometry_data(visibilities: np.ndarray,
                                  u_coords: np.ndarray,
                                  v_coords: np.ndarray) -> Dict:
    """
    Validate interferometry data for common issues.
    
    Args:
        visibilities: Visibility array
        u_coords: U coordinates
        v_coords: V coordinates
    
    Returns:
        dict: Validation results with warnings and errors
    """
    arr = np.asarray(visibilities)
    u = np.asarray(u_coords).reshape(-1)
    v = np.asarray(v_coords).reshape(-1)

    validation = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "visibility_shape": tuple(arr.shape),
        "time_axis_length": int(arr.shape[0]) if arr.ndim >= 1 else 0,
        "frequency_axis_length": int(arr.shape[1]) if arr.ndim >= 2 else 0,
        "baseline_axis_length": int(arr.shape[2]) if arr.ndim >= 3 else int(arr.shape[0]) if arr.ndim == 1 else 0,
        "polarization_axis_length": int(arr.shape[3]) if arr.ndim >= 4 else 0,
    }

    expected_baselines = validation["baseline_axis_length"]
    if expected_baselines <= 0:
        validation["is_valid"] = False
        validation["errors"].append("Visibility tensor must have at least one sample and a baseline axis")
    elif len(u) != expected_baselines or len(v) != expected_baselines:
        validation["is_valid"] = False
        validation["errors"].append(
            f"Visibility baseline axis length ({expected_baselines}) does not match uv coordinate lengths ({len(u)}, {len(v)})"
        )

    # Check for NaN values
    if np.any(np.isnan(arr)):
        validation["warnings"].append(f"{np.sum(np.isnan(arr))} NaN values in visibilities")
    
    # Check for infinite values
    if np.any(np.isinf(arr)):
        validation["warnings"].append(f"{np.sum(np.isinf(arr))} infinite values in visibilities")
    
    # Check UV coverage
    if u.size and v.size:
        uv_radius = np.sqrt(u**2 + v**2)
        if np.max(uv_radius) < 10:
            validation["warnings"].append("UV coverage is very small (< 10 wavelengths)")

    # Check for zero visibilities
    zero_fraction = np.sum(arr == 0) / arr.size if arr.size else 1.0
    if zero_fraction > 0.5:
        validation["warnings"].append(f"{zero_fraction*100:.1f}% of visibilities are zero")

    if validation["errors"]:
        validation["is_valid"] = False
    
    return validation
