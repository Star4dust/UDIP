"""\
Advanced RFI detection routines.

Capabilities:
  * Spectral kurtosis (SK) estimator - the canonical weak-signal RFI
    flagger for radio-astronomy waterfalls. Returns a 2D SK map and a
    boolean outlier mask using the Nita & Gary (2010) thresholds.
  * Drifting / satellite-track RFI detection using a 2D Hough-style
    transform on time-frequency maps. Returns the dominant drift
    rate in Hz/s and a per-track mask.

Both routines work on time-frequency matrices and integrate with the
existing RFI masking pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Spectral kurtosis
# ---------------------------------------------------------------------------

def spectral_kurtosis_map(
    waterfall: np.ndarray,
    *,
    window_time: int = 8,
    window_freq: int = 1,
) -> dict[str, Any]:
    """Compute the spectral kurtosis estimator on a waterfall.

    For a 2D waterfall W(t, f) of complex or real values, SK is
        SK = (M * sum_i |x_i|^2) / (|sum_i x_i|^2) - 1
    over a sliding rectangular window of M samples. SK ~ 1 for
    Gaussian noise (with the bias correction in Nita & Gary 2010).

    Returns
    -------
    dict with:
        sk_map : 2D ndarray of SK values
        threshold_low, threshold_high : bounds (Nita 2010)
        rfi_mask : 2D boolean mask of SK outliers
        n_flagged : int
        warnings : list[str]
    """
    w = np.asarray(waterfall, dtype=float)
    out_warnings: list[str] = []
    if w.ndim != 2:
        out_warnings.append("waterfall must be 2D")
        return {
            "sk_map": np.zeros_like(w) if w.ndim == 2 else np.array([]),
            "threshold_low": float("nan"),
            "threshold_high": float("nan"),
            "rfi_mask": np.zeros(w.shape, dtype=bool) if w.ndim == 2 else np.array([], dtype=bool),
            "n_flagged": 0,
            "warnings": out_warnings,
        }
    if window_time < 2:
        window_time = 2
    if window_freq < 1:
        window_freq = 1
    n_t, n_f = w.shape
    # For SK we use the squared magnitude.
    p = np.abs(w) ** 2
    # Use a 2D cumulative sum for fast sliding means.
    cs1 = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs2 = np.cumsum(np.cumsum(p ** 2, axis=0), axis=1)
    sk = np.zeros_like(p)
    for i in range(window_time - 1, n_t):
        for j in range(window_freq - 1, n_f):
            i0, j0 = i - window_time, j - window_freq
            s1 = cs1[i, j]
            s2 = cs2[i, j]
            if i0 >= 0:
                s1 -= cs1[i0, j]
                s2 -= cs2[i0, j]
            if j0 >= 0:
                s1 -= cs1[i, j0]
                s2 -= cs2[i, j0]
            if i0 >= 0 and j0 >= 0:
                s1 += cs1[i0, j0]
                s2 += cs2[i0, j0]
            m = float(window_time) * float(window_freq)
            denom = s1 ** 2
            if denom > 0 and m > 1.0:
                # Nita & Gary (2010) corrected SK estimator. For pure
                # Gaussian noise this is ~1; outliers (RFI bursts) push
                # the ratio far above 1.
                sk[i, j] = (m + 1.0) / (m - 1.0) * (m * s2 / denom) - (m + 1.0) / (m - 1.0)
            elif denom > 0:
                # m <= 1: undefined. Mark as 1.0 (neutral).
                sk[i, j] = 1.0
    # Nita & Gary 2010 threshold: 1 +/- 1/sqrt(M).
    m = float(window_time) * float(window_freq)
    th_lo = 1.0 - 1.0 / math.sqrt(m) if m > 0 else 0.5
    th_hi = 1.0 + 1.0 / math.sqrt(m) if m > 0 else 1.5
    # Use a robust criterion in addition to the textbook bounds: flag
    # cells where SK deviates from the median by > 5 * MAD. This catches
    # RFI that the small-M formula can't separate from the bulk because
    # the SK distribution is heavy-tailed for real radio data.
    finite_sk = sk[np.isfinite(sk)]
    if finite_sk.size:
        med = float(np.median(finite_sk))
        mad = float(np.median(np.abs(finite_sk - med)))
        if mad > 0:
            dev_mask = np.abs(sk - med) > 5.0 * 1.4826 * mad
        else:
            dev_mask = np.zeros_like(sk, dtype=bool)
    else:
        dev_mask = np.zeros_like(sk, dtype=bool)
    thresh_mask = (sk < th_lo) | (sk > th_hi)
    rfi_mask = thresh_mask | dev_mask
    return {
        "sk_map": sk,
        "threshold_low": float(th_lo),
        "threshold_high": float(th_hi),
        "robust_mad": float(mad) if finite_sk.size and mad > 0 else float("nan"),
        "rfi_mask": rfi_mask,
        "n_flagged": int(np.sum(rfi_mask)),
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Drifting / satellite RFI via 2D Hough transform
# ---------------------------------------------------------------------------

def _hough_line_kernel(rows: int, cols: int) -> np.ndarray:
    """Hough accumulator axis convention: rho in [-max, max], theta in [0, pi)."""
    diag = int(math.ceil(math.sqrt(rows * rows + cols * cols)))
    return np.zeros((2 * diag + 1, 180), dtype=float)


def _draw_line_in_hough(rho_bins: int, theta_count: int, x: float, y: float) -> np.ndarray:
    """Accumulate a single point (x, y) over theta in the Hough space."""
    acc = np.zeros((rho_bins, theta_count), dtype=float)
    for ti in range(theta_count):
        theta = math.pi * ti / theta_count
        rho = x * math.cos(theta) + y * math.sin(theta)
        ri = int(round(rho)) + rho_bins // 2
        if 0 <= ri < rho_bins:
            acc[ri, ti] += 1.0
    return acc


def drifting_rfi_detection(
    waterfall: np.ndarray,
    *,
    time_step_s: float = 1.0,
    freq_step_hz: float = 1.0,
    drift_rate_min_hz_per_s: float = 1e-3,
    drift_rate_max_hz_per_s: float = 1.0,
    time_axis_seconds: np.ndarray | None = None,
    frequency_axis_hz: np.ndarray | None = None,
) -> dict[str, Any]:
    """Detect drifting / satellite-track RFI in a 2D time-frequency map.

    Algorithm:
      1. Compute the per-time, per-freq power.
      2. Find the strongest non-Gaussian pixels (|z|>3).
      3. Run a 2D Hough transform on (time_idx, freq_idx) for those
         pixels. The dominant line gives a drift rate in Hz/s.

    Parameters
    ----------
    waterfall : ndarray
        2D time-frequency matrix.
    time_step_s, freq_step_hz : float
        Sample spacings used to convert (row, col) to (s, Hz).
    drift_rate_min_hz_per_s, drift_rate_max_hz_per_s : float
        Range of plausible drift rates (default 1e-3 .. 1 Hz/s).
    time_axis_seconds, frequency_axis_hz : ndarray, optional
        Explicit axes (override step sizes). Not yet used beyond
        keeping the metadata coherent.

    Returns
    -------
    dict with:
        drift_rate_hz_per_s : float
        drift_rate_hz_per_min : float
        dominant_line : tuple (slope, intercept) in (Hz, s)
        rfi_mask : 2D boolean mask covering the dominant line
        n_flagged : int
        warnings : list[str]
    """
    w = np.asarray(waterfall, dtype=float)
    out_warnings: list[str] = []
    if w.ndim != 2:
        out_warnings.append("waterfall must be 2D")
        return {
            "drift_rate_hz_per_s": float("nan"),
            "drift_rate_hz_per_min": float("nan"),
            "dominant_line": (float("nan"), float("nan")),
            "rfi_mask": np.zeros(w.shape, dtype=bool) if w.ndim == 2 else np.array([], dtype=bool),
            "n_flagged": 0,
            "warnings": out_warnings,
        }
    p = np.abs(w)
    # Robust z-score per row (time sample).
    mu = np.median(p, axis=1, keepdims=True)
    sigma = np.median(np.abs(p - mu), axis=1, keepdims=True) * 1.4826
    sigma = np.where(sigma > 0, sigma, 1.0)
    z = (p - mu) / sigma
    pixels = np.argwhere(z > 3.0)
    if pixels.size == 0:
        out_warnings.append("no non-Gaussian pixels above z=3")
        return {
            "drift_rate_hz_per_s": float("nan"),
            "drift_rate_hz_per_min": float("nan"),
            "dominant_line": (float("nan"), float("nan")),
            "rfi_mask": np.zeros(w.shape, dtype=bool),
            "n_flagged": 0,
            "warnings": out_warnings,
        }
    rows, cols = w.shape
    rho_bins = 2 * int(math.ceil(math.sqrt(rows * rows + cols * cols))) + 1
    theta_count = 180
    acc = _hough_line_kernel(rows, cols)
    for y, x in pixels:
        acc += _draw_line_in_hough(rho_bins, theta_count, float(x), float(y))
    # Find the dominant peak in Hough space.
    idx = np.unravel_index(int(np.argmax(acc)), acc.shape)
    rho_peak = idx[0] - rho_bins // 2
    theta_peak = math.pi * idx[1] / theta_count
    # Convert to image-plane line: y = m x + b
    sin_t = math.sin(theta_peak)
    if abs(sin_t) < 1e-6:
        out_warnings.append("nearly vertical Hough peak; drift rate undefined")
        slope_per_step = float("inf")
        intercept_per_step = float("nan")
    else:
        slope_per_step = -math.cos(theta_peak) / sin_t       # df/dt (per pixel)
        intercept_per_step = rho_peak / sin_t
    # Convert to physical units (Hz/s).
    slope = slope_per_step * (freq_step_hz / time_step_s)
    intercept = intercept_per_step * freq_step_hz
    drift_hz_per_s = slope
    drift_hz_per_min = drift_hz_per_s * 60.0

    # Build a mask of points within +/- 1 pixel of the dominant line.
    if abs(sin_t) >= 1e-6:
        yy, xx = np.indices(w.shape, dtype=float)
        line_y = slope_per_step * xx + intercept_per_step
        mask = np.abs(yy - line_y) <= 1.5
    else:
        # Vertical line: mask the column.
        col_int = int(round(rho_peak)) if abs(sin_t) < 1e-6 else 0
        col_int = max(0, min(col_int, cols - 1))
        mask = np.zeros(w.shape, dtype=bool)
        mask[:, col_int] = True

    # Sanity-check against plausible drift band.
    if not (drift_rate_min_hz_per_s <= abs(drift_hz_per_s) <= drift_rate_max_hz_per_s):
        out_warnings.append(
            f"dominant drift rate {drift_hz_per_s:.3g} Hz/s outside plausible band"
        )

    return {
        "drift_rate_hz_per_s": float(drift_hz_per_s),
        "drift_rate_hz_per_min": float(drift_hz_per_min),
        "dominant_line": (float(slope), float(intercept)),
        "hough_peak": (int(idx[0]), int(idx[1])),
        "rfi_mask": mask,
        "n_flagged": int(np.sum(mask)),
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def advanced_rfi_summary(
    waterfall: np.ndarray,
    *,
    time_step_s: float = 1.0,
    freq_step_hz: float = 1.0,
    sk_window_time: int = 8,
) -> dict[str, Any]:
    """One-call: spectral kurtosis + drifting RFI on the same waterfall."""
    out_warnings: list[str] = []
    sections: dict[str, Any] = {}
    sk = spectral_kurtosis_map(waterfall, window_time=sk_window_time)
    out_warnings.extend(sk["warnings"])
    sections["spectral_kurtosis"] = sk

    dr = drifting_rfi_detection(
        waterfall, time_step_s=time_step_s, freq_step_hz=freq_step_hz
    )
    out_warnings.extend(dr["warnings"])
    sections["drifting_rfi"] = dr

    # Union mask: SK outliers OR drifting track.
    if sk["rfi_mask"].size and dr["rfi_mask"].size and sk["rfi_mask"].shape == dr["rfi_mask"].shape:
        union = sk["rfi_mask"] | dr["rfi_mask"]
    else:
        union = sk["rfi_mask"]
    sections["union_mask"] = union
    sections["n_flagged_union"] = int(np.sum(union))
    sections["_warnings"] = out_warnings
    return sections
