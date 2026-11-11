"""\
HIRAX beam-geometry diagnostics.

Capabilities:
  * Per-dish beam main-axis direction from a calibrator transit.
  * Beam squint estimate (offset between two polarisation main axes).
  * Theoretical beam FWHM from dish diameter and frequency.
  * Galactic-transit detection (sinusoidal fit to time series of
    brightness temperature).
  * Expected vs measured peak comparison -> angular offset in degrees.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

# Speed of light (m/s).
_C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------------------
# Theoretical beam parameters
# ---------------------------------------------------------------------------

def theoretical_beam_fwhm_deg(dish_diameter_m: float, frequency_mhz: float) -> dict[str, Any]:
    """Airy pattern FWHM ~ 1.22 * lambda / D (in radians)."""
    out_warnings: list[str] = []
    d, d_ok = _safe_float(dish_diameter_m, float("nan"))
    f, f_ok = _safe_float(frequency_mhz, float("nan"))
    if not d_ok or not f_ok or d <= 0 or f <= 0:
        out_warnings.append("dish_diameter or frequency invalid; FWHM undefined")
        return {"fwhm_deg": float("nan"), "fwhm_arcmin": float("nan"), "warnings": out_warnings}
    lam = _C_LIGHT / (f * 1.0e6)
    fwhm_rad = 1.22 * lam / d
    fwhm_deg = math.degrees(fwhm_rad)
    return {
        "fwhm_deg": fwhm_deg,
        "fwhm_arcmin": fwhm_deg * 60.0,
        "fwhm_rad": fwhm_rad,
        "wavelength_m": lam,
        "dish_diameter_m": d,
        "frequency_mhz": f,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Transit (sinusoidal) fit
# ---------------------------------------------------------------------------

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


def fit_galactic_transit(
    time_hours: np.ndarray,
    t_measured_k: np.ndarray,
    *,
    period_hours: float = 23.9344696,   # sidereal day
) -> dict[str, Any]:
    """Fit a sinusoid of fixed period (sidereal) to T_measured(t).

    Returns fitted amplitude, phase, mean level, and the time of the
    peak. If the fit fails to converge, returns NaNs and warnings.
    """
    t = np.asarray(time_hours, dtype=float).ravel()
    y = np.asarray(t_measured_k, dtype=float).ravel()
    out_warnings: list[str] = []
    if t.size < 4 or y.size < 4:
        out_warnings.append("not enough samples for transit fit")
        return {
            "amplitude_k": float("nan"),
            "phase_rad": float("nan"),
            "mean_k": float("nan"),
            "peak_time_hours": float("nan"),
            "fit_ok": False,
            "warnings": out_warnings,
        }
    n = min(t.size, y.size)
    t = t[:n] - float(np.mean(t[:n]))
    y = y[:n]
    valid = np.isfinite(y)
    if not np.any(valid) or np.sum(valid) < 4:
        out_warnings.append("no finite samples for transit fit")
        return {
            "amplitude_k": float("nan"),
            "phase_rad": float("nan"),
            "mean_k": float("nan"),
            "peak_time_hours": float("nan"),
            "fit_ok": False,
            "warnings": out_warnings,
        }
    t = t[valid]
    y = y[valid]
    omega = 2.0 * math.pi / period_hours
    # Linear least squares for sin, cos, offset.
    M = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    a_sin, a_cos, offset = float(coef[0]), float(coef[1]), float(coef[2])
    amp = math.hypot(a_sin, a_cos)
    phase = math.atan2(a_cos, a_sin)            # at t=0 (centred)
    # Peak: omega*t + phase_offset = pi/2  -> t = (pi/2 - atan2(a_cos, a_sin)) / omega
    peak_t_rel = (math.pi / 2.0 - math.atan2(a_cos, a_sin)) / omega
    peak_t_abs = peak_t_rel + float(np.mean(np.asarray(time_hours, dtype=float).ravel()[:n]))
    return {
        "amplitude_k": float(amp),
        "phase_rad": float(phase),
        "mean_k": float(offset),
        "peak_time_hours": float(peak_t_abs),
        "fit_ok": True,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Beam squint estimate
# ---------------------------------------------------------------------------

def beam_squint_offset(
    pol_a_main_axis: tuple[float, float, float],
    pol_b_main_axis: tuple[float, float, float],
) -> dict[str, Any]:
    """Compute the angular offset between two polarisation main axes.

    Inputs are unit vectors (or any non-zero vectors) in any consistent
    coordinate frame. Returned offset is in degrees.
    """
    a = np.asarray(pol_a_main_axis, dtype=float)
    b = np.asarray(pol_b_main_axis, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    out_warnings: list[str] = []
    if na == 0 or nb == 0:
        out_warnings.append("zero-length main axis vector; offset undefined")
        return {"offset_deg": float("nan"), "offset_arcmin": float("nan"), "warnings": out_warnings}
    a = a / na
    b = b / nb
    cosang = float(np.clip(np.dot(a, b), -1.0, 1.0))
    offset_rad = math.acos(cosang)
    offset_deg = math.degrees(offset_rad)
    return {
        "offset_deg": offset_deg,
        "offset_arcmin": offset_deg * 60.0,
        "pol_a_main_axis_unit": a.tolist(),
        "pol_b_main_axis_unit": b.tolist(),
        "warnings": out_warnings,
    }


def beam_squint_from_calibrator_transits(
    peak_time_hours_pol_a: float,
    peak_time_hours_pol_b: float,
    *,
    hour_angle_deg_per_hour: float = 15.041,  # Earth rotation rate
) -> dict[str, Any]:
    """Infer beam squint from a single calibrator observed in two pols.

    The hour-angle offset between the two polarisation peaks gives a
    direct estimate of the on-sky separation along the scan direction.
    """
    a, a_ok = _safe_float(peak_time_hours_pol_a, float("nan"))
    b, b_ok = _safe_float(peak_time_hours_pol_b, float("nan"))
    out_warnings: list[str] = []
    if not a_ok or not b_ok or not math.isfinite(a) or not math.isfinite(b):
        out_warnings.append("non-finite peak times; offset undefined")
        return {
            "offset_deg": float("nan"),
            "offset_arcmin": float("nan"),
            "delta_time_hours": float("nan"),
            "warnings": out_warnings,
        }
    dt_hours = a - b
    ha_deg = dt_hours * hour_angle_deg_per_hour
    return {
        "offset_deg": abs(ha_deg),
        "offset_arcmin": abs(ha_deg) * 60.0,
        "delta_time_hours": dt_hours,
        "hour_angle_rate_deg_per_hour": hour_angle_deg_per_hour,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Peak-location comparison
# ---------------------------------------------------------------------------

def expected_galactic_peak_hour_angle(
    galactic_longitude_deg: float,
    galactic_latitude_deg: float,
    observer_longitude_deg: float,
    *,
    sidereal_day_hours: float = 23.9344696,
) -> dict[str, Any]:
    """Return the expected peak hour-angle in degrees for a given pointing.

    Simplified: we use a first-order meridian-transit approximation
    where the peak hour-angle ~ (longitude - 0) for a b=0 source. This
    is sufficient as a first-pass forecast; full Sidereal Time -> HA
    conversion requires a date which we do not propagate here.
    """
    l, _ = _safe_float(galactic_longitude_deg, 0.0)
    b, _ = _safe_float(galactic_latitude_deg, 0.0)
    o, _ = _safe_float(observer_longitude_deg, 0.0)
    out_warnings: list[str] = []
    if abs(b) > 20:
        out_warnings.append("|b|>20 deg: expected peak is approximate; off-plane boost weak")
    # Map Galactic longitude -> hour-angle at transit (approx).
    ha_peak = l - o
    # Wrap to (-180, 180].
    ha_peak = ((ha_peak + 180.0) % 360.0) - 180.0
    peak_time_hours = ha_peak / 15.041  # convert deg to hours
    return {
        "expected_peak_hour_angle_deg": ha_peak,
        "expected_peak_time_hours": peak_time_hours,
        "sidereal_day_hours": sidereal_day_hours,
        "warnings": out_warnings,
    }


def compare_peak_locations(
    expected_peak_time_hours: float,
    measured_peak_time_hours: float,
    *,
    hour_angle_deg_per_hour: float = 15.041,
) -> dict[str, Any]:
    """Angular difference between expected and measured peak locations."""
    e, e_ok = _safe_float(expected_peak_time_hours, float("nan"))
    m, m_ok = _safe_float(measured_peak_time_hours, float("nan"))
    out_warnings: list[str] = []
    if not e_ok or not m_ok or not math.isfinite(e) or not math.isfinite(m):
        out_warnings.append("non-finite peak times; angular offset undefined")
        return {"angular_offset_deg": float("nan"), "warnings": out_warnings}
    delta_hours = m - e
    delta_deg = delta_hours * hour_angle_deg_per_hour
    return {
        "angular_offset_deg": abs(delta_deg),
        "delta_time_hours": delta_hours,
        "hour_angle_deg_per_hour": hour_angle_deg_per_hour,
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Pipeline convenience
# ---------------------------------------------------------------------------

def beam_squint_summary(
    *,
    dish_diameter_m: float | None,
    frequency_mhz: float | None,
    attrs: Mapping[str, Any] | None,
    pol_a_transit: tuple[np.ndarray, np.ndarray] | None = None,
    pol_b_transit: tuple[np.ndarray, np.ndarray] | None = None,
    expected: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """One-call beam squint + transit + expected-vs-measured summary."""
    out_warnings: list[str] = []
    sections: dict[str, Any] = {}

    # Theoretical FWHM.
    fwhm = theoretical_beam_fwhm_deg(dish_diameter_m or float("nan"), frequency_mhz or float("nan"))
    out_warnings.extend(fwhm["warnings"])
    sections["theoretical_fwhm"] = fwhm

    # Per-pol transit fits.
    a_fit = {"fit_ok": False, "warnings": ["no data"]}
    b_fit = {"fit_ok": False, "warnings": ["no data"]}
    if pol_a_transit is not None and pol_b_transit is not None:
        a_fit = fit_galactic_transit(pol_a_transit[0], pol_a_transit[1])
        b_fit = fit_galactic_transit(pol_b_transit[0], pol_b_transit[1])
        out_warnings.extend(a_fit["warnings"])
        out_warnings.extend(b_fit["warnings"])
    sections["transit_pol_a"] = a_fit
    sections["transit_pol_b"] = b_fit

    # Squint.
    if a_fit.get("fit_ok") and b_fit.get("fit_ok"):
        squint = beam_squint_from_calibrator_transits(
            a_fit["peak_time_hours"], b_fit["peak_time_hours"]
        )
        out_warnings.extend(squint["warnings"])
    else:
        squint = {"offset_deg": float("nan"), "warnings": ["transit fits unavailable"]}
    sections["beam_squint"] = squint

    # Expected vs measured.
    if expected and a_fit.get("fit_ok"):
        expected_peak = expected.get("expected_peak_time_hours")
        if expected_peak is not None:
            cmp = compare_peak_locations(expected_peak, a_fit["peak_time_hours"])
            out_warnings.extend(cmp["warnings"])
        else:
            cmp = {"angular_offset_deg": float("nan"), "warnings": ["expected peak missing"]}
    else:
        cmp = {"angular_offset_deg": float("nan"), "warnings": ["insufficient data for comparison"]}
    sections["expected_vs_measured"] = cmp

    sections["_warnings"] = out_warnings
    return sections
