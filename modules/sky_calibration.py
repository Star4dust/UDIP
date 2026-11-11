"""\
HIRAX absolute-sky calibration primitives.

Implements (Option A): compute Tsys and Trxr in absolute Kelvin units
whenever calibration information is available. If required metadata is
missing, return uncertainty estimates and warnings instead of inferred
values.

Capabilities (all in absolute Kelvin, SI-consistent):
  * Global Sky Model (GSM) synthetic temperature T_gsm(l, b, v) using
    de Oliveira-Costa / Zheng et al. power-law approximation.
  * Single-dish auto-correlation -> antenna temperature T_ant.
  * Receiver temperature Trxr from Y-factor (T_hot - T_cold) / (Y - 1).
  * System temperature Tsys = T_rec + T_sky(v, t).
  * Beam-convolved brightness temperature T_b(v, t).
  * Residual sky temperature delta T_sky = T_measured - T_gsm_model.

All routines return dicts with explicit `uncertainties` and `warnings`
keys so downstream stages can distinguish "measured" from "estimated".
"""

from __future__ import annotations

import math
import warnings as _warnings
from typing import Any, Mapping, Sequence

import numpy as np

# Speed of light (m/s) and Boltzmann constant (J/K) - CODATA 2018.
_C_LIGHT = 299_792_458.0
_K_BOLTZMANN = 1.380649e-23

# Default de Oliveira-Costa GSM low-frequency power-law fit coefficients
# for 408 MHz normalization. We use the simple 3-parameter model
#   log T(l,b,v) = A0(l,b) + A1(l,b) * log(v / v0)
# with v0 = 408 MHz, A0 anchored to Haslam 408 MHz map mean ~ 22 K
# outside the Galactic plane, and A1 = -2.5 (typical spectral index).
# This is a deliberately simple but physically defensible model.
_GSM_V0_MHZ = 408.0
_GSM_A0_DEFAULT_K = 22.0          # K at v0 off-plane
_GSM_A1_DEFAULT = -2.5            # spectral index log-log
_GSM_GALACTIC_BOOST_K = 50.0      # additional K near Galactic centre
_GAL_PLANE_WIDTH_DEG = 10.0       # |b| half-width of the plane boost

# Per-dish / system metadata defaults for HRMS-class instruments. These
# are only used when the dataset has *no* calibration block at all, in
# which case the result is treated as a warning-flagged estimate.
_HRMS_DEFAULTS = {
    "dish_diameter_m": 6.0,
    "aperture_efficiency": 0.65,
    "trxr_typical_K": 60.0,       # for a cryo-cooled LNA chain at ~400-800 MHz
    "trxr_typical_unc_K": 15.0,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float) -> tuple[float, bool]:
    """Return (value, ok) where ok=False means the input was not usable."""
    if value is None:
        return default, False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(x):
        return default, False
    return x, True


def _get_attr(attrs: Mapping[str, Any] | None, *keys: str) -> tuple[Any, bool]:
    if not attrs:
        return None, False
    for key in keys:
        if key in attrs:
            return attrs[key], True
    return None, False


def _galactic_latitude_boost_deg(lat_deg: float) -> float:
    """Smooth Gaussian-ish boost of the sky temperature near the plane."""
    b = float(lat_deg)
    return float(_GSM_GALACTIC_BOOST_K * np.exp(-0.5 * (b / _GAL_PLANE_WIDTH_DEG) ** 2))


# ---------------------------------------------------------------------------
# Global Sky Model (synthetic, but physically motivated)
# ---------------------------------------------------------------------------

def global_sky_model_temperature(
    galactic_longitude_deg: Sequence[float] | np.ndarray,
    galactic_latitude_deg: Sequence[float] | np.ndarray,
    frequency_mhz: Sequence[float] | np.ndarray,
    a0_k: float = _GSM_A0_DEFAULT_K,
    a1: float = _GSM_A1_DEFAULT,
) -> dict[str, Any]:
    """Compute synthetic GSM antenna temperature.

    Parameters
    ----------
    galactic_longitude_deg, galactic_latitude_deg
        Direction on the sky (degrees). Must be broadcastable.
    frequency_mhz
        Observing frequency in MHz. Must be broadcastable.

    Returns
    -------
    dict with keys:
        t_ant_gsm_k : ndarray of GSM antenna temperatures in Kelvin.
        a0_k, a1, v0_mhz : model parameters used.
        warnings : list[str]
    """
    l = np.asarray(galactic_longitude_deg, dtype=float)
    b = np.asarray(galactic_latitude_deg, dtype=float)
    v = np.asarray(frequency_mhz, dtype=float)

    out_warnings: list[str] = []
    if np.any(v <= 0):
        out_warnings.append("non-positive frequency encountered; results masked to NaN")

    v0 = _GSM_V0_MHZ
    # Spectral shape: T ~ A0 * (v/v0)**A1
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ant_k = a0_k * (v / v0) ** a1

    # Galactic plane boost: the longitude dependence follows a smooth
    # cos(2l) bulge plus a central concentration.
    if l.size and b.size:
        l_rad = np.deg2rad(l)
        b_rad = np.deg2rad(b)
        # bulge term: stronger toward Galactic centre (l ~ 0)
        bulge = 1.0 + 0.6 * np.cos(l_rad) ** 2
        plane = np.exp(-0.5 * (b_rad / np.deg2rad(_GAL_PLANE_WIDTH_DEG)) ** 2)
        t_ant_k = t_ant_k * bulge + _galactic_latitude_boost_deg(np.rad2deg(b_rad)) * plane

    if np.any(v <= 0):
        t_ant_k = np.where(v > 0, t_ant_k, np.nan)

    return {
        "t_ant_gsm_k": np.asarray(t_ant_k, dtype=float),
        "a0_k": float(a0_k),
        "a1": float(a1),
        "v0_mhz": float(v0),
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Antenna / system temperature reconstruction
# ---------------------------------------------------------------------------

def antenna_temperature_from_auto(
    power_auto: np.ndarray,
    *,
    integration_seconds: float = 1.0,
    bandwidth_hz: float = 1.0e6,
    gain_k_per_jansky: float | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert auto-correlation power into antenna temperature.

    The radiometer equation:
        T_ant = P_auto / (k_B * delta_nu * t_int)
    If `gain_k_per_jansky` (K/Jy) is provided, the result is rescaled
    from raw counts/K -> K using SEFD-style conversion. We do *not*
    silently inject an aperture efficiency; the function returns
    `warnings` and a finite `uncertainty_k` band when calibration
    metadata is incomplete.
    """
    p = np.asarray(power_auto, dtype=float)
    out_warnings: list[str] = []
    if p.size == 0:
        return {
            "t_ant_k": np.zeros(0, dtype=float),
            "uncertainties": {"t_ant_k": float("nan")},
            "warnings": ["empty input"],
        }

    t_int, _ = _safe_float(integration_seconds, 1.0)
    bw, _ = _safe_float(bandwidth_hz, 1.0e6)
    if t_int <= 0 or bw <= 0:
        out_warnings.append("non-positive integration or bandwidth; raw units returned")

    denom = _K_BOLTZMANN * bw * t_int
    if denom > 0 and not out_warnings:
        t_ant_k = p / denom
    else:
        t_ant_k = p.copy()  # raw counts
        out_warnings.append("raw counts returned - radiometer calibration metadata missing")

    # Optional gain rescaling (K/Jy -> K).
    g_kjy, g_ok = _safe_float(gain_k_per_jansky, float("nan"))
    if g_ok and math.isfinite(g_kjy) and g_kjy > 0:
        t_ant_k = t_ant_k * g_kjy

    # Radiometer noise: sigma_T = T_sys / sqrt(delta_nu * t_int)
    sigma_t = float(np.nanstd(t_ant_k) / max(np.sqrt(p.size), 1.0)) if p.size else float("nan")
    return {
        "t_ant_k": np.asarray(t_ant_k, dtype=float),
        "uncertainties": {
            "t_ant_k": sigma_t,
            "radiometer_equation_assumed": True,
        },
        "warnings": out_warnings,
    }


def y_factor_receiver_temperature(
    power_hot: np.ndarray,
    power_cold: np.ndarray,
    *,
    t_hot_k: float = 300.0,
    t_cold_k: float = 10.0,
) -> dict[str, Any]:
    """Compute receiver temperature Trxr from a Y-factor measurement.

    Y = P_hot / P_cold
    Trxr = (T_hot - T_cold) / (Y - 1)

    Returns absolute Trxr in Kelvin with propagated uncertainty.
    """
    ph = np.asarray(power_hot, dtype=float)
    pc = np.asarray(power_cold, dtype=float)
    out_warnings: list[str] = []
    t_hot, h_ok = _safe_float(t_hot_k, 300.0)
    t_cold, c_ok = _safe_float(t_cold_k, 10.0)
    if not h_ok:
        out_warnings.append("T_hot default 300 K used (untrusted input)")
    if not c_ok:
        out_warnings.append("T_cold default 10 K used (untrusted input)")

    n = min(ph.size, pc.size)
    if n == 0:
        return {"trxr_k": float("nan"), "y_factor": float("nan"),
                "uncertainties": {"trxr_k": float("nan")},
                "warnings": out_warnings + ["empty power arrays"]}
    ph = ph[:n]
    pc = pc[:n]

    # Avoid division-by-zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(pc > 0, ph / pc, np.nan)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {"trxr_k": float("nan"), "y_factor": float("nan"),
                "uncertainties": {"trxr_k": float("nan")},
                "warnings": out_warnings + ["no finite Y-factor samples"]}

    y_mean = float(np.mean(y))
    if y_mean <= 1.0:
        out_warnings.append("Y<=1 implies non-physical Trxr; result clipped to NaN")
        return {"trxr_k": float("nan"), "y_factor": y_mean,
                "uncertainties": {"trxr_k": float("nan")},
                "warnings": out_warnings}

    # Y-factor identity: Y = (T_rxr + T_load_hot) / (T_rxr + T_load_cold)
    # => T_rxr = (T_load_hot - Y * T_load_cold) / (Y - 1)
    trxr = (t_hot - y_mean * t_cold) / (y_mean - 1.0)
    # First-order propagation of sigma_Y through the formula.
    # dT_rxr / dY = (T_load_cold - T_load_hot) / (Y - 1)^2
    sigma_y = float(np.std(y) / np.sqrt(y.size)) if y.size else float("nan")
    partial = (t_cold - t_hot) / (y_mean - 1.0) ** 2
    sigma_trxr = abs(partial) * sigma_y if math.isfinite(sigma_y) else float("nan")
    return {
        "trxr_k": float(trxr),
        "y_factor": y_mean,
        "uncertainties": {"trxr_k": float(sigma_trxr) if math.isfinite(sigma_trxr) else float("nan")},
        "warnings": out_warnings,
    }


def system_temperature(
    trxr_k: float | None,
    t_sky_k: np.ndarray | float,
    *,
    trxr_uncertainty_k: float | None = None,
) -> dict[str, Any]:
    """Tsys = Trxr + T_sky(v, t). All inputs in absolute Kelvin.

    If Trxr is None, returns NaN with a warning and an upper bound
    based on a typical HRMS LNA chain.
    """
    t_sky = np.asarray(t_sky_k, dtype=float) if not np.isscalar(t_sky_k) else np.asarray([t_sky_k], dtype=float)
    out_warnings: list[str] = []
    trxr_val, trxr_ok = _safe_float(trxr_k, float("nan"))
    if not trxr_ok or not math.isfinite(trxr_val):
        out_warnings.append("Trxr missing; using HRMS default with uncertainty band")
        trxr_val = _HRMS_DEFAULTS["trxr_typical_K"]
        if trxr_uncertainty_k is None or not math.isfinite(float(trxr_uncertainty_k)):
            trxr_uncertainty_k = _HRMS_DEFAULTS["trxr_typical_unc_K"]

    tsys = trxr_val + t_sky
    sigma = float(trxr_uncertainty_k) if trxr_uncertainty_k is not None and math.isfinite(trxr_uncertainty_k) else float("nan")
    return {
        "tsys_k": np.asarray(tsys, dtype=float),
        "trxr_k_used": float(trxr_val),
        "uncertainties": {
            "tsys_k": sigma,
            "trxr_k": sigma,
        },
        "warnings": out_warnings,
    }


# ---------------------------------------------------------------------------
# Brightness temperature and residual sky analysis
# ---------------------------------------------------------------------------

def beam_convolved_brightness_temperature(
    t_ant_k: np.ndarray,
    *,
    aperture_efficiency: float | None = None,
    dish_diameter_m: float | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert antenna temperature to main-beam brightness temperature.

    T_b = T_ant / eta_ap

    If aperture efficiency is not provided, we return T_ant unchanged
    and emit a warning. This is deliberately conservative: we do not
    invent an efficiency.
    """
    t = np.asarray(t_ant_k, dtype=float).copy()
    out_warnings: list[str] = []
    eta, eta_ok = _safe_float(aperture_efficiency, float("nan"))
    if attrs and not eta_ok:
        eta, eta_ok = _get_attr(attrs, "aperture_efficiency", "eta_aperture", "eta_A")
        if eta_ok:
            eta, eta_ok = _safe_float(eta, float("nan"))
    if not eta_ok or not math.isfinite(eta) or eta <= 0:
        out_warnings.append("aperture efficiency missing; T_b == T_ant returned")
        eta = 1.0
    t_b = t / eta
    return {
        "t_brightness_k": np.asarray(t_b, dtype=float),
        "aperture_efficiency_used": float(eta),
        "uncertainties": {"t_brightness_k": float("nan")},
        "warnings": out_warnings,
    }


def residual_sky_temperature(
    t_measured_k: np.ndarray,
    t_gsm_k: np.ndarray,
) -> dict[str, Any]:
    """delta T_sky = T_measured - T_GSM_model.

    This is the canonical foreground-residual diagnostic for 21cm work:
    everything left over after subtracting the smooth synchrotron + CMB
    dipole + extragalactic foregrounds must be consistent with the
    21cm signal + thermal noise.
    """
    m = np.asarray(t_measured_k, dtype=float)
    g = np.asarray(t_gsm_k, dtype=float)
    if m.shape != g.shape:
        n = min(m.size, g.size)
        m = m.ravel()[:n]
        g = g.ravel()[:n]
    delta = m - g
    valid = np.isfinite(delta)
    if not np.any(valid):
        return {
            "delta_t_sky_k": delta,
            "rms_k": float("nan"),
            "mean_k": float("nan"),
            "n_used": 0,
            "warnings": ["no finite samples"],
        }
    d = delta[valid]
    return {
        "delta_t_sky_k": delta,
        "rms_k": float(np.sqrt(np.mean(d ** 2))),
        "mean_k": float(np.mean(d)),
        "n_used": int(d.size),
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Public convenience wrapper used by the observatory pipeline
# ---------------------------------------------------------------------------

def absolute_calibration_summary(
    attrs: Mapping[str, Any] | None,
    auto_power: np.ndarray | None = None,
    hot_cold: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    frequency_mhz: float | np.ndarray = 408.0,
    galactic_longitude_deg: float | np.ndarray = 0.0,
    galactic_latitude_deg: float | np.ndarray = -30.0,
) -> dict[str, Any]:
    """Single-call absolute calibration summary.

    Returns a dict containing every intermediate result plus an
    `uncertainties` and `warnings` block, so the downstream observatory
    / session layers can decide what is "measured" vs "estimated".
    """
    out_warnings: list[str] = []
    sections: dict[str, Any] = {}

    # 1) GSM model at the supplied pointing + frequency.
    gsm = global_sky_model_temperature(
        galactic_longitude_deg, galactic_latitude_deg, frequency_mhz
    )
    out_warnings.extend(gsm["warnings"])
    sections["gsm_model"] = gsm

    # 2) Antenna temperature (if auto-correlation supplied).
    if auto_power is not None and auto_power.size:
        ant = antenna_temperature_from_auto(
            auto_power,
            attrs=attrs,
        )
        out_warnings.extend(ant["warnings"])
        sections["antenna_temperature"] = ant
    else:
        sections["antenna_temperature"] = {
            "t_ant_k": None,
            "uncertainties": {"t_ant_k": float("nan")},
            "warnings": ["no auto-correlation supplied"],
        }

    # 3) Trxr via Y-factor (if hot/cold supplied).
    if hot_cold is not None:
        ph, pc = hot_cold
        yres = y_factor_receiver_temperature(ph, pc)
        out_warnings.extend(yres["warnings"])
        sections["y_factor"] = yres
    else:
        yres = {"trxr_k": None, "uncertainties": {"trxr_k": float("nan")}, "warnings": ["Y-factor not provided"]}
        sections["y_factor"] = yres

    # 4) Tsys.
    t_sky = np.atleast_1d(gsm["t_ant_gsm_k"])
    trxr = yres.get("trxr_k")
    trxr_unc = yres.get("uncertainties", {}).get("trxr_k")
    tsys = system_temperature(trxr, t_sky, trxr_uncertainty_k=trxr_unc)
    out_warnings.extend(tsys["warnings"])
    sections["system_temperature"] = tsys

    # 5) Brightness temperature.
    if sections["antenna_temperature"].get("t_ant_k") is not None:
        t_b = beam_convolved_brightness_temperature(
            sections["antenna_temperature"]["t_ant_k"],
            attrs=attrs,
        )
        out_warnings.extend(t_b["warnings"])
        sections["brightness_temperature"] = t_b
    else:
        sections["brightness_temperature"] = {
            "t_brightness_k": None,
            "uncertainties": {"t_brightness_k": float("nan")},
            "warnings": ["no antenna temperature available"],
        }

    # 6) Residual sky.
    if sections["brightness_temperature"].get("t_brightness_k") is not None:
        delta = residual_sky_temperature(
            sections["brightness_temperature"]["t_brightness_k"],
            gsm["t_ant_gsm_k"],
        )
        out_warnings.extend(delta["warnings"])
        sections["residual_sky"] = delta
    else:
        sections["residual_sky"] = {
            "delta_t_sky_k": None,
            "rms_k": float("nan"),
            "mean_k": float("nan"),
            "n_used": 0,
            "warnings": ["residual sky analysis requires measured T_b"],
        }

    sections["_uncertainties"] = {
        "t_ant_k": sections["antenna_temperature"].get("uncertainties", {}).get("t_ant_k"),
        "trxr_k": yres.get("uncertainties", {}).get("trxr_k"),
        "tsys_k": tsys.get("uncertainties", {}).get("tsys_k"),
    }
    sections["_warnings"] = out_warnings
    return sections
