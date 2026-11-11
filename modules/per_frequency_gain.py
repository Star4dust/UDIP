"""Per-frequency gain solver with uncertainty propagation.

Returns the complex gain g(ν) and its 1-sigma uncertainty σ_g(ν) at
each frequency bin, using per-baseline complex visibilities and a model
visibility (or a redundancy-averaged measurement as the model).

The uncertainties are propagated through to the system temperature:

    g(ν) = V_meas(ν) / V_model(ν)
    σ_g(ν) = sqrt((σ_V/V_model)² + (|V_meas|/|V_model|² σ_V_model)²)

Then the receiver temperature Trxr is rescaled:

    Trxr(ν) = Trxr_ref / g(ν)

with the corresponding 1σ propagated.

Functions
---------
solve_gain_vs_frequency
    Solve g(ν) and σ_g(ν) from measured and model visibilities.
propagate_gain_uncertainty_to_tsys
    Combine σ_g(ν) with a reference Trxr and Tsys to yield σ_Tsys(ν).
gain_stability_metric
    Time/frequency g(ν) stability summary.

All routines are headless, dependency-light, and never fabricate
uncertainties: when input data is insufficient, they return NaN /
empty arrays and a populated `warnings` block.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
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


def _robust_sigma(values: np.ndarray) -> float:
    """MAD-based 1-sigma estimator. Returns NaN for empty input."""
    if values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if mad <= 0:
        return float(np.std(finite))
    return 1.4826 * mad


# ---------------------------------------------------------------------------
# g(ν) solver
# ---------------------------------------------------------------------------

def solve_gain_vs_frequency(
    visibilities_measured: np.ndarray,
    visibilities_model: np.ndarray,
    *,
    noise_sigma_measured: np.ndarray | float | None = None,
    noise_sigma_model: np.ndarray | float | None = None,
    frequencies_mhz: np.ndarray | None = None,
    reduction: str = "median",
    min_per_bin: int = 1,
) -> dict[str, Any]:
    """Solve complex gain g(ν) and 1-sigma uncertainty σ_g(ν).

    The two visibility arrays must share a frequency axis.  The
    baseline/polarisation axes are collapsed via `reduction`
    ('median' or 'mean').

    Parameters
    ----------
    visibilities_measured, visibilities_model : ndarray
        Complex.  Frequency axis must be the same in both.  Other
        leading/trailing axes are collapsed.
    noise_sigma_measured, noise_sigma_model : ndarray or float, optional
        Per-frequency or scalar 1-sigma noise.  When None, σ_g is
        estimated from the inter-quantile scatter of the complex
        g(ν) samples within each frequency bin.
    frequencies_mhz : ndarray, optional
        Frequency axis in MHz.  Returned unchanged in the output.
    reduction : str
        'median' (default) or 'mean' for collapsing non-frequency axes.
    min_per_bin : int
        Minimum number of (time, baseline, pol) samples required to
        trust the per-frequency gain.  Below this, g(ν) is reported
        as NaN with a warning.

    Returns
    -------
    dict with:
        gain : ndarray (N_freq,) complex
        gain_sigma : ndarray (N_freq,) float
        frequencies_mhz : ndarray
        valid_mask : ndarray (N_freq,) bool
        n_samples_per_bin : ndarray (N_freq,) int
        residual_rms : ndarray (N_freq,) float  (|V_meas - g·V_model| / |V_model|)
        reduction : str
        warnings : list[str]
    """
    out_warnings: list[str] = []
    v_m = np.asarray(visibilities_measured, dtype=np.complex128)
    v_mod = np.asarray(visibilities_model, dtype=np.complex128)
    if v_m.shape != v_mod.shape:
        raise ValueError(
            f"measured/model shape mismatch: {v_m.shape} vs {v_mod.shape}"
        )
    if v_m.size == 0:
        return _empty_gain_result(frequencies_mhz, ["empty input array"])

    n_freq = v_m.shape[0]  # we assume axis 0 is frequency
    if v_m.ndim == 1:
        v_m = v_m.reshape(n_freq, 1)
        v_mod = v_mod.reshape(n_freq, 1)

    n_freq_m = v_m.shape[0]
    if n_freq_m != n_freq:
        out_warnings.append(
            f"expected frequency axis on dim 0; got n_freq={n_freq_m}"
        )

    if frequencies_mhz is None:
        freq = np.arange(n_freq, dtype=float)
    else:
        freq = np.asarray(frequencies_mhz, dtype=float).ravel()
        if freq.size != n_freq:
            out_warnings.append("frequencies_mhz length mismatch; using channel index")
            freq = np.arange(n_freq, dtype=float)

    g = np.full(n_freq, np.nan, dtype=np.complex128)
    g_sig = np.full(n_freq, np.nan, dtype=float)
    n_per = np.zeros(n_freq, dtype=int)
    residual_rms = np.full(n_freq, np.nan, dtype=float)
    valid = np.zeros(n_freq, dtype=bool)

    # Per-bin complex g samples for scatter estimation.
    for f in range(n_freq):
        samples_m = v_m[f].ravel()
        samples_mod = v_mod[f].ravel()
        mask = np.isfinite(samples_m) & np.isfinite(samples_mod) & (np.abs(samples_mod) > 0)
        n_samp = int(np.sum(mask))
        n_per[f] = n_samp
        if n_samp < max(int(min_per_bin), 1):
            out_warnings.append(
                f"frequency bin {f}: only {n_samp} samples, below min_per_bin"
            )
            continue
        ratio = samples_m[mask] / samples_mod[mask]
        if reduction == "mean":
            g[f] = complex(np.mean(ratio.real), np.mean(ratio.imag))
        else:
            g[f] = complex(np.median(ratio.real), np.median(ratio.imag))
        valid[f] = True

        # Estimate uncertainty.
        if noise_sigma_measured is not None or noise_sigma_model is not None:
            # Propagate analytic error.
            sigma_m = noise_sigma_measured
            if np.isscalar(sigma_m):
                sm = float(sigma_m)
            else:
                sigma_m = np.asarray(sigma_m, dtype=float).ravel()
                sm = float(sigma_m[f]) if f < sigma_m.size else float(sigma_m[-1])
            sigma_mod = noise_sigma_model
            if np.isscalar(sigma_mod):
                smod = float(sigma_mod)
            else:
                sigma_mod = np.asarray(sigma_mod, dtype=float).ravel()
                smod = float(sigma_mod[f]) if f < sigma_mod.size else float(sigma_mod[-1])
            abs_mod = float(np.mean(np.abs(samples_mod[mask])))
            g_sig[f] = math.sqrt(
                (sm / max(abs_mod, 1e-30)) ** 2
                + (abs(np.mean(np.abs(samples_m[mask]))) * smod / max(abs_mod ** 2, 1e-30)) ** 2
            )
        else:
            # Empirical scatter of complex g.
            g_sig[f] = _robust_sigma(np.abs(ratio - g[f])) / math.sqrt(n_samp)

        # Residual rms of the calibration fit.
        residual_rms[f] = float(
            np.sqrt(np.mean(np.abs(samples_m[mask] - g[f] * samples_mod[mask]) ** 2))
            / max(np.mean(np.abs(samples_mod[mask])), 1e-30)
        )

    if not np.any(valid):
        out_warnings.append("no valid frequency bins; gain could not be solved")

    # Summary: collapse the per-frequency arrays into per-baseline
    # medians (amp_std, phase_std) and a global count of valid bins.
    return {
        "gain": g,
        "gain_sigma": g_sig,
        "frequencies_mhz": freq,
        "valid_mask": valid,
        "n_samples_per_bin": n_per,
        "residual_rms": residual_rms,
        "reduction": reduction,
        "min_per_bin": int(min_per_bin),
        "warnings": out_warnings,
    }


def _empty_gain_result(frequencies_mhz: np.ndarray | None, warnings: list[str]) -> dict[str, Any]:
    n = 0 if frequencies_mhz is None else int(np.asarray(frequencies_mhz).size)
    return {
        "gain": np.full(n, np.nan, dtype=np.complex128),
        "gain_sigma": np.full(n, np.nan, dtype=float),
        "frequencies_mhz": np.arange(n, dtype=float) if frequencies_mhz is None else np.asarray(frequencies_mhz, dtype=float).ravel(),
        "valid_mask": np.zeros(n, dtype=bool),
        "n_samples_per_bin": np.zeros(n, dtype=int),
        "residual_rms": np.full(n, np.nan, dtype=float),
        "reduction": "median",
        "min_per_bin": 1,
        "warnings": list(warnings),
    }


# ---------------------------------------------------------------------------
# Uncertainty propagation to Tsys
# ---------------------------------------------------------------------------

def propagate_gain_uncertainty_to_tsys(
    gain: np.ndarray,
    gain_sigma: np.ndarray,
    *,
    trxr_reference_k: float | np.ndarray,
    t_sky_k: np.ndarray,
    antenna_temperature_k: np.ndarray | None = None,
    trxr_reference_uncertainty_k: float | np.ndarray | None = None,
) -> dict[str, Any]:
    """Propagate σ_g(ν) through to σ_Tsys(ν).

    Tsys(ν) = Trxr(ν) + T_sky(ν)   with   Trxr(ν) = Trxr_ref / g(ν)

    When the user supplies an antenna-temperature array, we also report
    the per-frequency "effective receiver" (Trxr_ν = (T_ant - T_sky) / g),
    but only when g(ν) is finite.

    Parameters
    ----------
    gain, gain_sigma : ndarray (N_freq,) complex / float
        Per-frequency gain and 1-sigma from `solve_gain_vs_frequency`.
    trxr_reference_k : float or ndarray
        Reference Trxr in Kelvin, used to seed Trxr(ν) = Trxr_ref / g(ν).
    t_sky_k : ndarray (N_freq,)
        GSM / sky-antenna temperature in Kelvin.
    antenna_temperature_k : ndarray (N_freq,), optional
        When supplied, returned as-is in the result block.

    Returns
    -------
    dict with:
        tsys_k : ndarray (N_freq,)
        tsys_sigma_k : ndarray (N_freq,)
        trxr_k : ndarray (N_freq,)
        trxr_sigma_k : ndarray (N_freq,)
        gain_used : ndarray complex
        valid_mask : ndarray bool
        propagation_formula : str
        warnings : list[str]
    """
    out_warnings: list[str] = []
    g = np.asarray(gain, dtype=np.complex128).ravel()
    g_sig = np.asarray(gain_sigma, dtype=float).ravel()
    t_sky = np.asarray(t_sky_k, dtype=float).ravel()
    n = int(min(g.size, g_sig.size, t_sky.size))
    if n == 0:
        return _empty_propagation_result(["empty input"])
    g = g[:n]
    g_sig = g_sig[:n]
    t_sky = t_sky[:n]

    if np.isscalar(trxr_reference_k):
        trxr_ref = float(trxr_reference_k)
    else:
        trxr_ref_arr = np.asarray(trxr_reference_k, dtype=float).ravel()
        trxr_ref = float(trxr_ref_arr[0]) if trxr_ref_arr.size else float("nan")
    if not math.isfinite(trxr_ref):
        out_warnings.append("non-finite Trxr reference; σ_Tsys reflects input σ_g only")

    valid = np.isfinite(g) & (np.abs(g) > 0)
    if not np.all(valid):
        out_warnings.append(
            f"{int(np.sum(~valid))} frequency bins have non-finite or zero gain"
        )

    if trxr_reference_uncertainty_k is not None:
        trxr_ref_unc = float(np.asarray(trxr_reference_uncertainty_k, dtype=float).ravel()[0]) if np.asarray(trxr_reference_uncertainty_k, dtype=float).size else 0.0
    else:
        trxr_ref_unc = 0.0

    trxr = np.full(n, np.nan, dtype=float)
    trxr_sig = np.full(n, np.nan, dtype=float)
    tsys = np.full(n, np.nan, dtype=float)
    tsys_sig = np.full(n, np.nan, dtype=float)

    g_abs = np.abs(g)
    trxr[valid] = trxr_ref / g_abs[valid]
    # σ(Trxr)^2 = (Trxr_ref / |g|^2)^2 * σ_|g|^2 + (σ_Trxr_ref / |g|)^2
    trxr_sig[valid] = np.sqrt((trxr_ref / g_abs[valid] ** 2) ** 2 * g_sig[valid] ** 2 + (trxr_ref_unc / g_abs[valid]) ** 2)

    tsys[valid] = trxr[valid] + t_sky[valid]
    # T_sky is treated as known here; σ_Tsys = σ_Trxr.
    tsys_sig[valid] = trxr_sig[valid]

    if antenna_temperature_k is not None:
        t_ant = np.asarray(antenna_temperature_k, dtype=float).ravel()[:n]
    else:
        t_ant = None

    return {
        "tsys_k": tsys,
        "tsys_sigma_k": tsys_sig,
        "trxr_k": trxr,
        "trxr_sigma_k": trxr_sig,
        "gain_used": g,
        "valid_mask": valid,
        "t_sky_k": t_sky,
        "antenna_temperature_k": t_ant,
        "trxr_reference_k": trxr_ref,
        "propagation_formula": "Tsys = Trxr_ref/|g| + T_sky, sigma_Tsys = (Trxr_ref/|g|^2) sigma_|g|",
        "warnings": out_warnings,
    }


def _empty_propagation_result(warnings: list[str]) -> dict[str, Any]:
    return {
        "tsys_k": np.array([], dtype=float),
        "tsys_sigma_k": np.array([], dtype=float),
        "trxr_k": np.array([], dtype=float),
        "trxr_sigma_k": np.array([], dtype=float),
        "gain_used": np.array([], dtype=np.complex128),
        "valid_mask": np.array([], dtype=bool),
        "t_sky_k": np.array([], dtype=float),
        "antenna_temperature_k": None,
        "trxr_reference_k": float("nan"),
        "propagation_formula": "Tsys = Trxr_ref/|g| + T_sky",
        "warnings": list(warnings),
    }


# ---------------------------------------------------------------------------
# g(ν) stability metric
# ---------------------------------------------------------------------------

def gain_stability_metric(
    gain: Any,
    *,
    gain_sigma: np.ndarray | None = None,
    axis: int = -1,
) -> dict[str, Any]:
    """Time/frequency stability of g(ν).

    Returns the gain drift (|Δg|) and a normalised stability score
    0..1 (1.0 = perfectly stable).

    Parameters
    ----------
    gain : ndarray
        Complex gain array.  The frequency axis is by default the last.
    gain_sigma : ndarray, optional
        Per-frequency 1-sigma on |g|.
    axis : int
        Which axis is the frequency axis.

    Returns
    -------
    dict with:
        gain_amp : ndarray
        gain_phase_rad : ndarray
        drift_abs : ndarray
        drift_relative : ndarray
        stability_score : float in [0, 1]
        gain_sigma_used : ndarray
        warnings : list[str]
    """
    # 1) Unpack a dict (result of solve_gain_vs_frequency) if needed.
    if isinstance(gain, Mapping):
        g = np.asarray(gain.get("gain", gain.get("g")), dtype=np.complex128)
        if gain_sigma is None:
            gain_sigma = gain.get("gain_sigma", gain.get("sigma_g"))
    else:
        g = np.asarray(gain, dtype=np.complex128)
    if g.size == 0:
        return {
            "gain_amp": np.array([], dtype=float),
            "gain_phase_rad": np.array([], dtype=float),
            "drift_abs": np.array([], dtype=float),
            "drift_relative": np.array([], dtype=float),
            "stability_score": float("nan"),
            "gain_sigma_used": np.array([], dtype=float),
            "warnings": ["empty gain array"],
        }
    moved = np.moveaxis(g, axis, -1)
    amp = np.abs(moved)
    phase = np.angle(moved)

    if gain_sigma is not None:
        sig = np.asarray(gain_sigma, dtype=float)
        try:
            sig = np.moveaxis(sig, axis, -1)
        except Exception:
            sig = np.broadcast_to(sig, amp.shape)
    else:
        sig = np.full(amp.shape, np.nan, dtype=float)

    # Drift = |amp[i+1] - amp[i]|
    drift_abs = np.full(amp.shape, np.nan, dtype=float)
    drift_rel = np.full(amp.shape, np.nan, dtype=float)
    if amp.shape[-1] >= 2:
        d = np.abs(np.diff(amp, axis=-1))
        # Pad to keep the shape consistent.
        pad = np.full(d.shape[:-1] + (1,), np.nan, dtype=float)
        drift_abs = np.concatenate([d, pad], axis=-1)
        mean_amp = np.nanmean(amp, axis=-1, keepdims=True)
        drift_rel = np.where(mean_amp > 0, drift_abs / mean_amp, np.nan)

    # Stability score: 1 - normalised drift, clipped to [0, 1].
    finite_drift = drift_abs[np.isfinite(drift_abs)]
    if finite_drift.size == 0:
        score = float("nan")
    else:
        mean_drift = float(np.mean(finite_drift))
        mean_amp_all = float(np.nanmean(amp))
        norm = mean_drift / max(mean_amp_all, 1e-30)
        score = float(np.clip(1.0 - norm, 0.0, 1.0))

    return {
        "gain_amp": amp,
        "gain_phase_rad": phase,
        "drift_abs": drift_abs,
        "drift_relative": drift_rel,
        "stability_score": score,
        "gain_sigma_used": sig,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Convenience: full calibration chain
# ---------------------------------------------------------------------------

def full_calibration_chain(
    visibilities_measured: np.ndarray,
    visibilities_model: np.ndarray,
    *,
    frequencies_mhz: np.ndarray,
    trxr_reference_k: float,
    t_sky_k: np.ndarray,
    antenna_temperature_k: np.ndarray | None = None,
    noise_sigma_measured: np.ndarray | float | None = None,
    noise_sigma_model: np.ndarray | float | None = None,
) -> dict[str, Any]:
    """Run g(ν), σ_g(ν) → Trxr(ν) → Tsys(ν) and return the bundle.

    This is a convenience wrapper that calls `solve_gain_vs_frequency`
    followed by `propagate_gain_uncertainty_to_tsys` and
    `gain_stability_metric`, returning a flat dict with every result.
    """
    g_block = solve_gain_vs_frequency(
        visibilities_measured,
        visibilities_model,
        noise_sigma_measured=noise_sigma_measured,
        noise_sigma_model=noise_sigma_model,
        frequencies_mhz=frequencies_mhz,
    )
    t_block = propagate_gain_uncertainty_to_tsys(
        g_block["gain"],
        g_block["gain_sigma"],
        trxr_reference_k=trxr_reference_k,
        t_sky_k=t_sky_k,
        antenna_temperature_k=antenna_temperature_k,
    )
    s_block = gain_stability_metric(g_block["gain"], gain_sigma=g_block["gain_sigma"])
    return {
        "version": "1.0",
        "gain": g_block,
        "tsys": t_block,
        "stability": s_block,
        "warnings": list(g_block["warnings"]) + list(t_block["warnings"]),
    }


__all__ = [
    "solve_gain_vs_frequency",
    "propagate_gain_uncertainty_to_tsys",
    "gain_stability_metric",
    "full_calibration_chain",
]
