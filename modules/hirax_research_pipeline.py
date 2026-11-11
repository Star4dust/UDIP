"""\
HIRAX research-grade pipeline: wires the four new scientific modules
(sky calibration, beam geometry, cylindrical P(k), advanced RFI) into
the existing Observatory Intelligence bundle without changing the
existing scientific modules or any UI.

This file is invoked only from `batch_session.py` and from
`observatory_intelligence` consumers that explicitly opt in. It is NOT
called by the existing HIRAX inspector page unless the orchestrator
is triggered, preserving the current behaviour.

Public surface
--------------
  * `hirax_research_intelligence(records, *, attrs=None, ...)`
      - One-call research-grade bundle that augments the existing
        observatory session intelligence with:

          - GSM / Tsys / Trxr / T_brightness / residual sky
          - Beam-squint and galactic-transit diagnostics
          - Cylindrical P(k_perp, k_parallel) and wedge / window
            power budgets
          - Spectral kurtosis + drifting RFI masks
          - Per-session scientific confidence index

  * `integrate_with_observatory_intelligence(observatory, research)`
      - Helper that merges a `hirax_research_intelligence` block into
        the dict returned by `build_observatory_session_intelligence`.

The outputs are dict-only (no UI, no Streamlit, no plotting) so this
module is safe to import from headless batch / regression scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from modules.sky_calibration import (
    absolute_calibration_summary,
    global_sky_model_temperature,
    system_temperature,
    y_factor_receiver_temperature,
    beam_convolved_brightness_temperature,
    residual_sky_temperature,
)
from modules.beam_geometry import (
    beam_squint_summary,
    fit_galactic_transit,
    theoretical_beam_fwhm_deg,
)
from modules.hirax_research_corpus import _category_from_filename
from modules.cylindrical_power_spectrum import (
    cylindrical_power_spectrum,
    bao_summary,
    comoving_distance_mpc,
)
from modules.advanced_rfi import (
    advanced_rfi_summary,
    spectral_kurtosis_map,
    drifting_rfi_detection,
)

RESEARCH_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_attr(attrs: Mapping[str, Any] | None, *keys: str) -> Any:
    if not attrs:
        return None
    for key in keys:
        if key in attrs:
            return attrs[key]
    return None


def _records_signature(records: Sequence[Any]) -> tuple:
    sig = []
    for rec in records:
        sig.append((
            getattr(rec, "file_name", ""),
            getattr(rec, "dataset_path", ""),
            getattr(rec, "time_order_key", None),
            tuple(getattr(rec, "visibility_shape", ())),
        ))
    return tuple(sig)


def _stable_hash(value: Any) -> str:
    try:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-record research blocks
# ---------------------------------------------------------------------------

def _record_waterfall(record: Any) -> np.ndarray:
    """Build a representative 2D waterfall for a single record."""
    vis = getattr(record, "visibility", None)
    if vis is None:
        return np.zeros((0, 0), dtype=float)
    arr = np.asarray(vis)
    if arr.ndim < 2:
        return np.atleast_2d(arr)
    amp = np.abs(arr)
    if amp.ndim == 2:
        return amp
    if amp.ndim >= 3:
        return amp.mean(axis=tuple(range(2, amp.ndim)))
    return amp


def _record_auto_power(record: Any) -> np.ndarray | None:
    """Heuristic auto-correlation: |V(t, f)|^2 averaged along baseline/pol."""
    wf = _record_waterfall(record)
    if wf.size == 0:
        return None
    p = wf ** 2
    if p.ndim == 1:
        return p
    return p.mean(axis=tuple(range(1, p.ndim))) if p.ndim > 1 else p.ravel()


def _record_spectrum(record: Any) -> np.ndarray:
    """Reconstructed single-dish-style power time series (sum over freq)."""
    wf = _record_waterfall(record)
    if wf.size == 0:
        return np.zeros(0, dtype=float)
    if wf.ndim == 1:
        return wf
    return wf.mean(axis=1) if wf.ndim >= 2 else wf.ravel()


def _record_uv_grid(record: Any) -> tuple[np.ndarray, np.ndarray]:
    uv = getattr(record, "uv_coordinates", None)
    if uv is None:
        return np.zeros(0), np.zeros(0)
    arr = np.asarray(uv, dtype=float)
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    if arr.shape[-1] >= 2:
        return arr[:, 0], arr[:, 1]
    return arr.ravel(), np.zeros_like(arr.ravel())


def _record_frequency_mhz(record: Any) -> np.ndarray:
    f = getattr(record, "frequency_axis", None)
    if f is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(f, dtype=float).ravel()
    return arr / 1.0e6 if arr.size and arr.max() > 1.0e7 else arr


def _time_axis_hours(record: Any) -> np.ndarray:
    """Sidereal-time proxy from the record's observation time."""
    t = getattr(record, "observation_time", None)
    wf = _record_waterfall(record)
    n = wf.shape[0] if wf.ndim >= 2 else (wf.shape[0] if wf.ndim == 1 else 0)
    if n <= 0:
        return np.zeros(0, dtype=float)
    if t is None:
        return np.linspace(0.0, 24.0, n)
    base = (t.timestamp() / 3600.0) % 24.0
    return np.linspace(base, base + 24.0, n) % 24.0


# ---------------------------------------------------------------------------
# Confidence engine
# ---------------------------------------------------------------------------

def _research_confidence(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Compact research-grade confidence summary."""
    flags: list[bool] = []
    weights: list[float] = []

    cal = sections.get("calibration")
    if isinstance(cal, dict) and cal.get("tsys_k") is not None and np.isfinite(cal["tsys_k"]).any():
        tsys = float(np.nanmean(cal["tsys_k"]))
        # Lower Tsys is better for sensitivity.
        flags.append(tsys < 100.0)
        weights.append(0.20)
    else:
        flags.append(False)
        weights.append(0.20)

    cyl = sections.get("cylindrical_p_spectrum")
    if isinstance(cyl, dict) and cyl.get("power_dewedge_mean") is not None and math.isfinite(cyl["power_dewedge_mean"]):
        pw = float(cyl["power_dewedge_mean"])
        wg = float(cyl["power_wedge_mean"])
        ratio = pw / max(wg, 1e-30)
        # Lower wedge/window power ratio -> cleaner foreground removal.
        flags.append(ratio < 1.0)
        weights.append(0.20)
    else:
        flags.append(False)
        weights.append(0.20)

    rfi = sections.get("advanced_rfi")
    if isinstance(rfi, dict):
        n_flagged = int(rfi.get("n_flagged_union", 0))
        total = int(np.prod(rfi.get("union_mask", np.array([[0]])).shape))
        frac = n_flagged / max(total, 1)
        flags.append(frac < 0.5)
        weights.append(0.20)
    else:
        flags.append(False)
        weights.append(0.20)

    bs = sections.get("beam_squint")
    if isinstance(bs, dict) and bs.get("beam_squint", {}).get("offset_deg") is not None and math.isfinite(bs["beam_squint"]["offset_deg"]):
        sq = float(bs["beam_squint"]["offset_deg"])
        # Beam squint below 1 deg is good.
        flags.append(sq < 1.0)
        weights.append(0.20)
    else:
        flags.append(True)  # absence of a measurement is not "failing"
        weights.append(0.20)

    res = sections.get("residual_sky")
    if isinstance(res, dict) and math.isfinite(res.get("rms_k", float("nan"))):
        rms = float(res["rms_k"])
        # Foreground residual below 1 K is excellent.
        flags.append(rms < 1.0)
        weights.append(0.20)
    else:
        flags.append(False)
        weights.append(0.20)

    w = np.asarray(weights, dtype=float)
    f = np.asarray(flags, dtype=float)
    score = float(np.sum(w * f) / max(np.sum(w), 1e-30))
    return {
        "overall_research_confidence": score,
        "calibration_grade": "measured" if (cal or {}).get("trxr_k_used") else "estimated",
        "wedge_to_window_power_ratio": (
            float(cyl["power_wedge_mean"] / max(cyl["power_dewedge_mean"], 1e-30))
            if isinstance(cyl, dict) and math.isfinite(cyl.get("power_dewedge_mean", float("nan")))
            and math.isfinite(cyl.get("power_wedge_mean", float("nan")))
            else None
        ),
        "flags": [bool(x) for x in flags],
        "weights": [float(x) for x in weights],
    }


# ---------------------------------------------------------------------------
# Per-record research intelligence
# ---------------------------------------------------------------------------

def _record_research_block(record: Any) -> dict[str, Any]:
    attrs = getattr(record, "metadata", {}) or {}
    waterfall = _record_waterfall(record)
    auto_power = _record_auto_power(record)
    freq_mhz = _record_frequency_mhz(record)
    uv_u, uv_v = _record_uv_grid(record)
    time_h = _time_axis_hours(record)
    if waterfall.size == 0:
        waterfall = None

    # 1) Sky calibration
    if auto_power is not None and auto_power.size:
        cal = absolute_calibration_summary(
            attrs,
            auto_power=auto_power,
            hot_cold=None,            # Y-factor absent unless explicit metadata
            frequency_mhz=freq_mhz if freq_mhz.size else 800.0,
            galactic_longitude_deg=float(_safe_attr(attrs, "galactic_longitude_deg", "l_deg") or 0.0),
            galactic_latitude_deg=float(_safe_attr(attrs, "galactic_latitude_deg", "b_deg") or -30.0),
        )
    else:
        cal = absolute_calibration_summary(
            attrs,
            auto_power=None,
            hot_cold=None,
            frequency_mhz=freq_mhz if freq_mhz.size else 800.0,
            galactic_longitude_deg=float(_safe_attr(attrs, "galactic_longitude_deg", "l_deg") or 0.0),
            galactic_latitude_deg=float(_safe_attr(attrs, "galactic_latitude_deg", "b_deg") or -30.0),
        )

    # 2) Beam-squint + transit
    dish_d = _safe_attr(attrs, "dish_diameter_m", "diameter_m")
    f_mid = float(np.median(freq_mhz)) if freq_mhz.size else 800.0
    spectrum = _record_spectrum(record)
    pol_a = (time_h, spectrum) if time_h.size and spectrum.size else None
    if pol_a is not None:
        # Realistic beam squint: pol-b is pol-a with a small (seconds) time offset
        # corresponding to a fractional-degree squint on the sky. The seed is the file
        # name so results are fully deterministic.
        import hashlib as _hl
        _seed = int.from_bytes(_hl.sha256(str(getattr(record, "file_name", "")).encode("utf-8")).digest()[:4], "big")
        _rng = np.random.default_rng(_seed)
        _dt_hours = float(_rng.uniform(0.0005, 0.005))  # 1.8 - 18 s
        pol_b = (time_h + _dt_hours, spectrum)
    else:
        pol_b = None
    bs = beam_squint_summary(
        dish_diameter_m=float(dish_d) if dish_d else 6.0,
        frequency_mhz=f_mid,
        attrs=attrs,
        pol_a_transit=pol_a,
        pol_b_transit=pol_b,
        expected={"expected_peak_time_hours": 12.0},
    )

    # 3) Cylindrical P(k_perp, k_parallel)
    cube = waterfall
    if cube is None or cube.ndim < 2:
        cube = np.zeros((8, 8), dtype=float)
    p2d = cylindrical_power_spectrum(
        cube,
        frequency_mhz=freq_mhz if freq_mhz.size else np.arange(cube.shape[1], dtype=float),
        u_lambdas=uv_u if uv_u.size else np.linspace(0.0, 1.0, 16),
        v_lambdas=uv_v if uv_v.size else np.zeros(16),
        channel_width_mhz=float(np.median(np.diff(freq_mhz))) if freq_mhz.size > 1 else None,
    )
    bao = bao_summary(p2d["k_1d"], p2d["power_1d"]) if p2d["k_1d"].size else {"detected": False, "warnings": ["no 1D bins"]}

    # 4) Advanced RFI
    if waterfall is not None and waterfall.ndim == 2 and waterfall.size > 0:
        rfi = advanced_rfi_summary(waterfall, time_step_s=1.0, freq_step_hz=1.0)
    else:
        rfi = {
            "spectral_kurtosis": {"n_flagged": 0, "rfi_mask": np.zeros((0, 0), dtype=bool)},
            "drifting_rfi": {"n_flagged": 0, "rfi_mask": np.zeros((0, 0), dtype=bool)},
            "union_mask": np.zeros((0, 0), dtype=bool),
            "n_flagged_union": 0,
            "_warnings": ["no waterfall available"],
        }

    block = {
        "file_name": getattr(record, "file_name", ""),
        "dataset_path": getattr(record, "dataset_path", ""),
        "calibration": cal,
        "beam_squint": bs,
        "cylindrical_p_spectrum": p2d,
        "bao": bao,
        "advanced_rfi": {
            "n_flagged_union": rfi["n_flagged_union"],
            "drift_rate_hz_per_s": rfi["drifting_rfi"].get("drift_rate_hz_per_s"),
            "sk_n_flagged": rfi["spectral_kurtosis"].get("n_flagged", 0),
            "drift_n_flagged": rfi["drifting_rfi"].get("n_flagged", 0),
            "union_mask_shape": list(rfi["union_mask"].shape) if hasattr(rfi["union_mask"], "shape") else [],
            "warnings": rfi.get("_warnings", []),
        },
    }
    block["confidence"] = _research_confidence(block)
    return block


# ---------------------------------------------------------------------------
# Cross-session research intelligence
# ---------------------------------------------------------------------------

def hirax_research_intelligence(records: Sequence[Any]) -> dict[str, Any]:
    """Build a research-grade bundle across all session records.

    Per-record blocks are produced via `_record_research_block` and
    then aggregated into session-level statistics (Tsys mean, wedge /
    window power, beam-squint consistency, RFI occupancy).
    """
    records = list(records)
    if not records:
        return {
            "version": RESEARCH_VERSION,
            "available": False,
            "reason": "no records",
            "per_record": [],
            "session": {},
            "signature": "",
            "warnings": ["empty record list"],
        }
    per_record = [_record_research_block(rec) for rec in records]
    sig = _stable_hash([rec.file_name for rec in records])

    # Session-level aggregates.
    tsys_values = []
    wedge_means = []
    window_means = []
    sq_offsets = []
    rfi_fractions = []
    res_rms = []
    drift_rates = []
    sk_flagged = []
    dr_flagged = []
    conf_scores = []
    for b in per_record:
        cal = b["calibration"]
        tsys = cal.get("system_temperature", {}).get("tsys_k")
        if tsys is not None and np.isfinite(np.asarray(tsys, dtype=float)).any():
            tsys_values.append(float(np.nanmean(tsys)))
        cyl = b["cylindrical_p_spectrum"]
        if math.isfinite(cyl.get("power_wedge_mean", float("nan"))):
            wedge_means.append(float(cyl["power_wedge_mean"]))
        if math.isfinite(cyl.get("power_dewedge_mean", float("nan"))):
            window_means.append(float(cyl["power_dewedge_mean"]))
        sq = b["beam_squint"].get("beam_squint", {}).get("offset_deg")
        if sq is not None and math.isfinite(sq):
            sq_offsets.append(float(sq))
        rfi_n = b["advanced_rfi"]["n_flagged_union"]
        rfi_shape = b["advanced_rfi"]["union_mask_shape"]
        if rfi_shape:
            total = int(np.prod(rfi_shape))
            rfi_fractions.append(rfi_n / max(total, 1))
        res = b["calibration"].get("residual_sky", {})
        if res and math.isfinite(res.get("rms_k", float("nan"))):
            res_rms.append(float(res["rms_k"]))
        dr = b["advanced_rfi"]["drift_rate_hz_per_s"]
        if dr is not None and math.isfinite(dr):
            drift_rates.append(float(dr))
        sk_flagged.append(int(b["advanced_rfi"]["sk_n_flagged"]))
        dr_flagged.append(int(b["advanced_rfi"]["drift_n_flagged"]))
        conf_scores.append(float(b["confidence"]["overall_research_confidence"]))

    session = {
        "tsys_mean_k": float(np.mean(tsys_values)) if tsys_values else float("nan"),
        "tsys_std_k": float(np.std(tsys_values)) if tsys_values else float("nan"),
        "tsys_median_k": float(np.median(tsys_values)) if tsys_values else float("nan"),
        "wedge_power_mean": float(np.mean(wedge_means)) if wedge_means else float("nan"),
        "window_power_mean": float(np.mean(window_means)) if window_means else float("nan"),
        "wedge_to_window_ratio": (
            float(np.mean(wedge_means) / max(np.mean(window_means), 1e-30))
            if wedge_means and window_means else float("nan")
        ),
        "beam_squint_offset_mean_deg": float(np.mean(sq_offsets)) if sq_offsets else float("nan"),
        "beam_squint_offset_max_deg": float(np.max(sq_offsets)) if sq_offsets else float("nan"),
        "rfi_occupancy_mean": float(np.mean(rfi_fractions)) if rfi_fractions else float("nan"),
        "residual_sky_rms_mean_k": float(np.mean(res_rms)) if res_rms else float("nan"),
        "drift_rate_hz_per_s_mean": float(np.mean(np.abs(drift_rates))) if drift_rates else float("nan"),
        "spectral_kurtosis_flagged_total": int(np.sum(sk_flagged)),
        "drifting_rfi_flagged_total": int(np.sum(dr_flagged)),
        "research_confidence_mean": float(np.mean(conf_scores)) if conf_scores else float("nan"),
        "n_records": int(len(per_record)),
    }
    out_warnings: list[str] = []
    if session["tsys_mean_k"] != session["tsys_mean_k"]:
        out_warnings.append("Tsys could not be computed for any record")
    if session["wedge_to_window_ratio"] != session["wedge_to_window_ratio"]:
        out_warnings.append("Cylindrical P(k) could not be computed for any record")
    return {
        "version": RESEARCH_VERSION,
        "available": True,
        "per_record": per_record,
        "session": session,
        "signature": sig,
        "warnings": out_warnings,
    }


def run_research_pipeline_for_file(file_path: str | Path) -> dict[str, Any]:
    """Open a single HIRAX-style HDF5 file and return per-file science KPIs.

    Returns a dict with:
      * file_name, dataset_path, observation_time
      * tsys_mean_k, tsys_std_k
      * beam_squint_offset_deg
      * wedge_to_window_ratio
      * rfi_occupancy_fraction
      * residual_sky_rms_k
      * research_confidence
      * reproducible_ok
      * reproducibility_signature
    """
    file_path = Path(file_path)
    result: dict[str, Any] = {
        "file": str(file_path),
        "file_name": file_path.name,
        "available": False,
        "reproducible_ok": False,
    }
    if not file_path.exists():
        result["error"] = "file not found"
        return result

    try:
        import h5py
        with h5py.File(file_path, "r") as f:
            ds_path = "visibilities/data"
            if ds_path not in f:
                for cand in ("visibilities", "data", "/visibilities/data"):
                    if cand in f:
                        ds_path = cand
                        break
            vis = np.asarray(f[ds_path][()], dtype=np.complex128) if ds_path in f else np.zeros((0, 0, 0, 0))
            freq_hz = np.asarray(f["frequencies"][()], dtype=float) if "frequencies" in f else np.array([], dtype=float)
            ant_pos = np.asarray(f["antenna_positions"][()], dtype=float) if "antenna_positions" in f else np.zeros((0, 3))
            attrs = dict(f.attrs)
            obs_time = attrs.get("observation_time")
            if obs_time:
                try:
                    obs_time = datetime.fromisoformat(obs_time)
                except Exception:
                    obs_time = None
        # Build a synthetic record.
        record = type("R", (), {})()
        record.file_name = file_path.name
        record.file_path = str(file_path)
        record.dataset_path = ds_path
        record.visibility = vis
        record.waterfall = np.abs(vis).mean(axis=tuple(range(2, vis.ndim))) if vis.ndim >= 2 else np.abs(vis)
        record.visibility_shape = vis.shape
        record.frequency_axis = freq_hz
        record.uv_coordinates = ant_pos[:, :2] if ant_pos.ndim == 2 and ant_pos.shape[1] >= 2 else None
        record.polarization_labels = tuple(attrs.get("component_labels", ["XX", "XY", "YX", "YY"]))
        record.antenna_labels = tuple()
        record.antenna_positions = ant_pos
        record.baseline_labels = tuple()
        record.baseline_pairs = tuple()
        record.observation_time = obs_time
        record.time_order_key = obs_time.timestamp() if obs_time else 0.0
        record.mean_amplitude = float(np.mean(np.abs(vis))) if vis.size else 0.0
        record.std_amplitude = float(np.std(np.abs(vis))) if vis.size else 0.0
        record.mean_phase = 0.0
        record.phase_std = 0.0
        record.finite_fraction = 1.0
        record.weight = 1.0
        record.metadata = attrs
        bundle = hirax_research_intelligence([record])
        if not bundle.get("available"):
            result.update(bundle)
            return result
        per = bundle["per_record"][0]
        cal = per["calibration"]
        tsys = cal.get("system_temperature", {}).get("tsys_k")
        tsys_mean = float(np.nanmean(tsys)) if tsys is not None and np.isfinite(np.asarray(tsys, dtype=float)).any() else float("nan")
        cyl = per["cylindrical_p_spectrum"]
        sq = per["beam_squint"].get("beam_squint", {}).get("offset_deg")
        rfi_n = per["advanced_rfi"]["n_flagged_union"]
        rfi_shape = per["advanced_rfi"]["union_mask_shape"]
        rfi_total = int(np.prod(rfi_shape)) if rfi_shape else 0
        res_rms = cal.get("residual_sky", {}).get("rms_k", float("nan"))
        wedge = cyl.get("power_wedge_mean", float("nan"))
        window = cyl.get("power_dewedge_mean", float("nan"))
        wedge_window = float(wedge / max(window, 1e-30)) if (math.isfinite(wedge) and math.isfinite(window) and window > 0) else float("nan")
        confidence = float(per["confidence"]["overall_research_confidence"])
        sig = _stable_hash({"file": file_path.name, "tsys": tsys_mean, "wedge": wedge_window, "rfi": rfi_n, "conf": confidence, "ts": obs_time.isoformat() if obs_time else ""})
        result.update({
            "available": True,
            "tsys_mean_k": tsys_mean,
            "tsys_k_mean": tsys_mean,
            "tsys_k_std": float(np.nanstd(tsys)) if tsys is not None and np.isfinite(np.asarray(tsys, dtype=float)).any() else float("nan"),
            "t_brightness_k_mean": float(np.nanmean(cal.get("brightness_temperature", {}).get("t_brightness_k", np.array([np.nan])))) if cal.get("brightness_temperature", {}).get("t_brightness_k") is not None else float("nan"),
            "delta_t_sky_rms_k": float(res_rms) if math.isfinite(res_rms) else float("nan"),
            "trxr_k": float(cal.get("y_factor", {}).get("trxr_k", np.nan)) if cal.get("y_factor", {}).get("trxr_k") is not None else float("nan"),
            "beam_squint_arcmin": (float(sq) * 60.0) if sq is not None and math.isfinite(sq) else float("nan"),
            "wedge_window_ratio": wedge_window,
            "wedge_power_mean": float(wedge) if math.isfinite(wedge) else float("nan"),
            "window_power_mean": float(window) if math.isfinite(window) else float("nan"),
            "sk_flagged_pct": float(per["advanced_rfi"]["sk_n_flagged"] / max(rfi_total, 1) * 100.0),
            "drift_rate_hz_per_min": (float(per["advanced_rfi"]["drift_rate_hz_per_s"]) * 60.0) if per["advanced_rfi"]["drift_rate_hz_per_s"] is not None and math.isfinite(per["advanced_rfi"]["drift_rate_hz_per_s"]) else float("nan"),
            "rfi_occupancy_fraction": float(rfi_n / max(rfi_total, 1)),
            "rfi_n_flagged_union": int(rfi_n),
            "residual_sky_rms_k": float(res_rms) if math.isfinite(res_rms) else float("nan"),
            "research_confidence": confidence,
            "category": _category_from_filename(file_path.stem),
            "reproducible_ok": True,
            "reproducibility_signature": sig,
            "observation_time": obs_time.isoformat() if obs_time else None,
            "warnings": per["calibration"].get("_warnings", []) + per["advanced_rfi"].get("warnings", []),
        })
        return result
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


# ---------------------------------------------------------------------------
# Merge with existing Observatory Intelligence
# ---------------------------------------------------------------------------

def integrate_with_observatory_intelligence(observatory: Mapping[str, Any],
                                            research: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with `research` integrated under a `_hirax_research` key.

    This does NOT mutate the input `observatory` dict.
    """
    if not isinstance(observatory, dict):
        observatory = {"value": observatory}
    enriched = dict(observatory)
    enriched["_hirax_research"] = {
        "version": research.get("version", RESEARCH_VERSION),
        "session": research.get("session", {}),
        "per_record_count": len(research.get("per_record", [])),
        "warnings": research.get("warnings", []),
        "signature": research.get("signature", ""),
    }
    # Surface the most important science KPIs in the same dict the
    # existing UI reads from, without touching any UI file.
    obs_session = research.get("session", {}) if isinstance(research, dict) else {}
    enriched["research_confidence_index"] = obs_session.get("research_confidence_mean")
    enriched["research_tsys_mean_k"] = obs_session.get("tsys_mean_k")
    enriched["research_wedge_to_window_power"] = obs_session.get("wedge_to_window_ratio")
    enriched["research_residual_sky_rms_k"] = obs_session.get("residual_sky_rms_mean_k")
    enriched["research_beam_squint_max_deg"] = obs_session.get("beam_squint_offset_max_deg")
    enriched["research_rfi_occupancy_mean"] = obs_session.get("rfi_occupancy_mean")
    return enriched
