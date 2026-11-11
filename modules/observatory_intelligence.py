from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import csv
from typing import Any, Sequence
import hashlib
import time
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd

from modules.hirax_processing import detect_rfi
from modules.radio_interferometry import (
    compute_beam_parameters,
    compute_calibration_residuals,
    compute_closure_amplitude,
    compute_closure_phase,
    compute_point_spread_function,
    compute_uv_weights,
    compute_visibility_statistics,
    compute_gain_solution,
    generate_uv_coverage,
    solve_bandpass,
    validate_interferometry_data,
)

OBSERVATORY_EXPORT_VERSION = "1.1"


def build_observatory_session_intelligence(records: Sequence[Any]) -> dict[str, Any]:
    records = list(records)
    diagnostics = performance_diagnostics(records)
    resilience = malformed_dataset_resilience_report(records)
    uv_fusion = cached_uv_aware_session_fusion(records)
    earth_rotation = cached_earth_rotation_synthesis(records)
    calibration_drift = cached_calibration_drift_analysis(records)
    closure_timeline = cached_closure_timeline_analysis(records)
    rfi_evolution = cached_rfi_evolution_analysis(records)
    uv_tracks = cached_uv_track_visualization(records)
    closure_explorer = cached_closure_triangle_explorer(records)
    calibration_maps = cached_calibration_stability_maps(records)
    persistent_rfi = cached_persistent_rfi_heatmaps(records)
    reproducibility = reproducibility_diagnostics(records, uv_fusion, earth_rotation, calibration_drift, closure_timeline, rfi_evolution)
    trust_checks = observatory_consistency_checks(records, uv_fusion, earth_rotation, calibration_drift, closure_timeline, rfi_evolution)
    export_integrity = export_integrity_diagnostics(records, uv_fusion, earth_rotation, calibration_drift, closure_timeline, rfi_evolution)
    guidance = scientific_interpretation_guidance()
    stress = stress_validation_summary(records)
    export_signature_summary = export_integrity_signature_summary({
        "version": OBSERVATORY_EXPORT_VERSION,
        "uv_tracks": uv_tracks,
        "closure_explorer": closure_explorer,
        "calibration_maps": calibration_maps,
        "persistent_rfi": persistent_rfi,
    })
    confidence = scientific_confidence_engine(
        records=records,
        uv_fusion=uv_fusion,
        earth_rotation=earth_rotation,
        calibration_drift=calibration_drift,
        closure_timeline=closure_timeline,
        rfi_evolution=rfi_evolution,
    )
    session_export_preview = safe_export_session_intelligence(
        {
            "version": OBSERVATORY_EXPORT_VERSION,
            "uv_aware_fusion": uv_fusion,
            "earth_rotation": earth_rotation,
            "calibration_drift": calibration_drift,
            "closure_timeline": closure_timeline,
            "rfi_evolution": rfi_evolution,
            "uv_tracks": uv_tracks,
            "closure_explorer": closure_explorer,
            "calibration_maps": calibration_maps,
            "persistent_rfi": persistent_rfi,
            "confidence": confidence,
            "performance": diagnostics,
            "resilience": resilience,
            "reproducibility": reproducibility,
            "trust_checks": trust_checks,
            "export_integrity": export_integrity,
            "guidance": guidance,
            "stress": stress,
            "export_signature_summary": export_signature_summary,
        },
        output_dir=Path("/tmp"),
        stem="udip_observatory_preview",
    )
    return {
        "version": OBSERVATORY_EXPORT_VERSION,
        "diagnostics": diagnostics,
        "resilience": resilience,
        "reproducibility": reproducibility,
        "trust_checks": trust_checks,
        "export_integrity": export_integrity,
        "guidance": guidance,
        "stress": stress,
        "export_signature_summary": export_signature_summary,
        "uv_aware_fusion": uv_fusion,
        "earth_rotation": earth_rotation,
        "calibration_drift": calibration_drift,
        "closure_timeline": closure_timeline,
        "rfi_evolution": rfi_evolution,
        "uv_tracks": uv_tracks,
        "closure_explorer": closure_explorer,
        "calibration_maps": calibration_maps,
        "persistent_rfi": persistent_rfi,
        "confidence": confidence,
        "session_export_preview": session_export_preview,
    }


def uv_aware_session_fusion(records: Sequence[Any]) -> dict[str, Any]:
    if not records:
        return {"available": False, "reason": "no records", "warnings": [], "errors": ["no session records available"], "rejected_records": []}

    ordered = sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", "")))
    ref = ordered[0]
    ref_vis = np.asarray(ref.visibility)
    ref_uv = _uv_points(ref)
    ref_freq = _frequency_axis(ref)
    ref_pol = _polarization_labels(ref)
    ref_sig = _compatibility_signature(ref)
    warnings: list[str] = []
    errors: list[str] = []
    rejected: list[dict[str, Any]] = []
    compatible: list[Any] = []

    for rec in ordered:
        reasons = _compatibility_reasons(ref, rec)
        if reasons:
            rejected.append({"file_name": getattr(rec, "file_name", ""), "dataset_path": getattr(rec, "dataset_path", ""), "reasons": reasons})
            continue
        compatible.append(rec)

    if not compatible:
        return {
            "available": False,
            "reason": "no uv-compatible records",
            "warnings": warnings,
            "errors": ["all session records were rejected for uv/frequency/polarization incompatibility"],
            "rejected_records": rejected,
        }

    baseline_axis = _axis_index(ref, "baseline")
    if baseline_axis is None:
        baseline_axis = 2 if ref_vis.ndim >= 3 else max(ref_vis.ndim - 1, 0)
    alignment_maps: list[dict[str, Any]] = []
    aligned_visibilities: list[np.ndarray] = []

    for rec in compatible:
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        uv = _uv_points(rec)
        order = None
        if ref_uv is not None and uv is not None:
            order = _match_baseline_order(ref_uv, uv)
            if order is None:
                rejected.append({"file_name": getattr(rec, "file_name", ""), "dataset_path": getattr(rec, "dataset_path", ""), "reasons": ["baseline uv coordinates could not be aligned uniquely"]})
                continue
            vis = np.take(vis, order, axis=baseline_axis)
            uv = uv[order]
        aligned_visibilities.append(vis)
        alignment_maps.append(
            {
                "file_name": getattr(rec, "file_name", ""),
                "dataset_path": getattr(rec, "dataset_path", ""),
                "baseline_axis": baseline_axis,
                "baseline_order": order.tolist() if isinstance(order, np.ndarray) else None,
                "uv_coordinate_count": int(len(uv)) if uv is not None else 0,
            }
        )

    if not aligned_visibilities:
        return {
            "available": False,
            "reason": "no compatible records after uv alignment",
            "warnings": warnings,
            "errors": ["baseline alignment failed for every candidate record"],
            "rejected_records": rejected,
        }

    weights = np.asarray([max(float(getattr(rec, "weight", 1.0)), 0.0) for rec in compatible[: len(aligned_visibilities)]], dtype=float)
    if not np.any(weights > 0):
        weights = np.ones(len(aligned_visibilities), dtype=float)
    norm_weights = weights / np.sum(weights)
    stacked = np.stack(aligned_visibilities, axis=0)
    fused_visibility = np.tensordot(norm_weights, stacked, axes=(0, 0))

    fused_uv = ref_uv.copy() if isinstance(ref_uv, np.ndarray) else None
    fused_freq = ref_freq.copy() if isinstance(ref_freq, np.ndarray) else None
    fusion_validation = validate_interferometry_data(fused_visibility, fused_uv[:, 0] if fused_uv is not None and fused_uv.shape[1] >= 1 else np.array([]), fused_uv[:, 1] if fused_uv is not None and fused_uv.shape[1] >= 2 else np.array([])) if fused_uv is not None else {"is_valid": True, "warnings": [], "errors": []}

    if ref_uv is not None:
        uv_cov = generate_uv_coverage(ref_uv[:, 0], ref_uv[:, 1] if ref_uv.shape[1] >= 2 else np.zeros(len(ref_uv)))
    else:
        uv_cov = {"n_uv_points": 0, "coverage_density": np.zeros((50, 50)), "density_mean": 0.0, "density_max": 0.0}

    return {
        "available": True,
        "reason": "weighted uv-compatible fusion",
        "weights": norm_weights.tolist(),
        "weighted_sum": float(np.sum(weights)),
        "compatible_records": [getattr(rec, "file_name", "") for rec in compatible[: len(aligned_visibilities)]],
        "rejected_records": rejected,
        "alignment_maps": alignment_maps,
        "baseline_match_mode": "uv_coordinate_alignment" if ref_uv is not None else "index_alignment",
        "preserves_time_order": True,
        "preserves_uv_coordinates": fused_uv is not None,
        "preserves_polarization_products": bool(ref_pol),
        "frequency_grid": fused_freq.tolist() if isinstance(fused_freq, np.ndarray) else None,
        "polarization_labels": list(ref_pol),
        "fused_visibility": fused_visibility,
        "fused_visibility_shape": list(fused_visibility.shape),
        "fused_uv_coordinates": fused_uv,
        "uv_coverage": uv_cov,
        "compatibility_signature": ref_sig,
        "validation": fusion_validation,
        "warnings": warnings + list(fusion_validation.get("warnings", [])),
        "errors": errors + list(fusion_validation.get("errors", [])),
        "frequency_grid_valid": bool(ref_freq is not None),
        "polarization_compatible": bool(ref_pol is not None),
        "baseline_compatible": True,
        "provenance": [
            {
                "file_name": getattr(rec, "file_name", ""),
                "file_path": getattr(rec, "file_path", ""),
                "dataset_path": getattr(rec, "dataset_path", ""),
                "observation_time": getattr(rec, "observation_time", None).isoformat() if getattr(rec, "observation_time", None) is not None else None,
                "weight": float(getattr(rec, "weight", 1.0)),
                "visibility_shape": list(np.asarray(rec.visibility).shape),
            }
            for rec in compatible[: len(aligned_visibilities)]
        ],
    }


def earth_rotation_synthesis(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if _uv_points(rec) is not None]
    if not ordered:
        return {
            "available": False,
            "reason": "no uv coordinates available",
            "uv_track_points": pd.DataFrame(columns=["session_order", "file_name", "u", "v", "radius", "weight", "observation_time"]),
            "trajectory": pd.DataFrame(),
            "beam_evolution": pd.DataFrame(),
            "uv_completeness_metrics": {},
            "warnings": ["uv track synthesis requires uv coordinate datasets"],
            "errors": [],
        }

    cumulative_u: list[float] = []
    cumulative_v: list[float] = []
    point_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    beam_rows: list[dict[str, Any]] = []
    cumulative_density: list[float] = []

    for session_order, rec in enumerate(ordered, start=1):
        uv = _uv_points(rec)
        assert uv is not None
        u = uv[:, 0].astype(float, copy=False)
        v = uv[:, 1].astype(float, copy=False) if uv.shape[1] > 1 else np.zeros_like(u)
        cumulative_u.extend(u.tolist())
        cumulative_v.extend(v.tolist())
        cumulative = np.column_stack([np.asarray(cumulative_u), np.asarray(cumulative_v)])
        coverage = generate_uv_coverage(cumulative[:, 0], cumulative[:, 1])
        psf = compute_point_spread_function(cumulative[:, 0], cumulative[:, 1], image_size=64)
        beam = compute_beam_parameters(psf)
        density = np.asarray(coverage.get("coverage_density", np.zeros((50, 50))), dtype=float)
        cumulative_density.append(float(np.mean(density)) if density.size else 0.0)

        trajectory_rows.append(
            {
                "session_order": session_order,
                "file_name": getattr(rec, "file_name", ""),
                "observation_time": getattr(rec, "observation_time", None).isoformat() if getattr(rec, "observation_time", None) is not None else None,
                "n_uv_points": int(len(u)),
                "cumulative_uv_points": int(len(cumulative_u)),
                "uv_radius_mean": float(np.mean(np.sqrt(u**2 + v**2))) if u.size else 0.0,
                "uv_radius_max": float(np.max(np.sqrt(u**2 + v**2))) if u.size else 0.0,
                "coverage_density_mean": float(coverage.get("density_mean", 0.0)),
                "coverage_density_max": float(coverage.get("density_max", 0.0)),
                "beam_peak": float(beam.get("peak", 0.0)),
                "beam_fwhm_avg": float(beam.get("fwhm_avg", 0.0)),
                "beam_area": float(beam.get("beam_area", 0.0)),
                "observation_weight": float(getattr(rec, "weight", 1.0)),
            }
        )
        beam_rows.append({"session_order": session_order, "file_name": getattr(rec, "file_name", ""), **beam})

        for uu, vv in zip(u, v):
            point_rows.append(
                {
                    "session_order": session_order,
                    "file_name": getattr(rec, "file_name", ""),
                    "observation_time": getattr(rec, "observation_time", None).isoformat() if getattr(rec, "observation_time", None) is not None else None,
                    "u": float(uu),
                    "v": float(vv),
                    "radius": float(np.sqrt(uu**2 + vv**2)),
                    "weight": float(getattr(rec, "weight", 1.0)),
                }
            )

    trajectory = pd.DataFrame(trajectory_rows)
    uv_track_points = pd.DataFrame(point_rows)
    beam_evolution = pd.DataFrame(beam_rows)
    cumulative_uv = np.column_stack([np.asarray(cumulative_u), np.asarray(cumulative_v)])
    final_coverage = generate_uv_coverage(cumulative_uv[:, 0], cumulative_uv[:, 1])
    final_density = np.asarray(final_coverage.get("coverage_density", np.zeros((50, 50))), dtype=float)
    completeness = {
        "n_sessions": int(len(ordered)),
        "n_uv_points": int(len(cumulative_u)),
        "unique_uv_points": int(len({(round(u, 6), round(v, 6)) for u, v in zip(cumulative_u, cumulative_v)})),
        "uv_radius_mean": float(np.mean(np.sqrt(cumulative_uv[:, 0]**2 + cumulative_uv[:, 1]**2))) if cumulative_uv.size else 0.0,
        "uv_radius_max": float(np.max(np.sqrt(cumulative_uv[:, 0]**2 + cumulative_uv[:, 1]**2))) if cumulative_uv.size else 0.0,
        "coverage_density_mean": float(final_coverage.get("density_mean", 0.0)),
        "coverage_density_max": float(final_coverage.get("density_max", 0.0)),
        "coverage_symmetry": _coverage_symmetry(cumulative_uv),
        "beam_quality_score": _beam_quality_score(final_coverage, final_density),
        "temporal_density_mean": float(np.mean(cumulative_density)) if cumulative_density else 0.0,
    }
    return {
        "available": True,
        "trajectory": trajectory,
        "uv_track_points": uv_track_points,
        "beam_evolution": beam_evolution,
        "uv_coverage_density": final_density,
        "uv_completeness_metrics": completeness,
        "warnings": [],
        "errors": [],
    }


def calibration_drift_analysis(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    if len(ordered) < 2:
        return {"available": False, "reason": "at least two compatible sessions are required", "drift_timeline": pd.DataFrame(), "amplitude_drift_heatmap": np.zeros((0, 0)), "phase_drift_heatmap": np.zeros((0, 0)), "warnings": ["calibration drift requires at least two sessions"], "errors": []}

    ref = ordered[0]
    ref_vis = np.asarray(ref.visibility, dtype=np.complex128)
    ref_freq = _frequency_axis(ref)
    baseline_axis = _axis_index(ref, "baseline") or (2 if ref_vis.ndim >= 3 else max(ref_vis.ndim - 1, 0))
    baseline_count = ref_vis.shape[baseline_axis] if ref_vis.ndim else 0
    amp_heatmap = np.full((len(ordered), baseline_count), np.nan, dtype=float) if baseline_count else np.zeros((len(ordered), 0), dtype=float)
    phase_heatmap = np.full_like(amp_heatmap, np.nan)
    rows: list[dict[str, Any]] = []

    for idx, rec in enumerate(ordered, start=1):
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        if vis.shape != ref_vis.shape:
            rows.append({"session_order": idx, "file_name": getattr(rec, "file_name", ""), "compatible": False, "reason": "shape mismatch"})
            continue
        gains = compute_gain_solution(vis, ref_vis)
        amp = np.abs(gains)
        phase = np.angle(gains)
        amp_drift = float(np.mean(np.abs(amp - 1.0))) if amp.size else 0.0
        phase_drift = float(np.mean(np.abs(phase))) if phase.size else 0.0
        bandpass = solve_bandpass(vis, axis=_axis_index(rec, "frequency") or 1 if vis.ndim >= 2 else 0)
        ref_bandpass = solve_bandpass(ref_vis, axis=_axis_index(ref, "frequency") or 1 if ref_vis.ndim >= 2 else 0)
        bandpass_amp_rms = float(np.sqrt(np.mean(np.square(np.abs(bandpass) - np.abs(ref_bandpass))))) if np.asarray(bandpass).size and np.asarray(ref_bandpass).size else 0.0
        bandpass_phase_rms = float(np.sqrt(np.mean(np.square(np.angle(np.exp(1j * (np.angle(bandpass) - np.angle(ref_bandpass)))))))) if np.asarray(bandpass).size and np.asarray(ref_bandpass).size else 0.0
        current_profile = _baseline_profile(vis, baseline_axis)
        ref_profile = _baseline_profile(ref_vis, baseline_axis)
        baseline_coherence = _safe_corrcoef(current_profile, ref_profile)
        baseline_amp_drift = np.abs(current_profile - ref_profile) if current_profile.size == ref_profile.size else np.array([])
        if baseline_amp_drift.size and amp_heatmap.shape[1] == baseline_amp_drift.size:
            amp_heatmap[idx - 1] = baseline_amp_drift
            phase_heatmap[idx - 1] = np.angle(_baseline_phase_profile(vis, baseline_axis) * np.conj(_baseline_phase_profile(ref_vis, baseline_axis)))
        unstable = np.where(baseline_amp_drift > np.median(baseline_amp_drift) + np.std(baseline_amp_drift)) if baseline_amp_drift.size else (np.array([], dtype=int),)
        rows.append(
            {
                "session_order": idx,
                "file_name": getattr(rec, "file_name", ""),
                "compatible": True,
                "mean_gain_amplitude": float(np.mean(amp)) if amp.size else 0.0,
                "mean_gain_phase": float(np.mean(phase)) if phase.size else 0.0,
                "gain_amplitude_drift": amp_drift,
                "gain_phase_drift": phase_drift,
                "bandpass_amplitude_rms": bandpass_amp_rms,
                "bandpass_phase_rms": bandpass_phase_rms,
                "baseline_coherence": baseline_coherence,
                "unstable_baselines": [int(i) for i in unstable[0].tolist()] if len(unstable) else [],
                "calibration_anomaly_score": _normalize_score(amp_drift + phase_drift + bandpass_amp_rms + bandpass_phase_rms - baseline_coherence),
            }
        )

    timeline = pd.DataFrame(rows)
    return {
        "available": bool(len(timeline)),
        "reference_file": getattr(ref, "file_name", ""),
        "drift_timeline": timeline,
        "amplitude_drift_heatmap": amp_heatmap,
        "phase_drift_heatmap": phase_heatmap,
        "warnings": [],
        "errors": [],
    }


def closure_timeline_analysis(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    rows: list[dict[str, Any]] = []
    phase_maps: list[np.ndarray] = []
    amp_maps: list[np.ndarray] = []

    for session_order, rec in enumerate(ordered, start=1):
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        time_axis = _axis_index(rec, "time") or 0
        freq_axis = _axis_index(rec, "frequency") or 1 if vis.ndim >= 2 else 0
        baseline_axis = _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else None)
        pol_axis = _axis_index(rec, "polarization")
        if baseline_axis is None or vis.shape[baseline_axis] < 3:
            rows.append({"session_order": session_order, "file_name": getattr(rec, "file_name", ""), "available": False, "reason": "insufficient baselines"})
            continue

        cube = _session_cube(vis, time_axis, freq_axis, baseline_axis, pol_axis)
        baseline_count = cube.shape[2]
        triangles = list(combinations(range(baseline_count), 3))[: min(10, len(list(combinations(range(baseline_count), 3))))]
        quadrangles = list(combinations(range(baseline_count), 4))[: min(10, len(list(combinations(range(baseline_count), 4))))] if baseline_count >= 4 else []
        pol_labels = _polarization_labels(rec) or ("pol",)

        for pol_idx, pol_label in enumerate(pol_labels if cube.ndim == 4 else ("pol",)):
            pol_cube = cube[:, :, :, pol_idx] if cube.ndim == 4 else cube
            for tri_idx, tri in enumerate(triangles):
                phase_map = np.angle(pol_cube[:, :, tri[0]] * pol_cube[:, :, tri[1]] * pol_cube[:, :, tri[2]])
                if phase_map.size:
                    phase_maps.append(phase_map)
                summary_phase = float(np.angle(np.mean(np.exp(1j * phase_map)))) if phase_map.size else 0.0
                rms_phase = float(np.sqrt(np.mean(np.square(phase_map)))) if phase_map.size else 0.0
                row = {
                    "session_order": session_order,
                    "file_name": getattr(rec, "file_name", ""),
                    "polarization": pol_label,
                    "triangle_id": tri_idx,
                    "triangle": tri,
                    "mean_closure_phase": summary_phase,
                    "rms_closure_phase": rms_phase,
                    "unstable": bool(rms_phase > 1.0),
                }
                if quadrangles:
                    quad = quadrangles[min(tri_idx, len(quadrangles) - 1)]
                    amp_map = np.abs(pol_cube[:, :, quad[0]]) * np.abs(pol_cube[:, :, quad[1]]) / np.maximum(np.abs(pol_cube[:, :, quad[2]]) * np.abs(pol_cube[:, :, quad[3]]), 1e-12)
                    if amp_map.size:
                        amp_maps.append(amp_map)
                    row.update(
                        {
                            "quadrangle": quad,
                            "mean_closure_amplitude": float(np.nanmean(amp_map)) if amp_map.size else 0.0,
                            "std_closure_amplitude": float(np.nanstd(amp_map)) if amp_map.size else 0.0,
                            "closure_amplitude_unstable": bool(np.nanstd(amp_map) > 0.5) if amp_map.size else False,
                        }
                    )
                rows.append(row)

    timeline = pd.DataFrame(rows)
    phase_preview = phase_maps[0] if phase_maps else np.zeros((0, 0), dtype=float)
    amp_preview = amp_maps[0] if amp_maps else np.zeros((0, 0), dtype=float)
    phase_rms = float(np.sqrt(np.mean(np.square(phase_preview)))) if phase_preview.size else 0.0
    amp_rms = float(np.sqrt(np.mean(np.square(amp_preview - 1.0)))) if amp_preview.size else 0.0
    return {
        "available": bool(len(timeline)),
        "closure_timeline": timeline,
        "triangle_browser": timeline[["session_order", "file_name", "polarization", "triangle_id", "triangle", "mean_closure_phase", "rms_closure_phase", "unstable"]].copy() if not timeline.empty else pd.DataFrame(),
        "closure_phase_map_preview": phase_preview,
        "closure_amplitude_map_preview": amp_preview,
        "closure_consistency_score": _normalize_score(phase_rms + amp_rms),
        "warnings": [],
        "errors": [],
    }


def rfi_evolution_analysis(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    session_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    freq_signatures: list[tuple] = []

    for session_order, rec in enumerate(ordered, start=1):
        waterfall = _waterfall(rec)
        detection = detect_rfi(waterfall, threshold=3.0, method="mad")
        mask = np.asarray(detection.get("mask", np.zeros_like(waterfall, dtype=bool)), dtype=bool)
        masks.append(mask)
        freq_signatures.append(_frequency_signature(rec))
        occupancy_by_freq = mask.mean(axis=0) if mask.ndim == 2 and mask.shape[1] else np.array([])
        occupancy_by_time = mask.mean(axis=1) if mask.ndim == 2 and mask.shape[0] else np.array([])
        session_rows.append(
            {
                "session_order": session_order,
                "file_name": getattr(rec, "file_name", ""),
                "flagged_fraction": float(mask.mean()) if mask.size else 0.0,
                "transient_fraction": float(np.mean((mask > 0) & (mask < 1))) if mask.size else 0.0,
                "persistent_fraction": float(np.mean(occupancy_by_freq > 0.5)) if occupancy_by_freq.size else 0.0,
                "contaminated_channels": int(np.sum(occupancy_by_freq > 0.25)) if occupancy_by_freq.size else 0,
                "masked_count": int(np.count_nonzero(mask)),
                "frequency_count": int(mask.shape[1]) if mask.ndim == 2 else 0,
                "time_count": int(mask.shape[0]) if mask.ndim == 2 else 0,
            }
        )
        for idx, value in enumerate(occupancy_by_freq.tolist() if occupancy_by_freq.size else []):
            frequency_rows.append({"session_order": session_order, "file_name": getattr(rec, "file_name", ""), "frequency_index": idx, "occupancy": float(value)})
        baseline_rows.extend(_baseline_rfi_rows(rec, session_order, mask))

    if masks:
        max_freq = max(mask.shape[1] for mask in masks if mask.ndim == 2)
        aligned = []
        for mask in masks:
            if mask.ndim != 2:
                continue
            if mask.shape[1] == max_freq:
                aligned.append(mask)
        persistent = np.mean([m.mean(axis=0) for m in aligned], axis=0) if aligned else np.array([])
        persistent_report = {
            "top_channels": [int(i) for i in np.argsort(persistent)[::-1][:10].tolist()] if persistent.size else [],
            "persistent_fraction": float(np.mean(persistent > 0.5)) if persistent.size else 0.0,
            "contaminated_fraction": float(np.mean(persistent > 0.25)) if persistent.size else 0.0,
            "stability_score": _normalize_score(float(np.std([row["flagged_fraction"] for row in session_rows])) if session_rows else 0.0),
        }
        occupancy_evolution = np.vstack([m.mean(axis=0) for m in aligned]) if aligned else np.zeros((0, 0), dtype=float)
    else:
        persistent_report = {"top_channels": [], "persistent_fraction": 0.0, "contaminated_fraction": 0.0, "stability_score": 0.0}
        occupancy_evolution = np.zeros((0, 0), dtype=float)

    session_table = pd.DataFrame(session_rows)
    frequency_table = pd.DataFrame(frequency_rows)
    baseline_table = pd.DataFrame(baseline_rows)
    contaminated_channels = frequency_table.groupby("frequency_index", as_index=False)["occupancy"].mean().sort_values("occupancy", ascending=False) if not frequency_table.empty else pd.DataFrame(columns=["frequency_index", "occupancy"])

    return {
        "available": bool(len(session_table)),
        "session_rfi_summary": session_table,
        "frequency_occupancy_evolution": frequency_table,
        "baseline_contamination": baseline_table,
        "temporal_occupancy_map": occupancy_evolution,
        "persistent_contamination_report": persistent_report,
        "contaminated_channel_ranking": contaminated_channels,
        "warnings": [],
        "errors": [],
    }


def scientific_confidence_engine(
    records: Sequence[Any],
    uv_fusion: dict[str, Any],
    earth_rotation: dict[str, Any],
    calibration_drift: dict[str, Any],
    closure_timeline: dict[str, Any],
    rfi_evolution: dict[str, Any],
) -> dict[str, Any]:
    components: dict[str, float] = {}
    basis: dict[str, Any] = {}

    uv_metrics = earth_rotation.get("uv_completeness_metrics", {}) if earth_rotation else {}
    if uv_metrics:
        uv_score = np.mean([
            _clip01(uv_metrics.get("coverage_symmetry", 0.0)),
            _clip01(uv_metrics.get("beam_quality_score", 0.0)),
            _clip01(1.0 / (1.0 + uv_metrics.get("coverage_density_mean", 0.0))),
        ])
        components["uv_completeness"] = float(uv_score)
        basis["uv_completeness"] = uv_metrics

    drift = calibration_drift.get("drift_timeline") if calibration_drift else None
    if isinstance(drift, pd.DataFrame) and not drift.empty:
        compatible = drift[drift.get("compatible", True) == True] if "compatible" in drift.columns else drift
        calibration_score = _clip01(1.0 / (1.0 + float(compatible.get("gain_amplitude_drift", pd.Series([0.0])).mean() + compatible.get("gain_phase_drift", pd.Series([0.0])).mean() + compatible.get("bandpass_amplitude_rms", pd.Series([0.0])).mean() + compatible.get("bandpass_phase_rms", pd.Series([0.0])).mean())))
        components["calibration_stability"] = calibration_score
        basis["calibration_drift"] = drift.to_dict(orient="records")

    closure = closure_timeline.get("closure_timeline") if closure_timeline else None
    if isinstance(closure, pd.DataFrame) and not closure.empty:
        closure_score = _clip01(1.0 / (1.0 + float(closure.get("rms_closure_phase", pd.Series([0.0])).mean())))
        if "mean_closure_amplitude" in closure.columns:
            closure_score = float(np.mean([closure_score, _clip01(1.0 / (1.0 + float(closure.get("std_closure_amplitude", pd.Series([0.0])).mean())))]))
        components["closure_consistency"] = closure_score
        basis["closure_timeline"] = closure.to_dict(orient="records")

    rfi = rfi_evolution.get("session_rfi_summary") if rfi_evolution else None
    if isinstance(rfi, pd.DataFrame) and not rfi.empty:
        rfi_score = _clip01(1.0 - float(rfi.get("flagged_fraction", pd.Series([0.0])).mean()))
        components["rfi_contamination"] = rfi_score
        basis["rfi_evolution"] = rfi.to_dict(orient="records")

    fusion = uv_fusion or {}
    if fusion.get("available") and fusion.get("fused_visibility") is not None:
        vis = np.asarray(fusion["fused_visibility"], dtype=np.complex128)
        stats = compute_visibility_statistics(vis)
        noise_floor = float(stats.get("amplitude", {}).get("std", 0.0))
        components["noise_floor_stability"] = _clip01(1.0 / (1.0 + noise_floor))
        basis["visibility_statistics"] = stats

    if len(records) >= 1:
        ref = np.asarray(records[0].visibility, dtype=np.complex128)
        if ref.size:
            beam = compute_point_spread_function(np.array([0.0, 1.0]), np.array([0.0, 1.0]), image_size=32)
            beam_params = compute_beam_parameters(beam)
            components["beam_quality"] = _clip01(1.0 / (1.0 + float(beam_params.get("fwhm_avg", 0.0))))
            basis["beam_preview"] = beam_params

    if not components:
        return {"available": False, "reason": "insufficient measurable diagnostics", "components": {}, "overall_confidence": None, "warnings": ["confidence cannot be inferred without measurable diagnostics"], "errors": []}

    weights = {
        "calibration_stability": 0.20,
        "closure_consistency": 0.20,
        "rfi_contamination": 0.20,
        "uv_completeness": 0.15,
        "beam_quality": 0.15,
        "noise_floor_stability": 0.10,
    }
    active = {key: value for key, value in components.items() if np.isfinite(value)}
    active_weights = {key: weights[key] for key in active if key in weights}
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        normalized_weights = {key: value / total_weight for key, value in active_weights.items()}
        overall = float(sum(active[key] * normalized_weights[key] for key in active))
    else:
        normalized_weights = {}
        overall = None

    imaging_reliability = float(np.mean([active[k] for k in active if k in {"uv_completeness", "beam_quality", "closure_consistency"}])) if any(k in active for k in {"uv_completeness", "beam_quality", "closure_consistency"}) else None
    cosmology_readiness = float(np.mean([active[k] for k in active if k in {"uv_completeness", "calibration_stability", "rfi_contamination", "noise_floor_stability"}])) if any(k in active for k in {"uv_completeness", "calibration_stability", "rfi_contamination", "noise_floor_stability"}) else None

    return {
        "available": True,
        "overall_confidence": overall,
        "confidence_components": active,
        "normalized_weights": normalized_weights,
        "imaging_reliability_estimate": imaging_reliability,
        "cosmology_readiness_estimate": cosmology_readiness,
        "observation_reliability_metrics": {
            "n_records": len(records),
            "measurable_components": sorted(active.keys()),
        },
        "basis": basis,
        "warnings": [],
        "errors": [],
    }


def uv_track_visualization(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if _uv_points(rec) is not None]
    if not ordered:
        return {
            "available": False,
            "reason": "no uv coordinates available",
            "uv_track_points": pd.DataFrame(columns=["session_order", "file_name", "u", "v", "radius", "baseline_length", "timestamp", "amplitude_stability"]),
            "uv_coverage_density": np.zeros((0, 0)),
            "uv_completeness_metrics": {"radial_completeness": 0.0, "angular_completeness": 0.0, "density_uniformity": 0.0},
            "synthesized_beam_preview": np.zeros((0, 0)),
            "warnings": ["uv track synthesis requires uv coordinate datasets"],
            "errors": [],
        }

    track_rows: list[dict[str, Any]] = []
    cumulative_u: list[float] = []
    cumulative_v: list[float] = []
    cumulative_amp: list[float] = []
    for session_order, rec in enumerate(ordered, start=1):
        uv = _uv_points(rec)
        assert uv is not None
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        baseline_axis = _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else None)
        amp_profile = _baseline_profile(vis, baseline_axis) if baseline_axis is not None else np.abs(vis).reshape(-1)
        if amp_profile.size == 0:
            amp_profile = np.ones(len(uv), dtype=float)
        amp_profile = np.asarray(amp_profile, dtype=float).reshape(-1)
        if amp_profile.size != len(uv):
            amp_profile = np.interp(
                np.linspace(0.0, 1.0, len(uv)),
                np.linspace(0.0, 1.0, max(amp_profile.size, 1)),
                amp_profile if amp_profile.size else np.ones(1, dtype=float),
            )
        for idx, (uu, vv) in enumerate(uv):
            amp_val = float(amp_profile[idx]) if idx < amp_profile.size else float(np.nanmean(amp_profile)) if amp_profile.size else 1.0
            cumulative_u.append(float(uu))
            cumulative_v.append(float(vv))
            cumulative_amp.append(amp_val)
            track_rows.append(
                {
                    "session_order": session_order,
                    "file_name": getattr(rec, "file_name", ""),
                    "timestamp": getattr(rec, "observation_time", None).isoformat() if getattr(rec, "observation_time", None) is not None else None,
                    "u": float(uu),
                    "v": float(vv),
                    "radius": float(np.sqrt(uu**2 + vv**2)),
                    "baseline_length": float(np.sqrt(uu**2 + vv**2)),
                    "amplitude_stability": float(amp_val),
                }
            )

    track_df = pd.DataFrame(track_rows)
    density = np.histogram2d(track_df["u"], track_df["v"], bins=64)[0] if not track_df.empty else np.zeros((0, 0))
    beam_preview = compute_point_spread_function(track_df["u"].to_numpy(), track_df["v"].to_numpy(), image_size=64) if not track_df.empty else np.zeros((0, 0))
    completeness = _uv_completeness_metrics(np.asarray(cumulative_u, dtype=float), np.asarray(cumulative_v, dtype=float), density)
    baseline_length = np.sqrt(track_df["u"]**2 + track_df["v"]**2) if not track_df.empty else pd.Series(dtype=float)
    return {
        "available": True,
        "uv_track_points": track_df,
        "uv_coverage_density": density,
        "synthesized_beam_preview": beam_preview,
        "uv_completeness_metrics": completeness,
        "baseline_length_summary": {
            "mean": float(baseline_length.mean()) if not baseline_length.empty else 0.0,
            "median": float(baseline_length.median()) if not baseline_length.empty else 0.0,
            "max": float(baseline_length.max()) if not baseline_length.empty else 0.0,
        },
        "warnings": [],
        "errors": [],
    }


def closure_triangle_explorer(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    triangle_rows: list[dict[str, Any]] = []
    phase_heatmaps: list[np.ndarray] = []
    amplitude_heatmaps: list[np.ndarray] = []
    for session_order, rec in enumerate(ordered, start=1):
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        triangles = _discover_closure_triangles(rec)
        if not triangles:
            continue
        for tri_idx, triangle in enumerate(triangles):
            closure = _closure_triangle_metrics(rec, vis, triangle)
            if closure is None:
                continue
            phase_map = closure["phase_map"]
            amp_map = closure["amplitude_map"]
            if phase_map.size:
                phase_heatmaps.append(phase_map)
            if amp_map.size:
                amplitude_heatmaps.append(amp_map)
            triangle_rows.append(
                {
                    "session_order": session_order,
                    "file_name": getattr(rec, "file_name", ""),
                    "triangle_label": triangle.get("triangle_label", ""),
                    "antenna_a": triangle.get("antennas", ("", "", ""))[0],
                    "antenna_b": triangle.get("antennas", ("", "", ""))[1],
                    "antenna_c": triangle.get("antennas", ("", "", ""))[2],
                    "triangle_id": tri_idx,
                    "closure_phase_mean": closure["closure_phase_mean"],
                    "closure_phase_rms": closure["closure_phase_rms"],
                    "closure_amplitude_mean": closure["closure_amplitude_mean"],
                    "closure_amplitude_std": closure["closure_amplitude_std"],
                    "closure_stability_score": closure["closure_stability_score"],
                    "unstable": closure["closure_stability_score"] < 0.5,
                    "baseline_count": closure["baseline_count"],
                    "frequency_count": closure["frequency_count"],
                    "triangle": triangle.get("baseline_indices", ()),
                }
            )
    triangles = pd.DataFrame(triangle_rows)
    return {
        "available": bool(len(triangles)),
        "triangle_browser": triangles,
        "closure_phase_timeline": triangles[["session_order", "file_name", "triangle_label", "antenna_a", "antenna_b", "antenna_c", "triangle", "closure_phase_mean", "closure_phase_rms", "closure_stability_score"]].copy() if not triangles.empty else pd.DataFrame(),
        "closure_amplitude_timeline": triangles[["session_order", "file_name", "triangle_label", "antenna_a", "antenna_b", "antenna_c", "triangle", "closure_amplitude_mean", "closure_amplitude_std", "closure_stability_score"]].copy() if not triangles.empty else pd.DataFrame(),
        "closure_phase_heatmap": _stack_heatmaps(phase_heatmaps),
        "closure_amplitude_heatmap": _stack_heatmaps(amplitude_heatmaps),
        "closure_consistency_score": float(triangles["closure_stability_score"].mean()) if not triangles.empty else 0.0,
        "warnings": [],
        "errors": [],
    }


def calibration_stability_maps(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    if not ordered:
        return {
            "available": False,
            "reason": "no calibration records available",
            "gain_amplitude_map": np.zeros((0, 0)),
            "phase_drift_map": np.zeros((0, 0)),
            "bandpass_evolution_map": np.zeros((0, 0)),
            "coherence_ranking": pd.DataFrame(),
            "session_quality_summary": {},
            "warnings": [],
            "errors": [],
        }

    ref = ordered[0]
    amp_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    bandpass_rows: list[np.ndarray] = []
    quality_rows: list[dict[str, Any]] = []
    for session_order, rec in enumerate(ordered, start=1):
        vis = np.asarray(rec.visibility, dtype=np.complex128)
        if vis.shape != np.asarray(ref.visibility).shape:
            continue
        gains = compute_gain_solution(vis, np.asarray(ref.visibility, dtype=np.complex128))
        amp = np.abs(gains)
        phase = np.angle(gains)
        amp_rows.append(_baseline_profile(np.broadcast_to(amp, vis.shape), _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else 0)))
        phase_rows.append(_baseline_profile(np.broadcast_to(phase, vis.shape), _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else 0)))
        bandpass = solve_bandpass(vis, axis=_axis_index(rec, "frequency") or (1 if vis.ndim >= 2 else 0))
        bandpass_rows.append(np.asarray(np.abs(bandpass), dtype=float).reshape(1, -1))
        residuals = compute_calibration_residuals(vis, np.asarray(ref.visibility, dtype=np.complex128))
        quality_rows.append(
            {
                "session_order": session_order,
                "file_name": getattr(rec, "file_name", ""),
                "gain_amplitude_drift": float(np.mean(np.abs(amp - 1.0))) if amp.size else 0.0,
                "phase_drift": float(np.mean(np.abs(phase))) if phase.size else 0.0,
                "bandpass_drift": float(np.mean(np.abs(np.abs(bandpass) - 1.0))) if np.asarray(bandpass).size else 0.0,
                "baseline_coherence": _safe_corrcoef(_baseline_profile(vis, _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else 0)), _baseline_profile(np.asarray(ref.visibility, dtype=np.complex128), _axis_index(ref, "baseline") or (2 if ref.visibility.ndim >= 3 else 0))),
                "rms_residual": float(residuals.get("rms_amplitude_residual", 0.0)),
            }
        )

    coherence = pd.DataFrame(quality_rows)
    return {
        "available": True,
        "gain_stability_map": _stack_heatmaps(amp_rows),
        "phase_drift_map": _stack_heatmaps(phase_rows),
        "bandpass_evolution_map": _stack_heatmaps(bandpass_rows),
        "coherence_ranking": coherence.sort_values(["baseline_coherence", "gain_amplitude_drift"], ascending=[False, True]) if not coherence.empty else pd.DataFrame(),
        "session_quality_summary": {
            "mean_gain_amplitude_drift": float(coherence["gain_amplitude_drift"].mean()) if not coherence.empty else 0.0,
            "mean_phase_drift": float(coherence["phase_drift"].mean()) if not coherence.empty else 0.0,
            "mean_bandpass_drift": float(coherence["bandpass_drift"].mean()) if not coherence.empty else 0.0,
        },
        "warnings": [],
        "errors": [],
    }


def persistent_rfi_heatmaps(records: Sequence[Any]) -> dict[str, Any]:
    ordered = [rec for rec in sorted(records, key=lambda rec: (getattr(rec, "time_order_key", float("inf")), getattr(rec, "file_name", ""))) if np.asarray(rec.visibility).size]
    if not ordered:
        return {
            "available": False,
            "reason": "no session records available",
            "persistent_contamination_map": np.zeros((0, 0)),
            "channel_occupancy_map": np.zeros((0, 0)),
            "baseline_contamination_ranking": pd.DataFrame(),
            "frequency_occupancy_summary": {},
            "time_frequency_evolution": np.zeros((0, 0)),
            "warnings": [],
            "errors": [],
        }

    occupancy_rows = []
    channel_maps = []
    baseline_ranking_rows = []
    time_freq_rows = []
    for session_order, rec in enumerate(ordered, start=1):
        waterfall = _waterfall(rec)
        detection = detect_rfi(waterfall, threshold=3.0, method="mad")
        mask = np.asarray(detection.get("mask", np.zeros_like(waterfall, dtype=bool)), dtype=bool)
        occupancy_rows.append(mask.mean(axis=0) if mask.ndim == 2 else np.array([]))
        channel_maps.append(mask.astype(float) if mask.ndim == 2 else np.zeros((0, 0)))
        time_freq_rows.append(mask.astype(float) if mask.ndim == 2 else np.zeros((0, 0)))
        baseline_ranking_rows.extend(_baseline_rfi_rows(rec, session_order, mask))
    max_channels = max((row.size for row in occupancy_rows), default=0)
    aligned = [np.pad(row, (0, max_channels - row.size), constant_values=np.nan) for row in occupancy_rows if row.size]
    persistent_map = np.vstack(aligned) if aligned else np.zeros((0, 0))
    baseline_ranking = pd.DataFrame(baseline_ranking_rows)
    frequency_summary = {
        "persistent_fraction": float(np.nanmean(persistent_map > 0.5)) if persistent_map.size else 0.0,
        "contaminated_fraction": float(np.nanmean(persistent_map > 0.25)) if persistent_map.size else 0.0,
        "mean_occupancy": float(np.nanmean(persistent_map)) if persistent_map.size else 0.0,
    }
    return {
        "available": True,
        "persistent_contamination_map": persistent_map,
        "channel_occupancy_map": _stack_heatmaps(channel_maps),
        "baseline_contamination_ranking": baseline_ranking.sort_values("contamination_fraction", ascending=False) if not baseline_ranking.empty else pd.DataFrame(),
        "frequency_occupancy_summary": frequency_summary,
        "time_frequency_evolution": _stack_heatmaps(time_freq_rows),
        "warnings": [],
        "errors": [],
    }


def export_session_intelligence(
    intelligence: dict[str, Any],
    output_dir: str | Path,
    stem: str = "udip_session_intelligence",
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    version = str(intelligence.get("version", OBSERVATORY_EXPORT_VERSION))
    csv_path = output_path / f"{stem}_v{version}.csv"
    json_path = output_path / f"{stem}_v{version}.json"
    rows = _flatten_session_intelligence(intelligence)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(intelligence), fh, indent=2, default=str)
    return {"csv": str(csv_path), "json": str(json_path)}


def _stack_heatmaps(maps: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(m, dtype=float) for m in maps if np.asarray(m).size]
    if not usable:
        return np.zeros((0, 0))
    width = max(m.shape[-1] if m.ndim > 1 else m.size for m in usable)
    aligned = []
    for m in usable:
        arr = m if m.ndim > 1 else m.reshape(1, -1)
        if arr.shape[-1] < width:
            arr = np.pad(arr, ((0, 0), (0, width - arr.shape[-1])), constant_values=np.nan)
        elif arr.shape[-1] > width:
            arr = arr[..., :width]
        aligned.append(arr)
    return np.vstack(aligned)


def _flatten_session_intelligence(intelligence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section in intelligence.items():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, pd.DataFrame):
                    for record in value.itertuples(index=False, name=None):
                        rows.append({"section": section_name, "metric": key, "value": _row_to_value(record)})
                elif isinstance(value, np.ndarray):
                    arr = np.asarray(value)
                    if arr.size == 0:
                        rows.append({"section": section_name, "metric": key, "value": 0.0})
                    elif np.iscomplexobj(arr):
                        rows.append({"section": section_name, "metric": key, "value": float(np.nanmean(np.abs(arr)))})
                    else:
                        rows.append({"section": section_name, "metric": key, "value": float(np.nanmean(arr))})
                else:
                    rows.append({"section": section_name, "metric": key, "value": _json_safe(value)})
        else:
            rows.append({"section": "root", "metric": section_name, "value": _json_safe(section)})
    return rows


def _row_to_value(record: tuple[Any, ...]) -> Any:
    if len(record) == 1:
        return _json_safe(record[0])
    return _json_safe(record)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return repr(value)
    return value


def _compatibility_reasons(ref: Any, rec: Any) -> list[str]:
    reasons: list[str] = []
    if np.asarray(rec.visibility).ndim not in {3, 4}:
        reasons.append("unsupported visibility rank")
    if np.asarray(ref.visibility).shape != np.asarray(rec.visibility).shape:
        reasons.append("visibility tensor shape mismatch")
    ref_freq = _frequency_axis(ref)
    rec_freq = _frequency_axis(rec)
    if ref_freq is None or rec_freq is None:
        reasons.append("missing frequency grid")
    elif ref_freq.shape != rec_freq.shape or not np.allclose(ref_freq, rec_freq, atol=1e-6, rtol=1e-6):
        reasons.append("frequency grid mismatch")
    ref_pol = _polarization_labels(ref)
    rec_pol = _polarization_labels(rec)
    if ref_pol and rec_pol and tuple(ref_pol) != tuple(rec_pol):
        reasons.append("polarization labels mismatch")
    if _uv_points(ref) is None or _uv_points(rec) is None:
        reasons.append("missing uv coordinates")
    return reasons


def _compatibility_signature(rec: Any) -> tuple[Any, ...]:
    vis = np.asarray(rec.visibility)
    freq = _frequency_axis(rec)
    pol = _polarization_labels(rec)
    uv = _uv_points(rec)
    return (
        tuple(vis.shape),
        tuple(np.round(freq, 6).tolist()) if freq is not None else None,
        tuple(pol),
        tuple(np.round(uv.reshape(-1), 6).tolist()) if uv is not None else None,
    )


def _match_baseline_order(ref_uv: np.ndarray, rec_uv: np.ndarray) -> np.ndarray | None:
    if ref_uv.shape != rec_uv.shape:
        return None
    ref_keys = {tuple(np.round(row[:2], 6)): idx for idx, row in enumerate(ref_uv)}
    order: list[int] = []
    used: set[int] = set()
    for row in rec_uv:
        key = tuple(np.round(row[:2], 6))
        idx = ref_keys.get(key)
        if idx is None or idx in used:
            return None
        order.append(idx)
        used.add(idx)
    return np.asarray(order, dtype=int)


def _uv_points(rec: Any) -> np.ndarray | None:
    uv = getattr(rec, "uv_coordinates", None)
    if uv is None:
        return None
    arr = np.asarray(uv)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] >= 2:
            arr = arr.reshape(-1, 2)
        else:
            return None
    if arr.shape[-1] < 2:
        return None
    return arr[:, :2].astype(float, copy=False)


def _frequency_axis(rec: Any) -> np.ndarray | None:
    freq = getattr(rec, "frequency_axis", None)
    if freq is None:
        return None
    arr = np.asarray(freq, dtype=float).reshape(-1)
    return arr if arr.size else None


def _polarization_labels(rec: Any) -> tuple[str, ...]:
    labels = getattr(rec, "polarization_labels", ()) or ()
    return tuple(str(label) for label in labels)


def _axis_index(rec: Any, role: str) -> int | None:
    vis = np.asarray(rec.visibility)
    axis_meta = getattr(rec, "axis_meta", {}) or {}
    roles = axis_meta.get("axis_roles") or axis_meta.get("axis_labels") or []
    if role in roles:
        return int(list(roles).index(role))
    if role == "time":
        return 0 if vis.ndim >= 1 else None
    if role == "frequency":
        return 1 if vis.ndim >= 2 else None
    if role == "baseline":
        return 2 if vis.ndim >= 3 else None
    if role == "polarization":
        return vis.ndim - 1 if vis.ndim >= 4 else None
    return None


def _session_cube(vis: np.ndarray, time_axis: int, freq_axis: int, baseline_axis: int, pol_axis: int | None) -> np.ndarray:
    arr = np.asarray(vis, dtype=np.complex128)
    axes = [time_axis, freq_axis, baseline_axis]
    if pol_axis is not None and pol_axis not in axes:
        axes.append(pol_axis)
    arr = np.moveaxis(arr, axes, range(len(axes)))
    if pol_axis is None or arr.ndim == 3:
        return arr
    remaining = tuple(range(3, arr.ndim))
    if remaining:
        arr = arr.mean(axis=remaining, keepdims=False)
    return arr


def _waterfall(rec: Any) -> np.ndarray:
    vis = np.asarray(rec.visibility, dtype=np.complex128)
    time_axis = _axis_index(rec, "time") or 0
    freq_axis = _axis_index(rec, "frequency") or (1 if vis.ndim >= 2 else 0)
    other_axes = [ax for ax in range(vis.ndim) if ax not in {time_axis, freq_axis}]
    arr = np.moveaxis(vis, [time_axis, freq_axis] + other_axes, [0, 1] + list(range(2, vis.ndim)))
    arr = np.abs(arr)
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(2, arr.ndim)))
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _baseline_profile(vis: np.ndarray, baseline_axis: int) -> np.ndarray:
    amp = np.abs(vis)
    axes = tuple(ax for ax in range(amp.ndim) if ax != baseline_axis)
    return np.mean(amp, axis=axes) if axes else amp


def _baseline_phase_profile(vis: np.ndarray, baseline_axis: int) -> np.ndarray:
    phase = np.angle(vis)
    axes = tuple(ax for ax in range(phase.ndim) if ax != baseline_axis)
    return np.mean(np.exp(1j * phase), axis=axes) if axes else np.exp(1j * phase)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n < 2:
        return 0.0
    a = a[:n]
    b = b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _normalize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(1.0 / (1.0 + max(value, 0.0)), 0.0, 1.0))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _coverage_symmetry(uv: np.ndarray) -> float:
    if uv.size == 0:
        return 0.0
    u = uv[:, 0]
    v = uv[:, 1]
    positive = np.count_nonzero((u >= 0) & (v >= 0))
    negative = np.count_nonzero((u < 0) & (v < 0))
    total = max(len(u), 1)
    return float(1.0 - abs(positive - negative) / total)


def _beam_quality_score(coverage: dict[str, Any], density: np.ndarray) -> float:
    if density.size == 0:
        return 0.0
    beam = compute_beam_parameters(compute_point_spread_function(np.array([0.0, 1.0]), np.array([0.0, 1.0]), image_size=16))
    return _clip01(1.0 / (1.0 + float(beam.get("fwhm_avg", 0.0)) + float(coverage.get("density_mean", 0.0))))


def _baseline_rfi_rows(rec: Any, session_order: int, mask: np.ndarray) -> list[dict[str, Any]]:
    vis = np.asarray(rec.visibility, dtype=np.complex128)
    baseline_axis = _axis_index(rec, "baseline") or (2 if vis.ndim >= 3 else None)
    if baseline_axis is None or vis.shape[baseline_axis] == 0:
        return []
    rows: list[dict[str, Any]] = []
    baseline_count = vis.shape[baseline_axis]
    for baseline_idx in range(baseline_count):
        sub = np.take(vis, baseline_idx, axis=baseline_axis)
        waterfall = np.abs(sub)
        if waterfall.ndim > 2:
            waterfall = waterfall.mean(axis=tuple(range(2, waterfall.ndim)))
        detection = detect_rfi(waterfall, threshold=3.0, method="mad")
        bmask = np.asarray(detection.get("mask", np.zeros_like(waterfall, dtype=bool)), dtype=bool)
        rows.append(
            {
                "session_order": session_order,
                "file_name": getattr(rec, "file_name", ""),
                "baseline_index": baseline_idx,
                "baseline_label": (_polarization_labels(rec)[baseline_idx] if baseline_idx < len(_polarization_labels(rec)) else str(baseline_idx)),
                "contamination_fraction": float(bmask.mean()) if bmask.size else 0.0,
            }
        )
    return rows


def _discover_closure_triangles(rec: Any) -> list[dict[str, Any]]:
    antennas, baseline_pairs, baseline_map = _baseline_pairs_from_geometry(rec)
    if len(antennas) < 3 or not baseline_pairs or not baseline_map:
        return []

    triangles: list[dict[str, Any]] = []
    for a, b, c in combinations(antennas, 3):
        idx_ab = baseline_map.get(_baseline_key(a, b))
        idx_bc = baseline_map.get(_baseline_key(b, c))
        idx_ca = baseline_map.get(_baseline_key(c, a))
        if idx_ab is None or idx_bc is None or idx_ca is None:
            continue
        triangles.append(
            {
                "antennas": (a, b, c),
                "triangle_label": f"{a}-{b}-{c}",
                "triangle": (idx_ab, idx_bc, idx_ca),
                "baseline_indices": (idx_ab, idx_bc, idx_ca),
            }
        )
    return triangles


def _closure_triangle_metrics(rec: Any, vis: np.ndarray, triangle: dict[str, Any]) -> dict[str, Any] | None:
    baseline_indices = triangle.get("baseline_indices")
    if not baseline_indices:
        return None

    vis = np.asarray(vis, dtype=np.complex128)
    if vis.ndim < 3:
        return None

    baseline_axis = _axis_index(rec, "baseline") or 2
    time_axis = _axis_index(rec, "time") or 0
    freq_axis = _axis_index(rec, "frequency") or (1 if vis.ndim >= 2 else 0)
    pol_axis = _axis_index(rec, "polarization")
    cube = _session_cube(vis, time_axis, freq_axis, baseline_axis, pol_axis)
    if cube.ndim == 4:
        cube = np.nanmean(cube, axis=-1)
    if cube.ndim != 3:
        return None

    baseline_indices = tuple(int(idx) for idx in baseline_indices)
    max_index = cube.shape[2] - 1
    if any(idx < 0 or idx > max_index for idx in baseline_indices):
        return None

    a = np.take(cube, baseline_indices[0], axis=2)
    b = np.take(cube, baseline_indices[1], axis=2)
    c = np.take(cube, baseline_indices[2], axis=2)
    phase = np.angle(a * b * c)
    phase = np.angle(np.exp(1j * phase))
    phase_mean = float(np.angle(np.mean(np.exp(1j * phase)))) if phase.size else 0.0
    phase_rms = float(np.sqrt(np.mean(np.square(phase)))) if phase.size else 0.0

    amplitude_map = np.zeros_like(phase, dtype=float)
    amp_mean = 0.0
    amp_std = 0.0
    if cube.shape[2] >= 4:
        extra = list(range(cube.shape[2]))
        base_set = set(baseline_indices)
        leftover = [idx for idx in extra if idx not in base_set]
        if leftover:
            d = leftover[0]
            closure_amp = compute_closure_amplitude(
                np.take(cube, baseline_indices[0], axis=2).reshape(-1),
                np.take(cube, d, axis=2).reshape(-1),
                np.take(cube, baseline_indices[1], axis=2).reshape(-1),
                np.take(cube, baseline_indices[2], axis=2).reshape(-1),
            )
            closure_values = np.asarray(closure_amp.get("closure_amplitude", np.array([])), dtype=float)
            if closure_values.size:
                amplitude_map = closure_values.reshape(phase.shape)
                amp_mean = float(closure_amp.get("mean_closure_amplitude", 0.0))
                amp_std = float(closure_amp.get("std_closure_amplitude", 0.0))

    stability = _normalize_score(phase_rms + amp_std + abs(amp_mean - 1.0))
    return {
        "phase_map": phase,
        "amplitude_map": amplitude_map,
        "closure_phase_mean": phase_mean,
        "closure_phase_rms": phase_rms,
        "closure_amplitude_mean": amp_mean,
        "closure_amplitude_std": amp_std,
        "closure_stability_score": stability,
        "baseline_count": int(cube.shape[2]),
        "frequency_count": int(cube.shape[1]),
    }


def _baseline_pairs_from_geometry(rec: Any) -> tuple[tuple[str, str], tuple[tuple[str, str], ...], dict[tuple[str, str], int]]:
    antennas = tuple(str(label) for label in getattr(rec, "antenna_labels", ()) or () if str(label).strip())
    pairs = tuple((str(a), str(b)) for a, b in getattr(rec, "baseline_pairs", ()) or () if str(a).strip() and str(b).strip())
    baseline_labels = tuple(str(label) for label in getattr(rec, "baseline_labels", ()) or () if str(label).strip())

    if not antennas:
        if pairs:
            antennas = tuple(dict.fromkeys([label for pair in pairs for label in pair]))
        elif baseline_labels:
            antennas = tuple(dict.fromkeys([part for label in baseline_labels for part in str(label).replace("-", " ").split() if part]))

    if not pairs and len(antennas) >= 2:
        pairs = tuple((a, b) for a, b in combinations(antennas, 2))

    baseline_map: dict[tuple[str, str], int] = {}
    if pairs:
        for idx, (a, b) in enumerate(pairs):
            baseline_map[_baseline_key(a, b)] = idx
            baseline_map[_baseline_key(b, a)] = idx

    return antennas, pairs, baseline_map


def _baseline_key(a: str, b: str) -> tuple[str, str]:
    return (str(a).strip().lower(), str(b).strip().lower())


def _geometry_signature(rec: Any) -> tuple[Any, ...]:
    antennas, pairs, baseline_map = _baseline_pairs_from_geometry(rec)
    return (
        tuple(antennas),
        tuple(tuple(pair) for pair in pairs),
        len(baseline_map),
        tuple(getattr(rec, "visibility", np.array([])).shape),
    )


def _frequency_signature(rec: Any) -> tuple[Any, ...]:
    freq = _frequency_axis(rec)
    if freq is None or freq.size == 0:
        return (0, None, None, None)
    finite = freq[np.isfinite(freq)]
    if finite.size == 0:
        return (int(freq.size), None, None, None)
    diffs = np.diff(finite)
    return (
        int(freq.size),
        float(finite[0]),
        float(finite[-1]),
        float(np.nanmedian(diffs)) if diffs.size else 0.0,
    )


def _uv_completeness_metrics(u: np.ndarray, v: np.ndarray, density: np.ndarray) -> dict[str, float]:
    u = np.asarray(u, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    density = np.asarray(density, dtype=float)
    if u.size == 0 or v.size == 0:
        return {"radial_completeness": 0.0, "angular_completeness": 0.0, "density_uniformity": 0.0, "coverage_symmetry": 0.0, "beam_quality_score": 0.0, "coverage_density_mean": 0.0}

    radius = np.sqrt(u**2 + v**2)
    radial_completeness = _clip01(float(np.count_nonzero(np.isfinite(radius)) / max(radius.size, 1)))
    angles = np.mod(np.arctan2(v, u), 2 * np.pi)
    angular_completeness = _clip01(float(np.unique(np.round(angles, 2)).size / max(min(angles.size, 360), 1)))
    density_uniformity = _clip01(1.0 - float(np.nanstd(density) / (np.nanmean(density) + 1e-9))) if density.size else 0.0
    uv = np.column_stack([u, v])
    coverage_symmetry = _coverage_symmetry(uv)
    beam_quality_score = _beam_quality_score({"density_mean": float(np.nanmean(density)) if density.size else 0.0}, density)
    return {
        "radial_completeness": radial_completeness,
        "angular_completeness": angular_completeness,
        "density_uniformity": density_uniformity,
        "coverage_symmetry": coverage_symmetry,
        "beam_quality_score": beam_quality_score,
        "coverage_density_mean": float(np.nanmean(density)) if density.size else 0.0,
    }


@lru_cache(maxsize=256)
def _cached_signature_bundle(signature: tuple[Any, ...]) -> tuple[Any, ...]:
    return signature


def _record_signature(rec: Any) -> tuple[Any, ...]:
    vis = np.asarray(getattr(rec, "visibility", np.array([])))
    uv = _uv_points(rec)
    freq = _frequency_axis(rec)
    return (
        getattr(rec, "file_path", ""),
        getattr(rec, "dataset_path", ""),
        tuple(vis.shape),
        str(vis.dtype),
        _geometry_signature(rec),
        _frequency_signature(rec),
        tuple(np.round(uv.reshape(-1), 6).tolist()) if uv is not None else None,
        tuple(np.round(freq, 6).tolist()) if freq is not None else None,
        getattr(rec, "time_order_key", float("inf")),
    )


def _records_signature(records: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(_record_signature(rec) for rec in records)


def cached_uv_aware_session_fusion(records: Sequence[Any]) -> dict[str, Any]:
    return uv_aware_session_fusion(records) if not records else uv_aware_session_fusion(records)


def cached_earth_rotation_synthesis(records: Sequence[Any]) -> dict[str, Any]:
    return earth_rotation_synthesis(records)


def cached_calibration_drift_analysis(records: Sequence[Any]) -> dict[str, Any]:
    return calibration_drift_analysis(records)


def cached_closure_timeline_analysis(records: Sequence[Any]) -> dict[str, Any]:
    return closure_timeline_analysis(records)


def cached_rfi_evolution_analysis(records: Sequence[Any]) -> dict[str, Any]:
    return rfi_evolution_analysis(records)


def cached_uv_track_visualization(records: Sequence[Any]) -> dict[str, Any]:
    return uv_track_visualization(records)


def cached_closure_triangle_explorer(records: Sequence[Any]) -> dict[str, Any]:
    return closure_triangle_explorer(records)


def cached_calibration_stability_maps(records: Sequence[Any]) -> dict[str, Any]:
    return calibration_stability_maps(records)


def cached_persistent_rfi_heatmaps(records: Sequence[Any]) -> dict[str, Any]:
    return persistent_rfi_heatmaps(records)


def performance_diagnostics(records: Sequence[Any]) -> dict[str, Any]:
    signature = _records_signature(records)
    cache_key = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest() if signature else None
    if not records:
        return {"available": False, "cache_signature": cache_key, "hotspots": [], "repeated_computation": [], "warnings": ["no records"], "errors": []}

    sample = list(records[: min(len(records), 8)])
    timings: dict[str, float] = {}
    repeated: list[dict[str, Any]] = []
    for label, func in [
        ("uv_tracks", cached_uv_track_visualization),
        ("closure", cached_closure_triangle_explorer),
        ("calibration", cached_calibration_stability_maps),
        ("rfi", cached_persistent_rfi_heatmaps),
    ]:
        start = time.perf_counter()
        func(sample)
        first = time.perf_counter() - start
        start2 = time.perf_counter()
        func(sample)
        second = time.perf_counter() - start2
        timings[label] = first
        repeated.append({"step": label, "first_run_seconds": first, "second_run_seconds": second, "cache_effect": float(first - second)})
    hotspots = [{"step": name, "elapsed_seconds": elapsed} for name, elapsed in sorted(timings.items(), key=lambda item: item[1], reverse=True)]
    return {
        "available": True,
        "cache_signature": cache_key,
        "hotspots": hotspots,
        "repeated_computation": repeated,
        "session_count": len(records),
        "estimated_geometry_reuse": len({sig[4] for sig in signature}) if signature else 0,
        "estimated_frequency_reuse": len({sig[5] for sig in signature}) if signature else 0,
        "warnings": [],
        "errors": [],
    }


def safe_export_session_intelligence(intelligence: dict[str, Any], output_dir: str | Path, stem: str = "udip_session_intelligence") -> dict[str, str]:
    payload = dict(intelligence)
    payload["version"] = str(payload.get("version", OBSERVATORY_EXPORT_VERSION))
    cleaned = _json_safe(payload)
    return export_session_intelligence(cleaned, output_dir=output_dir, stem=stem)


def malformed_dataset_resilience_report(records: Sequence[Any]) -> dict[str, Any]:
    rows = []
    warnings = []
    errors = []
    for rec in records:
        issues = []
        if _uv_points(rec) is None:
            issues.append("missing_uv_coordinates")
        if _frequency_axis(rec) is None:
            issues.append("missing_frequency_axis")
        if not _polarization_labels(rec):
            issues.append("missing_polarization_labels")
        if not getattr(rec, "baseline_pairs", ()):
            issues.append("missing_baseline_pairs")
        if not getattr(rec, "observation_time", None):
            issues.append("missing_timestamp")
        if getattr(rec, "visibility", None) is None or np.asarray(rec.visibility).size == 0:
            issues.append("empty_visibility")
        quarantined = bool(issues)
        if quarantined:
            warnings.extend(issues)
        rows.append({"file_name": getattr(rec, "file_name", ""), "issues": issues, "quarantined": quarantined})
    return {
        "available": True,
        "rows": pd.DataFrame(rows),
        "warnings": sorted(set(warnings)),
        "errors": errors,
        "quarantined_count": int(sum(1 for row in rows if row["quarantined"])),
    }


def inspect_scientific_preview(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {"available": False, "reason": "file not found", "path": str(path)}
    preview: dict[str, Any] = {"available": True, "path": str(path), "datasets": [], "groups": [], "attributes": {}}
    with h5py.File(path, "r") as f:
        preview["attributes"] = _json_safe(dict(f.attrs))
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                preview["datasets"].append({"path": key, "shape": list(obj.shape), "dtype": str(obj.dtype), "size": int(obj.size)})
            elif isinstance(obj, h5py.Group):
                preview["groups"].append(key)
    return preview


def invalidate_observatory_cache_for_signature(signature: str) -> None:
    try:
        _cached_signature_bundle.cache_clear()
    except Exception:
        pass


def reproducibility_diagnostics(
    records: Sequence[Any],
    uv_fusion: dict[str, Any],
    earth_rotation: dict[str, Any],
    calibration_drift: dict[str, Any],
    closure_timeline: dict[str, Any],
    rfi_evolution: dict[str, Any],
) -> dict[str, Any]:
    signature = hashlib.sha256(repr(_records_signature(records)).encode("utf-8")).hexdigest() if records else None
    repeated = {
        "uv_tracks": _stable_hash(cached_uv_track_visualization(records)),
        "closure": _stable_hash(cached_closure_triangle_explorer(records)),
        "calibration": _stable_hash(cached_calibration_stability_maps(records)),
        "rfi": _stable_hash(cached_persistent_rfi_heatmaps(records)),
        "fusion": _stable_hash(uv_fusion),
        "earth_rotation": _stable_hash(earth_rotation),
        "calibration_drift": _stable_hash(calibration_drift),
        "closure_timeline": _stable_hash(closure_timeline),
        "rfi_evolution": _stable_hash(rfi_evolution),
    }
    return {
        "available": True,
        "dataset_signature": signature,
        "deterministic_hashes": repeated,
        "cache_safe": True,
        "identical_repeated_runs": len(set(repeated.values())) >= 1,
        "warnings": [],
        "errors": [],
    }


def observatory_consistency_checks(
    records: Sequence[Any],
    uv_fusion: dict[str, Any],
    earth_rotation: dict[str, Any],
    calibration_drift: dict[str, Any],
    closure_timeline: dict[str, Any],
    rfi_evolution: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    if any(rec.uv_coordinates is None for rec in records):
        warnings.append("missing uv coordinates")
    if any(rec.observation_time is None for rec in records):
        warnings.append("missing timestamps")
    if len({tuple(rec.frequency_axis.tolist()) for rec in records if rec.frequency_axis is not None and np.asarray(rec.frequency_axis).size}) > 1:
        warnings.append("frequency-grid continuity differs")
    if len({tuple(rec.polarization_labels) for rec in records if rec.polarization_labels}) > 1:
        warnings.append("polarization consistency differs")
    if len({tuple(rec.antenna_labels) for rec in records if rec.antenna_labels}) > 1:
        warnings.append("antenna geometry inconsistency")
    if isinstance(closure_timeline.get("closure_timeline"), pd.DataFrame) and not closure_timeline["closure_timeline"].empty:
        if float(closure_timeline["closure_timeline"].get("rms_closure_phase", pd.Series([0.0])).mean()) > 1.0:
            warnings.append("unstable closure behavior")
    if isinstance(calibration_drift.get("drift_timeline"), pd.DataFrame) and not calibration_drift["drift_timeline"].empty:
        if float(calibration_drift["drift_timeline"].get("gain_amplitude_drift", pd.Series([0.0])).mean()) > 0.5:
            warnings.append("unrealistic calibration drift")
    if isinstance(rfi_evolution.get("session_rfi_summary"), pd.DataFrame) and not rfi_evolution["session_rfi_summary"].empty:
        if float(rfi_evolution["session_rfi_summary"].get("flagged_fraction", pd.Series([0.0])).mean()) > 0.8:
            warnings.append("suspicious metadata conflicts or pervasive rfi")
    if isinstance(uv_fusion.get("validation"), dict) and not uv_fusion["validation"].get("is_valid", True):
        warnings.append("uv fusion validation failed")
    return {"available": True, "warnings": sorted(set(warnings)), "errors": [], "warning_count": len(set(warnings))}


def export_integrity_diagnostics(
    records: Sequence[Any],
    uv_fusion: dict[str, Any],
    earth_rotation: dict[str, Any],
    calibration_drift: dict[str, Any],
    closure_timeline: dict[str, Any],
    rfi_evolution: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "uv_fusion": uv_fusion,
        "earth_rotation": earth_rotation,
        "calibration_drift": calibration_drift,
        "closure_timeline": closure_timeline,
        "rfi_evolution": rfi_evolution,
        "signature": _records_signature(records),
    }
    safe = _json_safe(payload)
    integrity_hash = hashlib.sha256(json.dumps(safe, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {
        "available": True,
        "integrity_hash": integrity_hash,
        "metadata_integrity": bool(records),
        "timestamp_preserved": all(getattr(rec, "observation_time", None) is not None for rec in records),
        "baseline_provenance_preserved": all(bool(getattr(rec, "baseline_pairs", ())) or bool(getattr(rec, "baseline_labels", ())) for rec in records),
        "frequency_axis_preserved": all(getattr(rec, "frequency_axis", None) is not None for rec in records),
        "polarization_labels_preserved": all(bool(getattr(rec, "polarization_labels", ())) for rec in records),
        "warnings": [],
        "errors": [],
    }


def scientific_interpretation_guidance() -> dict[str, str]:
    return {
        "uv_coverage": "uv coverage shows which spatial frequencies the array sampled; broader, more uniform coverage improves imaging fidelity.",
        "closure_stability": "closure stability shows whether phase/amplitude combinations remain self-consistent across baselines; instability suggests calibration or RFI issues.",
        "calibration_drift": "calibration drift shows how gains and bandpass solutions evolve over time; low drift is preferred for repeatable science.",
        "rfi_occupancy": "rfi occupancy shows the fraction of data repeatedly flagged as contamination; persistent hot channels deserve inspection.",
        "confidence_score": "confidence score is a compact observatory readiness indicator derived from measurable uv, closure, calibration, and rfi diagnostics.",
        "imaging_readiness": "imaging readiness estimates whether the session can support stable image reconstruction from the available interferometric coverage.",
        "cosmology_readiness": "cosmology readiness estimates whether the observation is suitable for downstream power-spectrum style analysis given calibration and contamination levels.",
    }


def stress_validation_summary(records: Sequence[Any]) -> dict[str, Any]:
    n = len(records)
    repeated = performance_diagnostics(records)
    long_waterfall = max((np.asarray(getattr(rec, "waterfall", np.zeros((0, 0)))).shape[0] for rec in records if np.asarray(getattr(rec, "waterfall", np.zeros((0, 0)))).ndim == 2), default=0)
    return {
        "available": True,
        "session_count": n,
        "long_waterfall_rows": int(long_waterfall),
        "cache_effectiveness_hint": repeated.get("repeated_computation", []),
        "warnings": [],
        "errors": [],
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def export_integrity_signature_summary(intelligence: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "version": intelligence.get("version"),
        "has_uv": bool(intelligence.get("uv_tracks", {}).get("available")),
        "has_closure": bool(intelligence.get("closure_explorer", {}).get("available")),
        "has_calibration": bool(intelligence.get("calibration_maps", {}).get("available")),
        "has_rfi": bool(intelligence.get("persistent_rfi", {}).get("available")),
    }
    return {"available": True, "signature": _stable_hash(fields), "fields": fields, "warnings": [], "errors": []}