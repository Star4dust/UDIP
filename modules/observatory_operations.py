from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from modules.observation_session import reconstruct_observation_sessions
from modules.observatory_intelligence import (
    build_observatory_session_intelligence,
    cached_calibration_stability_maps,
    cached_closure_triangle_explorer,
    cached_persistent_rfi_heatmaps,
    cached_uv_track_visualization,
    export_session_intelligence,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "validation_corpus"
CORPUS_MANIFEST = CORPUS_ROOT / "manifest.json"


def load_validation_corpus_manifest() -> dict[str, Any]:
    if not CORPUS_MANIFEST.exists():
        return {"available": False, "corpus_root": str(CORPUS_ROOT), "datasets": [], "warnings": ["validation corpus manifest not found"], "errors": []}
    with CORPUS_MANIFEST.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    datasets = manifest.get("datasets", []) or []
    for item in datasets:
        item.setdefault("file", "")
        item.setdefault("category", "uncategorized")
        item.setdefault("expected_scientific_behavior", "")
        item.setdefault("observation_metadata_summary", {})
    manifest["available"] = True
    manifest["corpus_root"] = str(CORPUS_ROOT)
    manifest["dataset_count"] = len(datasets)
    return manifest


def validation_corpus_summary() -> dict[str, Any]:
    manifest = load_validation_corpus_manifest()
    datasets = manifest.get("datasets", []) or []
    counts = pd.Series([str(item.get("category", "uncategorized")) for item in datasets]).value_counts().sort_index()
    summary_rows = []
    for item in datasets:
        summary_rows.append(
            {
                "dataset_id": item.get("id", ""),
                "category": item.get("category", "uncategorized"),
                "file": item.get("file", ""),
                "expected_behavior": item.get("expected_scientific_behavior", ""),
                "observation_metadata_summary": json.dumps(item.get("observation_metadata_summary", {}), sort_keys=True),
            }
        )
    return {
        "available": bool(manifest.get("available")),
        "corpus_root": manifest.get("corpus_root", str(CORPUS_ROOT)),
        "dataset_count": int(manifest.get("dataset_count", 0)),
        "category_counts": counts.to_dict(),
        "datasets": pd.DataFrame(summary_rows),
        "manifest": manifest,
        "warnings": manifest.get("warnings", []),
        "errors": manifest.get("errors", []),
    }


def run_validation_corpus_regression(sample_limit: int | None = None) -> dict[str, Any]:
    manifest = load_validation_corpus_manifest()
    datasets = manifest.get("datasets", []) or []
    if sample_limit is not None:
        datasets = datasets[: int(sample_limit)]

    rows: list[dict[str, Any]] = []
    for item in datasets:
        file_path = CORPUS_ROOT / str(item.get("file", ""))
        if not file_path.exists():
            rows.append({"dataset_id": item.get("id", ""), "category": item.get("category", ""), "status": "missing_file", "file": str(file_path), "warnings": ["file not found"], "errors": ["missing file"], "reproducibility_signature": None})
            continue

        session = reconstruct_observation_sessions([str(file_path)], preferred_dataset_path=item.get("preferred_dataset_path") or None, target_freq_bins=int(item.get("target_freq_bins", 128)))
        observatory = build_observatory_session_intelligence(session.get("records", []))
        row = _regression_row(item, file_path, session, observatory)
        rows.append(row)

    frame = pd.DataFrame(rows)
    signature = _stable_hash(rows)
    return {
        "available": True,
        "corpus_summary": validation_corpus_summary(),
        "results": frame,
        "reproducibility_signature": signature,
        "warnings": sorted({warning for row in rows for warning in row.get("warnings", [])}),
        "errors": sorted({error for row in rows for error in row.get("errors", [])}),
    }


def benchmark_observatory_pipeline(
    file_paths: Sequence[str | Path],
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    file_paths = [Path(path) for path in file_paths]
    if not file_paths:
        return {"available": False, "reason": "no files provided", "warnings": ["no files"], "errors": []}

    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]

    t0 = time.perf_counter()
    session = reconstruct_observation_sessions([str(path) for path in file_paths], preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins)
    t1 = time.perf_counter()
    observatory = build_observatory_session_intelligence(session.get("records", []))
    t2 = time.perf_counter()
    uv_tracks = cached_uv_track_visualization(session.get("records", []))
    t3 = time.perf_counter()
    closure = cached_closure_triangle_explorer(session.get("records", []))
    t4 = time.perf_counter()
    calibration = cached_calibration_stability_maps(session.get("records", []))
    t5 = time.perf_counter()
    rfi = cached_persistent_rfi_heatmaps(session.get("records", []))
    t6 = time.perf_counter()
    export_paths = export_session_intelligence(observatory, export_dir or (CORPUS_ROOT / "benchmarks"), stem="observatory_benchmark")
    t7 = time.perf_counter()
    end_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    timings = {
        "session_reconstruction_seconds": float(t1 - t0),
        "observatory_summary_seconds": float(t2 - t1),
        "uv_track_generation_seconds": float(t3 - t2),
        "closure_diagnostics_seconds": float(t4 - t3),
        "calibration_summary_seconds": float(t5 - t4),
        "persistent_rfi_summary_seconds": float(t6 - t5),
        "export_generation_seconds": float(t7 - t6),
        "waterfall_render_seconds": float(_waterfall_render_seconds(session.get("combined_waterfall"))),
        "memory_growth_bytes": int(end_memory - start_memory),
        "peak_memory_bytes": int(peak_memory),
    }
    benchmark_rows = [{"metric": key, "value": value} for key, value in timings.items()]
    benchmark_signature = _stable_hash({"files": [str(path) for path in file_paths], "timings": benchmark_rows, "observatory_signature": observatory.get("export_signature_summary", {}).get("signature")})
    comparison_export = _export_benchmark_summary(benchmark_rows, export_paths, export_dir or (CORPUS_ROOT / "benchmarks"), signature=benchmark_signature)
    return {
        "available": True,
        "timings": pd.DataFrame(benchmark_rows),
        "reproducibility_signature": benchmark_signature,
        "comparison_export": comparison_export,
        "session_export": export_paths,
        "observatory_signature": observatory.get("export_signature_summary", {}).get("signature"),
        "warnings": session.get("warnings", []) + observatory.get("warnings", []) if isinstance(observatory, dict) else session.get("warnings", []),
        "errors": session.get("errors", []) + observatory.get("errors", []) if isinstance(observatory, dict) else session.get("errors", []),
    }


def hirax_instrument_profile() -> dict[str, Any]:
    return {
        "available": True,
        "name": "hirax",
        "axis_naming_conventions": {
            "time": ["time", "timestamp", "utc", "mjd", "jd"],
            "frequency": ["frequency", "freq", "channel", "channels"],
            "baseline": ["baseline", "baselines", "corr", "correlation"],
            "polarization": ["polarization", "pol", "component_labels"],
        },
        "baseline_conventions": {
            "preferred": "explicit antenna-pair baseline labels or ordered baseline coordinate arrays",
            "never_infer": "never fabricate baselines when metadata is absent or contradictory",
        },
        "uv_metadata_expectations": {
            "preferred_fields": ["uv_coordinates", "uvw", "baseline_pairs", "antenna_positions"],
            "validation_note": "guides validation only; explicit metadata always wins",
        },
        "polarization_expectations": {
            "preferred_labels": ["xx", "xy", "yx", "yy"],
            "validation_note": "pol labels must match the data rank; inconsistencies trigger warnings",
        },
        "timestamp_conventions": {
            "preferred_fields": ["observation_time", "start_time", "timestamp", "utc_start"],
            "fallback": "file modification time when necessary",
        },
        "calibration_metadata_expectations": {
            "preferred_fields": ["gain", "calibration", "bandpass", "solutions"],
            "validation_note": "used for validation summaries, not for hidden repair",
        },
        "validation_rules": [
            "prefer explicit metadata over heuristics",
            "isolate malformed datasets safely",
            "warn on ambiguous geometry",
            "preserve observatory realism",
        ],
    }


def compare_observing_sessions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_bundle = _normalize_session_bundle(left)
    right_bundle = _normalize_session_bundle(right)
    left_obs = left_bundle["observatory"]
    right_obs = right_bundle["observatory"]

    left_metrics = _extract_session_metrics(left_bundle)
    right_metrics = _extract_session_metrics(right_bundle)
    comparison = {
        "uv_completeness_delta": _delta_metric(left_metrics, right_metrics, "uv_completeness"),
        "calibration_stability_delta": _delta_metric(left_metrics, right_metrics, "calibration_stability"),
        "closure_stability_delta": _delta_metric(left_metrics, right_metrics, "closure_consistency"),
        "rfi_occupancy_delta": _delta_metric(left_metrics, right_metrics, "rfi_contamination", invert=True),
        "power_spectrum_consistency": _power_spectrum_consistency(left_bundle.get("session", {}), right_bundle.get("session", {})),
        "confidence_delta": _confidence_delta(left_obs, right_obs),
        "observational_consistency_summary": {
            "left_warnings": len(left_obs.get("trust_checks", {}).get("warnings", []) if isinstance(left_obs.get("trust_checks"), dict) else []),
            "right_warnings": len(right_obs.get("trust_checks", {}).get("warnings", []) if isinstance(right_obs.get("trust_checks"), dict) else []),
        },
    }
    return {
        "available": True,
        "left_signature": left_bundle["signature"],
        "right_signature": right_bundle["signature"],
        "comparison": comparison,
        "reproducibility_signature": _stable_hash(comparison),
        "warnings": [],
        "errors": [],
    }


def build_observatory_audit_trail(session_result: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    records = list(session_result.get("records", []))
    events = [
        {
            "event": "dataset_ingestion",
            "count": len(records),
            "files": [getattr(rec, "file_name", "") for rec in records],
        },
        {
            "event": "session_reconstruction",
            "ordered": bool(session_result.get("consistency_report", {}).get("ordered_times", True)),
            "session_count": session_result.get("consistency_report", {}).get("session_count", len(records)),
        },
        {
            "event": "calibration_actions",
            "available": bool(observatory.get("calibration_maps", {}).get("available")),
            "summary": observatory.get("calibration_maps", {}).get("session_quality_summary", {}),
        },
        {
            "event": "masking_operations",
            "available": bool(observatory.get("persistent_rfi", {}).get("available")),
            "persistent_fraction": observatory.get("persistent_rfi", {}).get("frequency_occupancy_summary", {}).get("persistent_fraction", 0.0),
        },
        {
            "event": "session_fusion",
            "available": bool(session_result.get("weighted_fusion", {}).get("available")),
            "reason": session_result.get("weighted_fusion", {}).get("reason", ""),
        },
        {
            "event": "exports_generated",
            "available": bool(observatory.get("session_export_preview")),
            "version": observatory.get("version", ""),
        },
        {
            "event": "diagnostics_generated",
            "available": True,
            "diagnostic_keys": sorted([key for key in observatory.keys() if key in {"diagnostics", "resilience", "reproducibility", "trust_checks", "export_integrity", "stress"}]),
        },
    ]
    signature = _stable_hash(events)
    return {
        "available": True,
        "events": pd.DataFrame(events),
        "signature": signature,
        "warnings": [],
        "errors": [],
    }


def export_audit_trail(audit_trail: dict[str, Any], output_dir: str | Path, stem: str = "udip_audit_trail") -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    events = audit_trail.get("events")
    if isinstance(events, pd.DataFrame):
        events.to_csv(csv_path, index=False)
        payload = events.to_dict(orient="records")
    else:
        pd.DataFrame(events or []).to_csv(csv_path, index=False)
        payload = events or []
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"signature": audit_trail.get("signature"), "events": payload}, fh, indent=2, default=str)
    return {"csv": str(csv_path), "json": str(json_path)}


def run_scientific_lockdown_audit() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    active_files = [repo_root / "app.py", repo_root / "pages", repo_root / "modules"]
    active_risks: list[str] = []
    inactive_legacy_modules: list[str] = []
    disallowed_terms = ["placeholder widget", "fake dashboard", "enterprise", "copilot", "ai operating system", "orchestration layer"]
    for candidate in [repo_root / "app.py", repo_root / "pages" / "hirax_inspector.py", repo_root / "modules" / "observatory_intelligence.py", repo_root / "modules" / "observatory_operations.py"]:
        if candidate.is_file():
            text = candidate.read_text(errors="ignore").lower()
            for term in disallowed_terms:
                if term in text:
                    active_risks.append(f"{candidate.name}: {term}")
    for legacy in ["workflow_copilot.py", "workflow_handoff.py"]:
        if (repo_root / "modules" / legacy).exists():
            inactive_legacy_modules.append(legacy)
    duplicate_pipelines = []
    if (repo_root / "modules" / "observatory_intelligence.py").exists() and (repo_root / "modules" / "observation_session.py").exists():
        duplicate_pipelines.append("observatory diagnostics and session reconstruction are both present as separate layers, as intended")
    locked_down = not active_risks
    return {
        "available": True,
        "locked_down": locked_down,
        "active_risks": active_risks,
        "inactive_legacy_modules": inactive_legacy_modules,
        "duplicate_pipeline_notes": duplicate_pipelines,
        "warnings": active_risks,
        "errors": [],
    }


def _regression_row(item: dict[str, Any], file_path: Path, session: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    consistency = session.get("consistency_report", {}) if isinstance(session, dict) else {}
    confidence = observatory.get("confidence", {}) if isinstance(observatory, dict) else {}
    return {
        "dataset_id": item.get("id", ""),
        "category": item.get("category", ""),
        "file": str(file_path),
        "reconstruction_ok": bool(session.get("records")),
        "consistent": bool(consistency.get("is_consistent", False)),
        "warnings": list(session.get("warnings", [])) + list(observatory.get("warnings", [])),
        "errors": list(session.get("errors", [])) + list(observatory.get("errors", [])),
        "uv_available": bool(observatory.get("uv_aware_fusion", {}).get("available")),
        "closure_available": bool(observatory.get("closure_explorer", {}).get("available")),
        "calibration_available": bool(observatory.get("calibration_maps", {}).get("available")),
        "rfi_available": bool(observatory.get("persistent_rfi", {}).get("available")),
        "confidence": confidence.get("overall_confidence"),
        "reproducibility_signature": _stable_hash({"session": session.get("consistency_report", {}), "confidence": confidence, "file": str(file_path)}),
        "expected_scientific_behavior": item.get("expected_scientific_behavior", ""),
    }


def _normalize_session_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    if "records" in bundle and "consistency_report" in bundle:
        observatory = bundle if "confidence" in bundle else build_observatory_session_intelligence(bundle.get("records", []))
        return {
            "session": bundle,
            "observatory": observatory,
            "signature": _stable_hash({"session": bundle.get("consistency_report", {}), "observatory": observatory.get("export_signature_summary", {})}),
        }
    if "session_result" in bundle and "observatory" in bundle:
        return {
            "session": bundle.get("session_result", {}),
            "observatory": bundle.get("observatory", {}),
            "signature": _stable_hash(bundle),
        }
    return {"session": {}, "observatory": {}, "signature": _stable_hash(bundle)}


def _extract_session_metrics(bundle: dict[str, Any]) -> dict[str, float]:
    observatory = bundle.get("observatory", {}) if isinstance(bundle, dict) else {}
    metrics: dict[str, float] = {}
    uv = observatory.get("uv_aware_fusion", {}) if isinstance(observatory, dict) else {}
    earth = observatory.get("earth_rotation", {}) if isinstance(observatory, dict) else {}
    calibration = observatory.get("calibration_maps", {}) if isinstance(observatory, dict) else {}
    closure = observatory.get("closure_explorer", {}) if isinstance(observatory, dict) else {}
    rfi = observatory.get("persistent_rfi", {}) if isinstance(observatory, dict) else {}
    metrics["uv_completeness"] = float(earth.get("uv_completeness_metrics", {}).get("beam_quality_score", 0.0) if isinstance(earth, dict) else 0.0)
    metrics["calibration_stability"] = float(1.0 - calibration.get("session_quality_summary", {}).get("mean_gain_amplitude_drift", 0.0) if isinstance(calibration, dict) else 0.0)
    metrics["closure_consistency"] = float(closure.get("closure_consistency_score", 0.0) if isinstance(closure, dict) else 0.0)
    metrics["rfi_contamination"] = float(rfi.get("frequency_occupancy_summary", {}).get("persistent_fraction", 0.0) if isinstance(rfi, dict) else 0.0)
    metrics["confidence"] = float(observatory.get("confidence", {}).get("overall_confidence") or 0.0) if isinstance(observatory.get("confidence"), dict) else 0.0
    return metrics


def _delta_metric(left: dict[str, float], right: dict[str, float], key: str, invert: bool = False) -> float:
    lv = float(left.get(key, 0.0))
    rv = float(right.get(key, 0.0))
    return float((lv - rv) if not invert else (rv - lv))


def _confidence_delta(left_obs: dict[str, Any], right_obs: dict[str, Any]) -> float:
    left_value = left_obs.get("confidence", {}).get("overall_confidence") if isinstance(left_obs.get("confidence"), dict) else None
    right_value = right_obs.get("confidence", {}).get("overall_confidence") if isinstance(right_obs.get("confidence"), dict) else None
    return float((left_value or 0.0) - (right_value or 0.0))


def _power_spectrum_consistency(left_session: dict[str, Any], right_session: dict[str, Any]) -> float:
    left_spec = _spectrum_signature(left_session)
    right_spec = _spectrum_signature(right_session)
    if left_spec.size == 0 or right_spec.size == 0:
        return 0.0
    n = min(left_spec.size, right_spec.size)
    left_spec = left_spec[:n]
    right_spec = right_spec[:n]
    if np.std(left_spec) == 0 or np.std(right_spec) == 0:
        return 0.0
    return float(np.corrcoef(left_spec, right_spec)[0, 1])


def _spectrum_signature(session: dict[str, Any]) -> np.ndarray:
    waterfall = np.asarray(session.get("combined_waterfall", np.zeros((0, 0))), dtype=float)
    if waterfall.size == 0:
        return np.array([])
    profile = np.nanmean(waterfall, axis=0)
    spectrum = np.abs(np.fft.rfft(profile))
    return spectrum.astype(float, copy=False)


def _waterfall_render_seconds(combined_waterfall: Any) -> float:
    arr = np.asarray(combined_waterfall, dtype=float)
    if arr.size == 0:
        return 0.0
    start = time.perf_counter()
    _ = np.log10(np.abs(arr) + 1e-10)
    return float(time.perf_counter() - start)


def _export_benchmark_summary(rows: list[dict[str, Any]], export_paths: dict[str, str], output_dir: str | Path, signature: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "observatory_benchmark_summary.csv"
    json_path = output_dir / "observatory_benchmark_summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"signature": signature, "metrics": rows, "session_export_paths": export_paths}, fh, indent=2)
    return {"csv": str(csv_path), "json": str(json_path)}


def _stable_hash(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()
