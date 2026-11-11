from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

from modules.observation_session import reconstruct_observation_sessions
from modules.observatory_intelligence import build_observatory_session_intelligence
from modules.observatory_operations import (
    benchmark_observatory_pipeline,
    build_observatory_audit_trail,
    compare_observing_sessions,
    load_validation_corpus_manifest,
    run_scientific_lockdown_audit,
    validation_corpus_summary,
)

SCHEMA_VERSION = "2.1"
DEFAULT_AVAILABLE_RAM_GB = 4.3
MAX_RAM_FRACTION = 0.55
SESSION_CACHE_DIR = Path("/home/workspace/udip_v3/.observatory_cache")
QUARANTINE_MANIFEST = SESSION_CACHE_DIR / "quarantine_manifest.json"
SESSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def streaming_reconstruct_observation_sessions(
    file_paths: Sequence[str | Path],
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
    batch_size: int | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    paths = [Path(path) for path in file_paths]
    if not paths:
        return {
            "available": False,
            "reason": "no files provided",
            "schema_version": SCHEMA_VERSION,
            "streaming_mode": True,
            "batch_count": 0,
            "batch_summaries": [],
            "warnings": ["no files"],
            "errors": [],
        }

    budget = estimate_memory_budget(paths)
    if batch_size is None:
        batch_size = int(budget.get("safe_batch_size_hint", 4))
    batch_size = max(int(batch_size), 1)
    cache_key = session_cache_key(paths, preferred_dataset_path, target_freq_bins, batch_size)
    if use_cache:
        cached = load_session_cache(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    batch_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []
    quarantined: list[dict[str, Any]] = []

    for batch_index in range(0, len(paths), batch_size):
        batch_paths = paths[batch_index : batch_index + batch_size]
        batch_session = reconstruct_observation_sessions(
            [str(path) for path in batch_paths],
            preferred_dataset_path=preferred_dataset_path,
            target_freq_bins=target_freq_bins,
        )
        records = list(batch_session.get("records", []))
        quarantine_rows = _quarantine_rows(batch_paths, batch_session)
        quarantined.extend(quarantine_rows)
        batch_summaries.append(
            {
                "batch_index": batch_index // batch_size,
                "file_count": len(batch_paths),
                "record_count": len(records),
                "warnings": list(batch_session.get("warnings", [])),
                "errors": list(batch_session.get("errors", [])),
                "quarantine_count": len(quarantine_rows),
                "signature": _stable_hash(
                    {
                        "paths": [str(path) for path in batch_paths],
                        "records": [getattr(rec, "summary", lambda: {})() for rec in records],
                        "warnings": batch_session.get("warnings", []),
                        "errors": batch_session.get("errors", []),
                    }
                ),
            }
        )
        warnings.extend(batch_session.get("warnings", []))
        errors.extend(batch_session.get("errors", []))

    combined_session = reconstruct_observation_sessions(
        [str(path) for path in paths],
        preferred_dataset_path=preferred_dataset_path,
        target_freq_bins=target_freq_bins,
    )
    observatory = build_observatory_session_intelligence(combined_session.get("records", []))
    result = {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "streaming_mode": True,
        "cache_hit": False,
        "cache_key": cache_key,
        "batch_count": len(batch_summaries),
        "batch_size": batch_size,
        "batch_summaries": batch_summaries,
        "session_result": combined_session,
        "observatory": observatory,
        "memory_budget": budget,
        "quarantine_manifest": update_quarantine_manifest(quarantined, cache_key),
        "warnings": warnings + list(combined_session.get("warnings", [])) + list(observatory.get("warnings", [])),
        "errors": errors + list(combined_session.get("errors", [])) + list(observatory.get("errors", [])),
        "stream_signature": _stable_hash(
            {
                "paths": [str(path) for path in paths],
                "batch_size": batch_size,
                "combined": combined_session.get("consistency_report", {}),
                "observatory": observatory.get("export_signature_summary", {}),
            }
        ),
    }
    if use_cache:
        save_session_cache(cache_key, result)
        evict_session_cache(max_entries=12)
    return result


def estimate_memory_budget(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    paths = [Path(path) for path in file_paths]
    file_sizes = []
    for path in paths:
        try:
            file_sizes.append(int(path.stat().st_size))
        except Exception:
            file_sizes.append(0)
    total_bytes = int(sum(file_sizes))
    estimated_ram_bytes = int(total_bytes * 6 + len(paths) * 5_000_000)
    pressure = _clip01(total_bytes / max(estimated_ram_bytes, 1))
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "file_count": len(paths),
        "file_sizes_bytes": file_sizes,
        "total_input_bytes": total_bytes,
        "estimated_ram_bytes": estimated_ram_bytes,
        "estimated_ram_mb": round(estimated_ram_bytes / 1_000_000.0, 2),
        "memory_pressure": pressure,
        "safe_batch_size_hint": max(1, min(16, 8 if total_bytes < 200_000_000 else 4)),
        "warnings": [],
        "errors": [],
    }


def hardware_profile_estimate(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    budget = estimate_memory_budget(file_paths)
    file_count = int(budget.get("file_count", 0))
    total_bytes = int(budget.get("total_input_bytes", 0))
    ram_mb = float(budget.get("estimated_ram_mb", 0.0))
    cpu_pressure = _clip01((file_count / 100.0) + (total_bytes / 2_000_000_000.0))
    throughput_hint = max(1.0, 100.0 / max(file_count, 1))
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "file_count": file_count,
        "estimated_ram_mb": ram_mb,
        "cpu_pressure": cpu_pressure,
        "throughput_hint_sessions_per_minute": throughput_hint,
        "scaling_profile": {
            "small": "single-pass reconstruction",
            "medium": "batch streaming reconstruction",
            "large": "batch streaming reconstruction with quarantine summaries",
        },
        "warnings": [],
        "errors": [],
    }


def session_consensus_engine(session_result: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    confidence = observatory.get("confidence", {}) if isinstance(observatory, dict) else {}
    trust = observatory.get("trust_checks", {}) if isinstance(observatory, dict) else {}
    consistency = session_result.get("consistency_report", {}) if isinstance(session_result, dict) else {}
    uv = observatory.get("uv_aware_fusion", {}) if isinstance(observatory, dict) else {}
    closure = observatory.get("closure_timeline", {}) if isinstance(observatory, dict) else {}
    calibration = observatory.get("calibration_maps", {}) if isinstance(observatory, dict) else {}
    rfi = observatory.get("persistent_rfi", {}) if isinstance(observatory, dict) else {}

    consensus = {
        "calibration_quality": _clip01(float(confidence.get("confidence_components", {}).get("calibration_stability", 0.0))),
        "uv_quality": _clip01(float(confidence.get("confidence_components", {}).get("uv_completeness", 0.0))),
        "rfi_quality": _clip01(float(confidence.get("confidence_components", {}).get("rfi_contamination", 0.0))),
        "cosmology_readiness": _clip01(float(confidence.get("cosmology_readiness_estimate", 0.0) or 0.0)),
        "imaging_readiness": _clip01(float(confidence.get("imaging_reliability_estimate", 0.0) or 0.0)),
    }
    deltas = {
        "warning_count_delta": int(len(trust.get("warnings", [])) - len(consistency.get("warnings", []))),
        "error_count_delta": int(len(trust.get("errors", [])) - len(consistency.get("errors", []))),
        "confidence_explanation": _confidence_explanation(observatory),
    }
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "consensus": consensus,
        "consensus_score": float(np.mean(list(consensus.values()))) if consensus else 0.0,
        "confidence_deltas": deltas,
        "source_signatures": {
            "session": _stable_hash(consistency),
            "uv": _stable_hash(uv),
            "closure": _stable_hash(closure),
            "calibration": _stable_hash(calibration),
            "rfi": _stable_hash(rfi),
        },
        "warnings": list(trust.get("warnings", [])),
        "errors": list(trust.get("errors", [])),
    }


def observation_health_timelines(observatory: dict[str, Any]) -> dict[str, Any]:
    timelines = {
        "calibration_drift_timeline": observatory.get("calibration_drift", {}).get("drift_timeline"),
        "rfi_contamination_timeline": observatory.get("rfi_evolution", {}).get("session_rfi_summary"),
        "uv_completeness_timeline": observatory.get("uv_aware_fusion", {}).get("uv_completeness_metrics"),
        "closure_stability_timeline": observatory.get("closure_timeline", {}).get("closure_timeline"),
        "imaging_readiness_timeline": observatory.get("confidence", {}).get("confidence_components", {}),
    }
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "timelines": timelines,
        "summary": {
            "calibration_drift": _timeline_summary(timelines["calibration_drift_timeline"]),
            "rfi_contamination": _timeline_summary(timelines["rfi_contamination_timeline"]),
            "uv_completeness": _timeline_summary(timelines["uv_completeness_timeline"]),
            "closure_stability": _timeline_summary(timelines["closure_stability_timeline"]),
        },
        "warnings": [],
        "errors": [],
    }


def scientific_alert_report(session_result: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []
    trust = observatory.get("trust_checks", {}) if isinstance(observatory, dict) else {}
    if trust.get("warnings"):
        for warning in trust.get("warnings", []):
            alerts.append({"severity": "warning", "kind": "consistency", "message": warning})

    calibration = observatory.get("calibration_drift", {}).get("drift_timeline") if isinstance(observatory, dict) else None
    if isinstance(calibration, pd.DataFrame) and not calibration.empty:
        if float(calibration.get("gain_amplitude_drift", pd.Series([0.0])).max()) > 0.5:
            alerts.append({"severity": "warning", "kind": "calibration_instability", "message": "sudden calibration instability detected"})
        if any((calibration.get("phase_drift", pd.Series([0.0])) > 1.0).tolist()):
            alerts.append({"severity": "warning", "kind": "phase_drift", "message": "abnormal phase drift detected"})

    rfi = observatory.get("persistent_rfi", {}).get("frequency_occupancy_summary", {}) if isinstance(observatory, dict) else {}
    if float(rfi.get("persistent_fraction", 0.0)) > 0.5:
        alerts.append({"severity": "warning", "kind": "rfi", "message": "persistent contaminated channels detected"})

    uv = observatory.get("uv_aware_fusion", {}).get("uv_coverage", {}) if isinstance(observatory, dict) else {}
    if float(uv.get("density_mean", 0.0)) < 0.05 and session_result.get("records"):
        alerts.append({"severity": "warning", "kind": "uv_sparsity", "message": "uv sparsity anomaly detected"})

    closure = observatory.get("closure_timeline", {}).get("closure_timeline") if isinstance(observatory, dict) else None
    if isinstance(closure, pd.DataFrame) and not closure.empty:
        if float(closure.get("rms_closure_phase", pd.Series([0.0])).max()) > 1.0:
            alerts.append({"severity": "warning", "kind": "closure", "message": "closure inconsistency spike detected"})

    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "alerts": alerts,
        "alert_count": len(alerts),
        "warnings": [alert["message"] for alert in alerts if alert["severity"] == "warning"],
        "errors": [],
    }


def strict_scientific_validation(session_result: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    consistency = session_result.get("consistency_report", {}) if isinstance(session_result, dict) else {}
    if not session_result.get("records"):
        reasons.append("no valid records")
    if not consistency.get("is_consistent", False):
        reasons.append("session consistency failed")
    if observatory.get("trust_checks", {}).get("warnings"):
        reasons.append("observatory trust warnings present")
    if observatory.get("confidence", {}).get("overall_confidence") is None:
        reasons.append("confidence could not be computed")
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "is_valid": len(reasons) == 0,
        "reasons": reasons,
        "warnings": reasons,
        "errors": [] if len(reasons) == 0 else reasons,
    }


def deployment_readiness_report(
    file_paths: Sequence[str | Path],
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
    batch_size: int = 8,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    streaming = streaming_reconstruct_observation_sessions(
        file_paths,
        preferred_dataset_path=preferred_dataset_path,
        target_freq_bins=target_freq_bins,
        batch_size=batch_size,
    )
    session_result = streaming["session_result"]
    observatory = streaming["observatory"]
    memory_profile = estimate_memory_budget(file_paths)
    hardware_profile = hardware_profile_estimate(file_paths)
    consensus = session_consensus_engine(session_result, observatory)
    health = observation_health_timelines(observatory)
    alerts = scientific_alert_report(session_result, observatory)
    strict = strict_scientific_validation(session_result, observatory)
    audit = build_observatory_audit_trail(session_result, observatory)
    lockdown = run_scientific_lockdown_audit()

    benchmark = benchmark_observatory_pipeline(
        [Path(path) for path in file_paths[: max(1, min(len(file_paths), batch_size))]],
        preferred_dataset_path=preferred_dataset_path,
        target_freq_bins=target_freq_bins,
        export_dir=export_dir or Path(os.getcwd()),
    )

    export_dir = Path(export_dir or Path.cwd() / "deployment_reports")
    export_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "streaming": streaming,
        "memory_profile": memory_profile,
        "hardware_profile": hardware_profile,
        "consensus": consensus,
        "health": health,
        "alerts": alerts,
        "strict_validation": strict,
        "audit": {"signature": audit.get("signature"), "event_count": len(audit.get("events", [])) if not isinstance(audit.get("events"), pd.DataFrame) else int(len(audit["events"]))},
        "lockdown": lockdown,
        "benchmark": {"signature": benchmark.get("reproducibility_signature"), "timings": _json_safe(benchmark.get("timings"))},
        "corpus_summary": validation_corpus_summary(),
    }
    csv_path = export_dir / f"deployment_readiness_v{SCHEMA_VERSION}.csv"
    json_path = export_dir / f"deployment_readiness_v{SCHEMA_VERSION}.json"
    pd.DataFrame(_flatten_report(report_payload)).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report_payload), fh, indent=2, default=str)

    bottlenecks = _bottleneck_summary(memory_profile, benchmark, observatory)
    reliability = _reliability_summary(consensus, strict, alerts, lockdown)
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "deployment_readiness": _clip01(reliability),
        "streaming": streaming,
        "memory_profile": memory_profile,
        "hardware_profile": hardware_profile,
        "session_consensus": consensus,
        "health_timelines": health,
        "scientific_alerts": alerts,
        "strict_validation": strict,
        "audit": audit,
        "lockdown": lockdown,
        "benchmark": benchmark,
        "bottleneck_summary": bottlenecks,
        "reproducibility_signature": _stable_hash(report_payload),
        "comparison_export": {"csv": str(csv_path), "json": str(json_path)},
        "warnings": streaming.get("warnings", []) + consensus.get("warnings", []) + alerts.get("warnings", []) + strict.get("warnings", []),
        "errors": streaming.get("errors", []) + consensus.get("errors", []) + alerts.get("errors", []) + strict.get("errors", []),
    }


def daily_observation_comparison(
    left_file_paths: Sequence[str | Path],
    right_file_paths: Sequence[str | Path],
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    left = streaming_reconstruct_observation_sessions(left_file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    right = streaming_reconstruct_observation_sessions(right_file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    comparison = compare_observing_sessions(
        {"session_result": left["session_result"], "observatory": left["observatory"]},
        {"session_result": right["session_result"], "observatory": right["observatory"]},
    )
    comparison["schema_version"] = SCHEMA_VERSION
    comparison["left_consensus"] = session_consensus_engine(left["session_result"], left["observatory"])
    comparison["right_consensus"] = session_consensus_engine(right["session_result"], right["observatory"])
    comparison["observational_consistency_summary"] = {
        "left_files": len(left_file_paths),
        "right_files": len(right_file_paths),
        "left_alerts": scientific_alert_report(left["session_result"], left["observatory"]),
        "right_alerts": scientific_alert_report(right["session_result"], right["observatory"]),
    }
    return comparison


def benchmark_reproducibility_check(file_paths: Sequence[str | Path], preferred_dataset_path: str | None = None, target_freq_bins: int | None = None, batch_size: int = 8) -> dict[str, Any]:
    first = deployment_readiness_report(file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    second = deployment_readiness_report(file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "first_signature": first["streaming"]["stream_signature"],
        "second_signature": second["streaming"]["stream_signature"],
        "identical": first["streaming"]["stream_signature"] == second["streaming"]["stream_signature"],
        "benchmark_delta": _benchmark_delta(first.get("benchmark", {}), second.get("benchmark", {})),
        "warnings": [],
        "errors": [],
    }


def export_reproducibility_check(file_paths: Sequence[str | Path], preferred_dataset_path: str | None = None, target_freq_bins: int | None = None, batch_size: int = 8) -> dict[str, Any]:
    first = deployment_readiness_report(file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    second = deployment_readiness_report(file_paths, preferred_dataset_path=preferred_dataset_path, target_freq_bins=target_freq_bins, batch_size=batch_size)
    first_payload = Path(first["comparison_export"]["json"]).read_text(encoding="utf-8") if Path(first["comparison_export"]["json"]).exists() else ""
    second_payload = Path(second["comparison_export"]["json"]).read_text(encoding="utf-8") if Path(second["comparison_export"]["json"]).exists() else ""
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "exports_match": first_payload == second_payload,
        "first_exports": first["comparison_export"],
        "second_exports": second["comparison_export"],
        "warnings": [],
        "errors": [],
    }


def lockdown_audit_report() -> dict[str, Any]:
    audit = run_scientific_lockdown_audit()
    audit["schema_version"] = SCHEMA_VERSION
    return audit


def session_cache_key(file_paths: Sequence[str | Path], preferred_dataset_path: str | None = None, target_freq_bins: int | None = None, batch_size: int | None = None) -> str:
    fields = []
    for path in file_paths:
        p = Path(path)
        try:
            stat = p.stat()
            fields.append({"path": str(p), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        except Exception:
            fields.append({"path": str(p), "missing": True})
    return _stable_hash({"files": fields, "preferred_dataset_path": preferred_dataset_path, "target_freq_bins": target_freq_bins, "batch_size": batch_size, "schema": SCHEMA_VERSION})


def _timeline_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {"available": True, "rows": int(len(value)), "columns": list(value.columns)}
    if isinstance(value, dict):
        return {"available": True, "keys": sorted(value.keys())}
    if isinstance(value, np.ndarray):
        return {"available": True, "shape": list(value.shape), "mean": float(np.nanmean(value)) if value.size else 0.0}
    return {"available": bool(value is not None), "type": type(value).__name__}


def _confidence_explanation(observatory: dict[str, Any]) -> str:
    confidence = observatory.get("confidence", {}) if isinstance(observatory, dict) else {}
    components = confidence.get("confidence_components", {}) if isinstance(confidence, dict) else {}
    if not components:
        return "insufficient measurable diagnostics"
    worst = sorted(components.items(), key=lambda item: item[1])[0]
    return f"lowest component is {worst[0]}={worst[1]:.3f}"


def _flatten_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, value in report.items():
        if isinstance(value, dict):
            for key, inner in value.items():
                rows.append({"section": section, "metric": key, "value": _json_safe(inner)})
        else:
            rows.append({"section": section, "metric": section, "value": _json_safe(value)})
    return rows


def _bottleneck_summary(memory_profile: dict[str, Any], benchmark: dict[str, Any], observatory: dict[str, Any]) -> dict[str, Any]:
    timings = benchmark.get("timings")
    if isinstance(timings, pd.DataFrame) and not timings.empty:
        top = timings.sort_values("value", ascending=False).head(3).to_dict(orient="records")
    else:
        top = []
    return {
        "top_timing_metrics": top,
        "memory_pressure": memory_profile.get("memory_pressure", 0.0),
        "session_count": int(observatory.get("diagnostics", {}).get("session_count", 0)) if isinstance(observatory, dict) else 0,
    }


def _reliability_summary(consensus: dict[str, Any], strict: dict[str, Any], alerts: dict[str, Any], lockdown: dict[str, Any]) -> float:
    score = float(consensus.get("consensus_score", 0.0) or 0.0)
    if strict.get("is_valid"):
        score += 0.2
    score -= min(0.2, 0.02 * float(alerts.get("alert_count", 0)))
    if lockdown.get("locked_down"):
        score += 0.1
    return float(np.clip(score, 0.0, 1.0))


def _benchmark_delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    def _seconds(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty and "value" in value.columns:
            return float(value.loc[value["metric"] == "session_reconstruction_seconds", "value"].iloc[0]) if any(value["metric"] == "session_reconstruction_seconds") else 0.0
        return 0.0

    return {
        "session_reconstruction_seconds_delta": _seconds(second, "timings") - _seconds(first, "timings"),
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, default=str).encode("utf-8")).hexdigest()


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
    return value


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def load_session_cache(cache_key: str) -> dict[str, Any] | None:
    path = SESSION_CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def save_session_cache(cache_key: str, result: dict[str, Any]) -> str:
    path = SESSION_CACHE_DIR / f"{cache_key}.json"
    payload = {
        "available": result.get("available", False),
        "schema_version": result.get("schema_version", SCHEMA_VERSION),
        "streaming_mode": result.get("streaming_mode", True),
        "cache_key": cache_key,
        "batch_count": result.get("batch_count", 0),
        "batch_size": result.get("batch_size", 0),
        "batch_summaries": result.get("batch_summaries", []),
        "session_result": _session_cache_summary(result.get("session_result", {})),
        "observatory": _observatory_cache_summary(result.get("observatory", {})),
        "memory_budget": result.get("memory_budget", {}),
        "quarantine_manifest": result.get("quarantine_manifest", {}),
        "warnings": result.get("warnings", []),
        "errors": result.get("errors", []),
        "stream_signature": result.get("stream_signature"),
        "cache_created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, default=str)
    return str(path)


def evict_session_cache(max_entries: int = 12) -> dict[str, Any]:
    files = sorted(SESSION_CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
    protected = {QUARANTINE_MANIFEST.resolve()}
    evicted = []
    for path in files[max_entries:]:
        if path.resolve() in protected:
            continue
        try:
            path.unlink()
            evicted.append(str(path))
        except Exception:
            pass
    return {"available": True, "evicted": evicted, "remaining": max(0, len(files) - len(evicted))}


def update_quarantine_manifest(rows: Sequence[dict[str, Any]], cache_key: str | None = None) -> dict[str, Any]:
    existing: list[dict[str, Any]] = []
    if QUARANTINE_MANIFEST.exists():
        try:
            existing_payload = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
            existing = list(existing_payload.get("rows", [])) if isinstance(existing_payload, dict) else []
        except Exception:
            existing = []
    stamped = []
    for row in rows:
        clean = dict(row)
        clean["cache_key"] = cache_key
        clean["recorded_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stamped.append(clean)
    combined = existing + stamped
    payload = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(combined),
        "rows": combined[-1000:],
        "signature": _stable_hash(combined[-1000:]),
    }
    QUARANTINE_MANIFEST.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def real_hardware_baseline_benchmark(file_paths: Sequence[str | Path], sample_count: int = 4) -> dict[str, Any]:
    paths = [Path(path) for path in file_paths[: max(1, sample_count)]]
    start = time.perf_counter()
    bytes_read = 0
    dataset_count = 0
    corrupted = 0
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                def _visit(_name: str, obj: Any) -> None:
                    nonlocal dataset_count
                    if isinstance(obj, h5py.Dataset):
                        dataset_count += 1
                handle.visititems(_visit)
            bytes_read += path.stat().st_size if path.exists() else 0
        except Exception:
            corrupted += 1
    elapsed = max(time.perf_counter() - start, 1e-9)
    throughput_mb_s = (bytes_read / 1_000_000.0) / elapsed
    return {
        "available": True,
        "schema_version": SCHEMA_VERSION,
        "sample_count": len(paths),
        "bytes_read": int(bytes_read),
        "dataset_count": int(dataset_count),
        "corrupted_count": int(corrupted),
        "elapsed_seconds": float(elapsed),
        "throughput_mb_s": float(throughput_mb_s),
        "platform": platform.platform(),
        "warnings": [],
        "errors": [],
    }


def quarantine_manifest_report() -> dict[str, Any]:
    if not QUARANTINE_MANIFEST.exists():
        return {"available": True, "schema_version": SCHEMA_VERSION, "row_count": 0, "rows": [], "warnings": [], "errors": []}
    try:
        payload = json.loads(QUARANTINE_MANIFEST.read_text(encoding="utf-8"))
        payload["available"] = True
        return payload
    except Exception as exc:
        return {"available": False, "schema_version": SCHEMA_VERSION, "row_count": 0, "rows": [], "warnings": [], "errors": [str(exc)]}


def _quarantine_rows(paths: Sequence[Path], session: dict[str, Any]) -> list[dict[str, Any]]:
    records = session.get("records", [])
    valid_paths = {str(Path(getattr(record, "file_path", ""))) for record in records}
    rows = []
    for path in paths:
        reasons = []
        if str(path) not in valid_paths:
            reasons.append("no_valid_visibility_record")
        if session.get("errors"):
            reasons.extend(str(err) for err in session.get("errors", []))
        if reasons:
            rows.append({"file_path": str(path), "reasons": sorted(set(reasons)), "quarantined": True})
    return rows


def _session_cache_summary(session: dict[str, Any]) -> dict[str, Any]:
    consistency = session.get("consistency_report", {}) if isinstance(session, dict) else {}
    records = session.get("records", []) if isinstance(session, dict) else []
    return {
        "available": bool(session),
        "record_count": len(records),
        "consistency_report": consistency,
        "warnings": session.get("warnings", []) if isinstance(session, dict) else [],
        "errors": session.get("errors", []) if isinstance(session, dict) else [],
    }


def _observatory_cache_summary(observatory: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(observatory),
        "export_signature_summary": observatory.get("export_signature_summary", {}) if isinstance(observatory, dict) else {},
        "confidence": observatory.get("confidence", {}) if isinstance(observatory, dict) else {},
        "trust_checks": observatory.get("trust_checks", {}) if isinstance(observatory, dict) else {},
        "diagnostics": observatory.get("diagnostics", {}) if isinstance(observatory, dict) else {},
        "warnings": observatory.get("warnings", []) if isinstance(observatory, dict) else [],
        "errors": observatory.get("errors", []) if isinstance(observatory, dict) else [],
    }
