from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from modules.core import read_scientific_hdf5, normalize_hdf5_attrs, infer_scientific_axis_metadata
from modules.radio_interferometry import compute_visibility_statistics

TIME_HINT_KEYS = (
    "observation_time",
    "observation_timestamp",
    "start_time",
    "start_timestamp",
    "utc_start",
    "utc",
    "time",
    "timestamp",
    "mjd",
    "jd",
)

VISIBILITY_HINTS = (
    "visibility",
    "visibilities",
    "vis",
    "corr",
    "correlation",
    "coherence",
)


@dataclass
class ObservationSessionRecord:
    file_path: str
    file_name: str
    dataset_path: str
    observation_time: datetime | None
    time_order_key: float
    visibility_shape: tuple[int, ...]
    waterfall: np.ndarray
    visibility: np.ndarray
    uv_coordinates: np.ndarray | None = None
    frequency_axis: np.ndarray | None = None
    polarization_labels: tuple[str, ...] = field(default_factory=tuple)
    axis_meta: dict[str, Any] = field(default_factory=dict)
    antenna_labels: tuple[str, ...] = field(default_factory=tuple)
    antenna_positions: np.ndarray | None = None
    baseline_labels: tuple[str, ...] = field(default_factory=tuple)
    baseline_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    mean_amplitude: float = 0.0
    std_amplitude: float = 0.0
    mean_phase: float = 0.0
    phase_std: float = 0.0
    finite_fraction: float = 0.0
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "dataset_path": self.dataset_path,
            "observation_time": self.observation_time.isoformat() if self.observation_time else None,
            "time_order_key": self.time_order_key,
            "visibility_shape": list(self.visibility_shape),
            "waterfall_shape": list(self.waterfall.shape),
            "mean_amplitude": self.mean_amplitude,
            "std_amplitude": self.std_amplitude,
            "mean_phase": self.mean_phase,
            "phase_std": self.phase_std,
            "finite_fraction": self.finite_fraction,
            "weight": self.weight,
            "metadata_keys": sorted(self.metadata.keys()),
        }


def reconstruct_observation_sessions(
    file_paths: Sequence[str | Path],
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
) -> dict[str, Any]:
    """Load files, order them in time, and reconstruct a continuous session timeline."""
    records: list[ObservationSessionRecord] = []
    warnings: list[str] = []
    errors: list[str] = []

    for file_path in file_paths:
        try:
            record = build_observation_record(
                file_path,
                preferred_dataset_path=preferred_dataset_path,
                target_freq_bins=target_freq_bins,
            )
            if record is not None:
                records.append(record)
        except Exception as exc:
            errors.append(f"{Path(file_path).name}: {exc}")

    records.sort(key=lambda rec: (rec.time_order_key, rec.file_name))
    timeline = build_temporal_visibility_evolution(records)
    consistency = multi_file_consistency_report(records)
    waterfall = combine_session_waterfall(records, target_freq_bins=target_freq_bins)
    fusion = weighted_visibility_fusion(records)

    warnings.extend(consistency.get("warnings", []))
    errors.extend(consistency.get("errors", []))

    return {
        "records": records,
        "records_summary": [record.summary() for record in records],
        "timeline": timeline,
        "combined_waterfall": waterfall["combined_waterfall"],
        "waterfall_time_labels": waterfall["time_labels"],
        "waterfall_frequency_bins": waterfall["frequency_bins"],
        "waterfall_session_slices": waterfall["session_slices"],
        "consistency_report": consistency,
        "weighted_fusion": fusion,
        "warnings": warnings,
        "errors": errors,
    }


def _preview_scientific_hdf5(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    preview: dict[str, Any] = {
        "metadata": {"filename": path.name, "source_path": str(path), "format": "hdf5", "source_type": "radio_astronomy", "attributes": {}},
        "datasets": {},
        "groups": {},
        "interferometry": {},
        "telescope": {},
        "frequencies": {},
        "time": {},
        "hierarchy": {},
        "axis_index": {},
        "axis_overrides": {},
        "file_path": str(path),
    }

    def _load_small_dataset(item: h5py.Dataset, item_path: str) -> Any | None:
        size = int(item.size)
        key = item_path.lower()
        if size == 0:
            return None
        if size <= 4096 or any(token in key for token in ("antenna", "baseline", "freq", "frequency", "time", "timestamp", "uv", "uvw", "label", "position", "coord")):
            try:
                return np.asarray(item[()])
            except Exception:
                return None
        return None

    def _walk(group: h5py.Group, path_prefix: str = "") -> dict[str, Any]:
        info = {"path": path_prefix, "attributes": normalize_hdf5_attrs(group.attrs) if group.attrs else {}, "datasets": {}, "subgroups": [], "children": {}}
        for key in group.keys():
            item_path = f"{path_prefix}/{key}" if path_prefix else key
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data_preview = _load_small_dataset(item, item_path)
                axis_meta = infer_scientific_axis_metadata(item_path, item.shape, normalize_hdf5_attrs(item.attrs) if item.attrs else {}, None)
                ds_info = {
                    "path": item_path,
                    "shape": tuple(int(x) for x in item.shape),
                    "dtype": str(item.dtype),
                    "size": int(item.size),
                    "attributes": normalize_hdf5_attrs(item.attrs) if item.attrs else {},
                    "data": data_preview,
                    "axis_meta": axis_meta,
                }
                info["datasets"][key] = {k: ds_info[k] for k in ("shape", "dtype", "size", "attributes", "axis_meta")}
                preview["datasets"][item_path] = ds_info
                preview["axis_index"][item_path] = axis_meta
                key_lower = item_path.lower()
                if "vis" in key_lower:
                    preview["interferometry"]["visibilities"] = ds_info
                elif "freq" in key_lower or "channel" in key_lower:
                    preview["frequencies"]["channels"] = ds_info
                elif "time" in key_lower or "timestamp" in key_lower:
                    preview["time"]["timestamps"] = ds_info
                elif "baseline" in key_lower or "antenna" in key_lower:
                    preview["interferometry"]["baselines"] = ds_info
                elif "uv" in key_lower or "uvw" in key_lower:
                    preview["interferometry"]["uv_coordinates"] = ds_info
                elif "gain" in key_lower or "calib" in key_lower:
                    preview["telescope"]["calibration"] = ds_info
            elif isinstance(item, h5py.Group):
                info["subgroups"].append(key)
                group_info = {"path": item_path, "attributes": normalize_hdf5_attrs(item.attrs) if item.attrs else {}, "datasets": {}, "groups": {}}
                preview["groups"][item_path] = group_info
                group_info["children"] = _walk(item, item_path)
                info["children"][key] = group_info["children"]
        return info

    with h5py.File(path, "r") as f:
        preview["metadata"]["attributes"] = normalize_hdf5_attrs(f.attrs) if f.attrs else {}
        preview["structure"] = _walk(f)
    return preview


def _load_selected_scientific_hdf5(file_path: str | Path, selected_dataset_path: str, preview: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(file_path)
    preview = preview or _preview_scientific_hdf5(path)
    include_paths = {selected_dataset_path, selected_dataset_path.lstrip("/")}
    scientific = {k: (v.copy() if isinstance(v, dict) else v) for k, v in preview.items() if k != "datasets"}
    scientific["datasets"] = {}
    scientific["file_path"] = str(path)

    def _should_load_small(item_path: str, item: h5py.Dataset) -> bool:
        key = item_path.lower()
        return int(item.size) <= 4096 or any(token in key for token in ("antenna", "baseline", "freq", "frequency", "time", "timestamp", "uv", "uvw", "label", "position", "coord"))

    def _walk(group: h5py.Group, path_prefix: str = "") -> None:
        for key in group.keys():
            item_path = f"{path_prefix}/{key}" if path_prefix else key
            item = group[key]
            if isinstance(item, h5py.Dataset):
                data = None
                if item_path in include_paths or item_path.lstrip("/") in include_paths:
                    try:
                        data = np.asarray(item[()])
                    except Exception:
                        data = None
                elif _should_load_small(item_path, item):
                    try:
                        data = np.asarray(item[()])
                    except Exception:
                        data = None
                axis_meta = infer_scientific_axis_metadata(item_path, item.shape, normalize_hdf5_attrs(item.attrs) if item.attrs else {}, None)
                ds_info = {
                    "path": item_path,
                    "shape": tuple(int(x) for x in item.shape),
                    "dtype": str(item.dtype),
                    "size": int(item.size),
                    "attributes": normalize_hdf5_attrs(item.attrs) if item.attrs else {},
                    "data": data,
                    "axis_meta": axis_meta,
                }
                scientific["datasets"][item_path] = ds_info
                scientific["axis_index"][item_path] = axis_meta
                key_lower = item_path.lower()
                if "vis" in key_lower:
                    scientific["interferometry"]["visibilities"] = ds_info
                elif "freq" in key_lower or "channel" in key_lower:
                    scientific["frequencies"]["channels"] = ds_info
                elif "time" in key_lower or "timestamp" in key_lower:
                    scientific["time"]["timestamps"] = ds_info
                elif "baseline" in key_lower or "antenna" in key_lower:
                    scientific["interferometry"]["baselines"] = ds_info
                elif "uv" in key_lower or "uvw" in key_lower:
                    scientific["interferometry"]["uv_coordinates"] = ds_info
                elif "gain" in key_lower or "calib" in key_lower:
                    scientific["telescope"]["calibration"] = ds_info
            elif isinstance(item, h5py.Group):
                _walk(item, item_path)

    with h5py.File(path, "r") as f:
        _walk(f)
    return scientific


def build_observation_record(
    file_path: str | Path,
    preferred_dataset_path: str | None = None,
    target_freq_bins: int | None = None,
) -> ObservationSessionRecord | None:
    path = Path(file_path)
    scientific_preview = _preview_scientific_hdf5(path)
    dataset_path, dataset_info = select_visibility_dataset(scientific_preview, preferred_dataset_path)
    if dataset_info is None:
        return None

    scientific = _load_selected_scientific_hdf5(path, dataset_path, preview=scientific_preview)
    dataset_info = scientific.get("datasets", {}).get(dataset_path) or dataset_info
    raw = dataset_info.get("data")
    if raw is None:
        return None

    visibility = np.asarray(raw)
    time_axis = _extract_time_axis(scientific, dataset_path, dataset_info, path)
    waterfall = _build_session_waterfall(visibility, target_freq_bins=target_freq_bins)
    waterfall = np.asarray(waterfall, dtype=float)

    stats = compute_visibility_statistics(visibility)
    amp_stats = stats.get("amplitude", {})
    phase_stats = stats.get("phase", {})
    finite_fraction = _finite_fraction(visibility)
    weight = _session_weight(visibility, finite_fraction, amp_stats)
    time_key = _datetime_to_order_key(time_axis)

    uv_coordinates = _extract_uv_coordinates(scientific, dataset_info)
    if uv_coordinates is None:
        uv_coordinates = _infer_uv_coordinates_from_geometry(scientific, dataset_info)
    frequency_axis = _extract_frequency_axis(scientific, dataset_info)
    polarization_labels = _extract_polarization_labels(dataset_info)
    antenna_labels, antenna_positions, baseline_pairs = _extract_geometry_metadata(scientific)
    baseline_labels = _extract_baseline_labels(dataset_info, baseline_pairs)
    axis_meta = dict(dataset_info.get("axis_meta", {}) or {})

    metadata = {
        "file_metadata": scientific.get("metadata", {}),
        "dataset_metadata": {
            "shape": list(dataset_info.get("shape", visibility.shape)),
            "dtype": dataset_info.get("dtype", str(visibility.dtype)),
            "size": dataset_info.get("size", int(visibility.size)),
            "attributes": dataset_info.get("attributes", {}),
            "axis_meta": dataset_info.get("axis_meta", {}),
            "lazy_loaded": True,
        },
    }
    return ObservationSessionRecord(
        file_path=str(path),
        file_name=path.name,
        dataset_path=dataset_path,
        observation_time=time_axis,
        time_order_key=time_key,
        visibility_shape=tuple(int(dim) for dim in visibility.shape),
        waterfall=waterfall,
        visibility=visibility,
        uv_coordinates=uv_coordinates,
        frequency_axis=frequency_axis,
        polarization_labels=polarization_labels,
        axis_meta=axis_meta,
        antenna_labels=antenna_labels,
        antenna_positions=antenna_positions,
        baseline_labels=baseline_labels,
        baseline_pairs=baseline_pairs,
        mean_amplitude=float(amp_stats.get("mean", 0.0)),
        std_amplitude=float(amp_stats.get("std", 0.0)),
        mean_phase=float(phase_stats.get("mean", 0.0)),
        phase_std=float(phase_stats.get("std", 0.0)),
        finite_fraction=finite_fraction,
        weight=weight,
        metadata=metadata,
    )


def _is_visibility_candidate(path: str, info: Mapping[str, Any]) -> bool:
    axis_meta = info.get("axis_meta", {}) or {}
    kind = str(axis_meta.get("scientific_kind", "")).lower()
    path_lower = path.lower()
    shape = tuple(info.get("shape", ()) or ())
    dtype = str(info.get("dtype", "")).lower()
    return bool(
        kind in {"visibility_tensor", "visibility_cube", "correlation_tensor", "correlation_cube"}
        or any(token in path_lower for token in VISIBILITY_HINTS)
        or ("complex" in dtype and len(shape) >= 3)
    )


def select_visibility_dataset(
    scientific_data: dict[str, Any],
    preferred_dataset_path: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    datasets = scientific_data.get("datasets", {}) or {}
    if not datasets:
        return "", None

    if preferred_dataset_path:
        preferred = datasets.get(preferred_dataset_path) or datasets.get(preferred_dataset_path.lstrip("/"))
        if preferred is not None and _is_visibility_candidate(preferred_dataset_path, preferred):
            return _normalize_dataset_path(preferred_dataset_path), preferred

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for path, info in datasets.items():
        if not _is_visibility_candidate(path, info):
            continue
        score = _dataset_score(path, info)
        ranked.append((score, path, info))
    if not ranked:
        return "", None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best_path, best_info = ranked[0]
    if best_score <= 0:
        return "", None
    return best_path, best_info


def build_temporal_visibility_evolution(records: Sequence[ObservationSessionRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "order",
                "file_name",
                "dataset_path",
                "observation_time",
                "time_gap_seconds",
                "mean_amplitude",
                "std_amplitude",
                "mean_phase",
                "phase_std",
                "finite_fraction",
                "weight",
                "visibility_shape",
            ]
        )

    rows: list[dict[str, Any]] = []
    previous_time: datetime | None = None
    for idx, record in enumerate(sorted(records, key=lambda rec: (rec.time_order_key, rec.file_name)), start=1):
        gap_seconds = None
        if previous_time and record.observation_time and previous_time.tzinfo and record.observation_time.tzinfo:
            gap_seconds = (record.observation_time - previous_time).total_seconds()
        elif previous_time and record.observation_time:
            gap_seconds = (record.observation_time.replace(tzinfo=timezone.utc) - previous_time.replace(tzinfo=timezone.utc)).total_seconds()
        rows.append(
            {
                "order": idx,
                "file_name": record.file_name,
                "dataset_path": record.dataset_path,
                "observation_time": record.observation_time.isoformat() if record.observation_time else None,
                "time_gap_seconds": gap_seconds,
                "mean_amplitude": record.mean_amplitude,
                "std_amplitude": record.std_amplitude,
                "mean_phase": record.mean_phase,
                "phase_std": record.phase_std,
                "finite_fraction": record.finite_fraction,
                "weight": record.weight,
                "visibility_shape": "×".join(str(dim) for dim in record.visibility_shape),
            }
        )
        previous_time = record.observation_time or previous_time

    return pd.DataFrame(rows)


def combine_session_waterfall(
    records: Sequence[ObservationSessionRecord],
    target_freq_bins: int | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "combined_waterfall": np.zeros((0, 0), dtype=float),
            "time_labels": [],
            "frequency_bins": 0,
            "session_slices": [],
        }

    waterfalls = [np.asarray(record.waterfall, dtype=float) for record in records]
    if target_freq_bins is None:
        target_freq_bins = max((waterfall.shape[1] if waterfall.ndim == 2 else 0) for waterfall in waterfalls)
    target_freq_bins = max(int(target_freq_bins or 0), 1)

    aligned: list[np.ndarray] = []
    session_slices: list[dict[str, Any]] = []
    start = 0
    for record, waterfall in zip(records, waterfalls):
        aligned_waterfall = _resample_waterfall(waterfall, target_freq_bins)
        aligned.append(aligned_waterfall)
        stop = start + aligned_waterfall.shape[0]
        session_slices.append(
            {
                "file_name": record.file_name,
                "dataset_path": record.dataset_path,
                "row_start": start,
                "row_stop": stop,
                "shape": list(aligned_waterfall.shape),
            }
        )
        start = stop

    combined = np.vstack(aligned) if aligned else np.zeros((0, target_freq_bins), dtype=float)
    time_labels = [
        record.observation_time.isoformat() if record.observation_time else record.file_name
        for record in records
    ]
    return {
        "combined_waterfall": combined,
        "time_labels": time_labels,
        "frequency_bins": target_freq_bins,
        "session_slices": session_slices,
    }


def weighted_visibility_fusion(records: Sequence[ObservationSessionRecord]) -> dict[str, Any]:
    if not records:
        return {
            "available": False,
            "reason": "no records",
            "weights": [],
            "fused_visibility": None,
            "fused_waterfall": None,
        }

    weights = np.asarray([max(record.weight, 0.0) for record in records], dtype=float)
    if not np.any(weights > 0):
        weights = np.ones(len(records), dtype=float)
    norm_weights = weights / np.sum(weights)

    same_shape = len({record.visibility.shape for record in records}) == 1
    fused_visibility = None
    visibility_reason = "shape mismatch"
    if same_shape:
        stacked = np.stack([np.asarray(record.visibility, dtype=np.complex128) for record in records], axis=0)
        fused_visibility = np.tensordot(norm_weights, stacked, axes=(0, 0))
        visibility_reason = "weighted complex average"

    waterfall_bundle = combine_session_waterfall(records)
    fused_waterfall = None
    if waterfall_bundle["combined_waterfall"].size:
        session_rows = [
            _resample_waterfall(record.waterfall, waterfall_bundle["frequency_bins"])
            for record in records
        ]
        stacked_waterfalls = np.stack(session_rows, axis=0)
        fused_waterfall = np.tensordot(norm_weights, stacked_waterfalls, axes=(0, 0))

    return {
        "available": True,
        "reason": visibility_reason,
        "weights": norm_weights.tolist(),
        "weighted_sum": float(np.sum(weights)),
        "fused_visibility": fused_visibility,
        "fused_waterfall": fused_waterfall,
        "fused_waterfall_shape": list(fused_waterfall.shape) if fused_waterfall is not None else None,
        "visible_shape_matched": same_shape,
    }


def multi_file_consistency_report(records: Sequence[ObservationSessionRecord]) -> dict[str, Any]:
    if not records:
        return {
            "session_count": 0,
            "is_consistent": False,
            "warnings": ["no sessions provided"],
            "errors": ["no session records available"],
        }

    shapes = [record.visibility.shape for record in records]
    dtypes = [str(record.visibility.dtype) for record in records]
    dataset_paths = [record.dataset_path for record in records]
    times = [record.observation_time for record in records if record.observation_time is not None]
    ordered_times = times == sorted(times) if len(times) > 1 else True
    warnings: list[str] = []
    errors: list[str] = []

    if len(set(shapes)) > 1:
        warnings.append(f"multiple visibility shapes detected: {sorted({str(shape) for shape in shapes})}")
    if len(set(dtypes)) > 1:
        warnings.append(f"multiple visibility dtypes detected: {sorted(set(dtypes))}")
    if any(record.observation_time is None for record in records):
        warnings.append("one or more sessions are missing timestamps; file modification time was used where necessary")
    if not ordered_times:
        warnings.append("sessions were reordered into timestamp order")
    if len(set(dataset_paths)) > 1:
        warnings.append("multiple dataset paths were selected across sessions")

    uv_shapes = {tuple(np.asarray(record.uv_coordinates).shape) for record in records if record.uv_coordinates is not None and np.asarray(record.uv_coordinates).size}
    if len(uv_shapes) > 1:
        warnings.append("uv track continuity differs across sessions")
    if any(record.uv_coordinates is None for record in records):
        warnings.append("missing uv coordinates in one or more sessions")

    freq_shapes = {tuple(np.asarray(record.frequency_axis).shape) for record in records if record.frequency_axis is not None and np.asarray(record.frequency_axis).size}
    if len(freq_shapes) > 1:
        warnings.append("frequency grid continuity differs across sessions")
    if any(record.frequency_axis is None for record in records):
        warnings.append("missing frequency axis in one or more sessions")

    pol_labels = {tuple(record.polarization_labels) for record in records if record.polarization_labels}
    if len(pol_labels) > 1:
        warnings.append("polarization consistency differs across sessions")

    antenna_geometry = {tuple(record.antenna_labels) for record in records if record.antenna_labels}
    if len(antenna_geometry) > 1:
        warnings.append("antenna geometry consistency differs across sessions")

    if any(record.visibility.size == 0 for record in records):
        errors.append("empty visibility data encountered")
    for record in records:
        if record.finite_fraction < 1.0:
            warnings.append(f"{record.file_name}: finite fraction {record.finite_fraction:.3f}")

    metadata_complete = all("file_metadata" in record.metadata and "dataset_metadata" in record.metadata for record in records)
    is_consistent = not errors and metadata_complete and len(records) > 0
    return {
        "session_count": len(records),
        "is_consistent": is_consistent,
        "visibility_shapes": [list(shape) for shape in shapes],
        "dataset_paths": dataset_paths,
        "ordered_times": ordered_times,
        "metadata_complete": metadata_complete,
        "warnings": sorted(set(warnings)),
        "errors": sorted(set(errors)),
    }


def _dataset_score(path: str, info: Mapping[str, Any]) -> float:
    score = 0.0
    path_lower = path.lower()
    if any(token in path_lower for token in VISIBILITY_HINTS):
        score += 10.0
    if "time" in path_lower:
        score += 1.0
    dtype = str(info.get("dtype", "")).lower()
    if "complex" in dtype:
        score += 5.0
    shape = info.get("shape", ())
    if isinstance(shape, Iterable):
        dims = list(shape)
        if len(dims) >= 2:
            score += 2.0
        if len(dims) >= 3:
            score += 1.0
    axis_meta = info.get("axis_meta", {}) or {}
    axis_roles = " ".join(str(v).lower() for v in axis_meta.values())
    if "time" in axis_roles:
        score += 2.0
    if "freq" in axis_roles:
        score += 2.0
    return score


def _normalize_dataset_path(path: str) -> str:
    return path if path.startswith("/") else path.lstrip("/")


def _extract_time_axis(
    scientific_data: dict[str, Any],
    dataset_path: str,
    dataset_info: Mapping[str, Any],
    file_path: Path,
) -> datetime | None:
    metadata = scientific_data.get("metadata", {}) or {}
    attributes = metadata.get("attributes", {}) or {}
    for key in TIME_HINT_KEYS:
        if key in attributes:
            parsed = _coerce_datetime(attributes[key])
            if parsed is not None:
                return parsed

    dataset_attrs = dataset_info.get("attributes", {}) or {}
    for key in TIME_HINT_KEYS:
        if key in dataset_attrs:
            parsed = _coerce_datetime(dataset_attrs[key])
            if parsed is not None:
                return parsed

    axis_meta = dataset_info.get("axis_meta", {}) or {}
    for value in axis_meta.values():
        if isinstance(value, Mapping):
            for key in TIME_HINT_KEYS:
                if key in value:
                    parsed = _coerce_datetime(value[key])
                    if parsed is not None:
                        return parsed
        elif isinstance(value, str):
            parsed = _coerce_datetime(value)
            if parsed is not None:
                return parsed

    try:
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (np.integer, int, np.floating, float)):
        numeric = float(value)
        if numeric > 1e12:
            numeric /= 1000.0
        if numeric > 1e10:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = pd.to_datetime(text, utc=True, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
        except Exception:
            return None
    return None


def _datetime_to_order_key(moment: datetime | None) -> float:
    if moment is None:
        return float("inf")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _finite_fraction(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr.real).sum() + np.isfinite(arr.imag).sum()) / float(arr.size * (2 if np.iscomplexobj(arr) else 1))


def _session_weight(arr: np.ndarray, finite_fraction: float, amp_stats: Mapping[str, Any]) -> float:
    std = float(amp_stats.get("std", 0.0)) if amp_stats else 0.0
    noise_floor = std * std + 1e-6
    quality = finite_fraction / noise_floor
    return float(max(quality, 1e-9))


def _build_session_waterfall(arr: np.ndarray, target_freq_bins: int | None = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=float)

    if arr.ndim == 1:
        waterfall = np.abs(arr.reshape(1, -1))
    elif arr.ndim == 2:
        waterfall = np.abs(arr)
    else:
        reduced = np.abs(arr)
        if reduced.ndim > 2:
            reduced = reduced.mean(axis=tuple(range(2, reduced.ndim)))
        waterfall = reduced if reduced.ndim == 2 else reduced.reshape(reduced.shape[0], -1)

    if target_freq_bins is not None and waterfall.shape[1] != target_freq_bins:
        waterfall = _resample_waterfall(waterfall, int(target_freq_bins))
    return np.asarray(waterfall, dtype=float)


def _resample_waterfall(waterfall: np.ndarray, target_freq_bins: int) -> np.ndarray:
    waterfall = np.asarray(waterfall, dtype=float)
    if waterfall.size == 0:
        return np.zeros((0, target_freq_bins), dtype=float)
    if waterfall.ndim == 1:
        waterfall = waterfall.reshape(1, -1)
    if waterfall.shape[1] == target_freq_bins:
        return waterfall
    if waterfall.shape[1] == 0:
        return np.zeros((waterfall.shape[0], target_freq_bins), dtype=float)

    x_old = np.linspace(0.0, 1.0, waterfall.shape[1])
    x_new = np.linspace(0.0, 1.0, target_freq_bins)
    resampled = np.empty((waterfall.shape[0], target_freq_bins), dtype=float)
    for i, row in enumerate(waterfall):
        resampled[i] = np.interp(x_new, x_old, row.astype(float, copy=False))
    return resampled


def _extract_uv_coordinates(scientific: dict[str, Any], dataset_info: Mapping[str, Any]) -> np.ndarray | None:
    candidates = []
    interferometry = scientific.get("interferometry", {}) or {}
    uv_info = interferometry.get("uv_coordinates")
    if isinstance(uv_info, Mapping):
        candidates.append(uv_info.get("data"))
    for path, info in (scientific.get("datasets", {}) or {}).items():
        axis_meta = info.get("axis_meta", {}) or {}
        kind = str(axis_meta.get("scientific_kind", "")).lower()
        if kind in {"uv_coordinates", "baseline_coordinates"} or any(token in path.lower() for token in ("uv", "uvw")):
            candidates.append(info.get("data"))
    for value in candidates:
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0:
            continue
        if arr.ndim == 1 and arr.shape[0] >= 2:
            arr = arr.reshape(-1, 2)
        if arr.ndim >= 2 and arr.shape[-1] >= 2:
            flat = arr.reshape(-1, arr.shape[-1])
            return flat[:, :2].astype(float, copy=False)
    return None


def _infer_uv_coordinates_from_geometry(
    scientific: dict[str, Any],
    dataset_info: Mapping[str, Any],
) -> np.ndarray | None:
    labels, positions, baseline_pairs = _extract_geometry_metadata(scientific)
    if positions is None or positions.size == 0:
        return None

    arr = np.asarray(positions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None

    expected = int(dataset_info.get("shape", ())[2]) if len(dataset_info.get("shape", ())) >= 3 else None
    if expected is not None and arr.shape[0] not in {expected, expected + 1}:
        return None

    uv = arr[:, :2]
    if not np.isfinite(uv).all():
        return None
    return uv


def _extract_frequency_axis(scientific: dict[str, Any], dataset_info: Mapping[str, Any]) -> np.ndarray | None:
    candidates = []
    for section in (scientific.get("frequencies", {}), scientific.get("time", {}), scientific.get("datasets", {})):
        if isinstance(section, Mapping):
            for path, info in section.items():
                if isinstance(info, Mapping):
                    axis_meta = info.get("axis_meta", {}) or {}
                    kind = str(axis_meta.get("scientific_kind", "")).lower()
                    if kind == "frequency_axis" or "freq" in str(path).lower() or "channel" in str(path).lower():
                        candidates.append(info.get("data"))
    axis_meta = dataset_info.get("axis_meta", {}) or {}
    labels = [str(label).lower() for label in axis_meta.get("axis_labels", []) or []]
    if "frequency" in labels and dataset_info.get("data") is not None:
        candidates.append(dataset_info.get("data"))
    for value in candidates:
        if value is None:
            continue
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        if arr.dtype.kind not in {"i", "u", "f", "c"}:
            coerced = pd.to_numeric(pd.Series(arr.reshape(-1)), errors="coerce").to_numpy(dtype=float)
            arr = coerced
        else:
            arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size:
            return arr
    return None


def _extract_polarization_labels(dataset_info: Mapping[str, Any]) -> tuple[str, ...]:
    axis_meta = dataset_info.get("axis_meta", {}) or {}
    component_labels = axis_meta.get("component_labels") or []
    labels = tuple(str(label) for label in component_labels if str(label).strip())
    return labels


def _extract_geometry_metadata(scientific: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray | None, tuple[tuple[str, str], ...]]:
    labels: list[str] = []
    positions: np.ndarray | None = None
    pairs: list[tuple[str, str]] = []
    for path, info in (scientific.get("datasets", {}) or {}).items():
        attrs = info.get("attributes", {}) or {}
        axis_meta = info.get("axis_meta", {}) or {}
        blob = f"{path} {attrs} {axis_meta}".lower()
        if any(token in blob for token in ("antenna", "antennas", "station", "array_geometry", "antpos")):
            data = info.get("data")
            arr = np.asarray(data) if data is not None else np.array([])
            if arr.ndim == 2 and arr.shape[1] in {2, 3} and arr.size:
                positions = arr.astype(float, copy=False)
            if arr.dtype.kind in {"U", "S", "O"} and arr.size:
                labels.extend([str(x) for x in arr.reshape(-1).tolist() if str(x).strip()])
            if isinstance(attrs.get("antenna_labels"), (list, tuple)):
                labels.extend([str(x) for x in attrs.get("antenna_labels", [])])
            if isinstance(attrs.get("baseline_pairs"), (list, tuple)):
                for pair in attrs.get("baseline_pairs", []):
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        pairs.append((str(pair[0]), str(pair[1])))
    labels = tuple(dict.fromkeys(labels))
    pairs = tuple(dict.fromkeys(pairs))
    return labels, positions, pairs


def _extract_baseline_labels(dataset_info: Mapping[str, Any], baseline_pairs: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    axis_meta = dataset_info.get("axis_meta", {}) or {}
    component_labels = [str(label) for label in axis_meta.get("component_labels", []) or [] if str(label).strip()]
    if component_labels:
        return tuple(component_labels)
    if baseline_pairs:
        return tuple(f"{a}-{b}" for a, b in baseline_pairs)
    return tuple()


def _baseline_pairs_from_geometry(rec: Any) -> tuple[tuple[str, str], tuple[tuple[str, str], ...], dict[tuple[str, str], int]]:
    antennas = tuple(str(label) for label in getattr(rec, "antenna_labels", ()) or () if str(label).strip())
    pairs = tuple((str(a), str(b)) for a, b in getattr(rec, "baseline_pairs", ()) or () if str(a).strip() and str(b).strip())
    baseline_labels = tuple(str(label) for label in getattr(rec, "baseline_labels", ()) or () if str(label).strip())

    if not antennas and pairs:
        antennas = tuple(dict.fromkeys([label for pair in pairs for label in pair]))

    if not pairs and baseline_labels:
        parsed_pairs: list[tuple[str, str]] = []
        for label in baseline_labels:
            parts = [part.strip() for part in str(label).replace("|", "-").replace(",", "-").split("-") if part.strip()]
            if len(parts) == 2 and parts[0] != parts[1]:
                parsed_pairs.append((parts[0], parts[1]))
        if parsed_pairs:
            pairs = tuple(parsed_pairs)
            if not antennas:
                antennas = tuple(dict.fromkeys([label for pair in pairs for label in pair]))

    baseline_map: dict[tuple[str, str], int] = {}
    if pairs:
        for idx, (a, b) in enumerate(pairs):
            baseline_map[_baseline_key(a, b)] = idx
            baseline_map[_baseline_key(b, a)] = idx

    return antennas, pairs, baseline_map
