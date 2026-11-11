from __future__ import annotations

import re
from typing import Any

import numpy as np

VISIBILITY_KEYWORDS = ("visibility", "visibilities", "vis", "corr", "correlation")
UV_KEYWORDS = ("uv", "uvw")
BASELINE_KEYWORDS = ("baseline", "baselines", "bl")
FREQUENCY_KEYWORDS = ("frequency", "frequencies", "freq", "chan", "channel", "spectral")
TIME_KEYWORDS = ("time", "timestamp", "timestamps", "integration", "sample", "scan")
POLARIZATION_KEYWORDS = ("polarization", "polarisation", "pol", "stokes")
CALIBRATION_KEYWORDS = ("gain", "gains", "calib", "calibration", "bandpass", "phase")
IMAGE_KEYWORDS = ("image", "cube", "dirty", "residual", "psf", "beam")


def normalize_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            return [normalize_hdf5_value(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalize_hdf5_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_hdf5_value(v) for k, v in value.items()}
    return value


def normalize_hdf5_attrs(attrs: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    try:
        items = attrs.items() if hasattr(attrs, "items") else dict(attrs).items()
    except Exception:
        return normalized
    for key, value in items:
        normalized[str(key)] = normalize_hdf5_value(value)
    return normalized


def _tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _text_blob(path: str, attrs: dict[str, Any]) -> str:
    parts = [path]
    for key in ("name", "label", "description", "units", "unit", "kind", "role", "axes", "axis_labels", "axis_roles"):
        if key in attrs:
            parts.append(str(attrs[key]))
    return " ".join(parts).lower()


def _explicit_axis_labels(attrs: dict[str, Any], ndim: int) -> list[str] | None:
    for key in ("axis_labels", "axis_roles", "axes", "dimension_labels", "DIMENSION_LABELS"):
        if key in attrs:
            labels = normalize_hdf5_value(attrs[key])
            if isinstance(labels, str):
                labels = [labels]
            if isinstance(labels, list) and len(labels) == ndim:
                return [str(label).strip().lower() for label in labels]
    return None


def _explicit_component_labels(attrs: dict[str, Any]) -> list[str] | None:
    for key in ("component_labels", "components", "axis_components"):
        if key in attrs:
            labels = normalize_hdf5_value(attrs[key])
            if isinstance(labels, str):
                labels = [labels]
            if isinstance(labels, list) and labels:
                return [str(label).strip().lower() for label in labels]
    return None


def apply_axis_overrides(axis_meta: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Apply explicit axis overrides without discarding inferred metadata."""
    if not overrides:
        return axis_meta

    merged = dict(axis_meta)
    overrides = normalize_hdf5_value(overrides) if isinstance(overrides, dict) else {}
    ndim = int(merged.get("ndim", 0) or 0)

    kind = overrides.get("scientific_kind")
    if kind:
        merged["scientific_kind"] = str(kind)

    axis_labels = overrides.get("axis_labels")
    if isinstance(axis_labels, str):
        axis_labels = [axis_labels]
    if isinstance(axis_labels, list) and (ndim == 0 or len(axis_labels) == ndim):
        merged["axis_labels"] = [str(label).strip().lower() for label in axis_labels]

    axis_roles = overrides.get("axis_roles")
    if isinstance(axis_roles, str):
        axis_roles = [axis_roles]
    if isinstance(axis_roles, list) and (ndim == 0 or len(axis_roles) == ndim):
        merged["axis_roles"] = [str(role).strip().lower().replace(" ", "_") for role in axis_roles]

    component_labels = overrides.get("component_labels")
    if isinstance(component_labels, str):
        component_labels = [component_labels]
    if isinstance(component_labels, list) and component_labels:
        merged["component_labels"] = [str(label).strip().lower() for label in component_labels]

    merged["axis_override_applied"] = True
    return merged


def infer_scientific_kind(path: str, shape: tuple[int, ...], attrs: dict[str, Any]) -> str:
    blob = _text_blob(path, attrs)
    tokens = set(_tokens(blob))
    if any(k in blob for k in VISIBILITY_KEYWORDS):
        return "visibility_tensor"
    if any(k in blob for k in UV_KEYWORDS):
        return "uv_coordinates"
    if any(k in blob for k in BASELINE_KEYWORDS):
        return "baseline_coordinates"
    if any(k in blob for k in FREQUENCY_KEYWORDS):
        return "frequency_axis"
    if any(k in blob for k in TIME_KEYWORDS):
        return "time_axis"
    if any(k in blob for k in POLARIZATION_KEYWORDS):
        return "polarization_axis"
    if any(k in blob for k in CALIBRATION_KEYWORDS):
        return "calibration_solution"
    if any(k in blob for k in IMAGE_KEYWORDS):
        return "image_cube"
    if len(shape) >= 3 and {"xx", "yy"}.issubset(tokens):
        return "visibility_tensor"
    return "generic_array"


def infer_scientific_axis_metadata(
    path: str,
    shape: tuple[int, ...],
    attrs: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs = dict(attrs or {})
    ndim = len(shape)
    kind = infer_scientific_kind(path, shape, attrs)
    explicit = _explicit_axis_labels(attrs, ndim)
    component_labels = _explicit_component_labels(attrs)
    lower_path = path.lower()

    axis_roles: list[str] = []
    axis_labels: list[str] = []

    if explicit:
        axis_labels = [str(label) for label in explicit]
        axis_roles = [str(label).strip().lower().replace(" ", "_") for label in explicit]
    elif kind == "visibility_tensor":
        if ndim >= 4:
            axis_labels = ["time", "frequency", "baseline", "polarization"][:ndim]
        elif ndim == 3:
            if any(k in lower_path for k in ("pol", "stokes")) or shape[-1] in {2, 4}:
                axis_labels = ["time", "frequency", "polarization"]
            else:
                axis_labels = ["time", "frequency", "baseline"]
        elif ndim == 2:
            axis_labels = ["time", "frequency"]
        else:
            axis_labels = ["visibility"]
        axis_roles = axis_labels[:]
    elif kind in {"uv_coordinates", "baseline_coordinates"}:
        if ndim == 2 and shape[-1] in {2, 3}:
            axis_labels = ["sample", "component"]
            axis_roles = ["sample", "component"]
            if component_labels is None:
                component_labels = ["u", "v", "w"][: shape[-1]]
        elif ndim == 1 and component_labels:
            axis_labels = ["component"]
            axis_roles = ["component"]
        else:
            axis_labels = [f"axis_{i}" for i in range(ndim)]
            axis_roles = axis_labels[:]
    elif kind == "calibration_solution":
        axis_labels = [f"axis_{i}" for i in range(ndim)]
        axis_roles = axis_labels[:]
    elif kind == "image_cube":
        axis_labels = ["time", "frequency", "image_y", "image_x"][:ndim]
        axis_roles = axis_labels[:]
    elif kind == "frequency_axis":
        axis_labels = ["frequency"] if ndim <= 1 else [f"axis_{i}" for i in range(ndim)]
        axis_roles = axis_labels[:]
    elif kind == "time_axis":
        axis_labels = ["time"] if ndim <= 1 else [f"axis_{i}" for i in range(ndim)]
        axis_roles = axis_labels[:]
    elif kind == "polarization_axis":
        axis_labels = ["polarization"] if ndim <= 1 else [f"axis_{i}" for i in range(ndim)]
        axis_roles = axis_labels[:]
    else:
        axis_labels = [f"axis_{i}" for i in range(ndim)]
        axis_roles = axis_labels[:]

    if not axis_roles:
        axis_roles = [f"axis_{i}" for i in range(ndim)]
    if not axis_labels:
        axis_labels = [f"axis_{i}" for i in range(ndim)]

    axis_meta = {
        "scientific_kind": kind,
        "axis_labels": axis_labels,
        "axis_roles": axis_roles,
        "component_labels": component_labels,
        "ndim": ndim,
        "shape": tuple(shape),
    }
    return apply_axis_overrides(axis_meta, overrides)


def apply_axis_overrides_to_scientific_data(
    scientific_data: dict[str, Any],
    axis_overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Mutate a scientific HDF5 structure with explicit axis overrides."""
    if not axis_overrides:
        return scientific_data

    datasets = scientific_data.get("datasets", {})
    axis_index = scientific_data.get("axis_index", {})
    for path, ds_info in datasets.items():
        override = axis_overrides.get(path) or axis_overrides.get(path.lstrip("/")) or None
        if override is None:
            continue
        axis_meta = ds_info.get("axis_meta", {})
        merged = apply_axis_overrides(axis_meta, override)
        ds_info["axis_meta"] = merged
        axis_index[path] = merged
    return scientific_data
