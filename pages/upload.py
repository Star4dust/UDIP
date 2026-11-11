import streamlit as st
import pandas as pd
import re
import io
import uuid
from pathlib import Path
from modules.core import (
    save_dataset, add_history, fmt_bytes, get_memory_mb, read_uploaded_file, read_scientific_hdf5,
    sanitize_varname, MEMORY_LIMIT_MB
)
from modules.scientific_hdf5 import apply_axis_overrides_to_scientific_data


def render_handoff_suggestions(*_args, **_kwargs):
    """Compatibility no-op for legacy upload-page handoff calls."""
    return None

SCI_KINDS = [
    "auto",
    "visibility_tensor",
    "uv_coordinates",
    "baseline_coordinates",
    "frequency_axis",
    "time_axis",
    "polarization_axis",
    "calibration_solution",
    "image_cube",
    "generic_array",
]


def _parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _axis_override_editor(scientific_hdf5: dict, file_key: str):
    st.markdown("#### axis mapping overrides")
    st.caption("use this only when the file has unusual or missing hdf5 metadata.")
    axis_overrides = scientific_hdf5.get("axis_overrides", {}) or {}
    dataset_paths = list(scientific_hdf5.get("datasets", {}).keys())
    if not dataset_paths:
        st.info("no scientific datasets were detected for axis mapping.")
        return

    selected_path = st.selectbox("dataset path", dataset_paths, key=f"axis_ds_{file_key}")
    ds_info = scientific_hdf5["datasets"].get(selected_path, {})
    axis_meta = ds_info.get("axis_meta", {})
    shape = ds_info.get("shape", ())
    ndim = len(shape)

    st.write("inferred axis metadata:")
    st.json({
        "scientific_kind": axis_meta.get("scientific_kind"),
        "axis_labels": axis_meta.get("axis_labels"),
        "axis_roles": axis_meta.get("axis_roles"),
        "component_labels": axis_meta.get("component_labels"),
        "axis_override_applied": axis_meta.get("axis_override_applied", False),
    })

    override_enabled = st.checkbox("override this dataset", key=f"axis_override_{file_key}_{selected_path}")
    if not override_enabled:
        return

    current_kind = axis_meta.get("scientific_kind", "generic_array")
    kind_index = SCI_KINDS.index(current_kind) if current_kind in SCI_KINDS else 0
    scientific_kind = st.selectbox("scientific kind", SCI_KINDS, index=kind_index, key=f"axis_kind_{file_key}_{selected_path}")

    current_labels = axis_meta.get("axis_labels", [f"axis_{i}" for i in range(ndim)])
    current_roles = axis_meta.get("axis_roles", current_labels)
    current_components = axis_meta.get("component_labels", [])

    label_text = st.text_input(
        "axis labels (comma-separated)",
        value=", ".join(current_labels),
        key=f"axis_labels_{file_key}_{selected_path}",
    )
    role_text = st.text_input(
        "axis roles (comma-separated)",
        value=", ".join(current_roles),
        key=f"axis_roles_{file_key}_{selected_path}",
    )
    component_text = st.text_input(
        "component labels (comma-separated)",
        value=", ".join(current_components),
        key=f"component_labels_{file_key}_{selected_path}",
    )

    labels = _parse_csv_list(label_text)
    roles = _parse_csv_list(role_text)
    components = _parse_csv_list(component_text)
    valid_lengths = (not labels or len(labels) == ndim) and (not roles or len(roles) == ndim)
    if not valid_lengths:
        st.error(f"axis labels and axis roles must have either 0 or {ndim} entries for this dataset.")
        return

    if st.button("apply axis override", key=f"apply_axis_{file_key}_{selected_path}"):
        axis_overrides[selected_path] = {
            "scientific_kind": scientific_kind if scientific_kind != "auto" else axis_meta.get("scientific_kind", "generic_array"),
            "axis_labels": labels or axis_meta.get("axis_labels", []),
            "axis_roles": roles or axis_meta.get("axis_roles", []),
            "component_labels": components or axis_meta.get("component_labels", []),
        }
        scientific_hdf5["axis_overrides"] = axis_overrides
        apply_axis_overrides_to_scientific_data(scientific_hdf5, axis_overrides)
        st.session_state[f"scientific_hdf5_{file_key}"] = scientific_hdf5
        ds_info["axis_meta"] = scientific_hdf5["datasets"][selected_path]["axis_meta"]
        st.success("axis override applied in this session. re-import to persist it as a new dataset entry.")


def _hdf5_summary_frame(scientific_data):
    rows = []
    for category, entries in scientific_data.items():
        if category in {"metadata", "structure", "file_path", "axis_index", "axis_overrides"}:
            continue
        if isinstance(entries, dict):
            for name, info in entries.items():
                if not isinstance(info, dict):
                    continue
                axis_meta = info.get("axis_meta", {})
                rows.append({
                    "category": category,
                    "name": name,
                    "path": info.get("path", ""),
                    "shape": str(info.get("shape", "")),
                    "dtype": str(info.get("dtype", "")),
                    "size": int(info.get("size", 0)) if info.get("size", 0) is not None else 0,
                    "scientific_kind": axis_meta.get("scientific_kind", ""),
                    "axis_labels": ", ".join(axis_meta.get("axis_labels", []) or []),
                    "component_labels": ", ".join(axis_meta.get("component_labels", []) or []),
                    "axis_override_applied": bool(axis_meta.get("axis_override_applied", False)),
                })
    return pd.DataFrame(rows)

def render():
    st.markdown('<div class="main-header">📁 Upload Datasets</div>', unsafe_allow_html=True)

    n_ds = len(st.session_state.datasets)
    total_rows = sum(v["df"].shape[0] for v in st.session_state.datasets.values())
    
    # Stats row
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{n_ds}</div><div class="stat-label">Datasets loaded</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total_rows:,}</div><div class="stat-label">Total rows</div></div>', unsafe_allow_html=True)
    with cols[2]:
        fmts = list({v.get("format", "processed") for v in st.session_state.datasets.values()})
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(fmts)}</div><div class="stat-label">Formats</div></div>', unsafe_allow_html=True)
    with cols[3]:
        total_mem_mb = get_memory_mb()
        st.markdown(f'<div class="stat-box"><div class="stat-number">{fmt_bytes(total_mem_mb * 1024 * 1024)}</div><div class="stat-label">Memory used</div></div>', unsafe_allow_html=True)

    # Processing engine control
    with st.expander("Processing Engine", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("**Polars acceleration**")
            st.caption("Use the Polars engine for large tabular datasets when available.")
        with c2:
            st.session_state.quantum_mode = st.toggle("Accelerate", value=st.session_state.get("quantum_mode", False), key="vanguard_toggle")
        
    st.divider()

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("### Upload files")
        st.markdown("Supports: **CSV, Excel, JSON, Parquet, HDF5, TSV, XML, Feather, Avro**")
        uploaded = st.file_uploader(
            "Drop files here or click Browse",
            accept_multiple_files=True,
            type=["csv", "xlsx", "xls", "json", "parquet", "h5", "hdf5", "tsv", "txt", "log", "xml", "feather", "avro", "pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if "uploader_cache" not in st.session_state:
            st.session_state.uploader_cache = set()

        groq_key = st.session_state.get("groq_key", "")

        if uploaded:
            for f in uploaded:
                file_key = f"{f.name}_{f.size}"
                if file_key not in st.session_state.uploader_cache:
                    with st.spinner(f"Loading {f.name}..."):
                        try:
                            _result = None
                            unstructured = f.name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
                            if unstructured:
                                st.info(f"⚠️ **Unstructured Content Detected**: {f.name}")
                                st.warning("AI-powered document extraction removed for radio astronomy focus. Please convert to CSV/Excel format.")
                            else:
                                if f.name.lower().endswith((".h5", ".hdf5")):
                                    scientific_hdf5 = read_scientific_hdf5(f)
                                    st.session_state[f"scientific_hdf5_{file_key}"] = scientific_hdf5
                                    _result = _hdf5_summary_frame(scientific_hdf5)
                                    metadata = {
                                        "source_type": "hirax_scientific_hdf5",
                                        "original_filename": f.name,
                                        "hdf5_metadata": scientific_hdf5.get("metadata", {}),
                                        "hdf5_groups": list(scientific_hdf5.get("groups", {}).keys()),
                                        "hdf5_datasets": list(scientific_hdf5.get("datasets", {}).keys()),
                                        "axis_overrides": scientific_hdf5.get("axis_overrides", {}),
                                    }
                                else:
                                    _result = read_uploaded_file(f)
                                    metadata = None

                            if _result is not None:
                                # Handle dict return (multi-table)
                                if isinstance(_result, dict):
                                    for sub_name, sub_df in _result.items():
                                        _register_dataset(sub_name, sub_df, f.name)
                                else:
                                    _register_dataset(f.name, _result, f.name, metadata=metadata if 'metadata' in locals() else None)
                                st.session_state.uploader_cache.add(file_key)
                                st.success(f"✅ Loaded {f.name}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load {f.name}: {e}")

        if uploaded:
            for f in uploaded:
                file_key = f"{f.name}_{f.size}"
                scientific_hdf5 = st.session_state.get(f"scientific_hdf5_{file_key}")
                if scientific_hdf5:
                    with st.expander(f"Scientific metadata: {f.name}", expanded=False):
                        st.dataframe(_hdf5_summary_frame(scientific_hdf5), use_container_width=True)
                        _axis_override_editor(scientific_hdf5, file_key)

        # Cloud source import section
        st.divider()
        st.markdown("### ☁️ Import from Connected Sources")
        st.markdown("Import datasets from your connected cloud storage providers.")
        
        # Check which providers are connected
        connected_providers = []
        provider_map = {
            "google_drive": "Google Drive",
            "aws": "AWS S3",
            "google": "Google Cloud Storage",
            "azure": "Azure Blob Storage"
        }
        
        for provider_key, provider_name in provider_map.items():
            if st.session_state.get(f"{provider_key}_connected", False):
                connected_providers.append((provider_key, provider_name))
        
        if not connected_providers:
            st.info("🔗 No cloud sources connected. Go to **Cloud Sources** page to connect a provider.")
        else:
            # Show tabs for each connected provider
            cloud_tab_names = [name for _, name in connected_providers]
            cloud_tab_keys = [key for key, _ in connected_providers]
            
            selected_cloud_tab = st.tabs(cloud_tab_names)
            
            for tab, (provider_key, provider_name) in zip(selected_cloud_tab, connected_providers):
                with tab:
                    if provider_key == "google_drive":
                        _render_google_drive_import()
                    else:
                        _render_object_storage_import(provider_key, provider_name)

    with right:
        st.markdown("### Loaded Datasets")
        if not st.session_state.datasets:
            st.info("No datasets loaded yet. Upload files to get started.")
        else:
            for ds_id, ds in list(st.session_state.datasets.items()):
                with st.container():
                    st.markdown(
                        f'<div class="ds-card">'
                        f'<div class="ds-name">{ds["name"]}</div>'
                        f'<div class="ds-meta">{ds["df"].shape[0]:,} rows × {ds["df"].shape[1]} cols · {ds.get("format","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("👁️ Preview", key=f"prev_{ds_id}"):
                        st.session_state._preview_ds = ds_id
                    if c2.button("🗑️ Remove", key=f"del_{ds_id}"):
                        from modules.core import delete_dataset_from_disk
                        delete_dataset_from_disk(ds_id)
                        del st.session_state.datasets[ds_id]
                        st.rerun()

    if st.session_state.get("_preview_ds"):
        pid = st.session_state._preview_ds
        if pid in st.session_state.datasets:
            ds = st.session_state.datasets[pid]
            st.divider()
            st.markdown(f"#### Preview: {ds['name']}")
            st.dataframe(ds["df"].head(100), width="stretch")
            if st.button("Close Preview"):
                del st.session_state._preview_ds
                st.rerun()

    # Workflow handoff suggestions are intentionally disabled in the stabilized radio-astronomy build.
    # Upload remains focused on ingestion, metadata preservation, and dataset registration.

def _register_dataset(name, df, source, metadata=None):
    import uuid
    ds_id = str(uuid.uuid4())[:8]
    entry = {
        "name": name,
        "df": df,
        "format": Path(source).suffix.replace(".", "") or "processed",
        "source": source,
        "varname": sanitize_varname(name),
        "metadata": metadata or {}
    }
    st.session_state.datasets[ds_id] = entry
    save_dataset(ds_id, entry)
    add_history("Upload", f"{name} ({len(df):,} rows)")
    
    # Workflow handoff suggestions are intentionally disabled in the stabilized radio-astronomy build.
    # Upload remains focused on ingestion, metadata preservation, and dataset registration.

def _render_google_drive_import():
    """Render Google Drive import interface."""
    st.info("Cloud import helpers are not configured in this local scientific build. Use file upload for CSV/Excel/HDF5 ingestion.")

def _render_object_storage_import(provider_key, provider_name):
    """Render object storage import interface."""
    st.info(f"{provider_name} import helpers are not configured in this local scientific build. Use file upload for CSV/Excel/HDF5 ingestion.")
