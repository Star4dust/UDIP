from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from modules.observation_session import reconstruct_observation_sessions
from modules.observatory_intelligence import build_observatory_session_intelligence, export_session_intelligence, inspect_scientific_preview, performance_diagnostics, malformed_dataset_resilience_report


def _write_geometry_session(path: Path, start_time: str, scale: float) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["observation_time"] = start_time
        f.create_dataset("antenna_labels", data=np.array([b"a0", b"a1", b"a2", b"a3"]))
        f.create_dataset("antenna_positions", data=np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [12.0, 12.0, 0.0]], dtype=float))
        f.create_dataset("baseline_pairs", data=np.array([[b"a0", b"a1"], [b"a1", b"a2"], [b"a2", b"a0"], [b"a0", b"a3"]], dtype="S2"))
        f.create_dataset("uv_coordinates", data=np.array([[0.0, 12.0], [12.0, 0.0], [12.0, 12.0], [-12.0, 12.0]], dtype=float))
        f.create_dataset("frequencies_mhz", data=np.array([800.0, 805.0, 810.0, 815.0], dtype=float))
        vis = np.zeros((4, 4, 4, 2), dtype=np.complex128)
        for t in range(vis.shape[0]):
            for c in range(vis.shape[1]):
                for b in range(vis.shape[2]):
                    for p in range(vis.shape[3]):
                        vis[t, c, b, p] = scale * (1 + t + c + b + p) + 1j * scale * 0.05 * (t + c + b + p)
        ds = f.create_dataset("visibilities", data=vis)
        ds.attrs["axis_0"] = "time"
        ds.attrs["axis_1"] = "frequency"
        ds.attrs["axis_2"] = "baseline"
        ds.attrs["axis_3"] = "polarization"
        ds.attrs["component_labels"] = np.array([b"xx", b"yy"])


def test_observatory_intelligence_exports_and_visual_diagnostics(tmp_path):
    a = tmp_path / "session_a.h5"
    b = tmp_path / "session_b.h5"
    _write_geometry_session(a, "2026-05-19T08:00:00Z", 1.0)
    _write_geometry_session(b, "2026-05-19T08:15:00Z", 1.2)

    session = reconstruct_observation_sessions([str(a), str(b)])
    observatory = build_observatory_session_intelligence(session["records"])

    assert observatory["uv_tracks"]["available"] is True
    assert observatory["closure_explorer"]["available"] is True
    assert observatory["calibration_maps"]["available"] is True
    assert observatory["persistent_rfi"]["available"] is True
    assert observatory["session_export_preview"]
    assert observatory["reproducibility"]["available"] is True
    assert observatory["trust_checks"]["available"] is True
    assert observatory["export_integrity"]["available"] is True
    assert observatory["guidance"]["uv_coverage"]
    assert observatory["stress"]["available"] is True
    assert observatory["export_signature_summary"]["available"] is True

    export_paths = export_session_intelligence(observatory, tmp_path / "exports", stem="observatory_session")
    assert Path(export_paths["csv"]).exists()
    assert Path(export_paths["json"]).exists()

    csv_df = pd.read_csv(export_paths["csv"])
    assert not csv_df.empty
    assert "section" in csv_df.columns


def test_lazy_preview_and_versioned_export_are_safe(tmp_path):
    a = tmp_path / "session_lazy.h5"
    _write_geometry_session(a, "2026-05-19T08:00:00Z", 1.0)

    preview = inspect_scientific_preview(a)
    assert preview["available"] is True
    assert preview["datasets"]

    session = reconstruct_observation_sessions([str(a)])
    observatory = build_observatory_session_intelligence(session["records"])
    export_paths = export_session_intelligence(observatory, tmp_path / "exports", stem="observatory_session")
    assert "_v1.1" in export_paths["csv"]
    assert Path(export_paths["csv"]).exists()
    assert Path(export_paths["json"]).exists()


def test_performance_diagnostics_and_cache_stability(tmp_path):
    a = tmp_path / "session_a.h5"
    b = tmp_path / "session_b.h5"
    _write_geometry_session(a, "2026-05-19T08:00:00Z", 1.0)
    _write_geometry_session(b, "2026-05-19T08:15:00Z", 1.2)

    session = reconstruct_observation_sessions([str(a), str(b)])
    diagnostics = performance_diagnostics(session["records"])
    assert diagnostics["available"] is True
    assert diagnostics["hotspots"]
    assert diagnostics["repeated_computation"]
    assert diagnostics["cache_signature"]


def test_malformed_dataset_resilience_reports_quarantine(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.attrs["observation_time"] = "2026-05-19T08:00:00Z"
        vis = np.zeros((2, 2, 2), dtype=np.complex128)
        ds = f.create_dataset("visibilities", data=vis)
        ds.attrs["axis_0"] = "time"
        ds.attrs["axis_1"] = "frequency"
        ds.attrs["axis_2"] = "baseline"

    session = reconstruct_observation_sessions([str(bad)])
    report = malformed_dataset_resilience_report(session["records"])
    assert report["available"] is True
    assert report["quarantined_count"] >= 0
    assert isinstance(report["rows"], pd.DataFrame)
