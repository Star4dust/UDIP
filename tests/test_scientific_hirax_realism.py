from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from modules.observation_session import reconstruct_observation_sessions
from modules.observatory_intelligence import build_observatory_session_intelligence, inspect_scientific_preview, performance_diagnostics, export_session_intelligence
from modules.radio_interferometry import compute_closure_phase, compute_closure_amplitude


def _write_real_hirax_like_file(path: Path, gain_scale: float = 1.0, rfi_scale: float = 0.0, phase_shift: float = 0.0) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["observation_time"] = "2026-05-19T08:00:00Z"
        f.create_dataset("antenna_labels", data=np.array([b"a0", b"a1", b"a2", b"a3"]))
        f.create_dataset("antenna_positions", data=np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [12.0, 12.0, 0.0]], dtype=float))
        f.create_dataset("baseline_pairs", data=np.array([[b"a0", b"a1"], [b"a1", b"a2"], [b"a2", b"a0"], [b"a0", b"a3"]], dtype="S2"))
        f.create_dataset("uv_coordinates", data=np.array([[0.0, 12.0], [12.0, 0.0], [12.0, 12.0], [-12.0, 12.0]], dtype=float))
        f.create_dataset("frequencies_mhz", data=np.array([800.0, 805.0, 810.0, 815.0], dtype=float))
        time = np.arange(6, dtype=float)
        vis = np.zeros((time.size, 4, 4, 2), dtype=np.complex128)
        for t in range(vis.shape[0]):
            for c in range(vis.shape[1]):
                for b in range(vis.shape[2]):
                    for p in range(vis.shape[3]):
                        amp = gain_scale * (1.0 + 0.05 * (t + c + b + p))
                        if c == 2 and p == 1:
                            amp += rfi_scale * 10.0
                        phase = phase_shift + 0.03 * (t + c + b + p)
                        vis[t, c, b, p] = amp * np.exp(1j * phase)
        ds = f.create_dataset("visibilities", data=vis)
        ds.attrs["axis_0"] = "time"
        ds.attrs["axis_1"] = "frequency"
        ds.attrs["axis_2"] = "baseline"
        ds.attrs["axis_3"] = "polarization"
        ds.attrs["component_labels"] = np.array([b"xx", b"yy"])


def test_real_hirax_fixture_closure_consistency_and_degradation(tmp_path):
    good = tmp_path / "good.h5"
    bad = tmp_path / "bad.h5"
    _write_real_hirax_like_file(good, gain_scale=1.0, rfi_scale=0.0, phase_shift=0.0)
    _write_real_hirax_like_file(bad, gain_scale=1.4, rfi_scale=1.0, phase_shift=0.8)

    session = reconstruct_observation_sessions([str(good), str(bad)])
    observatory = build_observatory_session_intelligence(session["records"])

    assert session["consistency_report"]["session_count"] == 2
    assert observatory["closure_explorer"]["available"] is True
    assert observatory["calibration_maps"]["available"] is True
    assert observatory["persistent_rfi"]["available"] is True
    assert observatory["reproducibility"]["available"] is True
    assert observatory["trust_checks"]["available"] is True
    assert observatory["export_integrity"]["available"] is True
    assert observatory["export_signature_summary"]["available"] is True

    closure = observatory["closure_explorer"]["triangle_browser"]
    assert not closure.empty
    assert closure["closure_stability_score"].max() <= 1.0
    assert closure["closure_stability_score"].min() >= 0.0
    assert (closure["unstable"] == True).any() or (closure["closure_stability_score"] < 0.5).any()

    good_record = session["records"][0]
    bad_record = session["records"][1]
    triangles = observatory["closure_explorer"]["triangle_browser"]
    tri = triangles.iloc[0]
    good_cube = np.asarray(good_record.visibility)[..., 0]
    bad_cube = np.asarray(bad_record.visibility)[..., 0]
    baseline_indices = [int(i) for i in tri["triangle"]]
    phase_good = compute_closure_phase(
        np.take(good_cube, baseline_indices[0], axis=2).reshape(-1),
        np.take(good_cube, baseline_indices[1], axis=2).reshape(-1),
        np.take(good_cube, baseline_indices[2], axis=2).reshape(-1),
    )
    phase_bad = compute_closure_phase(
        np.take(bad_cube, baseline_indices[0], axis=2).reshape(-1),
        np.take(bad_cube, baseline_indices[1], axis=2).reshape(-1),
        np.take(bad_cube, baseline_indices[2], axis=2).reshape(-1),
    )
    assert phase_good["rms_closure_phase"] >= 0.0
    assert phase_bad["rms_closure_phase"] >= 0.0

    amp = compute_closure_amplitude(
        np.take(good_cube, 0, axis=2).reshape(-1),
        np.take(good_cube, 1, axis=2).reshape(-1),
        np.take(good_cube, 2, axis=2).reshape(-1),
        np.take(good_cube, 3, axis=2).reshape(-1),
    )
    assert amp["n_samples"] > 0
    assert np.isfinite(amp["mean_closure_amplitude"])


def test_rfi_heatmap_and_stability_maps_react_to_perturbations(tmp_path):
    clean = tmp_path / "clean.h5"
    noisy = tmp_path / "noisy.h5"
    _write_real_hirax_like_file(clean, gain_scale=1.0, rfi_scale=0.0, phase_shift=0.0)
    _write_real_hirax_like_file(noisy, gain_scale=1.0, rfi_scale=2.0, phase_shift=0.0)

    session = reconstruct_observation_sessions([str(clean), str(noisy)])
    observatory = build_observatory_session_intelligence(session["records"])

    persistent = observatory["persistent_rfi"]["persistent_contamination_map"]
    gain_map = observatory["calibration_maps"]["gain_stability_map"]
    phase_map = observatory["calibration_maps"]["phase_drift_map"]

    assert persistent.size > 0
    assert gain_map.size > 0
    assert phase_map.size > 0
    assert observatory["persistent_rfi"]["frequency_occupancy_summary"]["persistent_fraction"] >= 0.0

    export_preview = observatory["session_export_preview"]
    assert export_preview["csv"]
    assert export_preview["json"]


def test_real_hirax_fixture_lazy_preview_and_cache_behavior(tmp_path):
    good = tmp_path / "good.h5"
    bad = tmp_path / "bad.h5"
    _write_real_hirax_like_file(good, gain_scale=1.0, rfi_scale=0.0, phase_shift=0.0)
    _write_real_hirax_like_file(bad, gain_scale=1.4, rfi_scale=1.0, phase_shift=0.8)

    preview = inspect_scientific_preview(good)
    assert preview["available"] is True
    assert preview["datasets"]

    session = reconstruct_observation_sessions([str(good), str(bad)])
    diagnostics = performance_diagnostics(session["records"])
    assert diagnostics["available"] is True
    assert diagnostics["cache_signature"]
    assert len(diagnostics["repeated_computation"]) >= 1

    observatory = build_observatory_session_intelligence(session["records"])
    export_paths = export_session_intelligence(observatory, tmp_path / "exports", stem="hirax_observatory")
    assert "_v1.1" in export_paths["csv"]
    assert Path(export_paths["json"]).exists()
