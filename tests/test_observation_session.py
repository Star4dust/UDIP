from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from modules.observation_session import reconstruct_observation_sessions


def _write_session_file(path: Path, start_time: str, scale: float) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["observation_time"] = start_time
        f.attrs["target"] = "hirax field"
        vis = np.zeros((4, 8, 2), dtype=np.complex128)
        for t in range(vis.shape[0]):
            for c in range(vis.shape[1]):
                vis[t, c, 0] = scale * (1 + t + c) + 1j * scale * (0.1 * (t + c))
                vis[t, c, 1] = scale * (2 + t + c) + 1j * scale * (0.2 * (t + c))
        ds = f.create_dataset("visibilities", data=vis)
        ds.attrs["kind"] = "visibility_tensor"
        ds.attrs["axis_0"] = "time"
        ds.attrs["axis_1"] = "frequency"
        ds.attrs["axis_2"] = "baseline"


def test_reconstruct_observation_sessions_orders_and_combines(tmp_path):
    later = tmp_path / "session_b.h5"
    earlier = tmp_path / "session_a.h5"
    _write_session_file(later, "2026-05-19T08:15:00Z", 2.0)
    _write_session_file(earlier, "2026-05-19T08:00:00Z", 1.0)

    result = reconstruct_observation_sessions([str(later), str(earlier)])

    assert result["consistency_report"]["is_consistent"] is True
    assert result["consistency_report"]["ordered_times"] is True
    assert len(result["records"]) == 2
    assert result["records_summary"][0]["file_name"] == "session_a.h5"
    assert result["timeline"].shape[0] == 2
    assert result["combined_waterfall"].shape[0] > 0
    assert result["combined_waterfall"].shape[1] == result["waterfall_frequency_bins"]
    assert result["weighted_fusion"]["available"] is True
    assert result["weighted_fusion"]["visible_shape_matched"] is True
    assert result["weighted_fusion"]["fused_visibility"].shape == (4, 8, 2)
    assert result["weighted_fusion"]["fused_waterfall"].shape[1] == result["waterfall_frequency_bins"]
    assert not result["errors"]
    assert result["warnings"]
    assert any("missing uv coordinates" in warning for warning in result["warnings"])


def test_reconstruct_observation_sessions_handles_missing_visibility_dataset(tmp_path):
    file_path = tmp_path / "non_visibility.h5"
    with h5py.File(file_path, "w") as f:
        f.attrs["observation_time"] = "2026-05-19T09:00:00Z"
        f.create_dataset("metadata", data=np.arange(4))

    result = reconstruct_observation_sessions([str(file_path)])

    assert result["records"] == []
    assert result["consistency_report"]["session_count"] == 0
    assert result["weighted_fusion"]["available"] is False
