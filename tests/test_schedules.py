"""Schedule persistence round-trip (regression test for missing load/save_schedules)."""
import modules.core as core


def test_schedules_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA_DIR", tmp_path)
    assert core.load_schedules() == []
    sample = [{"id": "abc", "name": "nightly export", "frequency": "daily"}]
    core.save_schedules(sample)
    assert core.load_schedules() == sample


def test_scheduler_module_imports():
    import modules.scheduler  # noqa: F401  (used to fail on missing core names)
