from __future__ import annotations

from pathlib import Path

from modules.observatory_operations import load_validation_corpus_manifest, run_validation_corpus_regression, benchmark_observatory_pipeline, hirax_instrument_profile, compare_observing_sessions, build_observatory_audit_trail, run_scientific_lockdown_audit
from modules.observation_session import reconstruct_observation_sessions
from modules.observatory_intelligence import build_observatory_session_intelligence


def test_validation_corpus_manifest_and_regression():
    manifest = load_validation_corpus_manifest()
    assert manifest['available'] is True
    assert manifest['dataset_count'] >= 8

    regression = run_validation_corpus_regression()
    assert regression['available'] is True
    assert not regression['results'].empty
    assert 'reproducibility_signature' in regression


def test_benchmark_and_export_trust(tmp_path):
    root = Path('/home/workspace/udip_v3/validation_corpus')
    files = [root / item['file'] for item in load_validation_corpus_manifest()['datasets'][:3]]
    bench = benchmark_observatory_pipeline(files, export_dir=tmp_path)
    assert bench['available'] is True
    assert not bench['timings'].empty
    assert bench['reproducibility_signature']
    assert Path(bench['comparison_export']['csv']).exists()
    assert Path(bench['comparison_export']['json']).exists()


def test_instrument_profile_and_comparison_audit():
    profile = hirax_instrument_profile()
    assert profile['available'] is True
    assert 'axis_naming_conventions' in profile

    root = Path('/home/workspace/udip_v3/validation_corpus')
    clean = root / 'clean_observation.h5'
    noisy = root / 'noisy_observation.h5'
    left = reconstruct_observation_sessions([str(clean)])
    right = reconstruct_observation_sessions([str(noisy)])
    left_obs = build_observatory_session_intelligence(left['records'])
    right_obs = build_observatory_session_intelligence(right['records'])
    comparison = compare_observing_sessions({'session_result': left, 'observatory': left_obs}, {'session_result': right, 'observatory': right_obs})
    assert comparison['available'] is True
    assert 'comparison' in comparison
    assert comparison['reproducibility_signature']

    audit = build_observatory_audit_trail(left, left_obs)
    assert audit['available'] is True
    assert not audit['events'].empty
    assert audit['signature']


def test_scientific_lockdown_audit():
    audit = run_scientific_lockdown_audit()
    assert audit['available'] is True
    assert 'locked_down' in audit
