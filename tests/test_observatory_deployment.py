from __future__ import annotations

from pathlib import Path

from modules.observatory_deployment import (
    benchmark_reproducibility_check,
    daily_observation_comparison,
    deployment_readiness_report,
    export_reproducibility_check,
    hardware_profile_estimate,
    lockdown_audit_report,
    observation_health_timelines,
    quarantine_manifest_report,
    real_hardware_baseline_benchmark,
    scientific_alert_report,
    session_consensus_engine,
    streaming_reconstruct_observation_sessions,
)


def _corpus_files(*names: str):
    root = Path('/home/workspace/udip_v3/validation_corpus')
    return [root / name for name in names]


def test_streaming_and_deployment_readiness():
    files = _corpus_files('clean_observation.h5', 'noisy_observation.h5', 'rfi_heavy_session.h5', 'missing_uv_geometry.h5')
    streaming = streaming_reconstruct_observation_sessions(files, batch_size=2, use_cache=False)
    assert streaming['available'] is True
    assert streaming['streaming_mode'] is True
    assert streaming['batch_count'] == 2
    assert streaming['memory_budget']['available'] is True
    assert streaming['quarantine_manifest']['schema_version']

    cached = streaming_reconstruct_observation_sessions(files, batch_size=2, use_cache=True)
    cached_again = streaming_reconstruct_observation_sessions(files, batch_size=2, use_cache=True)
    assert cached['stream_signature'] == cached_again['stream_signature']

    report = deployment_readiness_report(files, batch_size=2)
    assert report['available'] is True
    assert report['schema_version'] == '2.1'
    assert 0.0 <= report['deployment_readiness'] <= 1.0
    assert Path(report['comparison_export']['csv']).exists()
    assert Path(report['comparison_export']['json']).exists()


def test_comparison_and_timeline_consistency():
    left = _corpus_files('clean_observation.h5', 'noisy_observation.h5')
    right = _corpus_files('rfi_heavy_session.h5', 'partial_calibration_drift.h5')
    comparison = daily_observation_comparison(left, right, batch_size=2)
    assert comparison['available'] is True
    assert comparison['schema_version']
    assert 'uv_completeness_delta' in comparison['comparison']

    files = _corpus_files('clean_observation.h5', 'partial_calibration_drift.h5', 'rfi_heavy_session.h5')
    streaming = streaming_reconstruct_observation_sessions(files, batch_size=2)
    consensus = session_consensus_engine(streaming['session_result'], streaming['observatory'])
    health = observation_health_timelines(streaming['observatory'])
    alerts = scientific_alert_report(streaming['session_result'], streaming['observatory'])
    assert consensus['available'] is True
    assert health['available'] is True
    assert alerts['available'] is True


def test_reproducibility_hardware_and_lockdown():
    files = _corpus_files('clean_observation.h5', 'noisy_observation.h5', 'rfi_heavy_session.h5')
    bench = benchmark_reproducibility_check(files)
    export = export_reproducibility_check(files)
    hardware = hardware_profile_estimate(files)
    real_benchmark = real_hardware_baseline_benchmark(files)
    quarantine = quarantine_manifest_report()
    lockdown = lockdown_audit_report()
    assert bench['available'] is True
    assert export['available'] is True
    assert hardware['available'] is True
    assert real_benchmark['available'] is True
    assert quarantine['available'] is True
    assert lockdown['available'] is True
    assert 'locked_down' in lockdown
