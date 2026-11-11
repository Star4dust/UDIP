import numpy as np
import pandas as pd
import pytest

from modules.core import _prepare_dataframe_for_storage, _restore_complex_columns, read_scientific_hdf5
from modules.radio_interferometry import (
    compute_visibility_statistics,
    compute_baseline_lengths,
    generate_uv_coverage,
    compute_point_spread_function,
    compute_gain_solution,
    apply_gains,
    solve_phase_calibration,
    apply_phase_calibration,
    solve_bandpass,
    apply_bandpass,
    estimate_delay_phase_slope,
    apply_delay_correction,
    hogbom_clean,
    validate_interferometry_data,
)
from modules.hirax_processing import detect_rfi, clean_rfi
from modules.hirax_analysis import calibrate_signal
from modules.hirax_analysis import compute_fft, compute_power_spectrum
from modules.hirax_validation import generate_synthetic_signal, validate_spectral_analysis
from modules.cosmology_21cm import frequency_to_redshift, construct_hydrogen_cube, pca_foreground_clean, estimate_power_spectrum


@pytest.fixture()
def hirax_like_visibility_fixture():
    time = 4
    freq = 3
    baseline = 5
    pol = 2
    t = np.arange(time)[:, None, None, None]
    f = np.arange(freq)[None, :, None, None]
    b = np.arange(baseline)[None, None, :, None]
    p = np.arange(pol)[None, None, None, :]
    real = 1.0 + 0.1 * t + 0.01 * f + 0.001 * b
    imag = 0.5 + 0.05 * p + 0.02 * b
    vis = real + 1j * imag
    u = np.linspace(-20, 20, baseline)
    v = np.linspace(10, -10, baseline)
    return vis, u, v


@pytest.fixture()
def hdf5_file(tmp_path):
    import h5py
    path = tmp_path / 'hirax_fixture.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['telescope'] = 'HIRAX'
        obs = f.create_group('observation')
        vis = obs.create_dataset('visibilities', data=np.arange(24, dtype=np.complex128).reshape(2, 3, 4) + 1j)
        vis.attrs['axis_order'] = 'time,freq,baseline'
        obs.create_dataset('uv_coordinates', data=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
        obs.create_dataset('baselines', data=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64))
        obs.create_dataset('frequencies', data=np.array([700.0, 705.0, 710.0], dtype=np.float64))
        obs.create_dataset('polarization', data=np.array([b'XX', b'YY']))
        obs.create_dataset('timestamps', data=np.array([0.0, 1.0], dtype=np.float64))
    return path


@pytest.fixture()
def source_dataframe():
    return pd.DataFrame(
        {
            'vis': np.array([1 + 2j, 3 + 4j, 5 + 6j]),
            'u': [0.0, 1.0, 2.0],
            'v': [2.0, 1.0, 0.0],
            'freq_mhz': [700.0, 705.0, 710.0],
            'pol': ['XX', 'YY', 'XY'],
            'time_s': [0.0, 1.0, 2.0],
        }
    )


def test_complex_visibility_round_trip(source_dataframe):
    meta = {'name': 'fixture'}
    saved = _prepare_dataframe_for_storage(source_dataframe, meta)
    restored = _restore_complex_columns(saved, meta)
    assert np.allclose(restored['vis'].to_numpy(), source_dataframe['vis'].to_numpy())


def test_hdf5_metadata_round_trip(hdf5_file):
    class F:
        name = hdf5_file.name
        def read(self):
            return hdf5_file.read_bytes()
    result = read_scientific_hdf5(F())
    assert result['metadata']['attributes']['telescope'] == 'HIRAX'
    assert 'observation/visibilities' in result['datasets']
    assert result['datasets']['observation/visibilities']['shape'] == (2, 3, 4)
    assert result['datasets']['observation/visibilities']['dtype'].startswith('complex')
    assert 'observation/uv_coordinates' in result['datasets']


def test_visibility_and_baseline_validation(hirax_like_visibility_fixture):
    vis, u, v = hirax_like_visibility_fixture
    stats = compute_visibility_statistics(vis)
    assert 'amplitude' in stats and stats['amplitude']['max'] > stats['amplitude']['min']
    lengths = compute_baseline_lengths(np.column_stack([u, v]))
    assert lengths.shape == (len(u),)
    uv = generate_uv_coverage(u, v)
    assert uv['n_uv_points'] == len(u)
    validation = validate_interferometry_data(vis, u, v)
    assert validation['is_valid'] is True
    assert validation['visibility_shape'] == vis.shape
    assert validation['baseline_axis_length'] == vis.shape[2]


def test_imaging_and_cleaning_consistency(hirax_like_visibility_fixture):
    vis, u, v = hirax_like_visibility_fixture
    collapsed = vis.mean(axis=(1, 3))
    dirty = compute_point_spread_function(u, v, image_size=32)
    assert np.isclose(dirty.max(), 1.0)
    image = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.ifftshift(np.fft.fftshift(np.zeros((32, 32), dtype=complex))))))
    clean = hogbom_clean(np.real(np.eye(32)), dirty, gain=0.1, threshold=0.001, max_iterations=10)
    assert clean['residual'].shape == (32, 32)
    assert clean['components'].shape == (32, 32)
    assert clean['iterations'] >= 1


def test_calibration_invariants_and_bandpass(hirax_like_visibility_fixture):
    vis, u, v = hirax_like_visibility_fixture
    model = vis * (1 + 0j)
    gains = compute_gain_solution(vis, model)
    corrected = apply_gains(vis, gains)
    assert np.allclose(corrected, model)

    phase_solution = solve_phase_calibration(vis, model)
    phase_corrected = apply_phase_calibration(vis, phase_solution)
    assert np.allclose(np.abs(phase_corrected), np.abs(vis))

    bandpass = solve_bandpass(vis, axis=1)
    bp_corrected = apply_bandpass(vis, bandpass, axis=1)
    assert bp_corrected.shape == vis.shape

    freqs = np.array([700.0, 705.0, 710.0])
    delay = estimate_delay_phase_slope(vis[:, :, :, 0], freqs, axis=1)
    delayed = apply_delay_correction(vis[:, :, :, 0], freqs, delay, axis=1)
    assert delayed.shape == vis[:, :, :, 0].shape


def test_rfi_reproducibility_and_spectral_validation():
    data = generate_synthetic_signal(128, 4, signals=[{'freq': 12.0, 'amp': 2.0}], noise_level=0.05, rng_seed=7)
    r1 = detect_rfi(data, threshold=3.0, method='mad')
    r2 = detect_rfi(data, threshold=3.0, method='mad')
    assert np.array_equal(r1['mask'], r2['mask'])
    cleaned = clean_rfi(data, r1['mask'], method='median')
    assert cleaned.shape == data.shape
    calibrated = calibrate_signal(cleaned, window_size=9)
    assert calibrated.shape == data.shape
    validation = validate_spectral_analysis(data, sampling_rate=128.0)
    assert 'power_spectrum' in validation


def test_fft_and_cosmology_dimensions():
    data = np.arange(64, dtype=float).reshape(16, 4)
    fft = compute_fft(data, sampling_rate=16.0)
    power = compute_power_spectrum(data, sampling_rate=16.0)
    assert fft['magnitude'].shape[0] == 9
    assert power['power_avg'].shape[0] == 9

    freqs = np.array([700.0, 705.0, 710.0])
    z = frequency_to_redshift(freqs)
    assert z.shape == freqs.shape

    df = pd.DataFrame({'f700': np.arange(6), 'f705': np.arange(6) * 2, 'f710': np.arange(6) * 3})
    cube = construct_hydrogen_cube(df, ['f700', 'f705', 'f710'])
    cleaned = pca_foreground_clean(cube, n_components=1)
    ps = estimate_power_spectrum(cleaned['cleaned_cube'])
    assert cube.shape == (6, 3, 1)
    assert cleaned['cleaned_cube'].shape == cube.shape
    assert ps['power_cube'].shape == cube.shape
