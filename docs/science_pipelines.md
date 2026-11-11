# UDIP v3 Scientific Pipelines

This document details each implemented scientific analysis pipeline, including inputs, processing steps, outputs, and responsible modules.

---

## Pipeline: Interferometry & UV Coverage Analysis

### Overview
Analyzes spatial frequency (UV-plane) coverage from baseline coordinates. Critical for determining imaging fidelity and sensitivity.

### Inputs
- **u_coords**: Baseline U coordinates (spatial frequency, wavelengths)
- **v_coords**: Baseline V coordinates (spatial frequency, wavelengths)
- **Optional**: Time array for time-dependent analysis

### Processing Steps
1. **Generate UV Coverage** (`modules/radio_interferometry.generate_uv_coverage()`)
   - Compute 2D histogram of (u, v) points (density map)
   - Calculate coverage statistics:
     - n_uv_points: number of unique baselines
     - coverage_density: 2D numpy array (50×50 grid)
     - density_mean: average coverage
     - density_max: peak coverage

2. **Compute PSF** (`modules/radio_interferometry.compute_psf()`)
   - Point Spread Function from UV coverage
   - Indicates angular resolution and beam shape

3. **Baseline Analysis** (`modules/radio_interferometry.analyze_baseline_distribution()`)
   - Distribution of baselines in UV plane
   - Radial and angular coverage metrics

### Outputs
```python
{
  "n_uv_points": int,
  "coverage_density": np.ndarray,           # 2D histogram
  "density_mean": float,
  "density_max": float,
  "psf": {"peak": float, "sidelobe_ratio": float},
  "baseline_stats": {
    "min_baseline": float,
    "max_baseline": float,
    "mean_baseline": float
  }
}
```

### Module Responsible
- **Primary**: `modules/radio_interferometry.py`
- **UI Rendering**: `pages/visualize.py::_render_uv_coverage()`, `pages/hirax_inspector.py`
- **Orchestration**: `modules/observatory_intelligence.py`

### Pipeline Runner Action
```python
# In execution_plan:
{"action": "create_visualization", "type": "uv_coverage"}
# OR direct call:
from modules.radio_interferometry import generate_uv_coverage
result = generate_uv_coverage(u_coords, v_coords)
```

---

## Pipeline: RFI Detection & Cleaning

### Overview
Identifies and mitigates Radio Frequency Interference (RFI) that contaminates visibility data.

### Inputs
- **visibility_data**: Complex-valued visibility matrix (freq × time or freq × baseline)
- **threshold**: Detection threshold (default 3.0 σ)

### Processing Steps

#### Step 1: RFI Detection
```
modules/hirax_processing.detect_rfi(data, threshold=3.0)
```
- Per-frequency, per-time (or per-baseline) statistics
- Compute mean and standard deviation
- Flag points exceeding threshold × σ
- Return: binary mask (True = RFI, False = clean)

#### Step 2: RFI Cleaning
```
modules/hirax_processing.clean_rfi(data, mask, method="interpolate")
```
Supported methods:
- **"interpolate"**: Replace RFI with interpolated neighbors
- **"mask"**: Set to zero
- **"remove"**: Delete time/frequency slots

#### Step 3: Validation
- Compare before/after statistics
- Estimate percentage of data affected
- Log to history

### Outputs
```python
{
  "success": bool,
  "rfi_mask": np.ndarray,                  # Boolean array
  "percentage_flagged": float,              # % of data affected
  "cleaned_data": np.ndarray,               # Cleaned visibility data
  "method_used": str                        # "interpolate" | "mask" | "remove"
}
```

### Module Responsible
- **Detection/Cleaning**: `modules/hirax_processing.py`
- **Integration**: `modules/pipeline_runner.py` (actions: `detect_rfi`, `clean_rfi`)

### Pipeline Runner Actions
```python
# Step 1: Detect
{"action": "detect_rfi", "threshold": 3.5}
# Step 2: Clean
{"action": "clean_rfi", "method": "interpolate"}
```

### Session State Side Effects
- `st.session_state["hirax_rfi_mask"]` ← RFI mask for reuse
- `st.session_state["hirax_cleaned_data"]` ← Cleaned visibility data

---

## Pipeline: FFT & Power Spectrum Analysis

### Overview
Transforms visibility data from time/frequency domain to frequency domain (spectral analysis).

### Inputs
- **time_series_data**: Visibility data (1D array or 2D matrix)
- **sampling_rate**: Sampling frequency (Hz, default 1.0)
- **window_type**: Window function ("hann", "hamming", "blackman", "none")

### Processing Steps

#### Step 1: Apply Window Function
```
modules/hirax_analysis.apply_window(data, window_type="hann")
```
- Reduces spectral leakage
- Windowed data = data × window_coefficients

#### Step 2: Compute FFT
```
modules/hirax_analysis.compute_fft(data, sampling_rate=1.0)
```
- `numpy.fft.fft()` on 1D or per-column of 2D
- Return: complex-valued frequency spectrum

#### Step 3: Power Spectrum
```
modules/hirax_analysis.compute_power_spectrum(data, sampling_rate=1.0)
```
- Power = |FFT|²
- Frequently applies log-scale for visualization
- Optional averaging over time/baseline

### Outputs
```python
# FFT result:
{
  "frequencies": np.ndarray,                # Frequency bins (Hz)
  "spectrum": np.ndarray,                   # Complex FFT
  "power": np.ndarray                       # |spectrum|²
}

# Power spectrum result:
{
  "frequencies": np.ndarray,
  "power": np.ndarray,                      # Power (linear or log)
  "peak_frequency": float,
  "peak_power": float
}
```

### Module Responsible
- **Computation**: `modules/hirax_analysis.py`
- **Integration**: `modules/pipeline_runner.py` (actions: `apply_window`, `compute_fft`, `power_spectrum`)

### Pipeline Runner Actions
```python
# Full workflow:
{"action": "apply_window", "type": "hann"}
{"action": "compute_fft", "sampling_rate": 1.0}
{"action": "power_spectrum", "sampling_rate": 1.0}
```

### Session State Side Effects
- `st.session_state["hirax_windowed_data"]`
- `st.session_state["hirax_fft_result"]`
- `st.session_state["hirax_power_spectrum"]`

---

## Pipeline: Signal Calibration

### Overview
Corrects for instrumental effects, gain variations, and phase offsets.

### Inputs
- **visibility_data**: Raw or partially-cleaned visibilities
- **window_size**: Calibration window (time or frequency bins, default 50)
- **Optional**: Reference source or known gain curve

### Processing Steps

#### Step 1: Window-Based Statistics
- Divide data into sliding windows
- Compute gain and phase per window

#### Step 2: Correction
- Divide by estimated gain
- Subtract phase offset

#### Step 3: Validation
- Compare before/after amplitude distribution
- Check for overcorrection

### Outputs
```python
{
  "calibrated_data": np.ndarray,
  "gain_correction": np.ndarray,
  "phase_correction": np.ndarray,
  "residual_rms": float
}
```

### Module Responsible
- **Calibration**: `modules/hirax_analysis.calibrate_signal()`
- **Integration**: `modules/pipeline_runner.py` (action: `calibrate_signal`)

### Pipeline Runner Action
```python
{"action": "calibrate_signal", "window": 50}
```

### Session State Side Effects
- `st.session_state["hirax_calibrated_data"]`

---

## Pipeline: Peak Detection & Signal Classification

### Overview
Identifies spectral features (peaks, lines) and classifies signal types.

### Inputs
- **power_spectrum**: Already computed via FFT → power_spectrum pipeline
- **threshold**: Detection threshold relative to noise floor (default 0.0)

### Processing Steps

#### Step 1: Peak Detection
```
modules/hirax_analysis.detect_peaks(power, frequencies, threshold=0.0)
```
- Find local maxima above threshold
- Return: peak frequencies, peak powers, peak widths

#### Step 2: Signal Classification
```
modules/hirax_analysis.classify_signal(power)
```
- Classify by spectral shape:
  - **Broadband**: flat spectrum (RFI or Galactic)
  - **Narrowband**: sharp peaks (OH, H-alpha, etc.)
  - **Line**: single strong feature (pulsar, maser, astrophysical)
  - **Continuum**: smooth, extended

### Outputs
```python
# Peak detection:
{
  "peak_frequencies": np.ndarray,
  "peak_powers": np.ndarray,
  "peak_widths": np.ndarray,
  "n_peaks": int
}

# Classification:
{
  "signal_type": str,                       # broadband|narrowband|line|continuum
  "confidence": float,                      # 0.0–1.0
  "spectral_features": List[Dict]           # [{freq, power, type}, ...]
}
```

### Module Responsible
- **Analysis**: `modules/hirax_analysis.py`
- **Integration**: `modules/pipeline_runner.py` (actions: `detect_peaks`, `classify_signal`)

### Pipeline Runner Actions
```python
# Requires power_spectrum to be run first:
{"action": "detect_peaks", "threshold": 0.0}
{"action": "classify_signal"}
```

---

## End-to-End HIRAX Analysis Workflow

### Complete Pipeline Example

![UDIP scientific pipelines diagram](./diagrams/science_pipelines.svg)

### Execution Plan
```python
{
  "execution_plan": [
    {"action": "load_dataset", "file_path": "hirax_observation.h5"},
    {"action": "detect_rfi", "threshold": 3.5},
    {"action": "clean_rfi", "method": "interpolate"},
    {"action": "calibrate_signal", "window": 50},
    {"action": "apply_window", "type": "hann"},
    {"action": "compute_fft", "sampling_rate": 1.0},
    {"action": "power_spectrum", "sampling_rate": 1.0},
    {"action": "detect_peaks", "threshold": 0.0},
    {"action": "classify_signal"},
    {"action": "generate_report", "format": "html"}
  ]
}
```

---

## Scientific Module Dependency Map

```
┌─ visibility data (HDF5)
│
├─> hirax_processing.py (RFI detection/cleaning)
├─> hirax_analysis.py (calibration, FFT, windowing, peaks, classification)
├─> radio_interferometry.py (UV coverage, PSF, baselines)
│
├─> observable_intelligence.py (signal intelligence, metrics, guidance)
│
└─> execution_layer.py (log, save, statistics)
```

---

## Performance Characteristics

| Pipeline | Typical Input Size | Processing Time | Memory Peak |
|----------|-------------------|-----------------|-------------|
| UV Coverage | 1M baselines | 100–500 ms | 200 MB |
| RFI Detection | 100k×1k (freq×time) | 50–200 ms | 100 MB |
| FFT | 1D array (65k points) | 10–50 ms | 50 MB |
| Calibration | 100k×1k | 100–300 ms | 150 MB |
| Signal Classification | 65k spectrum | 5–20 ms | 30 MB |

---

## Known Limitations

1. **UV Coverage**: Assumes 2D UV plane; 3D (with W term) not yet supported
2. **RFI**: Simple sigma-clipping; no adaptive or morphological methods
3. **FFT**: 1D only; 2D spectrograms not yet implemented
4. **Calibration**: Window-based; no polynomial fitting
5. **Classification**: Rule-based; ML-based classifier planned

---

## Future Enhancements

- [ ] CLEAN imaging pipeline (interferometric deconvolution)
- [ ] 21 cm cosmology analysis (HI survey pipelines)
- [ ] Pulsar search (folding, periodogram)
- [ ] Multi-baseline phase calibration
- [ ] Real-time RFI monitoring (streaming mode)

