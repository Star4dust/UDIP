import numpy as np
from typing import Dict, Any, Tuple, List
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks

def calibrate_signal(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Calibrate signal by removing baseline drift and normalizing.
    
    1. Remove baseline drift (rolling mean subtraction along time axis/axis 0)
    2. Normalize (zero mean, unit variance)
    """
    if data.size == 0:
        return data

    arr = np.asarray(data)
    dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
    calibrated = arr.astype(dtype, copy=True)
    window_size = max(1, min(int(window_size), calibrated.shape[0]))

    def _rolling_mean(channel: np.ndarray) -> np.ndarray:
        if channel.size == 0:
            return channel
        if window_size <= 1:
            return np.full_like(channel, np.mean(channel), dtype=dtype)
        kernel = np.ones(window_size, dtype=float) / window_size
        if np.iscomplexobj(channel):
            return (
                np.convolve(channel.real, kernel, mode="same")
                + 1j * np.convolve(channel.imag, kernel, mode="same")
            )
        return np.convolve(channel, kernel, mode="same")
    
    if data.ndim == 2:
        for i in range(data.shape[1]):
            channel = calibrated[:, i]
            calibrated[:, i] = channel - _rolling_mean(channel)
    elif data.ndim == 1:
        calibrated = calibrated - _rolling_mean(calibrated)
    else:
        flat = calibrated.reshape(calibrated.shape[0], -1)
        for i in range(flat.shape[1]):
            flat[:, i] = flat[:, i] - _rolling_mean(flat[:, i])
        calibrated = flat.reshape(calibrated.shape)

    # 2. Normalize
    mean = np.mean(calibrated)
    std = np.std(np.abs(calibrated)) if np.iscomplexobj(calibrated) else np.std(calibrated)
    if std > 0:
        calibrated = (calibrated - mean) / std
    
    return calibrated

def compute_fft(data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Compute Fast Fourier Transform (FFT) along the time axis (axis 0).
    """
    if data.size == 0:
        return {"fft": np.array([]), "frequencies": np.array([])}

    # Compute FFT along time axis (axis 0). Complex visibilities require the
    # full FFT so negative frequency content is preserved.
    n = data.shape[0]
    if np.iscomplexobj(data):
        fft_values = np.fft.fft(data, axis=0)
        frequencies = np.fft.fftfreq(n, d=1.0/sampling_rate)
    else:
        fft_values = np.fft.rfft(data, axis=0)
        frequencies = np.fft.rfftfreq(n, d=1.0/sampling_rate)
    
    magnitude = np.abs(fft_values)
    
    return {
        "fft_raw": fft_values,
        "magnitude": magnitude,
        "frequencies": frequencies
    }

def compute_power_spectrum(data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Compute Power Spectrum (FFT magnitude squared).
    Averages across channels if 2D.
    """
    fft_res = compute_fft(data, sampling_rate)
    power = fft_res["magnitude"] ** 2
    
    # If 2D (time, freq_channels), average across channels to get a single spectrum
    if power.ndim == 2:
        power_avg = np.mean(power, axis=1)
    else:
        power_avg = power
    
    return {
        "power": power,
        "power_avg": power_avg,
        "frequencies": fft_res["frequencies"]
    }
        
def apply_window(data: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """
    Apply a window function to the data before FFT.
    Supported: 'hann', 'hamming', 'blackman'
    """
    if data.size == 0:
        return data
        
    n = data.shape[0]
    if window_type == "hann":
        win = np.hanning(n)
    elif window_type == "hamming":
        win = np.hamming(n)
    elif window_type == "blackman":
        win = np.blackman(n)
    else:
        return data
        
    # Reshape window for broadcasting if data is 2D
    if data.ndim == 2:
        win = win[:, np.newaxis]
        
    return data * win

def detect_peaks(power: np.ndarray, frequencies: np.ndarray, threshold: float = 0.0) -> Dict[str, Any]:
    """
    Identify local maxima in the power spectrum using prominence.
    """
    from scipy.signal import find_peaks

    if power.ndim > 1:
        power = np.mean(power, axis=tuple(range(1, power.ndim)))
    
    # Simple normalization for peak finding
    norm_power = (power - np.min(power)) / (np.max(power) - np.min(power) + 1e-10)
    
    # Detect peaks with a minimum prominence
    peaks, properties = find_peaks(norm_power, prominence=0.1, height=threshold)
    
    peak_list = []
    for p in peaks:
        peak_list.append({
            "frequency": float(frequencies[p]),
            "power": float(power[p]),
            "index": int(p)
        })
        
    # Sort by power descending
    peak_list = sorted(peak_list, key=lambda x: x["power"], reverse=True)
    
    return {
        "peaks": peak_list,
        "indices": peaks.tolist(),
        "count": len(peak_list)
    }

def classify_signal(power: np.ndarray) -> Dict[str, Any]:
    """
    Heuristic classification with confidence scoring.
    """
    if power.size == 0:
        return {"label": "unknown", "confidence": 0.0}
        
    mean_p = np.mean(power)
    std_p = np.std(power)
    max_p = np.max(power)
    papr = max_p / (mean_p + 1e-10)
    rel_std = std_p / (mean_p + 1e-10)
    
    # Heuristic confidence based on distance from thresholds
    if papr > 20 and rel_std > 2:
        label = "interference"
        conf = min(1.0, (papr / 40.0) * (rel_std / 5.0))
    elif rel_std > 0.5:
        label = "structured_signal"
        conf = min(1.0, rel_std / 2.0)
    else:
        label = "noise"
        conf = min(1.0, 1.0 - (rel_std / 0.5))
        
    return {"label": label, "confidence": float(conf)}

def run_batch_analysis(results_list: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results from multiple pipeline runs.
    """
    if not results_list:
        return {}
        
    # Aggregate power spectra
    all_powers = [r["power_spectrum"]["power"] for r in results_list if "power_spectrum" in r]
    if all_powers:
        avg_power = np.mean(all_powers, axis=0)
    else:
        avg_power = np.array([])
        
    # Peak consistency
    all_peaks = []
    for r in results_list:
        if "detected_peaks" in r:
            all_peaks.extend([p["frequency"] for p in r["detected_peaks"]["peaks"]])
            
    # Histogram of peak frequencies to find consistent signals
    peak_hist = {}
    if all_peaks:
        for f in all_peaks:
            f_rounded = round(f, 2) # 2 Hz resolution
            peak_hist[f_rounded] = peak_hist.get(f_rounded, 0) + 1
            
    return {
        "average_power": avg_power,
        "peak_consistency": sorted(peak_hist.items(), key=lambda x: x[1], reverse=True)[:10],
        "total_files": len(results_list)
    }

def export_to_csv(data: np.ndarray, filename: str, columns: List[str] = None):
    import pandas as pd
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

def export_to_numpy(data: np.ndarray, filename: str):
    np.save(filename, data)


def render():
    """
    Render the HIRAX Analysis interface for signal intelligence.
    """
    st.markdown('<div class="main-header">🛸 Signal Intelligence</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Advanced signal analysis and intelligence for HIRAX telescope data.
    Perform FFT analysis, signal classification, and peak detection.
    """)
    
    # Sample data generation for demonstration
    if st.button("🔬 Generate Sample Signal Data", key="generate_sample_signal"):
        # Create sample signal with noise and interference
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        noise = 0.2 * np.random.randn(len(t))
        signal_with_noise = signal + noise
        
        st.session_state['sample_signal'] = signal_with_noise
        st.session_state['sample_time'] = t
        st.success("Sample signal data generated!")
    
    # Check if we have signal data
    if 'sample_signal' in st.session_state:
        signal_data = st.session_state['sample_signal']
        time_data = st.session_state['sample_time']
        
        # Analysis parameters
        st.markdown("### 🔧 Analysis Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            window_size = st.slider("Window Size", 10, 100, 50, 5)
            window_type = st.selectbox("Window Type", ["hann", "hamming", "blackman"])
        
        with col2:
            sampling_rate = st.slider("Sampling Rate (Hz)", 100, 1000, 500, 50)
            peak_threshold = st.slider("Peak Threshold", 0.0, 1.0, 0.1, 0.05)
        
        with col3:
            calibrate = st.checkbox("Apply Calibration", True)
            should_apply_window = st.checkbox("Apply Window", True)
        
        # Perform analysis
        if st.button("🚀 Run Signal Analysis", key="run_signal_analysis"):
            with st.spinner("Analyzing signal..."):
                # Step 1: Calibration
                if calibrate:
                    calibrated_signal = calibrate_signal(signal_data, window_size)
                else:
                    calibrated_signal = signal_data
                
                # Step 2: Apply window
                if should_apply_window:
                    windowed_signal = apply_window(calibrated_signal, window_type)
                else:
                    windowed_signal = calibrated_signal
                
                # Step 3: FFT Analysis
                fft_result = compute_fft(windowed_signal, sampling_rate)
                
                # Step 4: Power Spectrum
                power_spectrum = compute_power_spectrum(windowed_signal, sampling_rate)
                
                # Step 5: Peak Detection
                peaks = detect_peaks(
                    power_spectrum["power_avg"], 
                    power_spectrum["frequencies"], 
                    peak_threshold
                )
                
                # Step 6: Signal Classification
                classification = classify_signal(power_spectrum["power_avg"])
                
                # Store results in session state
                st.session_state['analysis_results'] = {
                    'calibrated_signal': calibrated_signal,
                    'windowed_signal': windowed_signal,
                    'fft_result': fft_result,
                    'power_spectrum': power_spectrum,
                    'peaks': peaks,
                    'classification': classification
                }
        
        # Display results
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            st.markdown("### 📊 Analysis Results")
            
            # Signal Classification
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Signal Type", 
                    results['classification']['label'].title(),
                    f"Confidence: {results['classification']['confidence']:.2f}"
                )
            
            with col2:
                st.metric("Detected Peaks", results['peaks']['count'])
            
            with col3:
                max_power = np.max(results['power_spectrum']['power_avg'])
                st.metric("Max Power", f"{max_power:.2e}")
            
            # Visualizations
            st.markdown("### 📈 Signal Visualizations")
            
            # Original vs Calibrated Signal
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=time_data,
                y=signal_data,
                name='Original Signal',
                line=dict(color='blue', width=1)
            ))
            fig1.add_trace(go.Scatter(
                x=time_data,
                y=results['calibrated_signal'],
                name='Calibrated Signal',
                line=dict(color='red', width=1)
            ))
            fig1.update_layout(
                title="Signal Calibration",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Power Spectrum with Peaks
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=results['power_spectrum']['frequencies'],
                y=results['power_spectrum']['power_avg'],
                name='Power Spectrum',
                line=dict(color='purple', width=2)
            ))
            
            # Add detected peaks
            if results['peaks']['peaks']:
                peak_freqs = [p['frequency'] for p in results['peaks']['peaks'][:10]]  # Top 10 peaks
                peak_powers = [p['power'] for p in results['peaks']['peaks'][:10]]
                
                fig2.add_trace(go.Scatter(
                    x=peak_freqs,
                    y=peak_powers,
                    mode='markers',
                    name='Detected Peaks',
                    marker=dict(color='red', size=8)
                ))
            
            fig2.update_layout(
                title="Power Spectrum Analysis",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Peak Details
            if results['peaks']['peaks']:
                st.markdown("### 🔍 Detected Peaks")
                
                peak_data = []
                for i, peak in enumerate(results['peaks']['peaks'][:20]):  # Top 20 peaks
                    peak_data.append({
                        'Rank': i + 1,
                        'Frequency (Hz)': f"{peak['frequency']:.2f}",
                        'Power': f"{peak['power']:.2e}",
                        'Index': peak['index']
                    })
                
                df_peaks = pd.DataFrame(peak_data)
                st.dataframe(df_peaks, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export Power Spectrum", key="export_power"):
                    export_data = np.column_stack([
                        results['power_spectrum']['frequencies'],
                        results['power_spectrum']['power_avg']
                    ])
                    export_to_numpy(export_data, "hirax_power_spectrum.npy")
                    st.success("Power spectrum exported to hirax_power_spectrum.npy")
            
            with col2:
                if st.button("📈 Export Peaks", key="export_peaks"):
                    peak_export = np.array([
                        [p['frequency'], p['power']] 
                        for p in results['peaks']['peaks']
                    ])
                    export_to_numpy(peak_export, "hirax_peaks.npy")
                    st.success("Peaks exported to hirax_peaks.npy")
            
            with col3:
                if st.button("📋 Export Summary", key="export_summary"):
                    summary = {
                        'classification': results['classification'],
                        'peak_count': results['peaks']['count'],
                        'max_power': float(np.max(results['power_spectrum']['power_avg'])),
                        'sampling_rate': sampling_rate,
                        'window_type': window_type
                    }
                    pd.DataFrame([summary]).to_csv("hirax_analysis_summary.csv", index=False)
                    st.success("Summary exported to hirax_analysis_summary.csv")
    
    else:
        st.info("👆 Click 'Generate Sample Signal Data' to begin analysis")
        
        # Feature descriptions
        st.markdown("### 🎯 Signal Intelligence Features")
        
        features = [
            ("🔬 Signal Calibration", "Remove baseline drift and normalize signals"),
            ("📊 FFT Analysis", "Fast Fourier Transform for frequency domain analysis"),
            ("📈 Power Spectrum", "Compute power spectral density"),
            ("🔍 Peak Detection", "Identify significant signal peaks"),
            ("🏷️ Signal Classification", "Classify signals as noise, structured, or interference"),
            ("💾 Export Capabilities", "Export results in multiple formats")
        ]
        
        for feature, description in features:
            st.markdown(f"**{feature}**: {description}")
        
        st.markdown("### 📚 Analysis Pipeline")
        
        pipeline_steps = [
            "1. **Signal Loading** - Load or generate signal data",
            "2. **Preprocessing** - Apply calibration and windowing",
            "3. **FFT Analysis** - Transform to frequency domain",
            "4. **Power Spectrum** - Compute power spectral density",
            "5. **Peak Detection** - Identify significant peaks",
            "6. **Signal Classification** - Classify signal characteristics",
            "7. **Results Export** - Save analysis results"
        ]
        
        for step in pipeline_steps:
            st.markdown(step)
