import numpy as np
from typing import Dict, Any, List, Tuple
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.hirax_analysis import compute_power_spectrum, detect_peaks

def generate_synthetic_signal(n_samples: int, n_channels: int, 
                             signals: List[Dict[str, float]] = None, 
                             noise_level: float = 0.1,
                             rng_seed: int | None = 0) -> np.ndarray:
    """
    Generate synthetic HIRAX-like signal for validation.
    signals: List of {'freq': f, 'amp': a}
    """
    time = np.linspace(0, 1, n_samples)
    rng = np.random.default_rng(rng_seed)
    data = rng.normal(0, noise_level, (n_samples, n_channels))
    
    if signals:
        for s in signals:
            freq = s.get('freq', 10.0)
            amp = s.get('amp', 1.0)
            # Inject signal into all channels (simplified)
            # In real data, it might vary by channel (frequency)
            signal_wave = amp * np.sin(2 * np.pi * freq * time)
            for c in range(n_channels):
                data[:, c] += signal_wave
                
    return data

def measure_detection_accuracy(data: np.ndarray, ground_truth_freqs: List[float], 
                              sampling_rate: float = 1.0, tolerance: float = 0.5) -> Dict[str, Any]:
    """
    Measure how many ground truth frequencies were correctly detected.
    """
    p_res = compute_power_spectrum(data, sampling_rate=sampling_rate)
    peaks_res = detect_peaks(p_res["power"], p_res["frequencies"])
    detected_freqs = [p["frequency"] for p in peaks_res["peaks"]]
    
    matches = []
    for gt in ground_truth_freqs:
        found = any(abs(gt - df) <= tolerance for df in detected_freqs)
        matches.append(found)
        
    accuracy = sum(matches) / len(ground_truth_freqs) if ground_truth_freqs else 0.0
    
    return {
        "accuracy": accuracy,
        "true_positives": sum(matches),
        "total_expected": len(ground_truth_freqs),
        "false_positives": max(0, len(detected_freqs) - sum(matches))
    }


def validate_spectral_analysis(data: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """
    Comprehensive validation of spectral analysis pipeline.
    """
    # Compute power spectrum
    power_result = compute_power_spectrum(data, sampling_rate)
    
    # Detect peaks
    peaks_result = detect_peaks(power_result["power_avg"], power_result["frequencies"])
    
    # Spectral quality metrics
    power_array = power_result["power_avg"]
    if power_array.size == 0:
        return {
            "power_spectrum": power_result,
            "detected_peaks": peaks_result,
            "snr": 0.0,
            "spectral_flatness": 0.0,
            "peak_prominence": 0.0,
            "quality_score": 0.0,
        }

    snr = np.max(power_array) / (np.mean(power_array) + 1e-10)
    
    # Spectral flatness (measure of noise vs tonal content)
    geometric_mean = np.exp(np.mean(np.log(power_array + 1e-10)))
    arithmetic_mean = np.mean(power_array)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    # Peak prominence analysis
    peak_powers = [p["power"] for p in peaks_result["peaks"]]
    peak_prominence = np.std(peak_powers) / (np.mean(peak_powers) + 1e-10) if peak_powers else 0
    
    return {
        "power_spectrum": power_result,
        "detected_peaks": peaks_result,
        "snr": snr,
        "spectral_flatness": spectral_flatness,
        "peak_prominence": peak_prominence,
        "quality_score": min(1.0, (snr / 10.0) * (1.0 - spectral_flatness) * min(1.0, peak_prominence))
    }


def run_validation_suite(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run comprehensive validation suite on multiple test cases.
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        ground_truth = test_case.get("ground_truth", [])
        sampling_rate = test_case.get("sampling_rate", 1.0)
        
        # Run validation
        validation_result = validate_spectral_analysis(data, sampling_rate)
        
        # Measure accuracy if ground truth provided
        if ground_truth:
            accuracy_result = measure_detection_accuracy(data, ground_truth, sampling_rate)
            validation_result.update(accuracy_result)
        
        validation_result["test_case_id"] = i
        results.append(validation_result)
    
    # Aggregate statistics
    if results:
        avg_accuracy = np.mean([r.get("accuracy", 0) for r in results])
        avg_snr = np.mean([r["snr"] for r in results])
        avg_quality = np.mean([r["quality_score"] for r in results])
    else:
        avg_accuracy = avg_snr = avg_quality = 0.0
    
    return {
        "individual_results": results,
        "summary": {
            "total_tests": len(results),
            "average_accuracy": avg_accuracy,
            "average_snr": avg_snr,
            "average_quality_score": avg_quality
        }
    }


def render():
    """
    Render the HIRAX Validation interface for spectral analysis validation.
    """
    st.markdown('<div class="main-header">🌈 Spectral Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive validation and testing suite for HIRAX spectral analysis.
    Generate test signals, validate detection accuracy, and analyze spectral quality.
    """)
    
    # Validation mode selection
    st.markdown("### 🔬 Validation Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        validation_mode = st.selectbox(
            "Select Validation Mode",
            ["Synthetic Signal Testing", "Quality Assessment", "Batch Validation"]
        )
    
    with col2:
        sampling_rate = st.slider("Sampling Rate (Hz)", 100, 1000, 500, 50)
    
    if validation_mode == "Synthetic Signal Testing":
        st.markdown("### 🧪 Synthetic Signal Generation")
        
        # Signal parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.slider("Signal Length", 100, 5000, 1000, 100)
            n_channels = st.slider("Number of Channels", 1, 32, 4, 1)
        
        with col2:
            noise_level = st.slider("Noise Level", 0.01, 1.0, 0.1, 0.01)
            n_signals = st.slider("Number of Test Signals", 1, 10, 3, 1)
        
        with col3:
            min_freq = st.slider("Min Frequency (Hz)", 10, 100, 20, 5)
            max_freq = st.slider("Max Frequency (Hz)", 50, 400, 200, 10)
        
        # Generate test signals
        if st.button("🎲 Generate Test Signals", key="generate_test_signals"):
            test_signals = []
            for i in range(n_signals):
                freq = np.random.uniform(min_freq, max_freq)
                amp = np.random.uniform(0.5, 2.0)
                test_signals.append({"freq": freq, "amp": amp})
            
            synthetic_data = generate_synthetic_signal(
                n_samples, n_channels, test_signals, noise_level
            )
            
            st.session_state['validation_data'] = synthetic_data
            st.session_state['test_signals'] = test_signals
            st.session_state['ground_truth_freqs'] = [s["freq"] for s in test_signals]
            
            st.success(f"Generated {n_signals} test signals with {n_channels} channels")
    
    elif validation_mode == "Quality Assessment":
        st.markdown("### 📊 Spectral Quality Assessment")
        
        if st.button("🔍 Analyze Current Data", key="analyze_current"):
            if 'validation_data' in st.session_state:
                data = st.session_state['validation_data']
                
                with st.spinner("Analyzing spectral quality..."):
                    quality_result = validate_spectral_analysis(data, sampling_rate)
                    st.session_state['quality_result'] = quality_result
            else:
                st.warning("Please generate test data first")
    
    elif validation_mode == "Batch Validation":
        st.markdown("### 🔄 Batch Validation Suite")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_test_cases = st.slider("Number of Test Cases", 5, 50, 10, 5)
        
        with col2:
            complexity = st.selectbox("Test Complexity", ["Simple", "Medium", "Complex"])
        
        if st.button("🚀 Run Batch Validation", key="run_batch_validation"):
            with st.spinner("Running validation suite..."):
                test_cases = []
                
                for i in range(n_test_cases):
                    # Generate varied test cases
                    n_sig = 1 if complexity == "Simple" else (3 if complexity == "Medium" else 5)
                    noise = 0.05 if complexity == "Simple" else (0.2 if complexity == "Medium" else 0.5)
                    
                    signals = []
                    ground_truth = []
                    
                    for j in range(n_sig):
                        freq = np.random.uniform(20, 300)
                        amp = np.random.uniform(0.5, 2.0)
                        signals.append({"freq": freq, "amp": amp})
                        ground_truth.append(freq)
                    
                    data = generate_synthetic_signal(1000, 4, signals, noise)
                    
                    test_cases.append({
                        "data": data,
                        "ground_truth": ground_truth,
                        "sampling_rate": sampling_rate
                    })
                
                batch_results = run_validation_suite(test_cases)
                st.session_state['batch_results'] = batch_results
    
    # Display results
    if 'validation_data' in st.session_state:
        data = st.session_state['validation_data']
        
        st.markdown("### 📈 Signal Visualization")
        
        # Time domain plot
        time_axis = np.linspace(0, data.shape[0] / sampling_rate, data.shape[0])
        
        fig1 = go.Figure()
        for i in range(min(4, data.shape[1])):  # Show first 4 channels
            fig1.add_trace(go.Scatter(
                x=time_axis,
                y=data[:, i],
                name=f'Channel {i+1}',
                line=dict(width=1)
            ))
        
        fig1.update_layout(
            title="Time Domain Signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Validation results
        if 'quality_result' in st.session_state:
            result = st.session_state['quality_result']
            
            st.markdown("### 🎯 Validation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("SNR", f"{result['snr']:.2f}")
            
            with col2:
                st.metric("Spectral Flatness", f"{result['spectral_flatness']:.3f}")
            
            with col3:
                st.metric("Peak Prominence", f"{result['peak_prominence']:.2f}")
            
            with col4:
                st.metric("Quality Score", f"{result['quality_score']:.3f}")
            
            # Power spectrum
            power_data = result['power_spectrum']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=power_data['frequencies'],
                y=power_data['power_avg'],
                name='Power Spectrum',
                line=dict(color='purple', width=2)
            ))
            
            # Add detected peaks
            if result['detected_peaks']['peaks']:
                peak_freqs = [p['frequency'] for p in result['detected_peaks']['peaks'][:10]]
                peak_powers = [p['power'] for p in result['detected_peaks']['peaks'][:10]]
                
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
            
            # Detection accuracy if ground truth available
            if 'ground_truth_freqs' in st.session_state and 'accuracy' in result:
                st.markdown("### 🎯 Detection Accuracy")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                
                with col2:
                    st.metric("True Positives", result['true_positives'])
                
                with col3:
                    st.metric("False Positives", result['false_positives'])
                
                # Ground truth vs detected
                gt_freqs = st.session_state['ground_truth_freqs']
                detected_freqs = [p['frequency'] for p in result['detected_peaks']['peaks']]
                
                comparison_data = []
                for gt in gt_freqs:
                    closest = min(detected_freqs, key=lambda x: abs(x - gt)) if detected_freqs else None
                    match = closest is not None and abs(gt - closest) <= 0.5
                    
                    comparison_data.append({
                        'Ground Truth (Hz)': f"{gt:.2f}",
                        'Detected (Hz)': f"{closest:.2f}" if closest else "Not detected",
                        'Match': "✅" if match else "❌"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
        
        # Batch validation results
        if 'batch_results' in st.session_state:
            batch = st.session_state['batch_results']
            summary = batch['summary']
            
            st.markdown("### 📊 Batch Validation Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tests", summary['total_tests'])
            
            with col2:
                st.metric("Avg Accuracy", f"{summary['average_accuracy']:.2%}")
            
            with col3:
                st.metric("Avg SNR", f"{summary['average_snr']:.2f}")
            
            with col4:
                st.metric("Avg Quality", f"{summary['average_quality_score']:.3f}")
            
            # Individual test results
            st.markdown("### 📋 Individual Test Results")
            
            results_data = []
            for i, result in enumerate(batch['individual_results']):
                results_data.append({
                    'Test ID': i + 1,
                    'Accuracy': f"{result.get('accuracy', 0):.2%}",
                    'SNR': f"{result['snr']:.2f}",
                    'Quality Score': f"{result['quality_score']:.3f}",
                    'Detected Peaks': result['detected_peaks']['count']
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Quality distribution
            quality_scores = [r['quality_score'] for r in batch['individual_results']]
            
            fig3 = go.Figure(data=[go.Histogram(x=quality_scores, nbinsx=20)])
            fig3.update_layout(
                title="Quality Score Distribution",
                xaxis_title="Quality Score",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("👆 Select a validation mode and configure parameters to begin")
        
        # Feature descriptions
        st.markdown("### 🎯 Validation Features")
        
        features = [
            ("🧪 Synthetic Signal Generation", "Generate test signals with known properties"),
            ("📊 Quality Assessment", "Evaluate spectral analysis quality metrics"),
            ("🔄 Batch Validation", "Run comprehensive test suites"),
            ("🎯 Detection Accuracy", "Measure frequency detection accuracy"),
            ("📈 SNR Analysis", "Signal-to-noise ratio calculations"),
            ("🌈 Spectral Flatness", "Measure spectral characteristics")
        ]
        
        for feature, description in features:
            st.markdown(f"**{feature}**: {description}")
        
        st.markdown("### 📚 Validation Metrics")
        
        metrics = [
            "**Signal-to-Noise Ratio (SNR)**: Ratio of signal power to noise power",
            "**Spectral Flatness**: Measure of how flat the spectrum is (noise vs tonal)",
            "**Peak Prominence**: How prominent detected peaks are relative to background",
            "**Detection Accuracy**: Percentage of correctly identified frequencies",
            "**Quality Score**: Overall spectral analysis quality metric"
        ]
        
        for metric in metrics:
            st.markdown(metric)
