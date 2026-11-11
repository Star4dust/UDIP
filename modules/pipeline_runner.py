"""
UDIP pipeline runner.

This module executes orchestrator plans by dispatching each step to the
execution layer. It does not manipulate files or data outside the approved
execution_layer API surface.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Mapping

import pandas as pd

from modules import execution_layer
from modules.core import load_config
from modules.system_observer import SystemObserver
from modules.feature_alchemy import FeatureAlchemy
from modules.data_fusion import DataFusionEngine

_observer = SystemObserver()
_alchemy = FeatureAlchemy()
_fusion = DataFusionEngine()


logger = logging.getLogger("udip.pipeline_runner")
logger.addHandler(logging.NullHandler())

PIPELINE_PRESETS = {
    "hirax_clean_pipeline": [
        {"action": "load_dataset", "file_path": "{file_path}"},
        {"action": "detect_rfi", "threshold": 3.5},
        {"action": "clean_rfi", "method": "interpolate"},
        {"action": "create_visualization", "type": "waterfall"},
        {"action": "generate_report", "format": "pdf"}
    ]
}

def expand_preset(preset_name: str, params: dict) -> list[dict]:
    """Expand a pipeline preset into a list of steps with injected parameters."""
    if preset_name not in PIPELINE_PRESETS:
        return []
    
    steps = []
    for step in PIPELINE_PRESETS[preset_name]:
        new_step = step.copy()
        # Inject parameters into string values
        for k, v in new_step.items():
            if isinstance(v, str) and "{" in v:
                new_step[k] = v.format(**params)
        steps.append(new_step)
    return steps

ALLOWED_ACTIONS = {
    "load_dataset",
    "run_sql",
    "clean_data",
    "detect_anomalies",
    "save_dataset",
    "create_visualization",
    "generate_report",
    "engineer_features",
    "merge_datasets",
    "train_model",
    "run_clustering",
    "detect_schema",
    "generate_ai_report",
    "clarify_request",
    "detect_rfi",
    "clean_rfi",
    "calibrate_signal",
    "compute_fft",
    "power_spectrum",
    "apply_window",
    "detect_peaks",
    "classify_signal",
}


def run_pipeline(plan: dict) -> dict[str, Any]:
    """
    Execute an orchestrator plan sequentially through execution_layer functions.

    Returns a structured dictionary containing execution status, per-step
    summaries, and the final result object. The final result may be a
    pandas.DataFrame, anomaly dictionary, or save confirmation dictionary.
    """
    import streamlit as st
    
    logger.info("Pipeline execution requested.")
    validated_steps = _validate_plan(plan)
    run_id = plan.get("run_id", "unknown")
    start_time = time.time()

    # Initialize session state for real-time UI updates
    st.session_state["execution_logs"] = []
    st.session_state["execution_plan"] = plan
    st.session_state["agents_used"] = plan.get("agents_used", [])

    current_df: pd.DataFrame | None = None
    last_result: Any = None
    step_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for step_number, step in enumerate(validated_steps, start=1):
        action = step["action"]
        logger.info("Pipeline step started. step=%s action=%s", step_number, action)

        step_start = time.time()
        try:
            if action == "load_dataset":
                file_path = _require_string(step, "file_path", action)
                res = execution_layer.load_dataset(file_path)
                if not res["success"]: raise ValueError(res["error"])
                current_df = res["output"]
                last_result = current_df

            elif action == "run_sql":
                query = _require_string(step, "query", action)
                res = execution_layer.run_sql(query)
                if not res["success"]: raise ValueError(res["error"])
                current_df = res["output"]
                last_result = current_df

            elif action == "clean_data":
                current_df = _require_current_dataframe(current_df, action)
                method = _optional_string(step, "method", default="standard")
                res = execution_layer.clean_data(current_df, method)
                if not res["success"]: raise ValueError(res["error"])
                current_df = res["output"]
                last_result = current_df

            elif action == "detect_anomalies":
                current_df = _require_current_dataframe(current_df, action)
                res = execution_layer.detect_anomalies(current_df)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "save_dataset":
                current_df = _require_current_dataframe(current_df, action)
                name = _require_string(step, "name", action)
                res = execution_layer.save_dataset(current_df, name)
                if not res["success"]: raise ValueError(res["error"])
                last_result = {
                    "saved": True,
                    "name": name,
                    "rows": int(len(current_df)),
                    "columns": int(len(current_df.columns)),
                }

            elif action == "create_visualization":
                current_df = _require_current_dataframe(current_df, action)
                chart_type = _optional_string(step, "chart_type", default="line")
                res = execution_layer.create_visualization(current_df, chart_type)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "generate_report":
                report_input = last_result if isinstance(last_result, dict) else {}
                if isinstance(last_result, pd.DataFrame):
                    report_input = {"rows": len(last_result), "columns": len(last_result.columns)}
                res = execution_layer.generate_report(report_input)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "engineer_features":
                current_df = _require_current_dataframe(current_df, action)
                method = _optional_string(step, "method", default="generate")
                if method == "generate":
                    res = _alchemy.generate_features(current_df)
                    current_df = res["transformed_df"]
                    last_result = current_df
                else:
                    target = _require_string(step, "target", action)
                    res = _alchemy.select_features(current_df, target)
                    current_df = res["selected_df"]
                    last_result = current_df

            elif action == "merge_datasets":
                df1 = _require_current_dataframe(current_df, action)
                # For simplicity, we assume the second dataset is passed by name and loaded from execution_layer
                df2_name = _require_string(step, "dataset_name", action)
                load_res = execution_layer.load_dataset(df2_name)
                if not load_res["success"]: raise ValueError(load_res["error"])
                df2 = load_res["output"]
                
                key = _require_string(step, "key", action)
                res = _fusion.merge_datasets(df1, df2, key)
                current_df = res["merged_df"]
                last_result = current_df

            elif action == "train_model":
                current_df = _require_current_dataframe(current_df, action)
                target = _require_string(step, "target", action)
                features = step.get("features", [c for c in current_df.columns if c != target])
                task = _optional_string(step, "task", default="classification")
                res = execution_layer.train_model(current_df, target, features, task)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "run_clustering":
                current_df = _require_current_dataframe(current_df, action)
                features = step.get("features", current_df.select_dtypes(include="number").columns.tolist())
                n_clusters = step.get("n_clusters", 3)
                res = execution_layer.run_clustering(current_df, features, n_clusters)
                if not res["success"]: raise ValueError(res["error"])
                current_df = res["output"]
                last_result = current_df

            elif action == "detect_schema":
                current_df = _require_current_dataframe(current_df, action)
                res = execution_layer.detect_schema(current_df)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "generate_ai_report":
                current_df = _require_current_dataframe(current_df, action)
                ds_name = _optional_string(step, "ds_name", default="dataset")
                
                cfg = load_config()
                gkey = cfg.get("groq_key")
                if not gkey:
                    raise ValueError("Groq API key not configured. Enable AI in Settings.")
                    
                res = execution_layer.generate_ai_report(current_df, ds_name, gkey)
                if not res["success"]: raise ValueError(res["error"])
                last_result = res["output"]

            elif action == "detect_rfi":
                from modules import hirax_processing
                current_df = _require_current_dataframe(current_df, action)
                threshold = step.get("threshold", 3.0)
                res = hirax_processing.detect_rfi(current_df.values, threshold=threshold)
                last_result = res
                import streamlit as st
                st.session_state["hirax_rfi_mask"] = res["mask"]

            elif action == "clean_rfi":
                from modules import hirax_processing
                current_df = _require_current_dataframe(current_df, action)
                import streamlit as st
                mask = st.session_state.get("hirax_rfi_mask")
                if mask is None:
                    mask = hirax_processing.detect_rfi(current_df.values)["mask"]
                
                method = _optional_string(step, "method", default="interpolate")
                cleaned_values = hirax_processing.clean_rfi(current_df.values, mask, method=method)
                current_df = pd.DataFrame(cleaned_values, columns=current_df.columns, index=current_df.index)
                last_result = current_df
                st.session_state["hirax_cleaned_data"] = cleaned_values

            elif action == "calibrate_signal":
                from modules import hirax_analysis
                current_df = _require_current_dataframe(current_df, action)
                window = step.get("window", 50)
                calibrated = hirax_analysis.calibrate_signal(current_df.values, window_size=window)
                current_df = pd.DataFrame(calibrated, columns=current_df.columns, index=current_df.index)
                last_result = current_df
                import streamlit as st
                st.session_state["hirax_calibrated_data"] = calibrated

            elif action == "compute_fft":
                from modules import hirax_analysis
                current_df = _require_current_dataframe(current_df, action)
                s_rate = step.get("sampling_rate", 1.0)
                res = hirax_analysis.compute_fft(current_df.values, sampling_rate=s_rate)
                last_result = res
                import streamlit as st
                st.session_state["hirax_fft_result"] = res

            elif action == "power_spectrum":
                from modules import hirax_analysis
                current_df = _require_current_dataframe(current_df, action)
                s_rate = step.get("sampling_rate", 1.0)
                res = hirax_analysis.compute_power_spectrum(current_df.values, sampling_rate=s_rate)
                last_result = res
                import streamlit as st
                st.session_state["hirax_power_spectrum"] = res

            elif action == "apply_window":
                from modules import hirax_analysis
                current_df = _require_current_dataframe(current_df, action)
                w_type = _optional_string(step, "type", default="hann")
                windowed = hirax_analysis.apply_window(current_df.values, window_type=w_type)
                current_df = pd.DataFrame(windowed, columns=current_df.columns, index=current_df.index)
                last_result = current_df
                import streamlit as st
                st.session_state["hirax_windowed_data"] = windowed

            elif action == "detect_peaks":
                from modules import hirax_analysis
                import streamlit as st
                p_res = st.session_state.get("hirax_power_spectrum")
                if not p_res:
                    raise ValueError("Action 'detect_peaks' requires 'power_spectrum' to be run first.")
                threshold = step.get("threshold", 0.0)
                res = hirax_analysis.detect_peaks(p_res["power"], p_res["frequencies"], threshold=threshold)
                last_result = res
                st.session_state["hirax_detected_peaks"] = res

            elif action == "classify_signal":
                from modules import hirax_analysis
                import streamlit as st
                p_res = st.session_state.get("hirax_power_spectrum")
                if not p_res:
                    raise ValueError("Action 'classify_signal' requires 'power_spectrum' to be run first.")
                res = hirax_analysis.classify_signal(p_res["power"])
                last_result = res
                st.session_state["hirax_signal_classification"] = res

            elif action == "clarify_request":
                raise ValueError(step.get("message", "No supported execution-layer intent was detected."))

            else:
                raise ValueError(f"Unsupported pipeline action: {action}")

            step_duration = (time.time() - step_start) * 1000
            _observer.log_step(run_id, step_number, action, step_duration, success=True)

            summary = _summarise_result(last_result)
            log_entry = {
                "step": step_number,
                "action": action,
                "success": True,
                "summary": summary,
                "output": summary
            }
            step_results.append(log_entry)
            st.session_state["execution_logs"].append(log_entry)
            
            logger.info(
                "Pipeline step completed. step=%s action=%s summary=%s",
                step_number,
                action,
                summary,
            )

        except Exception as exc:
            step_duration = (time.time() - step_start) * 1000
            error_tier = "ExecutionError"
            if "SQL" in action or "DataFrame" in str(exc):
                error_tier = "DataError"
            elif action == "clarify_request":
                error_tier = "SystemError"

            _observer.log_step(run_id, step_number, action, step_duration, success=False, error_tier=error_tier, error_msg=str(exc))

            logger.exception(
                "Pipeline step failed. step=%s action=%s error=%s",
                step_number,
                action,
                exc,
            )
            failure = {
                "step": step_number,
                "action": action,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "output": f"Error: {str(exc)}"
            }
            step_results.append(failure)
            st.session_state["execution_logs"].append(failure)
            errors.append(failure)
            
            total_time = (time.time() - start_time) * 1000
            _observer.log_run_summary(run_id, plan.get("intent", []), plan.get("agents_used", []), total_time, success=False, step_count=len(step_results))
            
            st.session_state["final_output"] = last_result
            return {
                "steps_executed": step_results,
                "status": "failed",
                "errors": errors,
                "final_result": last_result,
                "final_summary": _summarise_result(last_result),
            }

    total_time = (time.time() - start_time) * 1000
    _observer.log_run_summary(run_id, plan.get("intent", []), plan.get("agents_used", []), total_time, success=True, step_count=len(step_results))
    
    logger.info("Pipeline execution completed. steps=%s", len(step_results))
    st.session_state["final_output"] = last_result
    result = {
        "steps_executed": step_results,
        "status": "success",
        "errors": [],
        "final_result": last_result,
        "final_summary": _summarise_result(last_result),
    }

    # Auto-log if it's a HIRAX run
    is_hirax = any(step["action"].endswith("_rfi") or step["action"] in ["calibrate_signal", "compute_fft", "detect_peaks"] for step in validated_steps)
    if is_hirax:
        try:
            from modules import experiment_tracker
            ds_id = plan.get("dataset_id", "unknown_hirax")
            experiment_tracker.log_experiment(ds_id, validated_steps, plan.get("params", {}), result)
        except Exception as e:
            logger.error(f"Failed to auto-log HIRAX experiment: {e}")

    return result

def run_batch_pipeline(file_paths: list[str], plan_template: dict) -> dict:
    """
    Run a pipeline across multiple files in parallel and aggregate results.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    results = []
    # Use max available cores minus 1 for safety
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in file_paths:
            plan = plan_template.copy()
            if plan["execution_plan"] and plan["execution_plan"][0]["action"] == "load_dataset":
                plan["execution_plan"][0]["file_path"] = path
            futures.append(executor.submit(run_pipeline, plan))
            
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error(f"Batch worker failed: {e}")
        
    from modules import hirax_analysis
    aggregated = hirax_analysis.run_batch_analysis(results)
    return aggregated


def _validate_plan(plan: dict) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary.")

    raw_steps = plan.get("execution_plan")
    preset_name = plan.get("preset")
    
    if preset_name and not raw_steps:
        params = plan.get("params", {})
        raw_steps = expand_preset(preset_name, params)
        if not raw_steps:
            raise ValueError(f"Unknown preset: {preset_name}")
            
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("plan must contain a non-empty 'execution_plan' list or a valid 'preset'.")

    validated_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            raise TypeError(f"step {index} must be a dictionary.")

        step = dict(raw_step)
        action = step.get("action")
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"step {index} must include a non-empty action.")

        action = action.strip()
        if action not in ALLOWED_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_ACTIONS))
            raise ValueError(f"step {index} has unsupported action '{action}'. Allowed: {allowed}.")

        step["action"] = action
        validated_steps.append(step)

    return validated_steps


def _require_string(step: Mapping[str, Any], field: str, action: str) -> str:
    value = step.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Action '{action}' requires a non-empty '{field}' value.")
    return value.strip()


def _optional_string(step: Mapping[str, Any], field: str, *, default: str) -> str:
    value = step.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Optional field '{field}' must be a non-empty string when provided.")
    return value.strip()


def _require_current_dataframe(current_df: pd.DataFrame | None, action: str) -> pd.DataFrame:
    if current_df is None:
        raise ValueError(
            f"Action '{action}' requires a current dataset. "
            "Run 'load_dataset' or 'run_sql' earlier in the plan."
        )
    if not isinstance(current_df, pd.DataFrame):
        raise TypeError(f"Current dataset for action '{action}' must be a pandas DataFrame.")
    return current_df


def _summarise_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {"type": "none"}

    if isinstance(result, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(result)),
            "columns": int(len(result.columns)),
            "column_names": [str(col) for col in result.columns[:20]],
        }

    if isinstance(result, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in result.keys()),
        }

    return {
        "type": type(result).__name__,
        "repr": repr(result)[:500],
    }
