"""
UDIP execution layer.

This module is the controlled boundary between AI agents and UDIP data.
Agents should call these functions instead of touching files, DuckDB, or
DataFrames through ad-hoc code paths.
"""

from __future__ import annotations

import json
import logging
import re
import datetime
import functools
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.dataset_memory import DatasetMemory, DatasetSnapshotManager, compare_snapshots

# Initialize memory system
_dataset_memory = DatasetMemory()
_snapshot_manager = DatasetSnapshotManager(_dataset_memory)


logger = logging.getLogger("udip.execution_layer")
logger.addHandler(logging.NullHandler())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = PROJECT_ROOT / "udip_data" / "execution_layer"
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

# ── Column name knowledge base — checked BEFORE value ranges ─────────────────
COLUMN_NAME_MAP = {
    "lat": ("latitude",    "degrees",  0.97, "Latitude coordinate"),
    "latitude": ("latitude", "degrees", 0.99, "Latitude coordinate"),
    "lon": ("longitude",   "degrees",  0.97, "Longitude coordinate"),
    "long": ("longitude",  "degrees",  0.96, "Longitude coordinate"),
    "longitude": ("longitude", "degrees", 0.99, "Longitude coordinate"),
    "elev": ("elevation_m", "metres",  0.95, "Elevation above sea level"),
    "elevation": ("elevation_m", "metres", 0.97, "Elevation above sea level"),
    "alt": ("altitude_m",  "metres",  0.90, "Altitude"),
    "altitude": ("altitude_m", "metres", 0.95, "Altitude"),
    "year": ("year",       "YYYY",    0.98, "Calendar year"),
    "month": ("month",     "1–12",    0.98, "Calendar month (1=Jan)"),
    "day": ("day",         "1–31",    0.98, "Day of month"),
    "hour": ("hour",       "0–23",    0.97, "Hour of day (24h)"),
    "minute": ("minute",   "0–59",    0.97, "Minute"),
    "second": ("second",   "0–59",    0.97, "Second"),
    "date": ("date",       "date",    0.95, "Date column"),
    "time": ("time",       "time",    0.92, "Time column"),
    "timestamp": ("timestamp", "datetime", 0.97, "Timestamp"),
    "temp": ("temperature_c", "°C",  0.93, "Temperature"),
    "precip": ("precipitation_mm", "mm", 0.95, "Precipitation"),
    "id": ("id",                "ID",  0.90, "Identifier"),
    "freq": ("frequency_mhz",  "MHz",  0.95, "Radio frequency"),
    "snr": ("signal_to_noise",  "dB",  0.93, "Signal-to-noise ratio"),
}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

SUPPORTED_LOAD_FORMATS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".xls",
    ".feather",
    ".h5",
    ".hdf5",
}

SUPPORTED_CLEANING_METHODS = {
    "drop_nulls",
    "fill_nulls",
    "drop_duplicates",
    "trim_strings",
    "normalize_columns",
    "infer_types",
    "remove_constant_columns",
    "standard",
    "advanced",
    "hirax",
}

READ_ONLY_SQL_STARTERS = {"select", "with", "show", "describe", "explain", "summarize"}
SQL_DENYLIST = {
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "export",
    "import",
    "insert",
    "install",
    "load",
    "pragma",
    "set",
    "truncate",
    "update",
}
SQL_FILE_ACCESS_PATTERNS = (
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_parquet",
    "read_blob",
    "read_text",
    "read_ndjson",
    "parquet_scan",
    "glob",
    "httpfs",
)

_DATASETS: dict[str, pd.DataFrame] = {}


def wrap_execution(func):
    """
    Decorator to wrap execution functions with logging, error handling,
    and a structured return format.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info("Execution started: %s", func_name)
        
        try:
            output = func(*args, **kwargs)
            
            # Extract summary for logging
            summary = "None"
            if isinstance(output, pd.DataFrame):
                summary = f"DataFrame(rows={len(output)}, cols={len(output.columns)})"
            elif isinstance(output, dict):
                summary = f"Dict(keys={list(output.keys())})"
            
            logger.info("Execution successful: %s. Output: %s", func_name, summary)
            
            return {
                "success": True,
                "output": output,
                "error": None
            }
        except Exception as e:
            error_msg = str(e)
            logger.error("Execution failed: %s. Error: %s", func_name, error_msg)
            return {
                "success": False,
                "output": None,
                "error": error_msg
            }
    return wrapper


@wrap_execution
def run_sql(query: str) -> pd.DataFrame:
    """
    Execute a DuckDB query against registered datasets.
    Validation is assumed to be handled by the caller.
    """
    with duckdb.connect(database=":memory:") as con:
        for table_name, frame in _DATASETS.items():
            con.register(table_name, frame)

        result = con.execute(query).fetchdf()

    table_name = f"sql_result_{datetime.datetime.now().strftime('%H%M%S')}"
    _DATASETS[table_name] = result.copy()
    return result


@wrap_execution
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a local dataset into a Pandas DataFrame and register it.
    Validation is assumed to be handled by the caller.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".txt":
        df = _read_text_table(path)
    elif suffix == ".json":
        df = _read_json(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".feather":
        df = pd.read_feather(path)
    elif suffix in {".h5", ".hdf5"}:
        df = _read_hdf5(path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    table_name = _safe_identifier(path.stem)
    _DATASETS[table_name] = df.copy()
    return df


@wrap_execution
def save_dataset(df: pd.DataFrame, name: str) -> None:
    """
    Save a DataFrame as Parquet and register it.
    Validation is assumed to be handled by the caller.
    """
    dataset_name = name.strip()
    output_path = STORAGE_DIR / f"{dataset_name}.parquet"
    temp_path = output_path.with_suffix(".parquet.tmp")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    storage_df = _prepare_for_storage(df)
    try:
        storage_df.to_parquet(temp_path, index=False)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    table_name = _safe_identifier(dataset_name)
    _DATASETS[table_name] = storage_df.copy()

@wrap_execution
def merge_datasets(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame, 
    how: str, 
    left_on: str | list[str] | None = None, 
    right_on: str | list[str] | None = None,
    on: str | list[str] | None = None
) -> pd.DataFrame:
    """
    Merge two DataFrames using the specified join type and keys.
    """
    return pd.merge(
        left_df, 
        right_df, 
        how=how, 
        left_on=left_on, 
        right_on=right_on, 
        on=on, 
        suffixes=("_left", "_right")
    )


@wrap_execution
def clean_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply one deterministic cleaning operation and return a new DataFrame.
    Validation is assumed to be handled by the caller.
    """
    clean_method = method.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = df.copy()

    if clean_method == "drop_nulls":
        cleaned = cleaned.dropna().reset_index(drop=True)
    elif clean_method == "fill_nulls":
        cleaned = _fill_nulls(cleaned)
    elif clean_method == "drop_duplicates":
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    elif clean_method == "trim_strings":
        cleaned = _trim_strings(cleaned)
    elif clean_method == "normalize_columns":
        cleaned.columns = _normalise_columns(cleaned.columns)
    elif clean_method == "infer_types":
        cleaned = _infer_types(cleaned)
    elif clean_method == "remove_constant_columns":
        cleaned = _remove_constant_columns(cleaned)
    elif clean_method == "standard":
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        cleaned = _fill_nulls(cleaned)
    elif clean_method == "advanced":
        cleaned.columns = _normalise_columns(cleaned.columns)
        cleaned = _infer_types(cleaned)
    elif clean_method == "hirax":
        cleaned = cleaned.fillna(0)
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_min = cleaned[col].min()
            col_max = cleaned[col].max()
            if col_max > col_min:
                cleaned[col] = (cleaned[col] - col_min) / (col_max - col_min)
            else:
                cleaned[col] = 0.0
    return cleaned

@wrap_execution
def train_model(df: pd.DataFrame, target: str, features: list[str], task: str) -> dict:
    """
    Train a simple Random Forest model for classification or regression.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
    
    X = df[features].copy()
    y = df[target].copy()

    # Handle categorical in features
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    if task.lower() == "classification":
        y_le = LabelEncoder()
        y = y_le.fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            "task": "classification",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred),
            "feature_importances": dict(zip(features, model.feature_importances_.tolist()))
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            "task": "regression",
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "feature_importances": dict(zip(features, model.feature_importances_.tolist()))
        }

@wrap_execution
def run_clustering(df: pd.DataFrame, features: list[str], n_clusters: int) -> pd.DataFrame:
    """
    Run K-Means clustering and append results to the DataFrame.
    """
    from sklearn.cluster import KMeans
    X = df[features].fillna(0)
    model = KMeans(n_clusters=n_clusters)
    result_df = df.copy()
    result_df['cluster'] = model.fit_predict(X)
    return result_df


@wrap_execution
def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Detect basic anomalies, provide a statistical summary, and check for drift.
    Validation is assumed to be handled by the caller.
    """
    # 1. Missing Values
    missing_values = {
        str(col): int(count)
        for col, count in df.isna().sum().items()
        if int(count) > 0
    }

    # 2. Statistical Summary & Outliers
    summary_stats = {}
    outliers = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        summary_stats[str(col)] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max())
        }

        if len(series) >= 3:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series.index[outlier_mask].tolist()

            if outlier_indices:
                outliers[str(col)] = {
                    "count": int(len(outlier_indices)),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "sample_indices": [_serialise_index(idx) for idx in outlier_indices[:20]]
                }

        # 3. Drift Detection against last snapshot using DatasetMemory
        drift_detected = False
        drift_details = {}
        comparison_results = {}
        
        dataset_name = df.attrs.get("dataset_name") or df.attrs.get("name")
        if dataset_name:
            try:
                # Use load_snapshot.output because load_snapshot is wrapped
                load_res = load_snapshot(dataset_name)
                if load_res["success"]:
                    last_snapshot = load_res["output"]
                    comparison_results = compare_snapshots(df, last_snapshot, _dataset_memory)
                    if comparison_results.get("drift_metrics") or comparison_results.get("distribution_changes") or comparison_results.get("row_drift_detected"):
                        drift_detected = True
                        drift_details = comparison_results
            except Exception:
                pass  # No snapshot exists or failed to compare

        result = {
            "missing_values": missing_values,
            "outliers": outliers,
            "summary_stats": summary_stats,
            "drift_detected": drift_detected,
        }
        if drift_detected:
            result["drift_details"] = drift_details

        return result


@wrap_execution
def create_visualization(df: pd.DataFrame, chart_type: str):
    """
    Create a simple matplotlib visualization of the given type.
    """
    logger.info("Visualization requested. type=%r", chart_type)
    try:
        _validate_dataframe(df, "dataset for visualization")
        supported_types = {"line", "bar", "histogram"}
        if chart_type not in supported_types:
            raise ValueError(f"Unsupported chart_type: {chart_type}. Allowed: {supported_types}")

        fig, ax = plt.subplots(figsize=(8, 5))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if chart_type == "line":
            if len(numeric_cols) > 0:
                df[numeric_cols].plot(kind="line", ax=ax)
            else:
                raise ValueError("Line chart requires numeric columns.")
        elif chart_type == "bar":
            if len(numeric_cols) > 0:
                # Sample first 20 rows to avoid cluttering the bar chart
                df[numeric_cols].head(20).plot(kind="bar", ax=ax)
            else:
                raise ValueError("Bar chart requires numeric columns.")
        elif chart_type == "histogram":
            if len(numeric_cols) > 0:
                df[numeric_cols].plot(kind="hist", alpha=0.7, bins=20, ax=ax)
            else:
                raise ValueError("Histogram requires numeric columns.")

        ax.set_title(f"Simple {chart_type.title()} Chart")
        plt.tight_layout()
        
        logger.info("Visualization created successfully.")
        return fig
    except Exception:
        logger.exception("Visualization failed.")
        raise


@wrap_execution
def save_snapshot(df: pd.DataFrame, name: str) -> None:
    """
    Save a point-in-time snapshot of a DataFrame locally as a Parquet file.
    Delegates to DatasetSnapshotManager.
    """
    return _snapshot_manager.save_snapshot(df, name)


@wrap_execution
def load_snapshot(name: str) -> pd.DataFrame:
    """
    Load the most recent snapshot for the given dataset name.
    Delegates to DatasetSnapshotManager.
    """
    return _snapshot_manager.load_snapshot(name)


@wrap_execution
def generate_report(results: dict) -> str:
    """
    Generate a plain text summary from a results dictionary.
    """
    logger.info("Report generation requested.")
    lines = []
    lines.append("=== DATA ANALYSIS REPORT ===")
    
    if "rows" in results or "row_count" in results:
        rows = results.get("rows", results.get("row_count", "N/A"))
        cols = results.get("columns", results.get("column_count", "N/A"))
        lines.append(f"Dataset Size: {rows} rows x {cols} columns")
        lines.append("")

    if "summary_stats" in results:
        lines.append("--- KEY STATISTICS ---")
        for col, stats in results["summary_stats"].items():
            lines.append(f"Column: {col}")
            lines.append(f"  Mean: {stats.get('mean', 'N/A'):.4f} | Std: {stats.get('std', 'N/A'):.4f}")
            lines.append(f"  Min:  {stats.get('min', 'N/A')} | Max: {stats.get('max', 'N/A')}")
        lines.append("")

    if "missing_values" in results and results["missing_values"]:
        lines.append("--- MISSING VALUES ---")
        for col, count in results["missing_values"].items():
            lines.append(f"  {col}: {count} missing")
        lines.append("")

    if "outliers" in results and results["outliers"]:
        lines.append("--- ANOMALIES (OUTLIERS) ---")
        for col, info in results["outliers"].items():
            lines.append(f"  {col}: {info.get('count', 0)} outliers detected")
        lines.append("")

    if "drift_detected" in results:
        drift = "YES" if results["drift_detected"] else "NO"
        lines.append(f"--- DATA DRIFT DETECTED: {drift} ---")
        if results.get("drift_details"):
            for col, details in results["drift_details"].items():
                lines.append(f"  {col} -> Mean diff: {details.get('mean_diff', 0):.4f}, Std diff: {details.get('std_diff', 0):.4f}")
        lines.append("")

    if len(lines) == 1:
        lines.append("No specific metrics found in results.")
        for k, v in results.items():
            if not isinstance(v, (dict, list)):
                lines.append(f"{k}: {v}")
            
    report = "\n".join(lines)
    logger.info("Report generated successfully.")
    return report


def _validate_dataframe(
    df: pd.DataFrame,
    label: str,
    *,
    allow_empty: bool = False,
    allow_duplicate_columns: bool = False,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    if not allow_empty and df.empty:
        raise ValueError(f"{label} cannot be empty.")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError(f"{label} cannot use MultiIndex columns.")
    if not allow_duplicate_columns and len(df.columns) != len(set(map(str, df.columns))):
        raise ValueError(f"{label} contains duplicate column names.")


def _validate_input_path(file_path: str) -> Path:
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string.")

    path = Path(file_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"file_path must point to a file: {path}")
    if path.suffix.lower() not in SUPPORTED_LOAD_FORMATS:
        raise ValueError(f"Unsupported dataset format: {path.suffix.lower()}")

    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"Dataset file is empty: {path}")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"Dataset file is too large: {file_size:,} bytes. "
            f"Limit is {MAX_FILE_SIZE_BYTES:,} bytes."
        )
    return path


def _validate_dataset_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")

    safe_name = _safe_identifier(Path(name).stem)
    if safe_name in {"con", "prn", "aux", "nul"}:
        raise ValueError(f"Reserved dataset name: {safe_name}")
    return safe_name


def _validate_cleaning_method(method: str) -> str:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string.")

    clean_method = method.strip().lower().replace("-", "_").replace(" ", "_")
    if clean_method not in SUPPORTED_CLEANING_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_CLEANING_METHODS))
        raise ValueError(f"Unsupported cleaning method: {method}. Allowed: {allowed}")
    return clean_method


def _validate_sql_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if "\x00" in query:
        raise ValueError("query contains invalid null bytes.")

    normalised = query.strip()
    normalised = normalised[:-1].strip() if normalised.endswith(";") else normalised
    if _has_unquoted_semicolon(normalised):
        raise ValueError("Only one SQL statement is allowed.")

    inspectable = _strip_sql_comments(normalised).strip()
    if not inspectable:
        raise ValueError("query contains no executable SQL.")

    first_token = re.match(r"^[A-Za-z_]+", inspectable)
    if not first_token or first_token.group(0).lower() not in READ_ONLY_SQL_STARTERS:
        allowed = ", ".join(sorted(READ_ONLY_SQL_STARTERS))
        raise ValueError(f"Only read-only SQL is allowed. Query must start with: {allowed}.")

    lowered = inspectable.lower()
    for keyword in SQL_DENYLIST:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            raise ValueError(f"SQL keyword is not allowed in execution layer: {keyword}")

    for pattern in SQL_FILE_ACCESS_PATTERNS:
        if re.search(rf"\b{re.escape(pattern)}\s*\(", lowered):
            raise ValueError(f"SQL file-access function is not allowed: {pattern}")

    return normalised


def _has_unquoted_semicolon(sql: str) -> bool:
    quote: str | None = None
    escaped = False
    for char in sql:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == ";":
            return True
    return False


def _strip_sql_comments(sql: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return re.sub(r"--.*?$", " ", without_block_comments, flags=re.MULTILINE)


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("Identifier cannot be empty after sanitisation.")
    if cleaned[0].isdigit():
        cleaned = f"dataset_{cleaned}"
    return cleaned[:80]


def _normalise_columns(columns: pd.Index) -> list[str]:
    seen: dict[str, int] = {}
    normalised = []
    for column in columns:
        base = _safe_identifier(str(column))
        count = seen.get(base, 0)
        seen[base] = count + 1
        normalised.append(base if count == 0 else f"{base}_{count}")
    return normalised


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if not cleaned[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            replacement = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(0 if pd.isna(replacement) else replacement)
        elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].ffill().bfill()
        else:
            mode = cleaned[col].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else ""
            cleaned[col] = cleaned[col].fillna(replacement)
    return cleaned


def _trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()
    return cleaned


def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.convert_dtypes()
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        col_name = str(col).lower()
        if any(token in col_name for token in ("date", "time", "timestamp")):
            parsed = pd.to_datetime(cleaned[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                cleaned[col] = parsed
    return cleaned


def _remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [col for col in df.columns if df[col].nunique(dropna=False) > 1]
    return df.loc[:, columns_to_keep].copy()


def _prepare_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    storage_df = df.copy()
    for col in storage_df.select_dtypes(include=["object"]).columns:
        if storage_df[col].map(lambda value: isinstance(value, (dict, list, set, tuple))).any():
            storage_df[col] = storage_df[col].map(
                lambda value: json.dumps(value, default=str)
                if isinstance(value, (dict, list, set, tuple))
                else value
            )
    return storage_df


def _read_text_table(path: Path) -> pd.DataFrame:
    for separator in ("\t", ",", ";", "|"):
        try:
            frame = pd.read_csv(path, sep=separator)
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _read_json(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def _read_hdf5(path: Path) -> pd.DataFrame:
    try:
        return pd.read_hdf(path)
    except Exception:
        pass

    import h5py

    candidates: list[tuple[str, int]] = []
    with h5py.File(path, "r") as h5_file:
        h5_file.visititems(
            lambda name, obj: candidates.append((name, int(obj.size)))
            if isinstance(obj, h5py.Dataset) and getattr(obj, "size", 0) > 0
            else None
        )

        if not candidates:
            raise ValueError(f"No readable datasets found in HDF5 file: {path}")

        dataset_name = max(candidates, key=lambda item: item[1])[0]
        data = h5_file[dataset_name][()]

    if not hasattr(data, "ndim"):
        return pd.DataFrame({Path(dataset_name).name: [data]})
    if data.ndim == 0:
        return pd.DataFrame({Path(dataset_name).name: [data.item()]})
    if data.ndim == 1:
        return pd.DataFrame({Path(dataset_name).name: data})

    flattened = data.reshape(data.shape[0], -1)
    column_prefix = _safe_identifier(Path(dataset_name).name)
    columns = [f"{column_prefix}_{idx}" for idx in range(flattened.shape[1])]
    return pd.DataFrame(flattened, columns=columns)


@wrap_execution
def detect_schema(df: pd.DataFrame) -> list[dict]:
    """
    Analyse each column and propose name, type, unit, confidence.
    """
    proposals = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        prop = {
            "original_name": col,
            "proposed_name": col,
            "dtype": str(series.dtype),
            "confidence": 0.5,
            "notes": [],
            "suggested_unit": "",
        }

        vals = series.dropna()
        if len(vals) == 0:
            prop["notes"].append("Column is empty")
            prop["confidence"] = 0.1
            proposals.append(prop)
            continue

        col_lower = col.lower().strip()
        col_clean = re.sub(r"[\s\(\°\%\[\]°\/\.\-]+.*$", "", col_lower).strip()

        name_matched = False
        for keyword, (proposed, unit, conf, note) in COLUMN_NAME_MAP.items():
            if col_clean == keyword or col_lower == keyword:
                prop["proposed_name"]  = proposed
                prop["suggested_unit"] = unit
                prop["confidence"]     = conf
                prop["notes"].append(f"Name matches known field: {note}")
                name_matched = True
                break

        if not name_matched and pd.api.types.is_numeric_dtype(series):
            vmin, vmax = vals.min(), vals.max()
            if 9.46e8 < vmin and vmax < 2.5e9:
                prop["proposed_name"] = "timestamp_unix"
                prop["suggested_unit"] = "Unix seconds"
                prop["confidence"] = 0.75
                prop["notes"].append("Values match Unix timestamp range")
            else:
                prop["confidence"] = 0.35

        proposals.append(prop)
    return proposals

@wrap_execution
def generate_ai_report(df: pd.DataFrame, ds_name: str, groq_key: str) -> str:
    """
    Generate a deep statistical analysis report using AI.
    """
    try:
        from modules.ai_engine import ask_groq
    except ImportError:
        return "AI analysis not available. Please use standard statistical analysis tools."
    # Build a more compact but informative summary
    # Use to_string() for tables as LLMs parse them better than nested dicts
    desc_str = df.describe().to_string()
    null_str = df.isna().sum().to_string()
    head_str = df.head(5).to_string()
    
    prompt = f"""### UDIP DATA SCIENCE TASK: DEEP ANALYTICAL REPORT
Dataset: {ds_name}
Size: {df.shape[0]} rows x {df.shape[1]} columns

### DATASET SUMMARY (STATISTICS):
{desc_str}

### MISSING VALUES:
{null_str}

### DATA PREVIEW (TOP 5 ROWS):
{head_str}

### INSTRUCTIONS:
As a Senior Data Scientist, perform a rigorous analysis of this data.
Your report MUST include these specific sections:
1. **Executive Summary**: High-level overview of the data quality and primary findings.
2. **Statistical Profile**: Analysis of the distributions, averages, and significant variances.
3. **Anomaly & Quality Audit**: Identification of outliers, missing patterns, and potential data integrity issues.
4. **Actionable Recommendations**: 3-5 specific, high-value next steps for a business or research team.

Use professional, technical yet accessible language. Format your response with clear Markdown headers (###).
"""
    reply, _, _ = ask_groq(user_message=prompt, groq_key=groq_key, include_dataset_context=False)
    return reply

def _serialise_index(index_value: Any) -> int | float | str:
    if isinstance(index_value, (np.integer,)):
        return int(index_value)
    if isinstance(index_value, (np.floating,)):
        return float(index_value)
    if isinstance(index_value, (str, int, float)):
        return index_value
    return str(index_value)
