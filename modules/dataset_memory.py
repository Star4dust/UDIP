import pandas as pd
import numpy as np
import json
import hashlib
import datetime
from pathlib import Path
import logging

logger = logging.getLogger("udip.dataset_memory")
logger.addHandler(logging.NullHandler())

STORAGE_DIR = Path("udip_data/execution_layer")

class DatasetMemory:
    """
    In-memory cache and disk persistence for dataset metadata.
    """
    def __init__(self):
        self._cache = {}

    def extract_metadata(self, df: pd.DataFrame) -> dict:
        metadata = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {str(col): int(df[col].isna().sum()) for col in df.columns},
            "timestamp": datetime.datetime.now().isoformat(),
            "row_hash": self._compute_row_hash(df)
        }
        return metadata

    def _compute_row_hash(self, df: pd.DataFrame) -> str:
        """
        A lightweight hash proxy for the dataset's exact row states.
        We hash the string representation of the head and tail, plus shape, to avoid full dataset hashing costs.
        If dataset is small, hash the whole thing.
        """
        if len(df) < 10000:
            content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        else:
            sample = pd.concat([df.head(5000), df.tail(5000)])
            content = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        return hashlib.sha256(content).hexdigest()

    def save_metadata(self, name: str, timestamp_str: str, metadata: dict) -> None:
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        metadata_dir = STORAGE_DIR / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{safe_name}_{timestamp_str}.json"
        path = metadata_dir / filename
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        # Update cache
        self._cache[name] = metadata
        
    def get_latest_metadata(self, name: str) -> dict | None:
        if name in self._cache:
            return self._cache[name]
            
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        metadata_dir = STORAGE_DIR / "metadata"
        if not metadata_dir.exists():
            return None
            
        candidates = list(metadata_dir.glob(f"{safe_name}_*.json"))
        if not candidates:
            return None
            
        latest_file = sorted(candidates)[-1]
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self._cache[name] = metadata
                return metadata
        except Exception:
            logger.exception("Failed to load metadata for %r", name)
            return None


class DatasetSnapshotManager:
    """
    Manages physical storage of snapshots.
    """
    def __init__(self, memory: DatasetMemory):
        self.memory = memory
        
    def _validate_dataset_name(self, name: str) -> str:
        if not name or not isinstance(name, str):
            raise ValueError("Dataset name must be a non-empty string.")
        return "".join(c if c.isalnum() else "_" for c in name).strip("_")

    def _prepare_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        storage_df = df.copy()
        storage_df.columns = storage_df.columns.astype(str)
        return storage_df

    def save_snapshot(self, df: pd.DataFrame, name: str) -> str:
        logger.info("Snapshot save requested. name=%r", name)
        safe_name = self._validate_dataset_name(name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.parquet"
        
        snapshot_dir = STORAGE_DIR / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = snapshot_dir / filename
        storage_df = self._prepare_for_storage(df)
        storage_df.to_parquet(output_path, index=False)
        
        # Save metadata via memory system
        metadata = self.memory.extract_metadata(df)
        self.memory.save_metadata(name, timestamp, metadata)
        
        return str(output_path)

    def load_snapshot(self, name: str) -> pd.DataFrame:
        logger.info("Snapshot load requested. name=%r", name)
        safe_name = self._validate_dataset_name(name)
        snapshot_dir = STORAGE_DIR / "snapshots"
        
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"No snapshots directory found: {snapshot_dir}")
            
        candidates = list(snapshot_dir.glob(f"{safe_name}_*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No snapshots found for dataset: {name}")
            
        latest_snapshot = sorted(candidates)[-1]
        df = pd.read_parquet(latest_snapshot)
        return df


def compare_snapshots(current_df: pd.DataFrame, historical_df: pd.DataFrame, memory: DatasetMemory = None) -> dict:
    """
    Compare two snapshots and return schema changes, distribution changes, and drift metrics.
    """
    result = {
        "schema_changes": {},
        "distribution_changes": {},
        "drift_metrics": {},
        "row_drift_detected": False,
        "is_identical": False
    }

    # Schema changes
    current_cols = set(current_df.columns)
    hist_cols = set(historical_df.columns)
    
    added_cols = current_cols - hist_cols
    removed_cols = hist_cols - current_cols
    
    if added_cols:
        result["schema_changes"]["added"] = list(added_cols)
    if removed_cols:
        result["schema_changes"]["removed"] = list(removed_cols)
        
    common_cols = current_cols.intersection(hist_cols)
    type_changes = {}
    for col in common_cols:
        if current_df[col].dtype != historical_df[col].dtype:
            type_changes[col] = {"from": str(historical_df[col].dtype), "to": str(current_df[col].dtype)}
    if type_changes:
        result["schema_changes"]["type_changes"] = type_changes

    # Row drift (hash proxy)
    if memory:
        current_hash = memory._compute_row_hash(current_df)
        hist_hash = memory._compute_row_hash(historical_df)
        result["row_drift_detected"] = (current_hash != hist_hash)
        if not result["row_drift_detected"] and not result["schema_changes"]:
            result["is_identical"] = True

    # Drift and Distribution changes
    numeric_cols = current_df.select_dtypes(include="number").columns
    hist_numeric = historical_df.select_dtypes(include="number").columns
    common_numeric = list(set(numeric_cols).intersection(set(hist_numeric)))

    for col in common_numeric:
        c_series = current_df[col].dropna()
        h_series = historical_df[col].dropna()
        
        if len(c_series) == 0 or len(h_series) == 0:
            continue
            
        c_mean, c_std = c_series.mean(), c_series.std()
        h_mean, h_std = h_series.mean(), h_series.std()
        
        # Distribution changes (min/max)
        c_min, c_max = c_series.min(), c_series.max()
        h_min, h_max = h_series.min(), h_series.max()
        
        if c_min != h_min or c_max != h_max:
            result["distribution_changes"][col] = {
                "min_diff": float(c_min - h_min),
                "max_diff": float(c_max - h_max)
            }
            
        # Statistical drift (e.g. mean shifted by > 2 std deviations of historical)
        if pd.notna(h_std) and h_std > 0:
            if abs(c_mean - h_mean) > (2 * h_std):
                result["drift_metrics"][col] = {
                    "mean_diff": float(c_mean - h_mean),
                    "std_diff": float(c_std - h_std) if pd.notna(c_std) else 0.0
                }

    return result
