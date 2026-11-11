"""
UDIP v3 — modules/consolidated_utils.py
Consolidated utility functions to reduce technical debt.

Centralizes common functions and imports used across multiple modules.
"""

import streamlit as st
import json
import os
import time
import hashlib
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import traceback
from functools import wraps


# === Session State Management ===

def initialize_session_state():
    """Initialize common session state variables."""
    common_keys = [
        'datasets',
        'active_page', 
        'dark_mode',
        'user_preferences',
        'context_engine',
        'command_palette',
        'workflow_handoffs',
        '_handoff_data',
        '_pmem_',
        '_saved_widget_states',
        'platform_health',
        'scalability_metrics'
    ]
    
    for key in common_keys:
        if key not in st.session_state:
            if key == 'datasets':
                st.session_state[key] = {}
            elif key == 'active_page':
                st.session_state[key] = "📌 Dashboard"
            elif key == 'dark_mode':
                st.session_state[key] = False
            elif key == 'user_preferences':
                st.session_state[key] = {}
            else:
                st.session_state[key] = None


# === Dataset Management ===

def save_dataset(name: str, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
    """
    Save a dataset to session state with metadata.
    
    Args:
        name: Dataset name
        df: DataFrame to save
        metadata: Optional metadata dictionary
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if name in st.session_state.datasets:
            st.warning(f"Dataset '{name}' already exists. Overwriting.")
        
        # Create dataset entry
        dataset_entry = {
            'df': df,
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'size_bytes': len(pickle.dumps(df)),
            'created_at': time.time(),
            'metadata': metadata or {}
        }
        
        st.session_state.datasets[name] = dataset_entry
        return True
        
    except Exception as e:
        st.error(f"Failed to save dataset '{name}': {e}")
        return False


def get_active_dataset() -> Optional[pd.DataFrame]:
    """Get the currently active dataset."""
    if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
        return st.session_state.context_engine.get_active_dataset()
    return None


def set_active_dataset(name: str) -> bool:
    """Set the active dataset by name."""
    try:
        if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
            return st.session_state.context_engine.set_active_dataset(name)
        
        # Fallback to direct session state
        if name in st.session_state.datasets:
            st.session_state._active_dataset_name = name
            return True
        return False
        
    except Exception as e:
        st.error(f"Failed to set active dataset '{name}': {e}")
        return False


def has_active_data() -> bool:
    """Check if there's active data available."""
    if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
        return st.session_state.context_engine.has_active_data()
    
    # Fallback check
    return ('datasets' in st.session_state and 
            len(st.session_state.datasets) > 0 and
            hasattr(st.session_state, '_active_dataset_name') and
            st.session_state._active_dataset_name in st.session_state.datasets)


# === AI Memory Management ===

def save_ai_memory(memory_type: str, content: Any, metadata: Dict[str, Any] = None) -> bool:
    """
    Save AI memory to session state.
    
    Args:
        memory_type: Type of memory (e.g., 'conversation', 'analysis', 'preferences')
        content: Content to save
        metadata: Optional metadata
    
    Returns:
        True if successful
    """
    try:
        if '_ai_memory' not in st.session_state:
            st.session_state._ai_memory = {}
        
        memory_entry = {
            'content': content,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        if memory_type not in st.session_state._ai_memory:
            st.session_state._ai_memory[memory_type] = []
        
        st.session_state._ai_memory[memory_type].append(memory_entry)
        
        # Keep only last 50 entries per type
        if len(st.session_state._ai_memory[memory_type]) > 50:
            st.session_state._ai_memory[memory_type] = st.session_state._ai_memory[memory_type][-50:]
        
        return True
        
    except Exception as e:
        st.error(f"Failed to save AI memory: {e}")
        return False


def load_ai_memory(memory_type: str) -> List[Dict[str, Any]]:
    """
    Load AI memory from session state.
    
    Args:
        memory_type: Type of memory to load
    
    Returns:
        List of memory entries
    """
    try:
        if '_ai_memory' not in st.session_state:
            return []
        
        return st.session_state._ai_memory.get(memory_type, [])
        
    except Exception as e:
        st.error(f"Failed to load AI memory: {e}")
        return []


# === Snapshot Management ===

def save_snapshot(snapshot_name: str, data: Any, metadata: Dict[str, Any] = None) -> bool:
    """
    Save a data snapshot.
    
    Args:
        snapshot_name: Name for the snapshot
        data: Data to snapshot
        metadata: Optional metadata
    
    Returns:
        True if successful
    """
    try:
        if '_snapshots' not in st.session_state:
            st.session_state._snapshots = {}
        
        snapshot_entry = {
            'data': data,
            'timestamp': time.time(),
            'size_bytes': len(pickle.dumps(data)),
            'metadata': metadata or {}
        }
        
        st.session_state._snapshots[snapshot_name] = snapshot_entry
        return True
        
    except Exception as e:
        st.error(f"Failed to save snapshot '{snapshot_name}': {e}")
        return False


def load_snapshot(snapshot_name: str) -> Optional[Any]:
    """
    Load a data snapshot.
    
    Args:
        snapshot_name: Name of the snapshot to load
    
    Returns:
        Snapshot data or None if not found
    """
    try:
        if '_snapshots' not in st.session_state:
            return None
        
        snapshot = st.session_state._snapshots.get(snapshot_name)
        return snapshot['data'] if snapshot else None
        
    except Exception as e:
        st.error(f"Failed to load snapshot '{snapshot_name}': {e}")
        return None


# === Error Handling ===

def handle_errors(func):
    """
    Decorator for consistent error handling across functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            
            # Log error if platform health is available
            if hasattr(st.session_state, 'platform_health') and st.session_state.platform_health:
                from modules.platform_health import log_platform_error
                log_platform_error(e, f"Function: {func.__name__}")
            
            return None
    return wrapper


# === Common Data Operations ===

def safe_dataframe_operation(operation_func):
    """
    Decorator for safe DataFrame operations.
    """
    @wraps(operation_func)
    def wrapper(df, *args, **kwargs):
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            
            if df.empty:
                st.warning("DataFrame is empty")
                return df
            
            return operation_func(df, *args, **kwargs)
            
        except Exception as e:
            st.error(f"DataFrame operation failed: {e}")
            return df
    
    return wrapper


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with DataFrame information
    """
    try:
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Add basic statistics for numeric columns
        if info['numeric_columns']:
            info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
        
        return info
        
    except Exception as e:
        st.error(f"Failed to get DataFrame info: {e}")
        return {}


# === File Operations ===

def safe_file_operation(operation_func):
    """
    Decorator for safe file operations.
    """
    @wraps(operation_func)
    def wrapper(file_path, *args, **kwargs):
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            return operation_func(file_path, *args, **kwargs)
            
        except Exception as e:
            st.error(f"File operation failed: {e}")
            return None
    
    return wrapper


@safe_file_operation
def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in {file_path}: {e}")
        return None


@safe_file_operation
def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """Safely save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to save JSON to {file_path}: {e}")
        return False


# === Performance Utilities ===

def measure_performance(func):
    """
    Decorator to measure function performance.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Log performance if scalability manager is available
        if hasattr(st.session_state, 'scalability_metrics'):
            if 'performance_log' not in st.session_state.scalability_metrics:
                st.session_state.scalability_metrics['performance_log'] = []
            
            st.session_state.scalability_metrics['performance_log'].append({
                'function': func.__name__,
                'execution_time': execution_time,
                'timestamp': time.time()
            })
        
        return result
    
    return wrapper


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}


# === Validation Utilities ===

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(df, pd.DataFrame):
            st.error("Input is not a pandas DataFrame")
            return False
        
        if df.empty:
            st.error("DataFrame is empty")
            return False
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
        
        return True
        
    except Exception as e:
        st.error(f"DataFrame validation failed: {e}")
        return False


def validate_parameters(params: Dict[str, Any], required_params: List[str]) -> bool:
    """
    Validate function parameters.
    
    Args:
        params: Parameters dictionary
        required_params: List of required parameter names
    
    Returns:
        True if valid, False otherwise
    """
    missing_params = set(required_params) - set(params.keys())
    if missing_params:
        st.error(f"Missing required parameters: {missing_params}")
        return False
    
    return True


# === Common UI Components ===

def render_metric_card(title: str, value: Any, delta: Any = None, help_text: str = None):
    """
    Render a consistent metric card.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
        help_text: Optional help text
    """
    if help_text:
        st.metric(title, value, delta, help=help_text)
    else:
        st.metric(title, value, delta)


def render_expandable_section(title: str, content_func: callable, expanded: bool = False):
    """
    Render an expandable section with consistent styling.
    
    Args:
        title: Section title
        content_func: Function to render content
        expanded: Whether to expand by default
    """
    with st.expander(title, expanded=expanded):
        content_func()


# === Data Export Utilities ===

def export_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> bool:
    """
    Export DataFrame to CSV.
    
    Args:
        df: DataFrame to export
        filename: Output filename
        index: Whether to include index
    
    Returns:
        True if successful
    """
    try:
        df.to_csv(filename, index=index)
        st.success(f"Data exported to {filename}")
        return True
    except Exception as e:
        st.error(f"Failed to export CSV: {e}")
        return False


def export_to_json(data: Any, filename: str) -> bool:
    """
    Export data to JSON.
    
    Args:
        data: Data to export
        filename: Output filename
    
    Returns:
        True if successful
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        st.success(f"Data exported to {filename}")
        return True
    except Exception as e:
        st.error(f"Failed to export JSON: {e}")
        return False


# === Logging Utilities ===

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_function_call(func):
    """
    Decorator to log function calls.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logging()
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper


# === Hash and Cache Utilities ===

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.
    
    Returns:
        MD5 hash of the arguments
    """
    key_data = f"{str(args)}:{str(sorted(kwargs.items()))}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_data_hash(data: Any) -> str:
    """
    Generate a hash for data objects.
    
    Args:
        data: Data to hash
    
    Returns:
        MD5 hash of the data
    """
    try:
        serialized = pickle.dumps(data)
        return hashlib.md5(serialized).hexdigest()
    except:
        return hashlib.md5(str(data).encode()).hexdigest()


# === Common Constants ===

class CommonConstants:
    """Common constants used across the application."""
    
    # File extensions
    CSV_EXTENSIONS = ['.csv']
    EXCEL_EXTENSIONS = ['.xlsx', '.xls']
    JSON_EXTENSIONS = ['.json']
    PARQUET_EXTENSIONS = ['.parquet', '.pq']
    
    # Data types
    NUMERIC_DTYPES = ['int64', 'int32', 'float64', 'float32', 'number']
    DATETIME_DTYPES = ['datetime64', 'datetime']
    CATEGORICAL_DTYPES = ['object', 'category', 'string']
    
    # Performance thresholds
    LARGE_DATASET_ROWS = 100000
    LARGE_DATASET_MB = 100
    MEMORY_WARNING_MB = 500
    
    # UI constants
    DEFAULT_CHART_HEIGHT = 400
    DEFAULT_TABLE_HEIGHT = 300
    MAX_DISPLAY_ROWS = 10000


# === Initialization Function ===

def initialize_consolidated_utils():
    """Initialize consolidated utilities and setup common state."""
    initialize_session_state()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Consolidated utilities initialized")
    
    return True
