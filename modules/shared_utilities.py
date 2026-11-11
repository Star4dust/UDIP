"""
UDIP v3 — modules/shared_utilities.py
Shared utilities to break circular dependencies.

This module contains common utilities that can be safely imported
by multiple modules without creating circular dependencies.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import json
import time
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class SafeLogger:
    """Thread-safe logging utility."""
    
    @staticmethod
    def log(message: str, level: str = "info"):
        """Safe logging without circular dependencies."""
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)


class StateValidator:
    """State validation utilities."""
    
    @staticmethod
    def validate_state_key(key: str) -> bool:
        """Validate state key format."""
        return isinstance(key, str) and len(key) > 0
    
    @staticmethod
    def validate_state_value(value: Any) -> bool:
        """Validate state value."""
        return value is not None
    
    @staticmethod
    def sanitize_key(key: str) -> str:
        """Sanitize state key."""
        return key.replace(" ", "_").lower()


class ImportHelper:
    """Helper utilities for import management."""
    
    @staticmethod
    def safe_import(module_name: str):
        """Safe import with error handling."""
        try:
            import importlib
            return importlib.import_module(module_name)
        except ImportError as e:
            SafeLogger.log(f"Failed to import {module_name}: {e}", "error")
            return None
    
    @staticmethod
    def check_circular_import(caller_module: str, target_module: str) -> bool:
        """Check for potential circular import."""
        return caller_module == target_module


class PerformanceTracker:
    """Performance tracking utilities."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        self.metrics[operation] = {"start": time.time()}
        return self.metrics[operation]["start"]
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation."""
        if operation in self.metrics:
            end_time = time.time()
            duration = end_time - self.metrics[operation]["start"]
            self.metrics[operation]["duration"] = duration
            return duration
        return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return self.metrics


class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration safely."""
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            SafeLogger.log(f"Failed to load config: {e}", "error")
            return {}
    
    @staticmethod
    def save_config(config_path: Path, config: Dict[str, Any]) -> bool:
        """Save configuration safely."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            SafeLogger.log(f"Failed to save config: {e}", "error")
            return False


class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_dataframe(df) -> tuple[bool, str]:
        """Validate DataFrame structure."""
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                return False, "Object is not a DataFrame"
            
            if df.empty:
                return False, "DataFrame is empty"
            
            return True, "Valid DataFrame"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def validate_numeric_column(series) -> tuple[bool, str]:
        """Validate numeric column."""
        try:
            if series.dtype in ['int64', 'float64', 'int32', 'float32']:
                return True, "Valid numeric column"
            return False, "Column is not numeric"
        except Exception as e:
            return False, f"Validation error: {e}"


class MemoryManager:
    """Memory management utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024   # MB
            }
        except ImportError:
            return {"rss": 0.0, "vms": 0.0}
    
    @staticmethod
    def cleanup_temp_files(temp_dir: Path) -> int:
        """Clean up temporary files."""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                return 1
            return 0
        except Exception:
            return 0


class ErrorReporter:
    """Error reporting utilities."""
    
    def __init__(self):
        self.errors = []
    
    def report_error(self, error: Exception, context: Dict[str, Any] = None):
        """Report an error with context."""
        error_report = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
            "context": context or {}
        }
        self.errors.append(error_report)
        SafeLogger.log(f"Error reported: {error_report}", "error")
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all reported errors."""
        return self.errors
    
    def clear_errors(self):
        """Clear all reported errors."""
        self.errors.clear()


# Global instances
performance_tracker = PerformanceTracker()
error_reporter = ErrorReporter()


def get_shared_logger():
    """Get shared logger instance."""
    return SafeLogger()


def get_performance_tracker():
    """Get shared performance tracker."""
    return performance_tracker


def get_error_reporter():
    """Get shared error reporter."""
    return error_reporter
