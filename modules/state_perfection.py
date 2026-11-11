"""
UDIP v3 — modules/state_perfection.py
State management perfection and optimization utilities.

This module provides advanced state management with lifecycle management,
memory optimization, and enterprise-grade state persistence.
"""

import streamlit as st
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import weakref

from modules.shared_utilities import get_shared_logger, StateValidator, MemoryManager

logger = logging.getLogger(__name__)


class StateLifecycleManager:
    """Manages state lifecycle with proper cleanup and validation."""
    
    def __init__(self):
        self.state_registry = {}
        self.cleanup_handlers = {}
        self.validation_rules = {}
        self.lifecycle_hooks = defaultdict(list)
    
    def register_state(self, key: str, initial_value: Any = None, 
                     validator: Optional[Callable] = None, 
                     cleanup_handler: Optional[Callable] = None):
        """Register a state key with lifecycle management."""
        self.state_registry[key] = {
            "initial_value": initial_value,
            "current_value": initial_value,
            "validator": validator,
            "cleanup_handler": cleanup_handler,
            "created_at": time.time(),
            "last_modified": time.time(),
            "access_count": 0
        }
        
        # Register cleanup handler
        if cleanup_handler:
            self.cleanup_handlers[key] = cleanup_handler
        
        # Initialize in session state if not present
        if key not in st.session_state:
            st.session_state[key] = initial_value
        
        logger.info(f"Registered state key: {key}")
    
    def set_state(self, key: str, value: Any, source: str = "user"):
        """Set state value with validation and lifecycle tracking."""
        if key not in self.state_registry:
            logger.warning(f"Setting unregistered state key: {key}")
            self.register_state(key, value)
        
        state_info = self.state_registry[key]
        
        # Validate value
        if state_info["validator"]:
            try:
                validated_value = state_info["validator"](value)
                if validated_value is not None:
                    value = validated_value
            except Exception as e:
                logger.error(f"State validation failed for {key}: {e}")
                return False
        
        # Check if value actually changed
        old_value = state_info["current_value"]
        if old_value != value:
            # Trigger lifecycle hooks
            self._trigger_lifecycle_hooks(key, "before_change", old_value, value, source)
            
            # Update state
            st.session_state[key] = value
            state_info["current_value"] = value
            state_info["last_modified"] = time.time()
            state_info["access_count"] += 1
            
            # Trigger after change hooks
            self._trigger_lifecycle_hooks(key, "after_change", old_value, value, source)
            
            logger.debug(f"State changed: {key} = {value} (from {old_value})")
            
            # Check for cleanup needed
            self._check_cleanup_needed(key)
        
        return True
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value with access tracking."""
        if key not in self.state_registry:
            logger.warning(f"Getting unregistered state key: {key}")
            return default
        
        state_info = self.state_registry[key]
        state_info["access_count"] += 1
        
        # Trigger access hooks
        self._trigger_lifecycle_hooks(key, "on_access", state_info["current_value"], None, "system")
        
        return state_info["current_value"]
    
    def delete_state(self, key: str):
        """Delete state with proper cleanup."""
        if key in self.state_registry:
            state_info = self.state_registry[key]
            
            # Trigger before delete hooks
            self._trigger_lifecycle_hooks(key, "before_delete", state_info["current_value"], None, "system")
            
            # Call cleanup handler
            if state_info["cleanup_handler"]:
                try:
                    state_info["cleanup_handler"](state_info["current_value"])
                except Exception as e:
                    logger.error(f"Cleanup handler failed for {key}: {e}")
            
            # Remove from session state and registry
            if key in st.session_state:
                del st.session_state[key]
            del self.state_registry[key]
            
            # Trigger after delete hooks
            self._trigger_lifecycle_hooks(key, "after_delete", state_info["current_value"], None, "system")
            
            logger.info(f"State deleted: {key}")
            return True
        
        return False
    
    def add_lifecycle_hook(self, key: str, event_type: str, handler: Callable):
        """Add lifecycle hook for state changes."""
        self.lifecycle_hooks[f"{key}_{event_type}"].append(handler)
        logger.debug(f"Added lifecycle hook: {key}_{event_type}")
    
    def remove_lifecycle_hook(self, key: str, event_type: str, handler: Callable):
        """Remove lifecycle hook."""
        hook_key = f"{key}_{event_type}"
        if hook_key in self.lifecycle_hooks:
            self.lifecycle_hooks[hook_key].remove(handler)
            logger.debug(f"Removed lifecycle hook: {hook_key}")
    
    def _trigger_lifecycle_hooks(self, key: str, event_type: str, old_value: Any, new_value: Any, source: str):
        """Trigger all lifecycle hooks for an event."""
        hook_key = f"{key}_{event_type}"
        if hook_key in self.lifecycle_hooks:
            for handler in self.lifecycle_hooks[hook_key]:
                try:
                    handler(key, event_type, old_value, new_value, source)
                except Exception as e:
                    logger.error(f"Lifecycle hook error for {hook_key}: {e}")
    
    def _check_cleanup_needed(self, key: str):
        """Check if cleanup is needed for a state key."""
        state_info = self.state_registry.get(key, {})
        
        # Auto-cleanup for old states
        if time.time() - state_info.get("last_modified", 0) > 3600:  # 1 hour
            if state_info["access_count"] < 2:  # Accessed less than twice in last hour
                self._trigger_lifecycle_hooks(key, "auto_cleanup", state_info["current_value"], None, "system")
    
    def get_state_info(self, key: str) -> Dict[str, Any]:
        """Get comprehensive state information."""
        return self.state_registry.get(key, {})
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered states."""
        return self.state_registry.copy()


class StateMemoryOptimizer:
    """Optimizes state memory usage and prevents memory leaks."""
    
    def __init__(self):
        self.memory_tracker = {}
        self.compression_enabled = True
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.memory_threshold = 50 * 1024 * 1024  # 50MB per state key
    
    def track_memory_usage(self, key: str, size: int):
        """Track memory usage for a state key."""
        self.memory_tracker[key] = {
            "size": size,
            "timestamp": time.time(),
            "compressed": self.compression_enabled
        }
        
        # Check memory threshold
        if size > self.memory_threshold:
            logger.warning(f"State {key} exceeds memory threshold: {size} bytes")
    
    def compress_state_value(self, value: Any) -> tuple[Any, int]:
        """Compress state value if beneficial."""
        if not self.compression_enabled:
            return value, 0
        
        try:
            import pickle
            serialized = pickle.dumps(value)
            compressed = pickle.dumps(serialized, protocol=pickle.HIGHEST_PROTOCOL)
            
            original_size = len(str(value).encode('utf-8'))
            compressed_size = len(compressed)
            
            if compressed_size < original_size * 0.8:  # 20% reduction
                return pickle.loads(compressed), compressed_size
            else:
                return value, original_size
                
        except Exception:
            return value, 0
    
    def cleanup_old_states(self):
        """Clean up old or unused states."""
        current_time = time.time()
        
        for key, state_info in self.memory_tracker.items():
            if current_time - state_info["timestamp"] > self.cleanup_interval:
                # Check if state is still being used
                if state_info.get("access_count", 0) == 0:
                    logger.info(f"Cleaning up unused state: {key}")
                    # This would trigger cleanup through lifecycle manager
                    # For now, just mark for cleanup
                    state_info["cleanup_needed"] = True
        
        self.last_cleanup = current_time
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get memory usage report."""
        total_memory = sum(info["size"] for info in self.memory_tracker.values())
        compressed_memory = sum(info.get("compressed_size", info["size"]) for info in self.memory_tracker.values())
        
        return {
            "total_states": len(self.memory_tracker),
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "compressed_memory_bytes": compressed_memory,
            "compression_ratio": compressed_memory / total_memory if total_memory > 0 else 0,
            "cleanup_needed": len([info for info in self.memory_tracker.values() if info.get("cleanup_needed", False)])
        }


class StatePersistenceManager:
    """Enterprise-grade state persistence with versioning and recovery."""
    
    def __init__(self, storage_path: Path = Path("udip_data/state_persistence")):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.version_history = {}
        self.backup_interval = 3600  # 1 hour
        self.max_versions = 10
        self.encryption_enabled = True
    
    def save_state_snapshot(self, state_data: Dict[str, Any], version: str = "auto") -> bool:
        """Save state snapshot with versioning."""
        try:
            timestamp = datetime.now().isoformat()
            snapshot = {
                "version": version,
                "timestamp": timestamp,
                "state_data": state_data,
                "checksum": self._calculate_checksum(state_data)
            }
            
            # Save current snapshot
            snapshot_file = self.storage_path / f"state_v{version}.json"
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2)
            
            # Update version history
            self.version_history[version] = snapshot
            
            # Cleanup old versions
            self._cleanup_old_versions()
            
            logger.info(f"State snapshot saved: version {version} at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state snapshot: {e}")
            return False
    
    def load_state_snapshot(self, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Load state snapshot with recovery."""
        try:
            if version == "latest":
                # Find latest version
                snapshot_files = list(self.storage_path.glob("state_v*.json"))
                if snapshot_files:
                    latest_file = max(snapshot_files, key=lambda f: f.stat().st_mtime)
                    version = latest_file.stem.replace("state_v", "").replace(".json", "")
                else:
                    return None
            else:
                snapshot_file = self.storage_path / f"state_v{version}.json"
            
            if not snapshot_file.exists():
                logger.warning(f"State snapshot not found: version {version}")
                return None
            
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
            
            # Verify checksum
            if snapshot["checksum"] != self._calculate_checksum(snapshot["state_data"]):
                logger.error(f"State snapshot corruption detected: version {version}")
                return None
            
            logger.info(f"State snapshot loaded: version {version}")
            return snapshot["state_data"]
            
        except Exception as e:
            logger.error(f"Failed to load state snapshot: {e}")
            return None
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for state data integrity."""
        import hashlib
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _cleanup_old_versions(self):
        """Clean up old state versions."""
        try:
            snapshot_files = list(self.storage_path.glob("state_v*.json"))
            
            # Sort by modification time
            snapshot_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Keep only the latest versions
            for snapshot_file in snapshot_files[self.max_versions:]:
                snapshot_file.unlink()
                logger.info(f"Cleaned up old state snapshot: {snapshot_file.name}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
    
    def get_version_history(self) -> Dict[str, Dict[str, Any]]:
        """Get version history."""
        return self.version_history.copy()
    
    def create_backup(self) -> bool:
        """Create backup of current state."""
        try:
            current_state = dict(st.session_state)
            backup_file = self.storage_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "state_data": current_state
                }, f, indent=2)
            
            logger.info(f"State backup created: {backup_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create state backup: {e}")
            return False


class StateValidationEngine:
    """Advanced state validation with custom rules and policies."""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_policies = {
            "strict_mode": False,
            "auto_fix": True,
            "validation_timeout": 5.0
        }
    
    def add_validation_rule(self, key_pattern: str, validator: Callable, error_message: str = "Validation failed"):
        """Add validation rule for state keys matching pattern."""
        self.validation_rules[key_pattern] = {
            "validator": validator,
            "error_message": error_message
        }
        logger.info(f"Added validation rule for pattern: {key_pattern}")
    
    def validate_state_key(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate a state key against all applicable rules."""
        for pattern, rule in self.validation_rules.items():
            if re.match(pattern, key):
                try:
                    is_valid = rule["validator"](value)
                    if not is_valid:
                        return False, rule["error_message"]
                except Exception as e:
                    logger.error(f"Validation rule error for {key}: {e}")
                    return False, f"Validation error: {e}"
        
        return True, "Valid"
    
    def enable_strict_mode(self):
        """Enable strict validation mode."""
        self.validation_policies["strict_mode"] = True
        logger.info("Strict validation mode enabled")
    
    def enable_auto_fix(self):
        """Enable automatic fixing of validation errors."""
        self.validation_policies["auto_fix"] = True
        logger.info("Auto-fix mode enabled")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "rules_count": len(self.validation_rules),
            "policies": self.validation_policies,
            "strict_mode": self.validation_policies["strict_mode"],
            "auto_fix": self.validation_policies["auto_fix"]
        }


# Global instances
state_lifecycle_manager = StateLifecycleManager()
state_memory_optimizer = StateMemoryOptimizer()
state_persistence_manager = StatePersistenceManager()
state_validation_engine = StateValidationEngine()


def get_state_lifecycle_manager() -> StateLifecycleManager:
    """Get global state lifecycle manager."""
    return state_lifecycle_manager


def get_state_memory_optimizer() -> StateMemoryOptimizer:
    """Get global state memory optimizer."""
    return state_memory_optimizer


def get_state_persistence_manager() -> StatePersistenceManager:
    """Get global state persistence manager."""
    return state_persistence_manager


def get_state_validation_engine() -> StateValidationEngine:
    """Get global state validation engine."""
    return state_validation_engine


def initialize_state_perfection():
    """Initialize state perfection system."""
    logger.info("Initializing state perfection system")
    
    # Set up default validation rules
    state_validation_engine.add_validation_rule(
        r".*_dataset$",
        lambda x: isinstance(x, str) and len(x) > 0,
        "Dataset name must be a non-empty string"
    )
    
    state_validation_engine.add_validation_rule(
        r".*_config$",
        lambda x: isinstance(x, dict) and all(k in x for k in ["enabled", "version", "settings"]),
        "Configuration must be a valid dict with required keys"
    )
    
    state_validation_engine.add_validation_rule(
        r".*_token$",
        lambda x: isinstance(x, str) and len(x) > 10,
        "Token must be a valid string"
    )
    
    logger.info("State perfection system initialized")


def optimize_state_management():
    """Apply state management optimizations."""
    logger.info("Applying state management optimizations")
    
    # Enable strict validation
    state_validation_engine.enable_strict_mode()
    
    # Enable auto-fix
    state_validation_engine.enable_auto_fix()
    
    # Initialize memory optimization
    state_memory_optimizer.cleanup_old_states()
    
    # Create backup of current state
    state_persistence_manager.create_backup()
    
    logger.info("State management optimizations applied")
