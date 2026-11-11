"""
UDIP v3 — modules/ui_components.py
Enhanced UI components with loading states and better user feedback.
"""

import streamlit as st
import time
import threading
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass


@dataclass
class LoadingState:
    """Manages loading states for different operations."""
    message: str
    progress: Optional[float] = None
    is_active: bool = False


class LoadingManager:
    """Centralized loading state manager."""
    
    def __init__(self):
        if "loading_states" not in st.session_state:
            st.session_state.loading_states = {}
    
    def start_loading(self, key: str, message: str, progress: Optional[float] = None):
        """Start a loading state."""
        st.session_state.loading_states[key] = LoadingState(message, progress, True)
        st.rerun()
    
    def update_progress(self, key: str, progress: float):
        """Update progress for a loading state."""
        if key in st.session_state.loading_states:
            st.session_state.loading_states[key].progress = progress
            st.rerun()
    
    def stop_loading(self, key: str):
        """Stop a loading state."""
        if key in st.session_state.loading_states:
            st.session_state.loading_states[key].is_active = False
            st.rerun()
    
    def is_loading(self, key: str) -> bool:
        """Check if a key is currently loading."""
        return (key in st.session_state.loading_states and 
                st.session_state.loading_states[key].is_active)
    
    def get_loading_state(self, key: str) -> Optional[LoadingState]:
        """Get loading state for a key."""
        return st.session_state.loading_states.get(key)


# Global loading manager instance
_loading_manager = LoadingManager()


def show_loading_spinner(message: str, key: str = "default") -> bool:
    """Show a loading spinner with message."""
    if _loading_manager.is_loading(key):
        loading_state = _loading_manager.get_loading_state(key)
        if loading_state.progress is not None:
            st.progress(loading_state.progress)
        st.spinner(loading_state.message)
        return True
    return False


def with_loading(message: str, key: str = "default"):
    """Decorator to add loading state to functions."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            _loading_manager.start_loading(key, message)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _loading_manager.stop_loading(key)
        return wrapper
    return decorator


def progress_bar_operation(operation: Callable, steps: List[str], key: str = "progress") -> Any:
    """Execute operation with progress bar."""
    _loading_manager.start_loading(key, steps[0], 0.0)
    
    try:
        for i, step in enumerate(steps):
            _loading_manager.update_progress(key, i / len(steps))
            if i < len(steps) - 1:
                time.sleep(0.1)  # Small delay for visual feedback
        
        result = operation()
        _loading_manager.stop_loading(key)
        return result
    except Exception as e:
        _loading_manager.stop_loading(key)
        raise e


def skeleton_loader(height: int = 100, count: int = 1):
    """Show skeleton loading placeholders."""
    for _ in range(count):
        st.markdown(f"""
        <div class="skeleton-loader" style="
            height: {height}px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 8px;
            margin: 8px 0;
        "></div>
        <style>
        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)


def smart_button(
    label: str,
    on_click: Callable,
    key: str,
    loading_message: str = "Processing...",
    show_progress: bool = False,
    **kwargs
) -> bool:
    """Enhanced button with loading state."""
    loading_key = f"{key}_loading"
    
    # Show loading state if active
    if _loading_manager.is_loading(loading_key):
        if show_progress:
            loading_state = _loading_manager.get_loading_state(loading_key)
            if loading_state and loading_state.progress is not None:
                st.progress(loading_state.progress)
        st.spinner(loading_message)
        return False
    
    # Show normal button
    if st.button(label, key=key, **kwargs):
        def execute_operation():
            try:
                result = on_click()
                return result
            except Exception as e:
                st.error(f"Operation failed: {e}")
                return False
            finally:
                _loading_manager.stop_loading(loading_key)
        
        # Start loading in background
        _loading_manager.start_loading(loading_key, loading_message)
        
        # Execute operation
        if show_progress:
            # For progress operations, we need to handle differently
            threading.Thread(target=execute_operation, daemon=True).start()
        else:
            result = execute_operation()
            if result is not False:
                st.rerun()
        
        return True
    
    return False


def smart_file_uploader(
    label: str,
    key: str,
    accept_multiple_files: bool = False,
    file_types: Optional[List[str]] = None,
    help_text: Optional[str] = None
) -> bool:
    """Enhanced file uploader with loading state."""
    loading_key = f"{key}_upload_loading"
    
    if _loading_manager.is_loading(loading_key):
        st.info("📁 Processing uploaded files...")
        return False
    
    uploaded_files = st.file_uploader(
        label,
        key=key,
        accept_multiple_files=accept_multiple_files,
        type=file_types,
        help=help_text
    )
    
    if uploaded_files:
        if accept_multiple_files:
            files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            if files:
                _loading_manager.start_loading(loading_key, f"Processing {len(files)} file(s)...")
                return True
        else:
            _loading_manager.start_loading(loading_key, "Processing file...")
            return True
    
    return False


def error_boundary(func: Callable) -> Callable:
    """Decorator to catch and display errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"""
            **Operation Failed**
            
            Error: `{str(e)}`
            
            Please try again or contact support if issue persists.
            """)
            return None
    return wrapper


def success_notification(message: str, duration: int = 3):
    """Show success notification."""
    st.success(f"✅ {message}")
    if duration > 0:
        # Auto-hide after duration (would need JavaScript implementation)
        pass


def warning_notification(message: str):
    """Show warning notification."""
    st.warning(f"⚠️ {message}")


def info_notification(message: str):
    """Show info notification."""
    st.info(f"ℹ️ {message}")


def render_command_center_header(current_page, n_ds, total_rows, ai_state, engine_state):
    """
    Renders main UDIP workspace shell/header.
    """
    st.markdown(
        f"""
        <div class="udip-shell">
            <div class="udip-shell__eyebrow">Radio Astronomy Operating System</div>
            <div class="udip-shell__title">UDIP Workspace</div>
            <div class="udip-shell__caption">
                Current workspace: <strong>{current_page}</strong>. UDIP keeps interferometric data ingestion,
                inspection, calibration, visualization, and reporting in one reproducible local workspace.
            </div>
            <div class="udip-shell__chips">
                <div class="udip-chip udip-chip--accent">{n_ds} dataset(s) loaded</div>
                <div class="udip-chip">{total_rows:,} rows in memory</div>
                <div class="udip-chip">{ai_state}</div>
                <div class="udip-chip udip-chip--warm">{engine_state}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_auth_gate(app_password):
    """
    Renders authentication gate UI.
    Returns True if authenticated, False otherwise.
    """
    st.markdown('<div class="main-header">🔐 UDIP Secure Access</div>', unsafe_allow_html=True)
    pwd = st.text_input("Enter App Password", type="password")
    if st.button("Unlock Platform"):
        if pwd == app_password:
            return True
        else:
            st.error("Invalid password.")
    return False


# Export loading manager for use in other modules
def get_loading_manager() -> LoadingManager:
    """Get the global loading manager instance."""
    return _loading_manager
