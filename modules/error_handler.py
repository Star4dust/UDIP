"""
UDIP v3 — modules/error_handler.py
Enhanced error handling and user feedback system.
"""

import streamlit as st
import traceback
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from functools import wraps
import logging


@dataclass
class ErrorReport:
    """Represents an error report with context."""
    error_type: str
    message: str
    details: Optional[str] = None
    timestamp: float = None
    user_action: Optional[str] = None
    context: Dict[str, Any] = None
    suggestions: Optional[list] = None


class ErrorHandler:
    """Centralized error handling and user feedback system."""
    
    def __init__(self):
        if "error_reports" not in st.session_state:
            st.session_state.error_reports = []
        if "error_settings" not in st.session_state:
            st.session_state.error_settings = {
                "show_details": False,
                "auto_retry": True,
                "max_retries": 3,
                "notify_user": True
            }
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None,
        suggestions: Optional[list] = None
    ) -> ErrorReport:
        """Handle an error with user-friendly feedback."""
        
        error_report = ErrorReport(
            error_type=type(error).__name__,
            message=str(error),
            details=traceback.format_exc(),
            timestamp=time.time(),
            user_action=user_action,
            context=context or {},
            suggestions=suggestions or self._generate_suggestions(error)
        )
        
        # Store error report
        st.session_state.error_reports.append(error_report)
        
        # Keep only last 50 errors
        if len(st.session_state.error_reports) > 50:
            st.session_state.error_reports = st.session_state.error_reports[-50:]
        
        # Show user feedback
        if st.session_state.error_settings["notify_user"]:
            self._display_error_to_user(error_report)
        
        return error_report
    
    def _generate_suggestions(self, error: Exception) -> list:
        """Generate contextual suggestions based on error type."""
        error_msg = str(error).lower()
        suggestions = []
        
        # Memory errors
        if "memory" in error_msg or "out of memory" in error_msg:
            suggestions.extend([
                "Try using data sampling for large datasets",
                "Close other applications to free memory",
                "Use Polars engine for better memory efficiency",
                "Restart UDIP to clear memory"
            ])
        
        # File errors
        elif "file" in error_msg or "permission" in error_msg:
            suggestions.extend([
                "Check if file is open in another program",
                "Verify file permissions",
                "Try copying file to a different location",
                "Check if file format is supported"
            ])
        
        # Network errors
        elif "network" in error_msg or "connection" in error_msg:
            suggestions.extend([
                "Check internet connection",
                "Verify firewall settings",
                "Try again in a few moments",
                "Check if API service is available"
            ])
        
        # Data errors
        elif "column" in error_msg or "dtype" in error_msg:
            suggestions.extend([
                "Check data format and column names",
                "Try cleaning the data first",
                "Verify all required columns are present",
                "Check for mixed data types in columns"
            ])
        
        # Import errors
        elif "import" in error_msg or "module" in error_msg:
            suggestions.extend([
                "Install missing package: pip install <package_name>",
                "Check Python version compatibility",
                "Restart UDIP after installing packages",
                "Verify package is in requirements.txt"
            ])
        
        # AI/LLM errors
        elif "api" in error_msg or "groq" in error_msg:
            suggestions.extend([
                "Check API key configuration",
                "Verify internet connection",
                "Check API service status",
                "Try regenerating API key"
            ])
        
        if not suggestions:
            suggestions = [
                "Try refreshing the page",
                "Check your data format",
                "Contact support if issue persists"
            ]
        
        return suggestions
    
    def _display_error_to_user(self, error_report: ErrorReport):
        """Display error to user in a friendly format."""
        
        # Main error message
        st.error(f"""
        **{error_report.error_type} Error**
        
        {error_report.message}
        """)
        
        # Show suggestions
        if error_report.suggestions:
            with st.expander("💡 Suggestions to fix this issue", expanded=True):
                for i, suggestion in enumerate(error_report.suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
        
        # Show details (optional)
        if st.session_state.error_settings["show_details"]:
            with st.expander("🔍 Technical Details", expanded=False):
                st.code(error_report.details or "No details available", language="text")
                
                if error_report.context:
                    st.markdown("**Context:**")
                    st.json(error_report.context)
        
        # Retry option
        if st.session_state.error_settings["auto_retry"]:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Retry", key=f"retry_{error_report.timestamp}"):
                    return True
            with col2:
                if st.button("❌ Dismiss", key=f"dismiss_{error_report.timestamp}"):
                    return False
        
        return None
    
    def show_error_history(self):
        """Display error history for debugging."""
        if not st.session_state.error_reports:
            st.info("No errors reported yet.")
            return
        
        st.markdown("### 📋 Error History")
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        with col1:
            error_filter = st.selectbox(
                "Filter by type",
                options=["All"] + list(set(r.error_type for r in st.session_state.error_reports)),
                key="error_filter"
            )
        
        with col2:
            time_filter = st.selectbox(
                "Time range",
                options=["All time", "Last hour", "Last 24 hours", "Last week"],
                key="error_time_filter"
            )
        
        # Display filtered errors
        filtered_errors = self._filter_errors(error_filter, time_filter)
        
        for error in reversed(filtered_errors[-20:]):  # Show last 20
            with st.expander(f"{error.error_type} - {time.strftime('%H:%M:%S', time.localtime(error.timestamp))}", expanded=False):
                st.markdown(f"**Error:** {error.message}")
                st.markdown(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(error.timestamp))}")
                
                if error.user_action:
                    st.markdown(f"**Action:** {error.user_action}")
                
                if error.suggestions:
                    st.markdown("**Suggestions:**")
                    for suggestion in error.suggestions:
                        st.markdown(f"- {suggestion}")
    
    def _filter_errors(self, error_type: str, time_range: str) -> list:
        """Filter errors based on type and time range."""
        errors = st.session_state.error_reports
        
        # Filter by type
        if error_type != "All":
            errors = [e for e in errors if e.error_type == error_type]
        
        # Filter by time
        current_time = time.time()
        if time_range == "Last hour":
            errors = [e for e in errors if current_time - e.timestamp < 3600]
        elif time_range == "Last 24 hours":
            errors = [e for e in errors if current_time - e.timestamp < 86400]
        elif time_range == "Last week":
            errors = [e for e in errors if current_time - e.timestamp < 604800]
        
        return errors
    
    def clear_error_history(self):
        """Clear error history."""
        st.session_state.error_reports = []
        st.success("Error history cleared.")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not st.session_state.error_reports:
            return {"total": 0, "by_type": {}, "recent": 0}
        
        errors = st.session_state.error_reports
        current_time = time.time()
        
        # Count by type
        by_type = {}
        for error in errors:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
        
        # Count recent (last hour)
        recent = sum(1 for e in errors if current_time - e.timestamp < 3600)
        
        return {
            "total": len(errors),
            "by_type": by_type,
            "recent": recent,
            "last_error": errors[-1] if errors else None
        }


# Global error handler instance
_error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    suggestions: Optional[list] = None
) -> ErrorReport:
    """Global error handling function."""
    return _error_handler.handle_error(error, context, user_action, suggestions)


def error_boundary(
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    suggestions: Optional[list] = None
):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _error_handler.handle_error(e, context, user_action, suggestions)
                return None
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    context: Optional[Dict[str, Any]] = None,
    user_action: Optional[str] = None,
    **kwargs
) -> tuple[bool, Any]:
    """Safely execute a function and return (success, result)."""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        _error_handler.handle_error(e, context, user_action)
        return False, None


def show_error_settings():
    """Display error handling settings."""
    st.markdown("### ⚙️ Error Handling Settings")
    
    # Initialize error settings if not present
    if "error_settings" not in st.session_state:
        st.session_state.error_settings = {
            "show_details": False,
            "auto_retry": True,
            "max_retries": 3,
            "notify_user": True
        }
    
    settings = st.session_state.error_settings
    
    settings["show_details"] = st.checkbox(
        "Show technical details",
        value=settings["show_details"],
        help="Display detailed error information for advanced users"
    )
    
    settings["auto_retry"] = st.checkbox(
        "Enable auto-retry",
        value=settings["auto_retry"],
        help="Show retry option for failed operations"
    )
    
    settings["notify_user"] = st.checkbox(
        "Show error notifications",
        value=settings["notify_user"],
        help="Display error messages to users"
    )
    
    if st.button("Clear Error History"):
        _error_handler.clear_error_history()
    
    # Show error statistics
    stats = _error_handler.get_error_stats()
    st.markdown("### 📊 Error Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Errors", stats["total"])
    with col2:
        st.metric("Recent (1h)", stats["recent"])
    with col3:
        if stats["by_type"]:
            most_common = max(stats["by_type"].items(), key=lambda x: x[1])
            st.metric("Most Common", f"{most_common[0]} ({most_common[1]})")
        else:
            st.metric("Most Common", "None")


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _error_handler


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
