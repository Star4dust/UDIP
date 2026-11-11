import streamlit as st
import os
from modules.core import (
    APP_VERSION, get_memory_mb, MEMORY_WARN_MB, load_config, HAS_POLARS, log
)
from modules.navigation import DEFAULT_PAGE, NAV_SECTIONS, normalize_page
from modules.global_search import render_global_search, get_search_manager
from modules.error_handler import show_error_settings, get_error_handler
from modules.performance_optimizer import render_performance_panel, render_chunk_navigation
from modules.undo_redo import render_undo_controls
from modules.performance_monitor import render_performance_monitoring
from modules.realtime_dashboards import render_realtime_dashboard
from modules.multi_user import render_multi_user_interface
from modules.api_server import render_api_server
from modules.context_engine import get_context_engine, render_context_aware_sidebar_info
from modules.command_palette import toggle_command_palette
from modules.workflow_handoff import render_handoff_suggestions
from modules.platform_health import get_health_status

def render_sidebar(page_map=None) -> str:
    """
    Renders UDIP sidebar and returns the selected page name.
    """
    from modules.state_manager import get_state_manager
    state_mgr = get_state_manager()
    
    # Get current page
    current_page = st.session_state.get("active_page", DEFAULT_PAGE)
    
    with st.sidebar:
        # Header
        st.markdown("### 📊 UDIP")
        st.markdown("---")
        
        # Theme toggle
        dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
        if dark_mode != st.session_state.get("dark_mode", False):
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        # Navigation
        for section_name, routes in NAV_SECTIONS.items():
            with st.expander(section_name, expanded=True):
                for route in routes:
                    if st.button(
                        f"{route.label}",
                        key=f"nav_{route.label}",
                        use_container_width=True,
                        help=f"Navigate to {route.label}"
                    ):
                        st.session_state.active_page = route.label
                        st.rerun()
        
        # Tools
        st.markdown("---")
        st.markdown("### 🔧 Tools")
        
        if st.button("🔍 Global Search", key="global_search", use_container_width=True):
            st.session_state.show_global_search = True
            st.rerun()
        
        if st.button("⚡ Command Palette", key="cmd_palette", use_container_width=True):
            toggle_command_palette()
            st.rerun()
        
        if st.button("🧠 Context Engine", key="context_engine_toggle", use_container_width=True):
            context_engine = get_context_engine()
            context_engine.toggle_context_panel()
            st.rerun()
        
        # Settings
        st.markdown("---")
        if st.button("⚙️ Settings", key="settings", use_container_width=True):
            st.session_state.active_page = "⚙️ Settings"
            st.rerun()
    
    return current_page
