"""
UDIP v3 — udip_pages/realtime_dashboard.py
Real-time dashboard page with live data streaming.
"""

import streamlit as st
from modules.realtime_dashboards import get_realtime_dashboard_manager


def render():
    """Render the real-time dashboard page."""
    st.markdown('<div class="main-header">📊 Real-Time Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Create and monitor live dashboards with real-time data streaming from datasets, APIs, and external sources.")
    
    # Get the dashboard manager
    dashboard_manager = get_realtime_dashboard_manager()
    
    # Render the dashboard interface
    render_realtime_dashboard()
