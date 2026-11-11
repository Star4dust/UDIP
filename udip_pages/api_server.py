"""
UDIP v3 — udip_pages/api_server.py
API server management page.
"""

import streamlit as st
from modules.api_server import get_api_server, render_api_server


def render():
    """Render the API server management page."""
    st.markdown('<div class="main-header">🌐 API Server</div>', unsafe_allow_html=True)
    st.markdown("Configure and manage the UDIP REST API server for external integrations and data access.")
    
    # Get API server instance
    server = get_api_server()
    
    # Render API server interface
    render_api_server()
