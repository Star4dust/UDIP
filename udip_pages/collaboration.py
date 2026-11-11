"""
UDIP v3 — udip_pages/collaboration.py
Workspace and Collaboration management.
"""

import streamlit as st
import os
from pathlib import Path
from modules.core import BASE_DATA_DIR, load_config, save_config, log

def render():
    st.markdown('<div class="main-header">🤝 Collaborative Hub</div>', unsafe_allow_html=True)
    st.markdown("Manage shared workspaces and security settings for team collaboration.")

    tab1, tab2 = st.tabs(["📂 Workspaces", "🔐 Security"])

    with tab1:
        st.subheader("Workspace Manager")
        st.markdown("Workspaces allow you to separate projects and data into isolated environments.")
        
        # List existing workspaces
        workspaces = [d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir()]
        current_ws = st.session_state.get("active_workspace", "default")
        
        selected_ws = st.selectbox("Switch Workspace", workspaces, index=workspaces.index(current_ws) if current_ws in workspaces else 0)
        
        if selected_ws != current_ws:
            if st.button("🚀 Switch Now"):
                st.session_state.active_workspace = selected_ws
                st.session_state._app_loaded = False # Force reload of all data
                st.success(f"Switched to workspace: {selected_ws}")
                st.rerun()

        st.divider()
        st.subheader("Create New Workspace")
        new_ws = st.text_input("Workspace Name (alphanumeric only)")
        if st.button("➕ Create Workspace"):
            if new_ws and new_ws.isalnum():
                new_path = BASE_DATA_DIR / new_ws
                new_path.mkdir(exist_ok=True)
                st.success(f"Workspace '{new_ws}' created!")
                st.rerun()
            else:
                st.error("Invalid workspace name. Use alphanumeric characters only.")

    with tab2:
        st.subheader("App Security")
        cfg = load_config()
        
        st.markdown("Set a simple password to gate access to this UDIP instance.")
        app_pass = st.text_input("App Password", type="password", value=cfg.get("app_password", ""))
        
        if st.button("💾 Save Security Settings"):
            cfg["app_password"] = app_pass
            save_config(cfg)
            st.success("Security settings saved!")
