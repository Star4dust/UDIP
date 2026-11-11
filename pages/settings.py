import streamlit as st
import os
from modules.core import load_config, save_config, BASE_DATA_DIR, update_paths

def render():
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)

    cfg = load_config()

    tab_api, tab_workspace, tab_ui = st.tabs(["🔑 API Keys", "📂 Workspace", "🎨 UI & Preferences"])

    with tab_api:
        st.markdown("### Groq API Configuration")
        st.caption("Required for AI Assistant, Omni-Extract, and Agentic workflows.")
        groq_key = st.text_input("Groq API Key", value=cfg.get("groq_key", ""), type="password")
        if st.button("Save API Key"):
            cfg["groq_key"] = groq_key
            save_config(cfg)
            os.environ["GROQ_API_KEY"] = groq_key
            st.success("API Key saved!")

    with tab_workspace:
        st.markdown("### Workspace Management")
        st.caption("Switch between different projects. Each workspace has its own datasets, history, and models.")
        
        # List directories in udip_data
        workspaces = [d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir()]
        active_ws = st.session_state.get("active_workspace", "default")
        
        new_ws = st.selectbox("Switch Workspace", workspaces, index=workspaces.index(active_ws) if active_ws in workspaces else 0)
        if new_ws != active_ws:
            if st.button(f"Switch to {new_ws}"):
                st.session_state.active_workspace = new_ws
                update_paths()
                st.session_state._app_loaded = False # Trigger reload
                st.rerun()
        
        st.divider()
        create_ws = st.text_input("Create New Workspace")
        if st.button("Create Workspace"):
            if create_ws:
                (BASE_DATA_DIR / create_ws).mkdir(exist_ok=True)
                st.success(f"Workspace '{create_ws}' created!")
                st.rerun()

    with tab_ui:
        st.markdown("### UI Preferences")
        dark_mode = st.checkbox("Dark Mode", value=st.session_state.get("dark_mode", False))
        if dark_mode != st.session_state.get("dark_mode", False):
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.divider()
        st.markdown("### System Limits")
        max_mem = st.number_input("Memory Warning Threshold (MB)", value=6000, step=1000)
        st.caption("UDIP will warn you when data in memory exceeds this limit.")
