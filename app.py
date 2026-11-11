"""
UDIP v3.0.0 — Universal Data Integration & Analysis Platform
Main application entry point.
All logic is split into modules/ and pages/ for maintainability.
"""

import sys
import os
from pathlib import Path

# Ensure modules/ and udip_pages/ are always findable regardless of how app is launched
_APP_DIR = Path(__file__).parent.resolve()
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
os.chdir(_APP_DIR)  # Set working dir to app folder so udip_data/ saves correctly

import streamlit as st
import pandas as pd
import numpy as np
import io, re, uuid, time

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="UDIP - Radio Astronomy OS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ── Module imports ─────────────────────────────────────────────────────────────
from modules.core import (
    APP_VERSION, init_session, load_config, HAS_POLARS, MEMORY_WARN_MB, log
)
from modules.state_manager import get_state_manager
from modules.navigation import build_page_map, normalize_page
from modules.context_engine import init_context_engine
from modules.command_palette import init_command_palette, handle_keyboard_shortcut, render_command_palette
from modules.workflow_handoff import init_workflow_handoffs
from modules.stability_improvements import apply_stability_improvements
from modules.platform_health import init_platform_health
from modules.scalability_manager import init_scalability_manager
from modules.consolidated_utils import initialize_consolidated_utils
from modules.keyboard_shortcuts import (
    get_keyboard_manager, check_keyboard_shortcut,
    render_keyboard_shortcuts_sidebar, inject_keyboard_javascript,
    ensure_keyboard_shortcut_state
)
from modules.ui_components import (
    render_command_center_header, render_auth_gate, get_loading_manager
)


# ── Theme ──────────────────────────────────────────────────────────────────────
from modules.theme_system import get_theme_variables, load_css
theme_vars, st_app_bg = get_theme_variables()
st.markdown(load_css(theme_vars, st_app_bg), unsafe_allow_html=True)

# ── Session init ───────────────────────────────────────────────────────────────
init_session()

# ── Load persistent state ─────────────────────────────────────────────────────
state_mgr = get_state_manager()
state_mgr.load_persistent_state()


def _safe_optional(label, func, *args, **kwargs):
    """Run an optional startup step. Never raises — failures are logged and ignored."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        try:
            log.warning(f"Optional startup component skipped ({label}): {exc}")
        except Exception:
            pass
        return None


# ── Auth Gate ──────────────────────────────────────────────────────────────────
cfg = load_config()
app_password = cfg.get("app_password", "")
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state.get("authenticated", False):
        if render_auth_gate(app_password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.stop()

# ── Global Widget State Preservation ──────────────────────────────────────────
if "_saved_widget_states" not in st.session_state:
    st.session_state["_saved_widget_states"] = {}

for k, v in st.session_state["_saved_widget_states"].items():
    if k not in st.session_state:
        st.session_state[k] = v

_exclude_keys = {
    "datasets", "history", "chat_history", "editor_code", "dashboard_pins",
    "ai_memory", "schedules", "active_page", "_last_run_error",
    "_pending_reply", "_last_ai_code", "groq_key", "_app_loaded",
    "_saved_widget_states", "quantum_mode", "uploader_cache"
}
_safe_bool_keys = {"vanguard_toggle", "show_code_tog", "dark_mode"}

for k, v in list(st.session_state.items()):
    if k not in _exclude_keys and not k.startswith("_"):
        if type(v) is bool and k not in _safe_bool_keys:
            continue
        if type(v) in (int, float, str, bool):
            st.session_state["_saved_widget_states"][k] = v
        elif type(v) in (list, tuple) and (not v or all(type(x) in (int, float, str, bool) for x in v)):
            st.session_state["_saved_widget_states"][k] = v

for k, v in list(st.session_state.items()):
    if k.startswith("_pmem_") and isinstance(v, dict):
        st.session_state["_saved_widget_states"][k] = v

# ── Optional Startup Enhancements ─────────────────────────────────────────────
_safe_optional("context engine",       init_context_engine)
_safe_optional("command palette",      init_command_palette)
_safe_optional("workflow handoffs",    init_workflow_handoffs)
_safe_optional("stability improvements", apply_stability_improvements)
_safe_optional("platform health",      init_platform_health)
_safe_optional("scalability manager",  init_scalability_manager)
_safe_optional("consolidated utils",   initialize_consolidated_utils)

# ── Sidebar & Routing ──────────────────────────────────────────────────────────
from modules.sidebar import render_sidebar
from modules.page_context import get_page_context

page_map = build_page_map({})
page = render_sidebar(page_map)
page = normalize_page(page, page_map)
st.session_state["active_page"] = page

_safe_optional("page context", get_page_context)

# ── Header stats ───────────────────────────────────────────────────────────────
_cfg = load_config()
n_ds = len(st.session_state.get("datasets", {}))
total_rows = sum(
    v["df"].shape[0]
    for v in st.session_state.get("datasets", {}).values()
    if isinstance(v, dict) and "df" in v
)
current_page = page.partition(" ")[2] or page
ai_state = "AI copilot online" if _cfg.get("groq_key") else "AI setup needed"
engine_state = "Polars acceleration active" if HAS_POLARS else "Pandas engine active"
render_command_center_header(current_page, n_ds, total_rows, ai_state, engine_state)

# ── Keyboard shortcuts (optional UX — must never block startup) ────────────────
_safe_optional("keyboard shortcut state",   ensure_keyboard_shortcut_state)
_safe_optional("keyboard shortcut check",   check_keyboard_shortcut)
_safe_optional("keyboard JS injection",     inject_keyboard_javascript)
_safe_optional("command palette keyboard",  handle_keyboard_shortcut)

# ── Command palette (optional — must never block startup) ──────────────────────
try:
    _cp = st.session_state.get("command_palette")
    if isinstance(_cp, dict) and _cp.get("is_open", False):
        render_command_palette()
except Exception as exc:
    _safe_optional("command palette render warning", log.warning, f"Command palette render skipped: {exc}")

# ── Page routing ───────────────────────────────────────────────────────────────
try:
    page_map[page]()
except Exception as exc:
    st.error("🚨 **Page Error**: Something went wrong while rendering this section.")
    with st.expander("Technical details"):
        st.code(str(exc), language="python")
    try:
        log.error(f"Route render failed for {page}: {exc}")
    except Exception:
        pass

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"UDIP v{APP_VERSION} | Local Workspace: {st.session_state.get('active_workspace', 'default')}")
