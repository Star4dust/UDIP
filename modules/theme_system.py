"""
UDIP v3 — modules/theme_system.py
Single source of truth for theme management and CSS styles.
Radio Astronomy OS — Unified Theme System
"""

import streamlit as st


# ============================================================================
# THEME STATE MANAGEMENT
# ============================================================================

def get_theme_mode():
    """
    Get the current theme mode from session state.
    
    Returns:
        bool: True if dark mode is enabled, False otherwise
    """
    return st.session_state.get("dark_mode", False)


def set_theme_mode(is_dark: bool):
    """
    Set the theme mode in session state.
    
    Args:
        is_dark: True for dark mode, False for light mode
    """
    st.session_state["dark_mode"] = is_dark


def get_theme_variables():
    """
    Get theme variables based on current theme mode.
    Single source of truth for theme calculations.
    
    Returns:
        tuple: (theme_vars dict, st_app_bg string)
    """
    is_dark = get_theme_mode()
    
    if is_dark:
        theme_vars = {
            "bg": "#020617",
            "surface": "rgba(30, 41, 59, 0.75)",
            "surface-strong": "#1e293b",
            "surface-soft": "#0f172a",
            "ink": "#f8fafc",
            "ink-muted": "#94a3b8",
            "line": "rgba(148, 163, 184, 0.12)",
            "shadow": "0 20px 50px rgba(0, 0, 0, 0.5)",
        }
        st_app_bg = "radial-gradient(circle at top left, rgba(15, 118, 110, 0.15), transparent 30%), radial-gradient(circle at top right, rgba(56, 189, 248, 0.10), transparent 25%), #020617"
    else:
        theme_vars = {
            "bg": "#f3f7f6",
            "surface": "rgba(255, 255, 255, 0.82)",
            "surface-strong": "#ffffff",
            "surface-soft": "#edf5f4",
            "ink": "#0f172a",
            "ink-muted": "#334155",
            "line": "rgba(15, 23, 42, 0.08)",
            "shadow": "0 18px 44px rgba(15, 23, 42, 0.08)",
        }
        st_app_bg = "radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 26%), radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 22%), #f8fbfc"
    
    return theme_vars, st_app_bg


# ============================================================================
# CSS STYLES
# ============================================================================

def load_css(theme_vars, st_app_bg):
    """
    Returns the main CSS style block for the UDIP platform.
    Single source of truth for all CSS.
    
    Args:
        theme_vars: Theme variables dictionary
        st_app_bg: Background gradient string
    
    Returns:
        str: Complete CSS block
    """
    return f'''
<style>
    :root {{
        --udip-bg: {theme_vars['bg']};
        --udip-surface: {theme_vars['surface']};
        --udip-surface-strong: {theme_vars['surface-strong']};
        --udip-surface-soft: {theme_vars['surface-soft']};
        --udip-ink: {theme_vars['ink']};
        --udip-ink-muted: {theme_vars['ink-muted']};
        --udip-line: {theme_vars['line']};
        --udip-line-strong: rgba(15, 118, 110, 0.20);
        --udip-shadow: {theme_vars['shadow']};
        --udip-accent: #0f766e;
        --udip-accent-strong: #115e59;
        --udip-accent-soft: rgba(15, 118, 110, 0.12);
        --udip-warm: #f59e0b;
        --udip-warm-soft: rgba(245, 158, 11, 0.12);
        --udip-sidebar-bg: #0f172a;
        --udip-sidebar-surface: rgba(148, 163, 184, 0.10);
        --udip-sidebar-line: rgba(148, 163, 184, 0.18);
    }}
    html, body, [class*="css"] {{
        font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    .stApp {{
        color: var(--udip-ink);
        background: {st_app_bg};
        transition: background 0.3s ease;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.28;
        background-image:
            linear-gradient(rgba(15, 23, 42, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.02) 1px, transparent 1px);
        background-size: 36px 36px;
    }}
    section[data-testid="stSidebar"] {{
        background:
            radial-gradient(circle at top, rgba(34, 211, 238, 0.10), transparent 24%),
            linear-gradient(180deg, #0f172a 0%, #111827 58%, #162033 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }}
    section[data-testid="stSidebar"] > div {{
        background: transparent;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {{
        display: none;
    }}
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: #94a3b8 !important;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] .stMarkdown h4 {{
        margin-top: 1.25rem;
        margin-bottom: 0.65rem;
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-size: 0.75rem;
        color: #67e8f9;
        opacity: 0.85;
        border-bottom: 1px solid rgba(103, 232, 249, 0.15);
        padding-bottom: 0.35rem;
    }}
    .udip-shell {{
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.45rem 1.25rem 1.45rem;
        margin: 0.2rem 0 1.25rem 0;
        border: 1px solid var(--udip-line-strong);
        border-radius: 24px;
        background:
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 26%),
            var(--udip-surface);
        box-shadow: var(--udip-shadow);
    }}
    .udip-shell::after {{
        content: "";
        position: absolute;
        inset: auto -20% -55% auto;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(15, 118, 110, 0.14), transparent 70%);
    }}
    .udip-shell__eyebrow {{
        position: relative;
        z-index: 1;
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--udip-accent);
    }}
    .udip-shell__title {{
        position: relative;
        z-index: 1;
        margin-top: 0.35rem;
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 2rem;
        line-height: 1.05;
        color: var(--udip-ink);
    }}
    .udip-shell__caption {{
        position: relative;
        z-index: 1;
        margin-top: 0.55rem;
        max-width: 920px;
        color: var(--udip-ink-muted);
        font-size: 0.98rem;
    }}
    .udip-shell__chips {{
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }}
    .udip-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        border: 1px solid var(--udip-line);
        background: var(--udip-surface-strong);
        color: var(--udip-ink);
        font-size: 0.84rem;
        font-weight: 600;
    }}
    .udip-chip--accent {{
        border-color: rgba(15, 118, 110, 0.18);
        background: var(--udip-accent-soft);
        color: var(--udip-accent-strong);
    }}
    .udip-chip--warm {{
        border-color: rgba(245, 158, 11, 0.20);
        background: var(--udip-warm-soft);
        color: #b45309;
    }}
    .main-header {{
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: var(--udip-ink);
        padding: 1rem 1.2rem;
        border: 1px solid var(--udip-line-strong);
        border-radius: 22px;
        margin: 0.25rem 0 1.35rem 0;
        background: var(--udip-surface);
        box-shadow: var(--udip-shadow);
    }}
    .stat-box {{
        background: var(--udip-surface);
        border: 1px solid var(--udip-line);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
    }}
    .stat-number {{
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--udip-accent-strong);
    }}
    .stat-label {{ 
        font-size: 0.85rem; 
        color: var(--udip-ink);
        font-weight: 600;
        margin-top: 2px; 
    }}
    .ds-card {{
        background: var(--udip-surface);
        border: 1px solid var(--udip-line);
        border-radius: 18px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
    }}
    .ds-name {{ font-weight: 700; font-size: 1rem; color: var(--udip-ink); }}
    .ds-meta {{ 
        font-size: 0.8rem; 
        color: var(--udip-ink-muted); 
        font-weight: 500;
        margin-top: 2px; 
    }}
    .history-item {{
        border-left: 3px solid var(--udip-accent); padding-left: 0.8rem;
        margin-bottom: 0.6rem;
    }}
    .history-action {{ font-weight: 600; font-size: 0.9rem; }}
    .history-detail {{ font-size: 0.82rem; color: var(--udip-ink-muted); }}
    .history-time {{ font-size: 0.75rem; color: #94a3b8; }}
    .chat-user {{
        background: linear-gradient(135deg, #0f766e, #115e59); color: white;
        padding: 0.7rem 1rem; border-radius: 12px 12px 4px 12px;
        margin: 0.4rem 0 0.4rem 3rem; font-size: 0.9rem;
    }}
    .chat-bot {{
        background: var(--udip-surface); color: var(--udip-ink);
        padding: 0.7rem 1rem; border-radius: 12px 12px 12px 4px;
        margin: 0.4rem 3rem 0.4rem 0; font-size: 0.9rem;
        border: 1px solid var(--udip-line);
    }}
    .stButton > button {{
        border-radius: 14px !important;
        border: 1px solid var(--udip-line) !important;
        background: rgba(255, 255, 255, 0.88) !important;
        color: var(--udip-ink) !important;
        font-weight: 600 !important;
        min-height: 2.7rem;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }}
    .stButton > button:hover {{
        border-color: rgba(15, 118, 110, 0.24) !important;
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.10);
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #0f766e, #115e59) !important;
        color: #f8fafc !important;
        border-color: transparent !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: var(--udip-sidebar-surface) !important;
        border-color: var(--udip-sidebar-line) !important;
        color: #f8fafc !important;
        box-shadow: none;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 1.25rem !important;
        height: 2.85rem !important;
        margin-bottom: 0.35rem !important;
        font-size: 0.92rem !important;
        border-radius: 12px !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        border-color: rgba(103, 232, 249, 0.44) !important;
        background: rgba(148, 163, 184, 0.18) !important;
        transform: translateX(4px);
    }}
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
        color: #ffffff !important;
        border-left: 4px solid #67e8f9 !important;
        padding-left: 1rem !important;
    }}
    .sidebar-signal,
    .sidebar-pill {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.8rem 0.9rem;
        border-radius: 16px;
        border: 1px solid var(--udip-sidebar-line);
        background: var(--udip-sidebar-surface);
        color: #e2e8f0;
        font-size: 0.84rem;
        font-weight: 600;
        text-align: center;
    }}
    .sidebar-signal__dot {{
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.65);
        animation: pulse-green 2s infinite;
    }}
    .sidebar-signal--off .sidebar-signal__dot {{
        background: #f59e0b;
        box-shadow: none;
        animation: none;
    }}
    .sidebar-pill--accent {{
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.88), rgba(20, 184, 166, 0.82));
        border-color: transparent;
        color: #f8fafc;
    }}
    [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stDateInput input,
    .stTimeInput input {{
        border-radius: 14px !important;
        border: 1px solid var(--udip-line) !important;
        background: var(--udip-surface-strong) !important;
        color: var(--udip-ink) !important;
    }}
    
    /* Active/Focus state */
    [data-baseweb="select"] > div:focus-within,
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus,
    .stDateInput input:focus,
    .stTimeInput input:focus {{
        border-color: var(--udip-accent) !important;
        box-shadow: 0 0 0 1px var(--udip-accent-soft) !important;
    }}

    /* Sidebar specific overrides for contrast */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: rgba(148, 163, 184, 0.08) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        color: #f8fafc !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        fill: #f8fafc !important;
    }}

    section[data-testid="stMain"] [data-testid="stWidgetLabel"],
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stMain"] .stSelectbox label,
    section[data-testid="stMain"] .stMultiSelect label,
    section[data-testid="stMain"] .stSlider label,
    section[data-testid="stMain"] .stTextInput label,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] .stCaption,
    section[data-testid="stMain"] .stMarkdown p {{
        color: var(--udip-ink) !important;
        opacity: 1 !important;
        text-shadow: none !important;
        font-weight: 600;
    }}
    section[data-testid="stMain"] [data-testid="stAlert"] *,
    section[data-testid="stMain"] [data-testid="stAlert"] p,
    section[data-testid="stMain"] [data-testid="stAlert"] span {{
        color: var(--udip-ink) !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetric"] {{
        border: 1px solid var(--udip-line);
        border-radius: 18px;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.05);
    }}
    [data-baseweb="tab-list"] {{
        gap: 0.45rem;
        padding: 0.35rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.04);
    }}
    button[data-baseweb="tab"] {{
        border-radius: 999px !important;
        min-height: 2.45rem;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #0f766e, #115e59) !important;
        color: #f8fafc !important;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid var(--udip-line);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.03);
        margin-bottom: 0.75rem;
    }}
    [data-testid="stExpander"] summary {{
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stFileUploader"] {{
        border: 1.6px dashed rgba(15, 118, 110, 0.42) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(234, 245, 243, 0.96));
    }}
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {{
        border: 1px solid var(--udip-line);
        border-radius: 18px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
    }}
    [data-testid="stAlert"] {{
        border-radius: 18px;
        border-width: 1px;
    }}
    .stCaption {{
        color: var(--udip-ink-muted);
    }}
    /* AI Chat Styling - Prevent oversized headers */
    [data-testid="stChatMessage"] h1, 
    [data-testid="stChatMessage"] h2 {{
        font-size: 1.45rem !important;
        margin-top: 1rem !important;
        color: var(--udip-accent) !important;
    }}
    [data-testid="stChatMessage"] h3 {{
        font-size: 1.15rem !important;
    }}
    [data-testid="stChatMessage"] p {{
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }}
    .udip-panel {{
        position: relative;
        overflow: hidden;
        margin: 0.25rem 0 1.1rem 0;
        padding: 1.15rem 1.2rem;
        border-radius: 22px;
        border: 1px solid var(--udip-line);
        background: var(--udip-surface);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.06);
    }}
    .udip-panel::after {{
        content: "";
        position: absolute;
        top: -30%;
        right: -8%;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(15, 118, 110, 0.10), transparent 70%);
    }}
    .udip-panel__eyebrow {{
        position: relative;
        z-index: 1;
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--udip-accent);
    }}
    .udip-panel__title {{
        position: relative;
        z-index: 1;
        margin-top: 0.3rem;
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 1.45rem;
        color: var(--udip-ink);
    }}
    .udip-panel__text {{
        position: relative;
        z-index: 1;
        margin-top: 0.45rem;
        color: var(--udip-ink-muted);
        font-size: 0.95rem;
        max-width: 920px;
    }}
    .udip-stat-strip {{
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }}
    .udip-stat-tile {{
        padding: 0.8rem 0.9rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.07);
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(6px);
    }}
    .udip-stat-tile__value {{
        font-family: "Bahnschrift", "Segoe UI", sans-serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--udip-ink);
    }}
    .udip-stat-tile__label {{
        margin-top: 0.2rem;
        font-size: 0.78rem;
        color: var(--udip-ink-muted);
    }}
    .udip-pill-row {{
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }}
    .udip-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        border: 1px solid var(--udip-line);
        background: var(--udip-surface-strong);
        backdrop-filter: blur(8px);
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--udip-ink);
    }}
    .udip-pill--accent {{
        border-color: rgba(15, 118, 110, 0.16);
        background: var(--udip-accent-soft);
        color: var(--udip-accent-strong);
    }}
    .udip-pill--warm {{
        border-color: rgba(245, 158, 11, 0.18);
        background: var(--udip-warm-soft);
        color: #b45309;
    }}
    /* Enhanced Navigation Buttons - Static Classes */
    .nav-button {{
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        background: rgba(148, 163, 184, 0.08);
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 600;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
        margin: 2px 0;
        box-sizing: border-box;
    }}
    .nav-button:hover {{
        border-color: rgba(103, 232, 249, 0.44);
        background: rgba(148, 163, 184, 0.18);
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }}
    .nav-button-active {{
        background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
        color: #ffffff !important;
        border-left: 4px solid #67e8f9 !important;
        padding-left: 0.8rem !important;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3);
    }}
    .nav-button-active:hover {{
        transform: translateX(2px);
        box-shadow: 0 8px 20px rgba(15, 118, 110, 0.4);
    }}
    /* Sidebar button overrides */
    section[data-testid="stSidebar"] .stButton > button {{
        background: rgba(148, 163, 184, 0.08) !important;
        border-color: rgba(148, 163, 184, 0.15) !important;
        color: #e2e8f0 !important;
        box-shadow: none;
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 1rem !important;
        height: 2.85rem !important;
        margin-bottom: 2px !important;
        font-size: 0.92rem !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        border-color: rgba(103, 232, 249, 0.44) !important;
        background: rgba(148, 163, 184, 0.18) !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }}
    section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
        color: #ffffff !important;
        border-left: 4px solid #67e8f9 !important;
        padding-left: 0.8rem !important;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3) !important;
    }}
    /* Hide the actual Streamlit buttons but keep them functional */
    section[data-testid="stSidebar"] .stButton button[style*="visibility: hidden"],
    section[data-testid="stSidebar"] .stButton button[style*="display: none"] {{
        position: absolute;
        opacity: 0;
        pointer-events: none;
    }}
</style>
'''


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def inject_theme():
    """
    Convenience function to inject the complete theme into the page.
    This is the single CSS injection point for the entire platform.
    """
    theme_vars, st_app_bg = get_theme_variables()
    css = load_css(theme_vars, st_app_bg)
    st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# For compatibility with existing imports
def load_styles(theme_vars, st_app_bg):
    """Backward compatibility alias for load_css"""
    return load_css(theme_vars, st_app_bg)
