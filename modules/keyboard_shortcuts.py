"""
UDIP v3 — modules/keyboard_shortcuts.py
Keyboard navigation and hotkey system for enhanced UX.

Startup safety rule:
Keyboard shortcut registration is stored in module memory, not Streamlit session state.
This prevents startup crashes from missing shortcut keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict

import streamlit as st


@dataclass
class Shortcut:
    """Keyboard shortcut definition."""

    key: str
    description: str
    action: Callable
    category: str = "General"


_SHORTCUT_REGISTRY: Dict[str, Shortcut] = {}


class KeyboardManager:
    """Manages keyboard shortcuts and navigation without relying on session-scoped shortcut storage."""

    def __init__(self):
        self.ensure_state()

    def ensure_state(self) -> bool:
        """Keep only non-critical UI flags in session state; shortcuts live in _SHORTCUT_REGISTRY."""
        try:
            if "shortcut_help" not in st.session_state:
                st.session_state["shortcut_help"] = False
        except Exception:
            pass
        return True

    def register_shortcut(self, shortcut: Shortcut) -> None:
        """Register a new keyboard shortcut."""
        _SHORTCUT_REGISTRY[shortcut.key] = shortcut

    def get_shortcuts(self) -> Dict[str, Shortcut]:
        """Get all registered shortcuts from the module registry."""
        if not _SHORTCUT_REGISTRY:
            init_default_shortcuts()
        return dict(_SHORTCUT_REGISTRY)

    def trigger_shortcut(self, key: str) -> bool:
        """Trigger a shortcut by key."""
        shortcuts = self.get_shortcuts()
        if key not in shortcuts:
            return False
        try:
            shortcuts[key].action()
            return True
        except Exception as exc:
            try:
                st.error(f"Shortcut '{key}' failed: {exc}")
            except Exception:
                pass
            return False

    def show_help(self) -> None:
        """Display keyboard shortcuts help."""
        shortcuts = self.get_shortcuts()
        if not shortcuts:
            st.info("No keyboard shortcuts registered.")
            return

        categories: dict[str, list[Shortcut]] = {}
        for shortcut in shortcuts.values():
            categories.setdefault(shortcut.category, []).append(shortcut)

        st.markdown("### ⌨️ Keyboard Shortcuts")
        for category, category_shortcuts in categories.items():
            st.markdown(f"#### {category}")
            for shortcut in category_shortcuts:
                st.markdown(f"**{shortcut.key}** - {shortcut.description}")
        st.markdown("---")
        st.markdown("*Hold `Ctrl` (or `Cmd` on Mac) + key to trigger shortcuts*")


_keyboard_manager = KeyboardManager()


def init_default_shortcuts() -> None:
    """Initialize default keyboard shortcuts idempotently."""
    defaults = [
        Shortcut("Ctrl+1", "Go to Dashboard", lambda: navigate_to_page("📌 Dashboard"), "Navigation"),
        Shortcut("Ctrl+2", "Go to Upload Data", lambda: navigate_to_page("📁 Upload Data"), "Navigation"),
        Shortcut("Ctrl+3", "Go to Data Lab", lambda: navigate_to_page("⚒️ Data Lab"), "Navigation"),
        Shortcut("Ctrl+4", "Go to Visualize", lambda: navigate_to_page("📊 Visualize"), "Navigation"),
        Shortcut("Ctrl+/", "Toggle Search", toggle_search, "Navigation"),
        Shortcut("Ctrl+S", "Save current work", save_current_work, "Actions"),
        Shortcut("Ctrl+Z", "Undo last action", undo_last_action, "Actions"),
        Shortcut("Ctrl+Y", "Redo last action", redo_last_action, "Actions"),
        Shortcut("Ctrl+R", "Refresh current page", refresh_current_page, "Actions"),
        Shortcut("Ctrl+H", "Toggle keyboard shortcuts help", toggle_help, "Help"),
    ]
    for shortcut in defaults:
        _keyboard_manager.register_shortcut(shortcut)


def navigate_to_page(page_name: str) -> None:
    """Navigate to a specific page."""
    try:
        st.session_state["active_page"] = page_name
        st.rerun()
    except Exception:
        return


def toggle_search() -> None:
    """Toggle global search visibility."""
    try:
        current = bool(st.session_state.get("search_visible", False))
        st.session_state["search_visible"] = not current
        st.rerun()
    except Exception:
        return


def toggle_help() -> None:
    """Toggle keyboard shortcuts help."""
    try:
        st.session_state["shortcut_help"] = not bool(st.session_state.get("shortcut_help", False))
        st.rerun()
    except Exception:
        return


def save_current_work() -> None:
    """Save current work based on active page."""
    current_page = st.session_state.get("active_page", "")

    if "📊 Visualize" in current_page:
        chart_config = st.session_state.get("chart_config")
        if chart_config is not None:
            save_chart_config(chart_config)
            st.success("Chart configuration saved!")
    elif "🐍 Code Editor" in current_page:
        editor_code = st.session_state.get("editor_code")
        if editor_code is not None:
            try:
                from modules.core import save_code

                save_code(editor_code)
                st.success("Code saved!")
            except Exception as exc:
                st.warning(f"Code save unavailable: {exc}")
    elif "📁 Upload Data" in current_page:
        st.success("Datasets auto-saved!")
    else:
        st.info("No saveable content on current page.")


def undo_last_action() -> None:
    """Undo the last action."""
    undo_stack = st.session_state.setdefault("undo_stack", [])
    if undo_stack:
        last_action = undo_stack.pop()
        if isinstance(last_action, dict) and "data" in last_action and "target" in last_action:
            st.session_state[last_action["target"]] = last_action["data"]
            st.success(f"Undid: {last_action.get('description', 'Action')}")
            st.rerun()
    else:
        st.info("Nothing to undo.")


def redo_last_action() -> None:
    """Redo the last undone action."""
    redo_stack = st.session_state.setdefault("redo_stack", [])
    if redo_stack:
        last_action = redo_stack.pop()
        if isinstance(last_action, dict) and "data" in last_action and "target" in last_action:
            st.session_state[last_action["target"]] = last_action["data"]
            st.success(f"Redid: {last_action.get('description', 'Action')}")
            st.rerun()
    else:
        st.info("Nothing to redo.")


def refresh_current_page() -> None:
    """Refresh the current page."""
    st.rerun()


def save_chart_config(config: dict) -> None:
    """Save chart configuration to session."""
    saved_charts = st.session_state.setdefault("saved_charts", [])
    saved_charts.append({"timestamp": time.time(), "config": config})


def add_undo_action(target: str, data: object, description: str = "Action") -> None:
    """Add an action to the undo stack."""
    undo_stack = st.session_state.setdefault("undo_stack", [])
    current_data = st.session_state.get(target)
    if current_data is not None:
        undo_stack.append({"target": target, "data": current_data, "description": description})
        redo_stack = st.session_state.get("redo_stack")
        if isinstance(redo_stack, list):
            redo_stack.clear()


def render_keyboard_shortcuts_sidebar() -> None:
    """Render keyboard shortcuts in sidebar."""
    if st.sidebar.checkbox("⌨️ Show Shortcuts", key="show_shortcuts"):
        _keyboard_manager.show_help()


def inject_keyboard_javascript() -> None:
    """Inject JavaScript for keyboard shortcuts handling."""
    shortcuts = _keyboard_manager.get_shortcuts()
    if not shortcuts:
        return

    js_code = f"""
    <script>
    document.addEventListener('keydown', function(e) {{
        let key = '';
        if (e.ctrlKey || e.metaKey) {{
            key += e.ctrlKey ? 'Ctrl+' : 'Cmd+';
        }}
        if (e.shiftKey) key += 'Shift+';
        if (e.altKey) key += 'Alt+';

        const keyMap = {{
            'Slash': '/', 'Minus': '-', 'Equal': '=', 'Comma': ',',
            'Period': '.', 'Backquote': '`', 'BracketLeft': '[',
            'BracketRight': ']', 'Backslash': '\\\\', 'Semicolon': ';',
            'Quote': "'", 'Enter': 'Return', 'Space': ' '
        }};

        let keyName = keyMap[e.code] || e.key;
        key += keyName;

        const shortcuts = {list(shortcuts.keys())};
        if (shortcuts.includes(key)) {{
            e.preventDefault();
            e.stopPropagation();
        }}
    }});
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)


def get_keyboard_manager() -> KeyboardManager:
    """Get the global keyboard manager instance."""
    return _keyboard_manager


def check_keyboard_shortcut() -> bool:
    """Check if a keyboard shortcut was triggered."""
    shortcut_key = st.session_state.get("keyboard_shortcut")
    if shortcut_key:
        try:
            del st.session_state["keyboard_shortcut"]
        except Exception:
            pass
        return _keyboard_manager.trigger_shortcut(shortcut_key)
    return False


def handle_keyboard_shortcut() -> bool:
    """Compatibility wrapper used by app.py."""
    return check_keyboard_shortcut()


def ensure_keyboard_shortcut_state() -> bool:
    """Compatibility guard. Shortcuts are module-scoped, not session-scoped."""
    if not _SHORTCUT_REGISTRY:
        init_default_shortcuts()
    _keyboard_manager.ensure_state()
    return True


try:
    init_default_shortcuts()
except Exception:
    pass
