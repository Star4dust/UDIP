"""
UDIP v3 — modules/state_manager.py
Unified state management across all pages for seamless navigation.
"""

import streamlit as st
from typing import Dict, Any, Optional
import json
import os
from modules.shared_utilities import get_shared_logger

class StateManager:
    """Manages persistent state across page navigation."""

    def __init__(self):
        self.state_file = "udip_data/app_state.json"
        self._initialize_state()

    def _initialize_state(self):
        """Initialize default state values."""
        defaults = {
            # Page states
            "visualize_state": {
                "ds_id": None,
                "cat_choice": "📊 Basic Charts",
                "chart_type": "Bar chart",
                "selections": {}
            },
            "ai_state": {
                "selected_agent": "udip_orchestrator",
                "task_mode": "auto",
                "response_style": "balanced",
                "focus_notes": "",
                "last_query": None
            },
            "agent_lab_state": {
                "blueprint": "Plan Mission",
                "task_brief": "",
                "constraints": "",
                "deliverable": "",
                "artifact_text": ""
            },
            "dashboard_state": {
                "selected_charts": [],
                "layout": "grid"
            },
            "sql_state": {
                "last_query": None,
                "selected_tables": []
            },
            "data_forge_state": {
                "operations": [],
                "preview_data": None
            },

            # Global shared state
            "global_filters": {},
            "active_datasets": [],
            "pinned_items": [],  # Global pinning system
            "user_preferences": {
                "theme": "light",
                "auto_save": True,
                "max_rows_display": 1000
            }
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def save_page_state(self, page_name: str, state: Dict[str, Any]):
        """Save state for a specific page."""
        st.session_state[f"{page_name}_state"] = state
        self._persist_state()

    def get_page_state(self, page_name: str) -> Dict[str, Any]:
        """Get state for a specific page."""
        return st.session_state.get(f"{page_name}_state", {})

    def update_global_filter(self, filter_name: str, value: Any):
        """Update a global filter that affects all pages."""
        if "global_filters" not in st.session_state:
            st.session_state.global_filters = {}
        st.session_state.global_filters[filter_name] = value
        self._persist_state()

    def get_global_filters(self) -> Dict[str, Any]:
        """Get all global filters."""
        return st.session_state.get("global_filters", {})

    def set_active_dataset(self, dataset_id: str):
        """Set the currently active dataset for cross-page operations."""
        st.session_state.active_dataset = dataset_id
        if dataset_id not in st.session_state.get("active_datasets", []):
            active = st.session_state.get("active_datasets", [])
            active.append(dataset_id)
            st.session_state.active_datasets = active[:5]  # Keep last 5
        self._persist_state()

    def get_active_dataset(self) -> Optional[str]:
        """Get the currently active dataset."""
        return st.session_state.get("active_dataset")

    def _persist_state(self):
        """Persist state to file for cross-session continuity."""
        try:
            state_to_save = {}
            # Only save specific keys that should persist
            persist_keys = [
                "visualize_state", "ai_state", "agent_lab_state", "dashboard_state",
                "sql_state", "data_forge_state", "global_filters",
                "active_datasets", "user_preferences"
            ]
            for key in persist_keys:
                if key in st.session_state:
                    state_to_save[key] = st.session_state[key]

            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2, default=str)
        except Exception as e:
            get_shared_logger().log(f"Failed to persist state: {e}", "error")

    def load_persistent_state(self):
        """Load persistent state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                for key, value in saved_state.items():
                    if key not in st.session_state:
                        st.session_state[key] = value
        except Exception as e:
            get_shared_logger().log(f"Failed to load persistent state: {e}", "error")

    def pin_item(self, item_type: str, item_data: Dict[str, Any], title: str):
        """Pin an item globally (chart, code, table, etc.)."""
        if "pinned_items" not in st.session_state:
            st.session_state.pinned_items = []

        pinned_item = {
            "id": f"{item_type}_{len(st.session_state.pinned_items)}",
            "type": item_type,
            "title": title,
            "data": item_data,
            "timestamp": str(__import__('datetime').datetime.now())
        }

        st.session_state.pinned_items.append(pinned_item)
        # Keep only last 20 pinned items
        st.session_state.pinned_items = st.session_state.pinned_items[-20:]
        self._persist_state()

    def unpin_item(self, item_id: str):
        """Remove a pinned item."""
        if "pinned_items" in st.session_state:
            st.session_state.pinned_items = [
                item for item in st.session_state.pinned_items
                if item["id"] != item_id
            ]
            self._persist_state()

    def reorder_item(self, item_id: str, direction: str):
        """Move a pinned item up or down in the list. direction: 'up' or 'down'."""
        items = st.session_state.get("pinned_items", [])
        idx = next((i for i, item in enumerate(items) if item["id"] == item_id), None)
        if idx is None:
            return
        if direction == "up" and idx > 0:
            items[idx], items[idx - 1] = items[idx - 1], items[idx]
        elif direction == "down" and idx < len(items) - 1:
            items[idx], items[idx + 1] = items[idx + 1], items[idx]
        else:
            return  # already at boundary — nothing to do
        st.session_state.pinned_items = items
        self._persist_state()

    def get_pinned_items(self, item_type: str = None) -> list:
        """Get all pinned items, optionally filtered by type."""
        items = st.session_state.get("pinned_items", [])
        if item_type:
            return [item for item in items if item["type"] == item_type]
        return items

    def clear_pinned_items(self, item_type: str = None):
        """Clear all pinned items or items of a specific type."""
        if item_type:
            st.session_state.pinned_items = [
                item for item in st.session_state.pinned_items
                if item["type"] != item_type
            ]
        else:
            st.session_state.pinned_items = []
        self._persist_state()


# Global instance
state_manager = StateManager()

def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    return state_manager


# ── PageMemory — lightweight per-page widget state helper ─────────────────────

class PageMemory:
    """
    Thin wrapper around a session_state dict for a single page.
    Keeps widget values alive across page navigation without touching
    any existing logic.

    Usage inside any render():
        mem = get_page_memory("sql")
        query = mem.get("query", DEFAULT_SQL)
        # ... render widget with value=query ...
        mem.set("query", query)
    """

    def __init__(self, page_key: str):
        self._key = f"_pmem_{page_key}"
        if self._key not in st.session_state:
            st.session_state[self._key] = {}

    def get(self, field: str, default=None):
        return st.session_state[self._key].get(field, default)

    def set(self, field: str, value):
        st.session_state[self._key][field] = value

    def get_index(self, field: str, options: list, default_idx: int = 0) -> int:
        """Return the saved value's index in options, or default_idx."""
        saved = self.get(field)
        if saved is not None and saved in options:
            return options.index(saved)
        return default_idx


def get_page_memory(page_key: str) -> PageMemory:
    """Get (or create) a PageMemory instance for the given page key."""
    return PageMemory(page_key)
