"""
UDIP v3 — modules/undo_redo.py
Undo/Redo system for data operations.
"""

import streamlit as st
import time
import copy
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class UndoAction:
    """Represents an undoable action."""
    action_type: str
    description: str
    target: str  # What this action affects
    before_state: Any
    after_state: Any
    timestamp: float
    metadata: Dict[str, Any] = None


class UndoRedoManager:
    """Manages undo/redo functionality."""
    
    def __init__(self):
        if "undo_stack" not in st.session_state:
            st.session_state.undo_stack = []
        if "redo_stack" not in st.session_state:
            st.session_state.redo_stack = []
        if "undo_settings" not in st.session_state:
            st.session_state.undo_settings = {
                "max_undo_items": 50,
                "auto_save": True,
                "track_data_changes": True,
                "track_ui_changes": True
            }
    
    def add_action(
        self,
        action_type: str,
        description: str,
        target: str,
        before_state: Any,
        after_state: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an action to the undo stack."""
        
        action = UndoAction(
            action_type=action_type,
            description=description,
            target=target,
            before_state=copy.deepcopy(before_state),
            after_state=copy.deepcopy(after_state),
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Add to undo stack
        st.session_state.undo_stack.append(action)
        
        # Clear redo stack when new action is performed
        st.session_state.redo_stack.clear()
        
        # Limit stack size
        max_items = st.session_state.undo_settings["max_undo_items"]
        if len(st.session_state.undo_stack) > max_items:
            st.session_state.undo_stack = st.session_state.undo_stack[-max_items:]
        
        # Show notification
        if st.session_state.undo_settings["auto_save"]:
            st.success(f"✅ {description}")
    
    def undo(self) -> bool:
        """Undo the last action."""
        if not st.session_state.undo_stack:
            st.info("Nothing to undo.")
            return False
        
        action = st.session_state.undo_stack.pop()
        
        try:
            # Restore the before state
            self._restore_state(action.target, action.before_state)
            
            # Add to redo stack
            st.session_state.redo_stack.append(action)
            
            st.success(f"↶️ Undid: {action.description}")
            return True
            
        except Exception as e:
            st.error(f"Failed to undo: {e}")
            # Put action back on undo stack
            st.session_state.undo_stack.append(action)
            return False
    
    def redo(self) -> bool:
        """Redo the last undone action."""
        if not st.session_state.redo_stack:
            st.info("Nothing to redo.")
            return False
        
        action = st.session_state.redo_stack.pop()
        
        try:
            # Restore the after state
            self._restore_state(action.target, action.after_state)
            
            # Add back to undo stack
            st.session_state.undo_stack.append(action)
            
            st.success(f"↷️ Redid: {action.description}")
            return True
            
        except Exception as e:
            st.error(f"Failed to redo: {e}")
            # Put action back on redo stack
            st.session_state.redo_stack.append(action)
            return False
    
    def _restore_state(self, target: str, state: Any):
        """Restore a state to its target."""
        if target.startswith("dataset_"):
            # Restore dataset
            ds_id = target.replace("dataset_", "")
            if ds_id in st.session_state.datasets:
                # Update DataFrame in place
                if isinstance(state, dict) and "df" in state:
                    st.session_state.datasets[ds_id]["df"] = state["df"]
                else:
                    st.session_state.datasets[ds_id] = state
        
        elif target.startswith("session_"):
            # Restore session state
            key = target.replace("session_", "")
            st.session_state[key] = state
        
        elif target in st.session_state:
            # Direct session state update
            st.session_state[target] = state
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(st.session_state.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(st.session_state.redo_stack) > 0
    
    def get_undo_history(self) -> List[UndoAction]:
        """Get undo history."""
        return st.session_state.undo_stack.copy()
    
    def get_redo_history(self) -> List[UndoAction]:
        """Get redo history."""
        return st.session_state.redo_stack.copy()
    
    def clear_history(self):
        """Clear undo/redo history."""
        st.session_state.undo_stack.clear()
        st.session_state.redo_stack.clear()
        st.success("Undo/Redo history cleared.")
    
    def get_action_history(self, limit: int = 20) -> List[UndoAction]:
        """Get combined action history."""
        all_actions = []
        
        # Add undo actions (reversed to show most recent first)
        for action in reversed(st.session_state.undo_stack[-limit:]):
            all_actions.append({
                "action": "UNDO",
                "description": action.description,
                "target": action.target,
                "timestamp": action.timestamp,
                "type": action.action_type
            })
        
        # Add redo actions
        for action in st.session_state.redo_stack[-limit:]:
            all_actions.append({
                "action": "REDO",
                "description": action.description,
                "target": action.target,
                "timestamp": action.timestamp,
                "type": action.action_type
            })
        
        return all_actions
    
    def render_undo_controls(self):
        """Render undo/redo controls."""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            undo_enabled = self.can_undo()
            if st.button(
                "↶️ Undo",
                key="undo_action",
                disabled=not undo_enabled,
                help="Undo last action (Ctrl+Z)"
            ):
                self.undo()
        
        with col2:
            redo_enabled = self.can_redo()
            if st.button(
                "↷️ Redo", 
                key="redo_action",
                disabled=not redo_enabled,
                help="Redo last undone action (Ctrl+Y)"
            ):
                self.redo()
        
        with col3:
            if st.button(
                "🗑️ Clear History",
                key="clear_undo_history",
                help="Clear all undo/redo history"
            ):
                self.clear_history()
    
    def render_history_panel(self):
        """Render detailed action history panel."""
        st.markdown("### 📋 Action History")
        
        # Settings
        settings = st.session_state.undo_settings
        col1, col2 = st.columns(2)
        
        with col1:
            settings["max_undo_items"] = st.number_input(
                "Max History Items",
                min_value=10,
                max_value=200,
                value=settings["max_undo_items"],
                help="Maximum number of actions to remember"
            )
        
        with col2:
            settings["auto_save"] = st.checkbox(
                "Auto-save Actions",
                value=settings["auto_save"],
                help="Automatically track actions for undo/redo"
            )
        
        st.session_state.undo_settings = settings
        
        # Action history
        history = self.get_action_history()
        
        if history:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                action_filter = st.selectbox(
                    "Filter by Action",
                    options=["All", "UNDO", "REDO"],
                    key="undo_action_filter"
                )
            
            with col2:
                time_filter = st.selectbox(
                    "Time Range",
                    options=["All time", "Last hour", "Last 24 hours", "Last week"],
                    key="undo_time_filter"
                )
            
            # Apply filters
            filtered_history = self._filter_history(history, action_filter, time_filter)
            
            # Display history
            for i, action in enumerate(filtered_history[:50]):  # Limit display
                with st.expander(
                    f"{action['action']}: {action['description']} - {time.strftime('%H:%M:%S', time.localtime(action['timestamp']))}",
                    expanded=False
                ):
                    st.markdown(f"**Action:** {action['action']}")
                    st.markdown(f"**Description:** {action['description']}")
                    st.markdown(f"**Target:** {action['target']}")
                    st.markdown(f"**Type:** {action['type']}")
                    st.markdown(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(action['timestamp']))}")
                    
                    if action.get('metadata'):
                        st.markdown("**Metadata:**")
                        st.json(action['metadata'])
        else:
            st.info("No action history available.")
    
    def _filter_history(self, history: List[Dict], action_filter: str, time_filter: str) -> List[Dict]:
        """Filter action history."""
        current_time = time.time()
        
        # Filter by action type
        if action_filter != "All":
            history = [h for h in history if h["action"] == action_filter]
        
        # Filter by time
        if time_filter == "Last hour":
            history = [h for h in history if current_time - h["timestamp"] < 3600]
        elif time_filter == "Last 24 hours":
            history = [h for h in history if current_time - h["timestamp"] < 86400]
        elif time_filter == "Last week":
            history = [h for h in history if current_time - h["timestamp"] < 604800]
        
        return history


# Global undo/redo manager instance
_undo_redo_manager = UndoRedoManager()


def track_data_change(
    action_type: str,
    description: str,
    dataset_id: str,
    before_df: Any,
    after_df: Any,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track a data change for undo/redo."""
    _undo_redo_manager.add_action(
        action_type=action_type,
        description=description,
        target=f"dataset_{dataset_id}",
        before_state={"df": before_df},
        after_state={"df": after_df},
        metadata=metadata
    )


def track_session_change(
    action_type: str,
    description: str,
    key: str,
    before_value: Any,
    after_value: Any,
    metadata: Optional[Dict[str, Any]] = None
):
    """Track a session state change for undo/redo."""
    _undo_redo_manager.add_action(
        action_type=action_type,
        description=description,
        target=f"session_{key}",
        before_state=before_value,
        after_state=after_value,
        metadata=metadata
    )


def undo_last_action() -> bool:
    """Undo the last action."""
    return _undo_redo_manager.undo()


def redo_last_action() -> bool:
    """Redo the last undone action."""
    return _undo_redo_manager.redo()


def can_undo() -> bool:
    """Check if undo is available."""
    return _undo_redo_manager.can_undo()


def can_redo() -> bool:
    """Check if redo is available."""
    return _undo_redo_manager.can_redo()


def render_undo_controls():
    """Render undo/redo controls."""
    _undo_redo_manager.render_undo_controls()


def render_undo_history():
    """Render undo/redo history panel."""
    _undo_redo_manager.render_history_panel()


def get_undo_manager() -> UndoRedoManager:
    """Get the global undo/redo manager."""
    return _undo_redo_manager
