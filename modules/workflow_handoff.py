"""
UDIP v3 — modules/workflow_handoff.py
Cross-Page Smart Handoffs for seamless workflow transitions.

Enables intelligent context passing between modules, creating a connected
workspace experience instead of isolated tools.
"""

import streamlit as st
import time
import uuid
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

# Safe imports to avoid circular dependencies
try:
    from modules.context_engine import get_context_engine
    from modules.navigation import normalize_page
    from modules.experiment_tracker import get_experiment_tracker
except ImportError:
    get_context_engine = None
    normalize_page = lambda x: x
    get_experiment_tracker = None


class HandoffType(Enum):
    """Types of handoff payloads."""
    DATAFRAME = "dataframe"
    CHART = "chart"
    SQL_RESULT = "sql_result"
    ML_MODEL = "ml_model"
    HIRAX_SPECTRAL = "hirax_spectral"
    HIRAX_FFT = "hirax_fft"
    HIRAX_RFI = "hirax_rfi"
    EXPERIMENT = "experiment"
    REPORT = "report"
    TRANSFORMATION = "transformation"
    ANALYSIS = "analysis"


class HandoffPriority(Enum):
    """Priority levels for handoff suggestions."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class HandoffPayload:
    """Represents a handoff payload containing context and data."""
    id: str
    type: HandoffType
    source_page: str
    target_page: str
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)  # Previous handoff IDs
    created_at: float = field(default_factory=time.time)
    priority: HandoffPriority = HandoffPriority.MEDIUM
    icon: str = "🔄"
    requires_action: bool = False


@dataclass
class HandoffAction:
    """Represents a handoff action that can be performed."""
    id: str
    title: str
    description: str
    target_page: str
    payload_type: HandoffType
    icon: str = "→"
    priority: HandoffPriority = HandoffPriority.MEDIUM
    condition: Optional[Callable] = None  # Function to check if action is available
    action: Optional[Callable] = None  # Custom action function


@dataclass
class WorkflowTransition:
    """Represents a workflow transition between stages."""
    from_stage: str
    to_stage: str
    trigger_action: str
    suggested_actions: List[HandoffAction] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)


class WorkflowHandoffManager:
    """
    Manages cross-page smart handoffs and workflow transitions.
    
    Provides seamless context passing between modules with intelligent
    recommendations and lineage tracking.
    """
    
    def __init__(self):
        """Initialize the workflow handoff manager."""
        self._logger = logging.getLogger(__name__)
        self._active_handoffs: Dict[str, HandoffPayload] = {}
        self._handoff_history: List[HandoffPayload] = []
        self._handoff_actions: Dict[str, HandoffAction] = {}
        self._workflow_transitions: Dict[str, WorkflowTransition] = {}
        self._initialize_session_state()
        self._initialize_handoff_actions()
        self._initialize_workflow_transitions()
    
    def _initialize_session_state(self):
        """Initialize session state for workflow handoffs."""
        if "workflow_handoffs" not in st.session_state:
            st.session_state.workflow_handoffs = {
                "active_payloads": {},
                "handoff_history": [],
                "pending_suggestions": [],
                "last_handoff": None,
                "workflow_lineage": {}
            }
    
    def _initialize_handoff_actions(self):
        """Initialize all available handoff actions."""
        # Data-focused handoffs
        self._register_handoff_action(HandoffAction(
            id="send_to_visualize",
            title="Send to Visualize",
            description="Create visualizations from this data",
            target_page="📊 Visualize",
            payload_type=HandoffType.DATAFRAME,
            icon="📊",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_data(),
            action=self._action_send_to_visualize
        ))
        
        self._register_handoff_action(HandoffAction(
            id="send_to_data_forge",
            title="Open in Data Forge",
            description="Clean and transform this data",
            target_page="⚒️ Data Forge",
            payload_type=HandoffType.DATAFRAME,
            icon="🧹",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_data(),
            action=self._action_send_to_data_forge
        ))
        
        self._register_handoff_action(HandoffAction(
            id="send_to_ml",
            title="Open in Machine Learning",
            description="Train ML models on this data",
            target_page="🧠 Machine Learning",
            payload_type=HandoffType.DATAFRAME,
            icon="🤖",
            priority=HandoffPriority.MEDIUM,
            condition=lambda: self._has_active_data(),
            action=self._action_send_to_ml
        ))
        
        self._register_handoff_action(HandoffAction(
            id="send_to_sql",
            title="Open in SQL Editor",
            description="Query this data with SQL",
            target_page="📝 SQL Editor",
            payload_type=HandoffType.DATAFRAME,
            icon="🗄️",
            priority=HandoffPriority.MEDIUM,
            condition=lambda: self._has_active_data(),
            action=self._action_send_to_sql
        ))
        
        # AI-focused handoffs
        self._register_handoff_action(HandoffAction(
            id="send_to_ai_assistant",
            title="Send to AI Assistant",
            description="Get AI insights about this data",
            target_page="🤖 AI Assistant",
            payload_type=HandoffType.DATAFRAME,
            icon="🧠",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_data(),
            action=self._action_send_to_ai_assistant
        ))
        
        # Chart and analysis handoffs
        self._register_handoff_action(HandoffAction(
            id="send_chart_to_report",
            title="Add to Report",
            description="Include this chart in a report",
            target_page="📖 Report Studio",
            payload_type=HandoffType.CHART,
            icon="📋",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_charts(),
            action=self._action_send_chart_to_report
        ))
        
        self._register_handoff_action(HandoffAction(
            id="send_analysis_to_report",
            title="Generate Report",
            description="Create report from analysis results",
            target_page="📖 Report Studio",
            payload_type=HandoffType.ANALYSIS,
            icon="📝",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_analysis(),
            action=self._action_send_analysis_to_report
        ))
        
        # HIRAX-specific handoffs
        self._register_handoff_action(HandoffAction(
            id="hirax_waterfall_to_fft",
            title="Analyze with FFT",
            description="Perform FFT analysis on waterfall data",
            target_page="🔭 HIRAX Inspector",
            payload_type=HandoffType.HIRAX_SPECTRAL,
            icon="🌊",
            priority=HandoffPriority.CRITICAL,
            condition=lambda: self._has_hirax_waterfall(),
            action=self._action_hirax_waterfall_to_fft
        ))
        
        self._register_handoff_action(HandoffAction(
            id="hirax_fft_to_spectrum",
            title="Generate Power Spectrum",
            description="Create power spectrum from FFT",
            target_page="🔭 HIRAX Inspector",
            payload_type=HandoffType.HIRAX_FFT,
            icon="📈",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_hirax_fft(),
            action=self._action_hirax_fft_to_spectrum
        ))
        
        self._register_handoff_action(HandoffAction(
            id="hirax_spectrum_to_signal",
            title="Signal Intelligence",
            description="Analyze signals in power spectrum",
            target_page="🛸 Signal Intelligence",
            payload_type=HandoffType.HIRAX_SPECTRAL,
            icon="🛸",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_hirax_spectrum(),
            action=self._action_hirax_spectrum_to_signal
        ))
        
        self._register_handoff_action(HandoffAction(
            id="hirax_to_experiment",
            title="Track Experiment",
            description="Add to experiment tracking",
            target_page="🔬 Experiment Tracker",
            payload_type=HandoffType.HIRAX_SPECTRAL,
            icon="🔬",
            priority=HandoffPriority.MEDIUM,
            condition=lambda: self._has_hirax_data(),
            action=self._action_hirax_to_experiment
        ))
        
        # ML and export handoffs
        self._register_handoff_action(HandoffAction(
            id="ml_to_report",
            title="Export Model Report",
            description="Generate comprehensive ML report",
            target_page="📖 Report Studio",
            payload_type=HandoffType.ML_MODEL,
            icon="📊",
            priority=HandoffPriority.HIGH,
            condition=lambda: self._has_active_ml_models(),
            action=self._action_ml_to_report
        ))
        
        self._register_handoff_action(HandoffAction(
            id="export_dashboard",
            title="Generate Dashboard",
            description="Create interactive dashboard",
            target_page="🏗️ Dashboard Builder",
            payload_type=HandoffType.CHART,
            icon="📊",
            priority=HandoffPriority.MEDIUM,
            condition=lambda: self._has_active_charts(),
            action=self._action_export_dashboard
        ))
    
    def _initialize_workflow_transitions(self):
        """Initialize workflow stage transitions."""
        # Data Ingestion → Data Preparation
        self._register_workflow_transition(WorkflowTransition(
            from_stage="data_ingestion",
            to_stage="data_preparation",
            trigger_action="dataset_uploaded",
            suggested_actions=[
                self._handoff_actions.get("send_to_data_forge"),
                self._handoff_actions.get("send_to_visualize"),
                self._handoff_actions.get("send_to_ai_assistant")
            ],
            context_requirements={"has_active_data": True}
        ))
        
        # Data Preparation → Exploration
        self._register_workflow_transition(WorkflowTransition(
            from_stage="data_preparation",
            to_stage="exploration",
            trigger_action="data_cleaned",
            suggested_actions=[
                self._handoff_actions.get("send_to_visualize"),
                self._handoff_actions.get("send_to_ml"),
                self._handoff_actions.get("send_to_sql")
            ],
            context_requirements={"has_transformed_data": True}
        ))
        
        # Exploration → Analysis
        self._register_workflow_transition(WorkflowTransition(
            from_stage="exploration",
            to_stage="analysis",
            trigger_action="charts_created",
            suggested_actions=[
                self._handoff_actions.get("send_chart_to_report"),
                self._handoff_actions.get("send_analysis_to_report"),
                self._handoff_actions.get("export_dashboard")
            ],
            context_requirements={"has_charts": True}
        ))
        
        # Analysis → Modeling
        self._register_workflow_transition(WorkflowTransition(
            from_stage="analysis",
            to_stage="modeling",
            trigger_action="insights_generated",
            suggested_actions=[
                self._handoff_actions.get("send_to_ml"),
                self._handoff_actions.get("ml_to_report")
            ],
            context_requirements={"has_analysis": True}
        ))
        
        # HIRAX-specific transitions
        self._register_workflow_transition(WorkflowTransition(
            from_stage="hirax_waterfall",
            to_stage="hirax_fft",
            trigger_action="waterfall_loaded",
            suggested_actions=[
                self._handoff_actions.get("hirax_waterfall_to_fft")
            ],
            context_requirements={"has_hirax_waterfall": True}
        ))
        
        self._register_workflow_transition(WorkflowTransition(
            from_stage="hirax_fft",
            to_stage="hirax_spectrum",
            trigger_action="fft_completed",
            suggested_actions=[
                self._handoff_actions.get("hirax_fft_to_spectrum")
            ],
            context_requirements={"has_hirax_fft": True}
        ))
        
        self._register_workflow_transition(WorkflowTransition(
            from_stage="hirax_spectrum",
            to_stage="hirax_signal",
            trigger_action="spectrum_generated",
            suggested_actions=[
                self._handoff_actions.get("hirax_spectrum_to_signal"),
                self._handoff_actions.get("hirax_to_experiment")
            ],
            context_requirements={"has_hirax_spectrum": True}
        ))
    
    def _register_handoff_action(self, action: HandoffAction):
        """Register a handoff action."""
        self._handoff_actions[action.id] = action
    
    def _register_workflow_transition(self, transition: WorkflowTransition):
        """Register a workflow transition."""
        key = f"{transition.from_stage}_to_{transition.to_stage}"
        self._workflow_transitions[key] = transition
    
    # ── Handoff Creation and Management ───────────────────────────────────────────
    
    def create_handoff(self, 
                      payload_type: HandoffType,
                      source_page: str,
                      title: str,
                      description: str,
                      data: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None,
                      priority: HandoffPriority = HandoffPriority.MEDIUM) -> HandoffPayload:
        """Create a new handoff payload."""
        handoff_id = str(uuid.uuid4())
        
        # Get lineage from current context
        lineage = []
        if get_context_engine:
            try:
                context = get_context_engine()
                # Add current active items to lineage
                if context.get_active_dataset():
                    lineage.append(f"dataset:{context.get_active_dataset().id}")
                if context.get_active_dataframe_info():
                    lineage.append(f"dataframe:{context.get_active_dataframe_info().id}")
            except Exception as e:
                self._logger.warning(f"Failed to get context for lineage: {e}")
        
        payload = HandoffPayload(
            id=handoff_id,
            type=payload_type,
            source_page=source_page,
            target_page="",  # Will be set when action is chosen
            title=title,
            description=description,
            data=data,
            metadata=metadata or {},
            lineage=lineage,
            priority=priority
        )
        
        # Store in session state
        st.session_state.workflow_handoffs["active_payloads"][handoff_id] = payload
        st.session_state.workflow_handoffs["last_handoff"] = payload
        
        return payload
    
    def execute_handoff(self, handoff_id: str, action_id: str) -> bool:
        """Execute a handoff action."""
        if handoff_id not in st.session_state.workflow_handoffs["active_payloads"]:
            return False
        
        if action_id not in self._handoff_actions:
            return False
        
        payload = st.session_state.workflow_handoffs["active_payloads"][handoff_id]
        action = self._handoff_actions[action_id]
        
        # Check conditions
        if action.condition and not action.condition():
            return False
        
        try:
            # Execute action
            if action.action:
                success = action.action(payload)
                if success:
                    # Update payload target
                    payload.target_page = action.target_page
                    
                    # Move to history
                    st.session_state.workflow_handoffs["handoff_history"].append(payload)
                    del st.session_state.workflow_handoffs["active_payloads"][handoff_id]
                    
                    # Track in context engine
                    if get_context_engine:
                        try:
                            context = get_context_engine()
                            context._track_workflow(
                                f"handoff_{action_id}",
                                payload.source_page,
                                context.ContextType.EXPERIMENT,  # Use experiment type for handoffs
                                handoff_id,
                                {"action": action.title, "target": action.target_page}
                            )
                        except:
                            pass
                    
                    return True
            else:
                # Default navigation action
                st.session_state.active_page = action.target_page
                payload.target_page = action.target_page
                
                # Store handoff data for target page
                st.session_state._handoff_data = {
                    "payload": payload,
                    "action": action
                }
                
                # Move to history
                st.session_state.workflow_handoffs["handoff_history"].append(payload)
                del st.session_state.workflow_handoffs["active_payloads"][handoff_id]
                
                st.rerun()
                return True
        
        except Exception as e:
            self._logger.error(f"Handoff execution failed: {e}")
            return False
        
        return False
    
    # ── Smart Recommendations ─────────────────────────────────────────────────────
    
    def get_handoff_suggestions(self, current_page: str = None) -> List[HandoffAction]:
        """Get smart handoff suggestions based on current context."""
        suggestions = []
        
        # Get current context
        if get_context_engine:
            try:
                context = get_context_engine()
                workflow_stage = context.get_current_workflow_stage().value
                
                # Check for workflow transitions
                for transition in self._workflow_transitions.values():
                    if transition.from_stage == workflow_stage:
                        # Check if transition conditions are met
                        if self._check_transition_conditions(transition.context_requirements):
                            suggestions.extend([
                                action for action in transition.suggested_actions
                                if action.condition and action.condition()
                            ])
                
                # Add context-specific suggestions
                if context.has_active_data():
                    suggestions.extend([
                        self._handoff_actions.get("send_to_visualize"),
                        self._handoff_actions.get("send_to_ai_assistant"),
                        self._handoff_actions.get("send_to_data_forge")
                    ])
                
                if context.get_active_charts():
                    suggestions.extend([
                        self._handoff_actions.get("send_chart_to_report"),
                        self._handoff_actions.get("export_dashboard")
                    ])
                
                if context.get_active_ml_models():
                    suggestions.append(self._handoff_actions.get("ml_to_report"))
                
                if context.get_active_hirax_session():
                    hirax_actions = [
                        self._handoff_actions.get("hirax_waterfall_to_fft"),
                        self._handoff_actions.get("hirax_fft_to_spectrum"),
                        self._handoff_actions.get("hirax_spectrum_to_signal")
                    ]
                    suggestions.extend([action for action in hirax_actions if action.condition and action.condition()])
                
            except Exception as e:
                self._logger.warning(f"Failed to get context for suggestions: {e}")
        
        # Remove None values and sort by priority
        suggestions = [action for action in suggestions if action is not None]
        suggestions.sort(key=lambda x: x.priority.value, reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def trigger_workflow_transition(self, trigger_action: str) -> List[HandoffAction]:
        """Trigger workflow transition based on action."""
        suggestions = []
        
        # Find matching transitions
        for transition in self._workflow_transitions.values():
            if transition.trigger_action == trigger_action:
                if self._check_transition_conditions(transition.context_requirements):
                    suggestions.extend([
                        action for action in transition.suggested_actions
                        if action.condition and action.condition()
                    ])
        
        # Store suggestions for display
        st.session_state.workflow_handoffs["pending_suggestions"] = suggestions
        
        return suggestions
    
    # ── Handoff Actions ─────────────────────────────────────────────────────────
    
    def _action_send_to_visualize(self, payload: HandoffPayload) -> bool:
        """Action: Send data to Visualize."""
        try:
            st.session_state.active_page = "📊 Visualize"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "visualize_data"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send to visualize: {e}")
            return False
    
    def _action_send_to_data_forge(self, payload: HandoffPayload) -> bool:
        """Action: Send data to Data Forge."""
        try:
            st.session_state.active_page = "⚒️ Data Forge"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "clean_data"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send to data forge: {e}")
            return False
    
    def _action_send_to_ml(self, payload: HandoffPayload) -> bool:
        """Action: Send data to Machine Learning."""
        try:
            st.session_state.active_page = "🧠 Machine Learning"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "train_model"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send to ML: {e}")
            return False
    
    def _action_send_to_sql(self, payload: HandoffPayload) -> bool:
        """Action: Send data to SQL Editor."""
        try:
            st.session_state.active_page = "📝 SQL Editor"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "query_data"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send to SQL: {e}")
            return False
    
    def _action_send_to_ai_assistant(self, payload: HandoffPayload) -> bool:
        """Action: Send data to AI Assistant."""
        try:
            st.session_state.active_page = "🤖 AI Assistant"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "analyze_with_ai"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send to AI Assistant: {e}")
            return False
    
    def _action_send_chart_to_report(self, payload: HandoffPayload) -> bool:
        """Action: Send chart to Report Studio."""
        try:
            st.session_state.active_page = "📖 Report Studio"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "include_chart"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send chart to report: {e}")
            return False
    
    def _action_send_analysis_to_report(self, payload: HandoffPayload) -> bool:
        """Action: Send analysis to Report Studio."""
        try:
            st.session_state.active_page = "📖 Report Studio"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "generate_report"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send analysis to report: {e}")
            return False
    
    def _action_hirax_waterfall_to_fft(self, payload: HandoffPayload) -> bool:
        """Action: HIRAX waterfall to FFT."""
        try:
            st.session_state.active_page = "🔭 HIRAX Inspector"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "run_fft"
            }
            st.session_state._hirax_action = "fft"
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to run FFT: {e}")
            return False
    
    def _action_hirax_fft_to_spectrum(self, payload: HandoffPayload) -> bool:
        """Action: HIRAX FFT to spectrum."""
        try:
            st.session_state.active_page = "🔭 HIRAX Inspector"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "generate_spectrum"
            }
            st.session_state._hirax_action = "spectrum"
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to generate spectrum: {e}")
            return False
    
    def _action_hirax_spectrum_to_signal(self, payload: HandoffPayload) -> bool:
        """Action: HIRAX spectrum to signal intelligence."""
        try:
            st.session_state.active_page = "🛸 Signal Intelligence"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "analyze_signals"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to analyze signals: {e}")
            return False
    
    def _action_hirax_to_experiment(self, payload: HandoffPayload) -> bool:
        """Action: HIRAX to experiment tracking."""
        try:
            st.session_state.active_page = "🔬 Experiment Tracker"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "track_experiment"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to track experiment: {e}")
            return False
    
    def _action_ml_to_report(self, payload: HandoffPayload) -> bool:
        """Action: ML model to report."""
        try:
            st.session_state.active_page = "📖 Report Studio"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "ml_report"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to generate ML report: {e}")
            return False
    
    def _action_export_dashboard(self, payload: HandoffPayload) -> bool:
        """Action: Export dashboard."""
        try:
            st.session_state.active_page = "🏗️ Dashboard Builder"
            st.session_state._handoff_data = {
                "payload": payload,
                "action": "build_dashboard"
            }
            st.rerun()
            return True
        except Exception as e:
            self._logger.error(f"Failed to build dashboard: {e}")
            return False
    
    # ── Helper Methods ───────────────────────────────────────────────────────────
    
    def _has_active_data(self) -> bool:
        """Check if there's active data."""
        if get_context_engine:
            try:
                return get_context_engine().has_active_data()
            except:
                pass
        return bool(st.session_state.get("datasets", {}))
    
    def _has_active_charts(self) -> bool:
        """Check if there are active charts."""
        if get_context_engine:
            try:
                return bool(get_context_engine().get_active_charts())
            except:
                pass
        return False
    
    def _has_active_analysis(self) -> bool:
        """Check if there's active analysis."""
        return bool(st.session_state.get("_analysis_results", {}))
    
    def _has_active_ml_models(self) -> bool:
        """Check if there are active ML models."""
        if get_context_engine:
            try:
                return bool(get_context_engine().get_active_ml_models())
            except:
                pass
        return False
    
    def _has_hirax_waterfall(self) -> bool:
        """Check if there's HIRAX waterfall data."""
        return bool(st.session_state.get("_hirax_waterfall", {}))
    
    def _has_hirax_fft(self) -> bool:
        """Check if there's HIRAX FFT data."""
        return bool(st.session_state.get("_hirax_fft", {}))
    
    def _has_hirax_spectrum(self) -> bool:
        """Check if there's HIRAX spectrum data."""
        return bool(st.session_state.get("_hirax_spectrum", {}))
    
    def _has_hirax_data(self) -> bool:
        """Check if there's any HIRAX data."""
        return any([
            self._has_hirax_waterfall(),
            self._has_hirax_fft(),
            self._has_hirax_spectrum()
        ])
    
    def _check_transition_conditions(self, requirements: Dict[str, Any]) -> bool:
        """Check if workflow transition conditions are met."""
        for key, value in requirements.items():
            if key == "has_active_data" and value:
                if not self._has_active_data():
                    return False
            elif key == "has_transformed_data" and value:
                # Check if any transformations have been applied
                if not st.session_state.get("_transformations_applied", False):
                    return False
            elif key == "has_charts" and value:
                if not self._has_active_charts():
                    return False
            elif key == "has_analysis" and value:
                if not self._has_active_analysis():
                    return False
            elif key == "has_hirax_waterfall" and value:
                if not self._has_hirax_waterfall():
                    return False
            elif key == "has_hirax_fft" and value:
                if not self._has_hirax_fft():
                    return False
            elif key == "has_hirax_spectrum" and value:
                if not self._has_hirax_spectrum():
                    return False
        
        return True
    
    def get_handoff_history(self, limit: int = 20) -> List[HandoffPayload]:
        """Get handoff history."""
        history = st.session_state.workflow_handoffs.get("handoff_history", [])
        return sorted(history, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def get_active_handoffs(self) -> List[HandoffPayload]:
        """Get active handoffs."""
        return list(st.session_state.workflow_handoffs.get("active_payloads", {}).values())
    
    def clear_handoffs(self):
        """Clear all handoffs."""
        st.session_state.workflow_handoffs["active_payloads"] = {}
        st.session_state.workflow_handoffs["pending_suggestions"] = []


# ── Global Instance ─────────────────────────────────────────────────────────────

_global_handoff_manager: Optional[WorkflowHandoffManager] = None


def get_workflow_handoff_manager() -> WorkflowHandoffManager:
    """Get the global workflow handoff manager instance."""
    global _global_handoff_manager
    if _global_handoff_manager is None:
        _global_handoff_manager = WorkflowHandoffManager()
    return _global_handoff_manager


def init_workflow_handoffs():
    """Initialize workflow handoffs (call once at app startup)."""
    return get_workflow_handoff_manager()


# ── UI Components ───────────────────────────────────────────────────────────────

def render_handoff_suggestions(current_page: str = None):
    """Render handoff suggestions UI."""
    manager = get_workflow_handoff_manager()
    suggestions = manager.get_handoff_suggestions(current_page)
    
    if not suggestions:
        return
    
    st.markdown("### 🔄 Continue Workflow")
    
    for action in suggestions:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.markdown(f"**{action.icon}**")
        
        with col2:
            if st.button(action.title, key=f"handoff_{action.id}", help=action.description):
                # Create handoff payload
                payload = manager.create_handoff(
                    payload_type=action.payload_type,
                    source_page=current_page or st.session_state.get("active_page", ""),
                    title=f"Handoff to {action.target_page}",
                    description=action.description,
                    data={"source": current_page, "timestamp": time.time()},
                    priority=action.priority
                )
                
                # Execute handoff
                if manager.execute_handoff(payload.id, action.id):
                    st.success(f"Handoff to {action.target_page} successful!")
                else:
                    st.error("Handoff failed!")


def render_handoff_actions(payload_type: HandoffType, 
                          source_page: str,
                          title: str,
                          description: str,
                          data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None):
    """Render handoff action buttons for a specific payload."""
    manager = get_workflow_handoff_manager()
    
    # Create handoff payload
    payload = manager.create_handoff(
        payload_type=payload_type,
        source_page=source_page,
        title=title,
        description=description,
        data=data,
        metadata=metadata
    )
    
    # Get available actions for this payload type
    available_actions = [
        action for action in manager._handoff_actions.values()
        if action.payload_type == payload_type and 
           (not action.condition or action.condition())
    ]
    
    if not available_actions:
        return
    
    st.markdown("### 📤 Send To...")
    
    cols = st.columns(min(len(available_actions), 3))
    for i, action in enumerate(available_actions):
        with cols[i % len(cols)]:
            if st.button(f"{action.icon} {action.title}", 
                        key=f"action_{payload.id}_{action.id}",
                        help=action.description):
                if manager.execute_handoff(payload.id, action.id):
                    st.success(f"Sent to {action.target_page}!")
                else:
                    st.error("Handoff failed!")


def render_workflow_lineage():
    """Render workflow lineage visualization."""
    manager = get_workflow_handoff_manager()
    history = manager.get_handoff_history(limit=10)
    
    if not history:
        return
    
    st.markdown("### 📍 Workflow Lineage")
    
    for i, handoff in enumerate(history):
        with st.expander(f"{handoff.icon} {handoff.title}", expanded=False):
            st.markdown(f"**From:** {handoff.source_page}")
            st.markdown(f"**To:** {handoff.target_page}")
            st.markdown(f"**When:** {time.ctime(handoff.created_at)}")
            
            if handoff.lineage:
                st.markdown("**Lineage:**")
                for item in handoff.lineage:
                    st.caption(f"  → {item}")


# ── Convenience Functions ───────────────────────────────────────────────────────

def trigger_data_uploaded_handoff(dataset_name: str, dataset_info: Dict):
    """Trigger handoff suggestions after data upload."""
    manager = get_workflow_handoff_manager()
    return manager.trigger_workflow_transition("dataset_uploaded")


def trigger_data_cleaned_handoff(transformation_info: Dict):
    """Trigger handoff suggestions after data cleaning."""
    manager = get_workflow_handoff_manager()
    st.session_state._transformations_applied = True
    return manager.trigger_workflow_transition("data_cleaned")


def trigger_charts_created_handoff(chart_info: Dict):
    """Trigger handoff suggestions after chart creation."""
    manager = get_workflow_handoff_manager()
    return manager.trigger_workflow_transition("charts_created")


def trigger_hirax_waterfall_loaded_handoff(waterfall_info: Dict):
    """Trigger handoff suggestions after HIRAX waterfall loaded."""
    manager = get_workflow_handoff_manager()
    st.session_state._hirax_waterfall = waterfall_info
    return manager.trigger_workflow_transition("waterfall_loaded")


def trigger_hirax_fft_completed_handoff(fft_info: Dict):
    """Trigger handoff suggestions after HIRAX FFT completed."""
    manager = get_workflow_handoff_manager()
    st.session_state._hirax_fft = fft_info
    return manager.trigger_workflow_transition("fft_completed")


def trigger_hirax_spectrum_generated_handoff(spectrum_info: Dict):
    """Trigger handoff suggestions after HIRAX spectrum generated."""
    manager = get_workflow_handoff_manager()
    st.session_state._hirax_spectrum = spectrum_info
    return manager.trigger_workflow_transition("spectrum_generated")


def get_handoff_data() -> Optional[Dict[str, Any]]:
    """Get handoff data passed to current page."""
    return st.session_state.get("_handoff_data")


def clear_handoff_data():
    """Clear handoff data from current page."""
    if "_handoff_data" in st.session_state:
        del st.session_state._handoff_data
