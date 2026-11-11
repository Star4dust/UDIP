"""
UDIP v3 — modules/adaptive_workspace.py
Adaptive Workspace UI system for intelligent interface adaptation.

Dynamically reorganizes and prioritizes interface elements based on:
- User's current workflow
- Dataset type
- Analytical intent
- HIRAX scientific context
- Personalized preferences
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import json


class AdaptiveMode(Enum):
    """Adaptive workspace modes based on workflow context."""
    DATA_ENGINEERING = "data_engineering"
    ANALYTICS = "analytics"
    AI_WORKSPACE = "ai_workspace"
    HIRAX_SCIENTIFIC = "hirax_scientific"
    GENERAL = "general"


@dataclass
class WorkspaceContext:
    """Current workspace context for adaptation decisions."""
    active_dataset: Optional[str] = None
    dataset_type: Optional[str] = None
    workflow_stage: Optional[str] = None
    current_page: Optional[str] = None
    recent_actions: List[str] = field(default_factory=list)
    hirax_active: bool = False
    ai_active: bool = False
    visualization_active: bool = False
    analysis_active: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptiveLayout:
    """Adaptive layout configuration."""
    mode: AdaptiveMode
    prioritized_tools: List[str]
    contextual_shortcuts: List[Dict[str, Any]]
    smart_panels: List[Dict[str, Any]]
    sidebar_order: List[str]
    theme_colors: Dict[str, str]
    layout_config: Dict[str, Any]


@dataclass
class UserPreference:
    """User preference for workspace personalization."""
    favorite_tools: Set[str] = field(default_factory=set)
    recent_workflows: List[str] = field(default_factory=list)
    preferred_layouts: Dict[AdaptiveMode, Dict[str, Any]] = field(default_factory=dict)
    frequently_used_pipelines: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class AdaptiveWorkspace:
    """
    Adaptive Workspace UI system for intelligent interface adaptation.
    
    Dynamically reorganizes and prioritizes interface elements based on context.
    """
    
    def __init__(self):
        """Initialize the adaptive workspace system."""
        self._logger = logging.getLogger(__name__)
        self._current_context = WorkspaceContext()
        self._current_mode = AdaptiveMode.GENERAL
        self._user_preference = UserPreference()
        self._layout_cache: Dict[str, AdaptiveLayout] = {}
        self._adaptation_history: List[Dict[str, Any]] = []
        
        # Mode-specific configurations
        self._mode_configurations = self._initialize_mode_configurations()
        
        # Initialize session state for adaptive workspace
        if "adaptive_workspace" not in st.session_state:
            st.session_state.adaptive_workspace = {
                "enabled": True,
                "auto_adapt": True,
                "show_context_indicators": True,
                "smooth_transitions": True,
                "personalization_enabled": True,
                "last_adaptation": 0.0,
                "adaptation_interval": 10.0  # Adapt every 10 seconds
            }
    
    def _initialize_mode_configurations(self) -> Dict[AdaptiveMode, Dict[str, Any]]:
        """Initialize mode-specific configurations."""
        return {
            AdaptiveMode.DATA_ENGINEERING: {
                "prioritized_tools": [
                    "⚒️ Data Forge",
                    "🔮 Data Fusion", 
                    "✅ Data Validation",
                    "🔍 SQL Editor",
                    "📋 Schema Inspector"
                ],
                "contextual_shortcuts": [
                    {"action": "clean_data", "label": "Quick Clean", "icon": "🧹"},
                    {"action": "merge_datasets", "label": "Merge Data", "icon": "🔗"},
                    {"action": "validate_schema", "label": "Validate", "icon": "✅"},
                    {"action": "transform_data", "label": "Transform", "icon": "🔄"}
                ],
                "smart_panels": [
                    {"type": "data_quality", "title": "Data Quality Monitor"},
                    {"type": "transformation_history", "title": "Recent Transformations"},
                    {"type": "schema_analysis", "title": "Schema Analysis"}
                ],
                "theme_colors": {
                    "primary": "#2563eb",
                    "secondary": "#7c3aed",
                    "accent": "#0891b2"
                }
            },
            
            AdaptiveMode.ANALYTICS: {
                "prioritized_tools": [
                    "📊 Visualize",
                    "💓 Data Pulse",
                    "🔍 Auto Analysis",
                    "🧠 Machine Learning",
                    "📈 Statistical Analysis"
                ],
                "contextual_shortcuts": [
                    {"action": "create_chart", "label": "Quick Chart", "icon": "📈"},
                    {"action": "statistical_summary", "label": "Stats", "icon": "📊"},
                    {"action": "correlation_analysis", "label": "Correlations", "icon": "🔗"},
                    {"action": "ml_model", "label": "ML Model", "icon": "🤖"}
                ],
                "smart_panels": [
                    {"type": "recommended_charts", "title": "Recommended Charts"},
                    {"type": "statistical_insights", "title": "Statistical Insights"},
                    {"type": "analysis_progress", "title": "Analysis Progress"}
                ],
                "theme_colors": {
                    "primary": "#059669",
                    "secondary": "#0891b2", 
                    "accent": "#10b981"
                }
            },
            
            AdaptiveMode.AI_WORKSPACE: {
                "prioritized_tools": [
                    "🤖 AI Assistant",
                    "🧪 Agent Lab",
                    "🧭 Agent Pipeline",
                    "⚡ Execution UI",
                    "📝 Code Editor"
                ],
                "contextual_shortcuts": [
                    {"action": "ai_query", "label": "Ask AI", "icon": "🤖"},
                    {"action": "generate_code", "label": "Generate Code", "icon": "📝"},
                    {"action": "run_pipeline", "label": "Run Pipeline", "icon": "🚀"},
                    {"action": "agent_orchestrate", "label": "Orchestrate", "icon": "🎯"}
                ],
                "smart_panels": [
                    {"type": "ai_insights", "title": "AI Insights"},
                    {"type": "agent_status", "title": "Agent Status"},
                    {"type": "pipeline_queue", "title": "Pipeline Queue"}
                ],
                "theme_colors": {
                    "primary": "#7c3aed",
                    "secondary": "#ec4899",
                    "accent": "#a855f7"
                }
            },
            
            AdaptiveMode.HIRAX_SCIENTIFIC: {
                "prioritized_tools": [
                    "🔭 HIRAX Inspector",
                    "🛸 Signal Intelligence",
                    "🌈 Spectral Analysis",
                    "📡 RFI Detection",
                    "🧪 Validation Workbench"
                ],
                "contextual_shortcuts": [
                    {"action": "fft_analysis", "label": "FFT Analysis", "icon": "📊"},
                    {"action": "rfi_detection", "label": "RFI Detection", "icon": "📡"},
                    {"action": "waterfall_plot", "label": "Waterfall", "icon": "🌊"},
                    {"action": "spectral_calibration", "label": "Calibrate", "icon": "⚙️"}
                ],
                "smart_panels": [
                    {"type": "hirax_status", "title": "HIRAX Status"},
                    {"type": "spectral_monitor", "title": "Spectral Monitor"},
                    {"type": "experiment_tracker", "title": "Experiment Tracker"}
                ],
                "theme_colors": {
                    "primary": "#dc2626",
                    "secondary": "#ea580c",
                    "accent": "#f97316"
                }
            },
            
            AdaptiveMode.GENERAL: {
                "prioritized_tools": [
                    "📌 Dashboard",
                    "📊 Visualize",
                    "🤖 AI Assistant",
                    "⚒️ Data Forge",
                    "🧠 Machine Learning"
                ],
                "contextual_shortcuts": [
                    {"action": "upload_data", "label": "Upload", "icon": "📤"},
                    {"action": "quick_explore", "label": "Explore", "icon": "🔍"},
                    {"action": "create_report", "label": "Report", "icon": "📋"},
                    {"action": "settings", "label": "Settings", "icon": "⚙️"}
                ],
                "smart_panels": [
                    {"type": "workspace_overview", "title": "Workspace Overview"},
                    {"type": "recent_activity", "title": "Recent Activity"},
                    {"type": "quick_actions", "title": "Quick Actions"}
                ],
                "theme_colors": {
                    "primary": "#3b82f6",
                    "secondary": "#6366f1",
                    "accent": "#8b5cf6"
                }
            }
        }
    
    def detect_adaptive_mode(self) -> AdaptiveMode:
        """
        Detect the appropriate adaptive mode based on current context.
        
        Returns:
            Detected adaptive mode
        """
        context = self._get_current_context()
        
        # Priority-based mode detection
        
        # HIRAX Scientific Mode (highest priority)
        if context.hirax_active or self._is_hirax_context():
            return AdaptiveMode.HIRAX_SCIENTIFIC
        
        # AI Workspace Mode
        if context.ai_active or self._is_ai_context():
            return AdaptiveMode.AI_WORKSPACE
        
        # Data Engineering Mode
        if self._is_data_engineering_context():
            return AdaptiveMode.DATA_ENGINEERING
        
        # Analytics Mode
        if self._is_analytics_context():
            return AdaptiveMode.ANALYTICS
        
        # Default to General Mode
        return AdaptiveMode.GENERAL
    
    def _get_current_context(self) -> WorkspaceContext:
        """Get current workspace context."""
        try:
            # Update context with current state
            context = WorkspaceContext()
            
            # Active dataset information
            if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
                context.active_dataset = st.session_state.context_engine.get_active_dataset_name()
                context.workflow_stage = st.session_state.context_engine.get_current_workflow_stage().value if st.session_state.context_engine.get_current_workflow_stage() else None
            
            # Current page
            context.current_page = st.session_state.get('active_page', '')
            
            # Dataset type detection
            if context.active_dataset and context.active_dataset in st.session_state.datasets:
                from modules.workflow_copilot import get_workflow_copilot
                copilot = get_workflow_copilot()
                if context.active_dataset in copilot._dataset_insights:
                    insight = copilot._dataset_insights[context.active_dataset]
                    context.dataset_type = insight.dataset_type.value
            
            # Activity detection
            context.hirax_active = self._is_hirax_context()
            context.ai_active = self._is_ai_context()
            context.visualization_active = self._is_visualization_context()
            context.analysis_active = self._is_analysis_context()
            
            # Recent actions (from workflow history)
            if hasattr(get_workflow_copilot(), '_workflow_history'):
                recent_actions = get_workflow_copilot()._workflow_history[-5:]
                context.recent_actions = [action.get('action', '') for action in recent_actions]
            
            self._current_context = context
            return context
            
        except Exception as e:
            self._logger.error(f"Failed to get current context: {e}")
            return WorkspaceContext()
    
    def _is_hirax_context(self) -> bool:
        """Check if current context is HIRAX-related."""
        try:
            # Check page
            current_page = st.session_state.get('active_page', '')
            hirax_pages = ['HIRAX Inspector', 'Signal Intelligence', 'Spectral Analysis']
            if any(page in current_page for page in hirax_pages):
                return True
            
            # Check dataset type
            if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
                dataset_name = st.session_state.context_engine.get_active_dataset_name()
                if dataset_name and dataset_name in st.session_state.datasets:
                    from modules.workflow_copilot import get_workflow_copilot
                    copilot = get_workflow_copilot()
                    if dataset_name in copilot._dataset_insights:
                        insight = copilot._dataset_insights[dataset_name]
                        return 'hirax' in insight.dataset_type.value.lower()
            
            return False
            
        except Exception:
            return False
    
    def _is_ai_context(self) -> bool:
        """Check if current context is AI-related."""
        try:
            current_page = st.session_state.get('active_page', '')
            ai_pages = ['AI Assistant', 'Agent Lab', 'Agent Pipeline', 'Execution UI']
            return any(page in current_page for page in ai_pages)
        except Exception:
            return False
    
    def _is_data_engineering_context(self) -> bool:
        """Check if current context is data engineering."""
        try:
            current_page = st.session_state.get('active_page', '')
            engineering_pages = ['Data Forge', 'Data Fusion', 'Data Validation', 'SQL Editor']
            return any(page in current_page for page in engineering_pages)
        except Exception:
            return False
    
    def _is_analytics_context(self) -> bool:
        """Check if current context is analytics."""
        try:
            current_page = st.session_state.get('active_page', '')
            analytics_pages = ['Visualize', 'Data Pulse', 'Auto Analysis', 'Machine Learning']
            return any(page in current_page for page in analytics_pages)
        except Exception:
            return False
    
    def _is_visualization_context(self) -> bool:
        """Check if current context involves visualization."""
        try:
            current_page = st.session_state.get('active_page', '')
            return 'Visualize' in current_page or 'Dashboard' in current_page
        except Exception:
            return False
    
    def _is_analysis_context(self) -> bool:
        """Check if current context involves analysis."""
        try:
            current_page = st.session_state.get('active_page', '')
            analysis_pages = ['Auto Analysis', 'Machine Learning', 'Statistical Analysis']
            return any(page in current_page for page in analysis_pages)
        except Exception:
            return False
    
    def get_adaptive_layout(self) -> AdaptiveLayout:
        """
        Get adaptive layout based on current mode and context.
        
        Returns:
            Adaptive layout configuration
        """
        mode = self.detect_adaptive_mode()
        config = self._mode_configurations[mode]
        
        # Create adaptive layout
        layout = AdaptiveLayout(
            mode=mode,
            prioritized_tools=self._get_prioritized_tools(mode),
            contextual_shortcuts=self._get_contextual_shortcuts(mode),
            smart_panels=self._get_smart_panels(mode),
            sidebar_order=self._get_sidebar_order(mode),
            theme_colors=config["theme_colors"],
            layout_config=self._get_layout_config(mode)
        )
        
        # Cache layout
        cache_key = f"{mode.value}_{hash(str(self._current_context))}"
        self._layout_cache[cache_key] = layout
        
        return layout
    
    def _get_prioritized_tools(self, mode: AdaptiveMode) -> List[str]:
        """Get prioritized tools for the given mode."""
        base_tools = self._mode_configurations[mode]["prioritized_tools"]
        
        # Add user favorites
        if self._user_preference.favorite_tools:
            # Move favorites to top
            prioritized = []
            for tool in self._user_preference.favorite_tools:
                if tool in base_tools:
                    prioritized.append(tool)
            
            # Add remaining tools
            for tool in base_tools:
                if tool not in prioritized:
                    prioritized.append(tool)
            
            return prioritized
        
        return base_tools
    
    def _get_contextual_shortcuts(self, mode: AdaptiveMode) -> List[Dict[str, Any]]:
        """Get contextual shortcuts for the given mode."""
        base_shortcuts = self._mode_configurations[mode]["contextual_shortcuts"]
        
        # Add context-specific shortcuts
        contextual_shortcuts = base_shortcuts.copy()
        
        # Dataset-specific shortcuts
        if self._current_context.active_dataset:
            contextual_shortcuts.extend([
                {"action": "dataset_info", "label": "Dataset Info", "icon": "📊"},
                {"action": "dataset_export", "label": "Export", "icon": "📤"},
                {"action": "dataset_share", "label": "Share", "icon": "🔗"}
            ])
        
        # HIRAX-specific shortcuts
        if mode == AdaptiveMode.HIRAX_SCIENTIFIC:
            contextual_shortcuts.extend([
                {"action": "hirax_calibration", "label": "Calibrate", "icon": "⚙️"},
                {"action": "hirax_validation", "label": "Validate", "icon": "✅"},
                {"action": "hirax_experiment", "label": "Experiment", "icon": "🧪"}
            ])
        
        return contextual_shortcuts
    
    def _get_smart_panels(self, mode: AdaptiveMode) -> List[Dict[str, Any]]:
        """Get smart panels for the given mode."""
        base_panels = self._mode_configurations[mode]["smart_panels"]
        
        # Add context-specific panels
        smart_panels = base_panels.copy()
        
        # Add workflow status panel
        if self._current_context.workflow_stage:
            smart_panels.append({
                "type": "workflow_status",
                "title": f"Workflow: {self._current_context.workflow_stage.replace('_', ' ').title()}"
            })
        
        # Add dataset insights panel
        if self._current_context.active_dataset:
            smart_panels.append({
                "type": "dataset_insights",
                "title": f"Dataset: {self._current_context.active_dataset}"
            })
        
        return smart_panels
    
    def _get_sidebar_order(self, mode: AdaptiveMode) -> List[str]:
        """Get sidebar order for the given mode."""
        prioritized_tools = self._get_prioritized_tools(mode)
        
        # Get all available tools from navigation
        try:
            from modules.navigation import iter_routes
            all_routes = list(iter_routes())
            all_tools = [route.get('name', '') for route in all_routes]
            
            # Create ordered list
            ordered_tools = []
            
            # Add prioritized tools first
            for tool in prioritized_tools:
                if tool in all_tools:
                    ordered_tools.append(tool)
            
            # Add remaining tools
            for tool in all_tools:
                if tool not in ordered_tools:
                    ordered_tools.append(tool)
            
            return ordered_tools
            
        except Exception:
            return prioritized_tools
    
    def _get_layout_config(self, mode: AdaptiveMode) -> Dict[str, Any]:
        """Get layout configuration for the given mode."""
        config = {
            "sidebar_width": 280,
            "main_content_width": "full",
            "show_context_bar": True,
            "show_smart_panels": True,
            "panel_position": "right",
            "animation_speed": "0.3s"
        }
        
        # Mode-specific adjustments
        if mode == AdaptiveMode.HIRAX_SCIENTIFIC:
            config.update({
                "sidebar_width": 320,  # Wider for scientific tools
                "show_context_bar": True,
                "panel_position": "right"
            })
        elif mode == AdaptiveMode.AI_WORKSPACE:
            config.update({
                "sidebar_width": 300,  # Wider for AI tools
                "animation_speed": "0.2s"  # Faster for AI interactions
            })
        elif mode == AdaptiveMode.DATA_ENGINEERING:
            config.update({
                "show_smart_panels": True,
                "panel_position": "left"  # Left panel for data engineering
            })
        
        return config
    
    def should_adapt(self) -> bool:
        """Check if workspace should adapt."""
        settings = st.session_state.adaptive_workspace
        
        if not settings.get('enabled', True):
            return False
        
        if not settings.get('auto_adapt', True):
            return False
        
        # Check adaptation interval
        current_time = time.time()
        last_adaptation = settings.get('last_adaptation', 0.0)
        interval = settings.get('adaptation_interval', 10.0)
        
        return (current_time - last_adaptation) > interval
    
    def adapt_workspace(self) -> bool:
        """
        Perform workspace adaptation.
        
        Returns:
            True if adaptation was performed
        """
        try:
            if not self.should_adapt():
                return False
            
            # Detect new mode
            new_mode = self.detect_adaptive_mode()
            
            # Check if mode changed
            if new_mode != self._current_mode:
                self._current_mode = new_mode
                
                # Update adaptation history
                self._adaptation_history.append({
                    'timestamp': time.time(),
                    'mode': new_mode.value,
                    'context': self._current_context,
                    'reason': 'mode_change'
                })
                
                # Update last adaptation time
                st.session_state.adaptive_workspace['last_adaptation'] = time.time()
                
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to adapt workspace: {e}")
            return False
    
    def update_user_preferences(self, action: str, details: Dict[str, Any] = None):
        """Update user preferences based on actions."""
        try:
            # Track frequently used tools
            if action == 'tool_access':
                tool_name = details.get('tool', '')
                if tool_name:
                    self._user_preference.favorite_tools.add(tool_name)
            
            # Track recent workflows
            elif action == 'workflow_complete':
                workflow_name = details.get('workflow', '')
                if workflow_name:
                    if workflow_name not in self._user_preference.recent_workflows:
                        self._user_preference.recent_workflows.append(workflow_name)
                    
                    # Keep only last 10 workflows
                    if len(self._user_preference.recent_workflows) > 10:
                        self._user_preference.recent_workflows = self._user_preference.recent_workflows[-10:]
            
            # Track frequently used pipelines
            elif action == 'pipeline_execute':
                pipeline_name = details.get('pipeline', '')
                if pipeline_name:
                    if pipeline_name not in self._user_preference.frequently_used_pipelines:
                        self._user_preference.frequently_used_pipelines.append(pipeline_name)
                    
                    # Keep only last 5 pipelines
                    if len(self._user_preference.frequently_used_pipelines) > 5:
                        self._user_preference.frequently_used_pipelines = self._user_preference.frequently_used_pipelines[-5:]
            
            # Update timestamp
            self._user_preference.last_updated = time.time()
            
        except Exception as e:
            self._logger.error(f"Failed to update user preferences: {e}")
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of current adaptation state."""
        return {
            'current_mode': self._current_mode.value,
            'context': {
                'active_dataset': self._current_context.active_dataset,
                'dataset_type': self._current_context.dataset_type,
                'workflow_stage': self._current_context.workflow_stage,
                'current_page': self._current_context.current_page
            },
            'adaptation_count': len(self._adaptation_history),
            'user_preferences': {
                'favorite_tools': len(self._user_preference.favorite_tools),
                'recent_workflows': len(self._user_preference.recent_workflows),
                'frequently_used_pipelines': len(self._user_preference.frequently_used_pipelines)
            },
            'last_adaptation': st.session_state.adaptive_workspace.get('last_adaptation', 0.0)
        }


# Global instance
_global_adaptive_workspace: Optional[AdaptiveWorkspace] = None


def get_adaptive_workspace() -> AdaptiveWorkspace:
    """Get the global adaptive workspace instance."""
    global _global_adaptive_workspace
    if _global_adaptive_workspace is None:
        _global_adaptive_workspace = AdaptiveWorkspace()
    return _global_adaptive_workspace


def init_adaptive_workspace() -> AdaptiveWorkspace:
    """Initialize the global adaptive workspace."""
    global _global_adaptive_workspace
    _global_adaptive_workspace = AdaptiveWorkspace()
    return _global_adaptive_workspace


def render_adaptive_sidebar():
    """
    Render adaptive sidebar with dynamic prioritization.
    
    Reorders and highlights tools based on current mode and context.
    """
    workspace = get_adaptive_workspace()
    
    # Adapt workspace if needed
    workspace.adapt_workspace()
    
    # Get adaptive layout
    layout = workspace.get_adaptive_layout()
    
    # Apply adaptive styling
    _apply_adaptive_styling(layout.theme_colors)
    
    # Render prioritized tools
    st.markdown(f"### 🎯 {layout.mode.value.replace('_', ' ').title()} Mode")
    
    # Context indicator
    if st.session_state.adaptive_workspace.get('show_context_indicators', True):
        context = workspace._current_context
        context_info = []
        
        if context.active_dataset:
            context_info.append(f"📁 {context.active_dataset}")
        if context.dataset_type:
            context_info.append(f"📊 {context.dataset_type}")
        if context.workflow_stage:
            context_info.append(f"🔄 {context.workflow_stage}")
        
        if context_info:
            st.markdown(f"**Context:** {' | '.join(context_info)}")
    
    # Render prioritized tools
    for tool in layout.prioritized_tools[:10]:  # Show top 10
        # Highlight favorite tools
        is_favorite = tool in workspace._user_preference.favorite_tools
        
        if is_favorite:
            st.markdown(f"⭐ **{tool}**")
        else:
            st.markdown(f"• {tool}")
    
    # Render contextual shortcuts
    if layout.contextual_shortcuts:
        st.markdown("### ⚡ Quick Actions")
        
        cols = st.columns(2)
        for i, shortcut in enumerate(layout.contextual_shortcuts[:6]):  # Show top 6
            with cols[i % 2]:
                if st.button(f"{shortcut['icon']} {shortcut['label']}", 
                           key=f"shortcut_{shortcut['action']}",
                           help=shortcut['action']):
                    _execute_shortcut_action(shortcut['action'])
    
    # Mode switcher
    st.markdown("### 🔄 Switch Mode")
    
    mode_options = {
        AdaptiveMode.DATA_ENGINEERING: "🔧 Data Engineering",
        AdaptiveMode.ANALYTICS: "📊 Analytics", 
        AdaptiveMode.AI_WORKSPACE: "🤖 AI Workspace",
        AdaptiveMode.HIRAX_SCIENTIFIC: "🔭 HIRAX Scientific",
        AdaptiveMode.GENERAL: "🏠 General"
    }
    
    selected_mode = st.selectbox(
        "Select Mode:",
        options=list(mode_options.keys()),
        format_func=lambda x: mode_options[x],
        index=list(mode_options.keys()).index(layout.mode)
    )
    
    if selected_mode != layout.mode:
        # Force mode change
        workspace._current_mode = selected_mode
        st.rerun()


def render_adaptive_smart_panels():
    """
    Render smart panels based on current context and mode.
    
    Displays recommended charts, analyses, and insights.
    """
    workspace = get_adaptive_workspace()
    layout = workspace.get_adaptive_layout()
    
    if not layout.smart_panels:
        return
    
    # Panel configuration
    settings = st.session_state.adaptive_workspace
    show_panels = settings.get('show_smart_panels', True)
    panel_position = layout.layout_config.get('panel_position', 'right')
    
    if not show_panels:
        return
    
    # Render panels
    for panel in layout.smart_panels[:3]:  # Show top 3 panels
        panel_type = panel['type']
        panel_title = panel['title']
        
        with st.expander(panel_title, expanded=False):
            _render_smart_panel_content(panel_type, panel_title)


def _render_smart_panel_content(panel_type: str, panel_title: str):
    """Render content for a specific smart panel."""
    workspace = get_adaptive_workspace()
    context = workspace._current_context
    
    if panel_type == "data_quality_monitor":
        _render_data_quality_panel(context)
    elif panel_type == "recommended_charts":
        _render_recommended_charts_panel(context)
    elif panel_type == "workflow_status":
        _render_workflow_status_panel(context)
    elif panel_type == "hirax_status":
        _render_hirax_status_panel(context)
    elif panel_type == "ai_insights":
        _render_ai_insights_panel(context)
    elif panel_type == "dataset_insights":
        _render_dataset_insights_panel(context)
    elif panel_type == "workspace_overview":
        _render_workspace_overview_panel(context)
    else:
        st.info(f"Panel content for {panel_type} coming soon...")


def _render_data_quality_panel(context):
    """Render data quality monitoring panel."""
    if not context.active_dataset:
        st.info("No active dataset")
        return
    
    try:
        from modules.workflow_copilot import get_workflow_copilot
        copilot = get_workflow_copilot()
        
        if context.active_dataset in copilot._dataset_insights:
            insight = copilot._dataset_insights[context.active_dataset]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality Score", f"{insight.data_quality_score:.2f}")
                st.metric("Missing Data", f"{insight.missing_data_ratio:.1%}")
            
            with col2:
                st.metric("ML Suitability", f"{insight.ml_suitability:.2f}")
                st.metric("Outliers", f"{insight.outlier_ratio:.1%}")
            
            # Quality recommendations
            if insight.recommendations:
                st.markdown("**Recommendations:**")
                for rec in insight.recommendations[:3]:
                    st.write(f"• {rec}")
    
    except Exception:
        st.error("Unable to load quality metrics")


def _render_recommended_charts_panel(context):
    """Render recommended charts panel."""
    if not context.active_dataset:
        st.info("No active dataset")
        return
    
    # Chart recommendations based on dataset type
    recommendations = []
    
    if context.dataset_type == "time_series":
        recommendations = ["Line Chart", "Time Series Decomposition", "Seasonal Plot"]
    elif context.dataset_type == "hirax_spectral":
        recommendations = ["Power Spectrum", "Waterfall Plot", "Frequency Distribution"]
    elif context.dataset_type == "geospatial":
        recommendations = ["Map Visualization", "Geospatial Heatmap", "Coordinate Plot"]
    else:
        recommendations = ["Histogram", "Scatter Plot", "Box Plot", "Correlation Matrix"]
    
    for chart in recommendations:
        if st.button(f"📊 {chart}", key=f"chart_{chart.lower().replace(' ', '_')}"):
            st.session_state.active_page = "📊 Visualize"
            st.session_state.selected_chart = chart
            st.rerun()


def _render_workflow_status_panel(context):
    """Render workflow status panel."""
    if not context.workflow_stage:
        st.info("No active workflow")
        return
    
    st.markdown(f"**Current Stage:** {context.workflow_stage.replace('_', ' ').title()}")
    
    # Progress indicator
    stages = ["data_ingestion", "data_preparation", "exploration", "analysis", "modeling", "reporting"]
    current_index = stages.index(context.workflow_stage) if context.workflow_stage in stages else 0
    
    progress = (current_index + 1) / len(stages)
    st.progress(progress)
    
    st.markdown(f"**Progress:** {current_index + 1}/{len(stages)} stages")
    
    # Next stage suggestion
    if current_index < len(stages) - 1:
        next_stage = stages[current_index + 1]
        st.markdown(f"**Next:** {next_stage.replace('_', ' ').title()}")


def _render_hirax_status_panel(context):
    """Render HIRAX-specific status panel."""
    if not context.hirax_active:
        st.info("HIRAX not active")
        return
    
    st.markdown("### 🔭 HIRAX Status")
    
    # HIRAX-specific metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mode", "Scientific")
        st.metric("Data Type", context.dataset_type or "Unknown")
    
    with col2:
        st.metric("Processing", "Active")
        st.metric("Quality", "Good")
    
    # HIRAX quick actions
    st.markdown("**Quick Actions:**")
    
    hirax_actions = [
        ("FFT Analysis", "fft_analysis"),
        ("RFI Detection", "rfi_detection"),
        ("Calibration", "calibration"),
        ("Validation", "validation")
    ]
    
    cols = st.columns(2)
    for i, (label, action) in enumerate(hirax_actions):
        with cols[i % 2]:
            if st.button(label, key=f"hirax_{action}"):
                _execute_hirax_action(action)


def _render_ai_insights_panel(context):
    """Render AI insights panel."""
    if not context.ai_active:
        st.info("AI not active")
        return
    
    st.markdown("### 🤖 AI Insights")
    
    # AI status
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Status", "Online")
        st.metric("Models", "Active")
    
    with col2:
        st.metric("Queries", "0")
        st.metric("Response Time", "< 1s")
    
    # AI quick actions
    st.markdown("**Quick Actions:**")
    
    ai_actions = [
        ("Ask AI", "ai_query"),
        ("Generate Code", "generate_code"),
        ("Analyze Data", "analyze_data"),
        ("Get Help", "get_help")
    ]
    
    cols = st.columns(2)
    for i, (label, action) in enumerate(ai_actions):
        with cols[i % 2]:
            if st.button(label, key=f"ai_{action}"):
                _execute_ai_action(action)


def _render_dataset_insights_panel(context):
    """Render dataset insights panel."""
    if not context.active_dataset:
        st.info("No active dataset")
        return
    
    st.markdown(f"### 📊 {context.active_dataset}")
    
    try:
        if context.active_dataset in st.session_state.datasets:
            dataset_info = st.session_state.datasets[context.active_dataset]
            df = dataset_info['df']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
                st.metric("Columns", df.shape[1])
            
            with col2:
                st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                st.metric("Type", context.dataset_type or "Unknown")
            
            # Column types
            numeric_cols = df.select_dtypes(include=['number']).shape[1]
            text_cols = df.select_dtypes(include=['object']).shape[1]
            
            st.markdown("**Column Types:**")
            st.write(f"• Numeric: {numeric_cols}")
            st.write(f"• Text: {text_cols}")
            
    except Exception:
        st.error("Unable to load dataset info")


def _render_workspace_overview_panel(context):
    """Render workspace overview panel."""
    st.markdown("### 🏠 Workspace Overview")
    
    # Current status
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mode", context.workflow_stage or "General")
        st.metric("Datasets", len(st.session_state.datasets))
    
    with col2:
        st.metric("Active Page", context.current_page or "None")
        st.metric("Adaptations", len(get_adaptive_workspace()._adaptation_history))
    
    # Recent activity
    if context.recent_actions:
        st.markdown("**Recent Actions:**")
        for action in context.recent_actions[-3:]:
            st.write(f"• {action}")


def _apply_adaptive_styling(theme_colors: Dict[str, str]):
    """Apply adaptive styling based on current mode."""
    primary = theme_colors.get('primary', '#3b82f6')
    secondary = theme_colors.get('secondary', '#6366f1')
    accent = theme_colors.get('accent', '#8b5cf6')
    
    st.markdown(f"""
    <style>
    .adaptive-primary {{
        background-color: {primary} !important;
        color: white !important;
    }}
    .adaptive-secondary {{
        background-color: {secondary} !important;
        color: white !important;
    }}
    .adaptive-accent {{
        background-color: {accent} !important;
        color: white !important;
    }}
    .adaptive-border {{
        border-color: {primary} !important;
    }}
    .adaptive-text {{
        color: {primary} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def _execute_shortcut_action(action: str):
    """Execute a shortcut action."""
    try:
        if action == "clean_data":
            st.session_state.active_page = "⚒️ Data Forge"
        elif action == "merge_datasets":
            st.session_state.active_page = "🔮 Data Fusion"
        elif action == "validate_schema":
            st.session_state.active_page = "✅ Data Validation"
        elif action == "create_chart":
            st.session_state.active_page = "📊 Visualize"
        elif action == "statistical_summary":
            st.session_state.active_page = "💓 Data Pulse"
        elif action == "ml_model":
            st.session_state.active_page = "🧠 Machine Learning"
        elif action == "ai_query":
            st.session_state.active_page = "🤖 AI Assistant"
        elif action == "generate_code":
            st.session_state.active_page = "🤖 AI Assistant"
        elif action == "fft_analysis":
            st.session_state.active_page = "🛸 Signal Intelligence"
        elif action == "rfi_detection":
            st.session_state.active_page = "📡 RFI Detection"
        elif action == "dataset_info":
            # Show dataset info in current page
            pass
        elif action == "dataset_export":
            # Export current dataset
            pass
        
        # Update user preferences
        workspace = get_adaptive_workspace()
        workspace.update_user_preferences('shortcut_action', {'action': action})
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to execute action {action}: {e}")


def _execute_hirax_action(action: str):
    """Execute HIRAX-specific action."""
    hirax_pages = {
        "fft_analysis": "🛸 Signal Intelligence",
        "rfi_detection": "📡 RFI Detection",
        "calibration": "🔭 HIRAX Inspector",
        "validation": "🌈 Spectral Analysis"
    }
    
    if action in hirax_pages:
        st.session_state.active_page = hirax_pages[action]
        st.rerun()


def _execute_ai_action(action: str):
    """Execute AI-specific action."""
    if action in ["ai_query", "generate_code", "analyze_data", "get_help"]:
        st.session_state.active_page = "🤖 AI Assistant"
        st.rerun()


def render_adaptive_workspace():
    """
    Main function to render the complete adaptive workspace.
    
    Integrates sidebar, smart panels, and adaptive features.
    """
    workspace = get_adaptive_workspace()
    
    # Auto-adapt if needed
    if workspace.should_adapt():
        workspace.adapt_workspace()
    
    # Render adaptive sidebar (this would replace the standard sidebar)
    render_adaptive_sidebar()
    
    # Render smart panels
    render_adaptive_smart_panels()
    
    # Update user preferences based on current page
    current_page = st.session_state.get('active_page', '')
    if current_page:
        workspace.update_user_preferences('page_visit', {'page': current_page})


def get_adaptive_summary() -> Dict[str, Any]:
    """Get summary of adaptive workspace state."""
    workspace = get_adaptive_workspace()
    return workspace.get_adaptation_summary()
