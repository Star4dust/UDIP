"""
UDIP v3 — modules/command_palette.py
Universal Command Palette for intelligent platform control.

Provides VS Code/Cursor/Notion-style command interface with natural language
processing, intelligent routing, and context-aware suggestions.
"""

import streamlit as st
import re
import time
import uuid
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from difflib import SequenceMatcher

# Safe imports to avoid circular dependencies
try:
    from modules.context_engine import get_context_engine
    from modules.orchestrator import get_orchestrator
    from modules.navigation import NAV_SECTIONS, normalize_page
    from modules.ai_engine import ask_groq
except ImportError:
    get_context_engine = None
    get_orchestrator = None
    NAV_SECTIONS = {}
    normalize_page = lambda x: x
    ask_groq = None


class CommandType(Enum):
    """Types of commands supported by the palette."""
    NAVIGATION = "navigation"
    DATA_ACTION = "data_action"
    AI_ACTION = "ai_action"
    HIRAX_ACTION = "hirax_action"
    WORKFLOW_ACTION = "workflow_action"
    EXPORT_ACTION = "export_action"
    SYSTEM_ACTION = "system_action"


class CommandPriority(Enum):
    """Priority levels for command suggestions."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class Command:
    """Represents a single command in the palette."""
    id: str
    title: str
    description: str
    category: CommandType
    keywords: List[str] = field(default_factory=list)
    action: Callable = None
    target_page: Optional[str] = None
    requires_data: bool = False
    requires_hirax: bool = False
    priority: CommandPriority = CommandPriority.MEDIUM
    icon: str = "⚡"
    shortcut: Optional[str] = None


@dataclass
class CommandExecution:
    """Result of a command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UniversalCommandPalette:
    """
    Universal Command Palette for UDIP.
    
    Provides intelligent command routing, natural language processing,
    and context-aware suggestions.
    """
    
    def __init__(self):
        """Initialize the command palette."""
        self._logger = logging.getLogger(__name__)
        self._commands: Dict[str, Command] = {}
        self._recent_commands: List[str] = []
        self._initialize_commands()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for command palette."""
        if "command_palette" not in st.session_state or not isinstance(st.session_state.get("command_palette"), dict):
            st.session_state["command_palette"] = {
                "is_open": False,
                "query": "",
                "selected_index": 0,
                "filtered_commands": [],
                "execution_history": [],
                "last_execution": None
            }
        else:
            state = st.session_state["command_palette"]
            state.setdefault("is_open", False)
            state.setdefault("query", "")
            state.setdefault("selected_index", 0)
            state.setdefault("filtered_commands", [])
            state.setdefault("execution_history", [])
            state.setdefault("last_execution", None)
    
    def _initialize_commands(self):
        """Initialize all available commands."""
        # Navigation Commands
        self._register_command(Command(
            id="nav_dashboard",
            title="Open Dashboard",
            description="Navigate to the main dashboard",
            category=CommandType.NAVIGATION,
            keywords=["dashboard", "home", "main", "start"],
            target_page="📌 Dashboard",
            icon="📊",
            priority=CommandPriority.HIGH
        ))
        
        self._register_command(Command(
            id="nav_upload",
            title="Upload Dataset",
            description="Open the data upload page",
            category=CommandType.NAVIGATION,
            keywords=["upload", "import", "add", "load", "dataset"],
            target_page="📁 Upload Data",
            icon="📁",
            priority=CommandPriority.HIGH
        ))
        
        self._register_command(Command(
            id="nav_visualize",
            title="Open Visualize",
            description="Navigate to data visualization studio",
            category=CommandType.NAVIGATION,
            keywords=["visualize", "chart", "plot", "graph", "viz"],
            target_page="📊 Visualize",
            requires_data=True,
            icon="📈",
            priority=CommandPriority.HIGH
        ))
        
        self._register_command(Command(
            id="nav_data_forge",
            title="Open Data Forge",
            description="Navigate to data cleaning and preparation",
            category=CommandType.NAVIGATION,
            keywords=["clean", "prepare", "forge", "transform", "preprocess"],
            target_page="⚒️ Data Forge",
            requires_data=True,
            icon="🧹",
            priority=CommandPriority.HIGH
        ))
        
        self._register_command(Command(
            id="nav_sql_editor",
            title="Open SQL Editor",
            description="Navigate to SQL query interface",
            category=CommandType.NAVIGATION,
            keywords=["sql", "query", "database", "editor"],
            target_page="📝 SQL Editor",
            requires_data=True,
            icon="🗄️",
            priority=CommandPriority.MEDIUM
        ))
        
        self._register_command(Command(
            id="nav_ai_assistant",
            title="Open AI Assistant",
            description="Navigate to AI-powered data assistant",
            category=CommandType.NAVIGATION,
            keywords=["ai", "assistant", "chat", "help", "copilot"],
            target_page="🤖 AI Assistant",
            icon="🤖",
            priority=CommandPriority.HIGH
        ))
        
        self._register_command(Command(
            id="nav_hirax",
            title="Open HIRAX Inspector",
            description="Navigate to HIRAX telescope data analysis",
            category=CommandType.NAVIGATION,
            keywords=["hirax", "telescope", "radio", "astronomy", "inspector"],
            target_page="🔭 HIRAX Inspector",
            icon="🔭",
            priority=CommandPriority.MEDIUM
        ))
        
        self._register_command(Command(
            id="nav_merge",
            title="Open Merge Wizard",
            description="Navigate to dataset merging interface",
            category=CommandType.NAVIGATION,
            keywords=["merge", "join", "combine", "union"],
            target_page="🔗 Merge Wizard",
            requires_data=True,
            icon="🔗",
            priority=CommandPriority.MEDIUM
        ))
        
        self._register_command(Command(
            id="nav_ml",
            title="Open Machine Learning",
            description="Navigate to ML modeling interface",
            category=CommandType.NAVIGATION,
            keywords=["ml", "machine learning", "model", "train", "predict"],
            target_page="🧠 Machine Learning",
            requires_data=True,
            icon="🧠",
            priority=CommandPriority.MEDIUM
        ))
        
        self._register_command(Command(
            id="nav_export",
            title="Open Export Center",
            description="Navigate to data export interface",
            category=CommandType.NAVIGATION,
            keywords=["export", "save", "download", "output"],
            target_page="💾 Export Center",
            icon="💾",
            priority=CommandPriority.MEDIUM
        ))
        
        # Data Action Commands
        self._register_command(Command(
            id="data_clean_missing",
            title="Clean Missing Values",
            description="Remove or impute missing values in active dataset",
            category=CommandType.DATA_ACTION,
            keywords=["clean", "missing", "null", "nan", "impute"],
            requires_data=True,
            icon="🧹",
            priority=CommandPriority.HIGH,
            action=self._action_clean_missing_values
        ))
        
        self._register_command(Command(
            id="data_remove_duplicates",
            title="Remove Duplicates",
            description="Remove duplicate rows from active dataset",
            category=CommandType.DATA_ACTION,
            keywords=["duplicates", "unique", "dedupe", "remove"],
            requires_data=True,
            icon="🔄",
            priority=CommandPriority.HIGH,
            action=self._action_remove_duplicates
        ))
        
        self._register_command(Command(
            id="data_sample",
            title="Sample Dataset",
            description="Create a random sample of the active dataset",
            category=CommandType.DATA_ACTION,
            keywords=["sample", "subset", "random", "reduce"],
            requires_data=True,
            icon="🎲",
            priority=CommandPriority.MEDIUM,
            action=self._action_sample_dataset
        ))
        
        self._register_command(Command(
            id="data_describe",
            title="Describe Dataset",
            description="Generate statistical summary of active dataset",
            category=CommandType.DATA_ACTION,
            keywords=["describe", "summary", "stats", "statistics"],
            requires_data=True,
            icon="📊",
            priority=CommandPriority.HIGH,
            action=self._action_describe_dataset
        ))
        
        # AI Action Commands
        self._register_command(Command(
            id="ai_analyze",
            title="Run Auto Analysis",
            description="Run AI-powered automated data analysis",
            category=CommandType.AI_ACTION,
            keywords=["analyze", "auto", "automatic", "ai analysis"],
            requires_data=True,
            icon="🔬",
            priority=CommandPriority.HIGH,
            action=self._action_auto_analysis
        ))
        
        self._register_command(Command(
            id="ai_generate_code",
            title="Generate Code",
            description="Generate Python/R code for data analysis",
            category=CommandType.AI_ACTION,
            keywords=["generate", "code", "python", "script", "program"],
            requires_data=True,
            icon="🐍",
            priority=CommandPriority.MEDIUM,
            action=self._action_generate_code
        ))
        
        self._register_command(Command(
            id="ai_explain_data",
            title="Explain Dataset",
            description="Get AI explanation of dataset patterns",
            category=CommandType.AI_ACTION,
            keywords=["explain", "understand", "patterns", "insights"],
            requires_data=True,
            icon="💡",
            priority=CommandPriority.MEDIUM,
            action=self._action_explain_dataset
        ))
        
        # HIRAX Action Commands
        self._register_command(Command(
            id="hirax_fft",
            title="Run FFT Analysis",
            description="Perform Fast Fourier Transform on HIRAX data",
            category=CommandType.HIRAX_ACTION,
            keywords=["fft", "fourier", "frequency", "spectrum"],
            requires_hirax=True,
            icon="🌊",
            priority=CommandPriority.HIGH,
            action=self._action_hirax_fft
        ))
        
        self._register_command(Command(
            id="hirax_rfi",
            title="Detect RFI",
            description="Detect Radio Frequency Interference in data",
            category=CommandType.HIRAX_ACTION,
            keywords=["rfi", "interference", "radio", "noise"],
            requires_hirax=True,
            icon="📡",
            priority=CommandPriority.HIGH,
            action=self._action_hirax_rfi
        ))
        
        self._register_command(Command(
            id="hirax_spectral",
            title="Spectral Analysis",
            description="Perform spectral classification and analysis",
            category=CommandType.HIRAX_ACTION,
            keywords=["spectral", "classify", "classification", "spectrum"],
            requires_hirax=True,
            icon="🌈",
            priority=CommandPriority.MEDIUM,
            action=self._action_hirax_spectral
        ))
        
        # Workflow Action Commands
        self._register_command(Command(
            id="workflow_create_pipeline",
            title="Create Pipeline",
            description="Create a new data processing pipeline",
            category=CommandType.WORKFLOW_ACTION,
            keywords=["pipeline", "workflow", "automate", "sequence"],
            icon="⚡",
            priority=CommandPriority.MEDIUM,
            action=self._action_create_pipeline
        ))
        
        self._register_command(Command(
            id="workflow_run_pipeline",
            title="Run Pipeline",
            description="Execute existing data processing pipeline",
            category=CommandType.WORKFLOW_ACTION,
            keywords=["run", "execute", "pipeline", "workflow"],
            icon="▶️",
            priority=CommandPriority.MEDIUM,
            action=self._action_run_pipeline
        ))
        
        # Export/Reporting Commands
        self._register_command(Command(
            id="export_generate_report",
            title="Generate Report",
            description="Generate comprehensive data analysis report",
            category=CommandType.EXPORT_ACTION,
            keywords=["report", "summary", "document", "export"],
            icon="📋",
            priority=CommandPriority.HIGH,
            action=self._action_generate_report
        ))
        
        self._register_command(Command(
            id="export_dashboard",
            title="Export Dashboard",
            description="Export current dashboard as interactive file",
            category=CommandType.EXPORT_ACTION,
            keywords=["dashboard", "export", "share", "interactive"],
            icon="📊",
            priority=CommandPriority.MEDIUM,
            action=self._action_export_dashboard
        ))
        
        # System Commands
        self._register_command(Command(
            id="system_clear_cache",
            title="Clear Cache",
            description="Clear application cache and temporary data",
            category=CommandType.SYSTEM_ACTION,
            keywords=["clear", "cache", "reset", "clean"],
            icon="🗑️",
            priority=CommandPriority.LOW,
            action=self._action_clear_cache
        ))
        
        self._register_command(Command(
            id="system_settings",
            title="Open Settings",
            description="Navigate to application settings",
            category=CommandType.SYSTEM_ACTION,
            keywords=["settings", "config", "preferences", "options"],
            target_page="⚙️ Settings",
            icon="⚙️",
            priority=CommandPriority.MEDIUM
        ))
    
    def _register_command(self, command: Command):
        """Register a command in the palette."""
        self._commands[command.id] = command
    
    # ── Command Processing ────────────────────────────────────────────────────────
    
    def process_query(self, query: str) -> List[Command]:
        """Process user query and return matching commands."""
        if not query.strip():
            return self._get_default_suggestions()
        
        query_lower = query.lower()
        matches = []
        
        for command in self._commands.values():
            score = self._calculate_match_score(query_lower, command)
            if score > 0:
                matches.append((command, score))
        
        # Sort by score and priority
        matches.sort(key=lambda x: (x[1], x[0].priority.value), reverse=True)
        
        # Add recent commands boost
        recent_boost = self._get_recent_commands_boost(query_lower)
        for i, (cmd, score) in enumerate(matches):
            if cmd.id in recent_boost:
                matches[i] = (cmd, score + recent_boost[cmd.id])
        
        # Re-sort after boosting
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return [cmd for cmd, _ in matches[:10]]  # Top 10 results
    
    def _calculate_match_score(self, query: str, command: Command) -> float:
        """Calculate match score between query and command."""
        score = 0.0
        
        # Exact title match
        if query == command.title.lower():
            score += 1.0
        
        # Title contains query
        if query in command.title.lower():
            score += 0.8
        
        # Keywords match
        for keyword in command.keywords:
            if query == keyword:
                score += 0.9
            elif query in keyword:
                score += 0.6
            elif keyword in query:
                score += 0.4
        
        # Description match (lower weight)
        if query in command.description.lower():
            score += 0.3
        
        # Fuzzy matching for typos
        title_ratio = SequenceMatcher(None, query, command.title.lower()).ratio()
        if title_ratio > 0.6:
            score += title_ratio * 0.3
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_default_suggestions(self) -> List[Command]:
        """Get default command suggestions based on context."""
        suggestions = []
        
        # Add context-aware suggestions first
        if get_context_engine:
            try:
                context = get_context_engine()
                workflow_stage = context.get_current_workflow_stage()
                
                # Stage-based suggestions
                if workflow_stage.value == "data_ingestion":
                    # Just uploaded data
                    suggestions.extend([
                        self._commands.get("nav_data_forge"),
                        self._commands.get("data_clean_missing"),
                        self._commands.get("data_describe"),
                        self._commands.get("nav_visualize")
                    ])
                elif workflow_stage.value == "data_preparation":
                    # Data being prepared
                    suggestions.extend([
                        self._commands.get("nav_visualize"),
                        self._commands.get("ai_analyze"),
                        self._commands.get("nav_sql_editor"),
                        self._commands.get("data_remove_duplicates")
                    ])
                elif workflow_stage.value == "exploration":
                    # Data exploration active
                    suggestions.extend([
                        self._commands.get("ai_generate_code"),
                        self._commands.get("nav_ml"),
                        self._commands.get("workflow_create_pipeline"),
                        self._commands.get("export_generate_report")
                    ])
                elif workflow_stage.value == "analysis":
                    # Analysis in progress
                    suggestions.extend([
                        self._commands.get("export_generate_report"),
                        self._commands.get("nav_export"),
                        self._commands.get("workflow_run_pipeline")
                    ])
                
                # If has active charts, suggest reporting
                if context.get_active_charts():
                    suggestions.append(self._commands.get("export_generate_report"))
                
                # If has active ML models, suggest reporting
                if context.get_active_ml_models():
                    suggestions.append(self._commands.get("export_generate_report"))
                
                # If HIRAX session, suggest HIRAX actions
                if context.get_active_hirax_session():
                    hirax_actions = [
                        self._commands.get("hirax_fft"),
                        self._commands.get("hirax_rfi"),
                        self._commands.get("hirax_spectral")
                    ]
                    suggestions.extend(hirax_actions)
                
                # If SQL results, suggest further analysis
                if context.get_active_sql_result():
                    suggestions.extend([
                        self._commands.get("nav_visualize"),
                        self._commands.get("ai_explain_data")
                    ])
                
            except Exception as e:
                self._logger.warning(f"Failed to get context for suggestions: {e}")
        
        # Add high priority navigation commands if no context-specific ones
        if not suggestions:
            for cmd in self._commands.values():
                if cmd.priority == CommandPriority.HIGH and cmd.category == CommandType.NAVIGATION:
                    suggestions.append(cmd)
        
        # Remove None values and limit
        suggestions = [cmd for cmd in suggestions if cmd is not None]
        return suggestions[:8]  # Limit to 8 suggestions
    
    def _get_recent_commands_boost(self, query: str) -> Dict[str, float]:
        """Get boost scores for recently used commands."""
        boost = {}
        for cmd_id in self._recent_commands[-5:]:  # Last 5 commands
            if cmd_id in self._commands:
                cmd = self._commands[cmd_id]
                if query in cmd.title.lower() or any(query in kw for kw in cmd.keywords):
                    boost[cmd_id] = 0.2
        return boost
    
    # ── Command Execution ────────────────────────────────────────────────────────
    
    def execute_command(self, command_id: str, query: str = "") -> CommandExecution:
        """Execute a command by ID."""
        if command_id not in self._commands:
            return CommandExecution(
                success=False,
                message="Command not found",
                error=f"Command ID '{command_id}' not registered"
            )
        
        command = self._commands[command_id]
        
        # Check prerequisites
        if command.requires_data and not self._has_active_data():
            return CommandExecution(
                success=False,
                message="No active dataset",
                error="This command requires an active dataset"
            )
        
        if command.requires_hirax and not self._has_active_hirax():
            return CommandExecution(
                success=False,
                message="No active HIRAX session",
                error="This command requires an active HIRAX session"
            )
        
        # Track command usage
        self._track_command_usage(command_id)
        
        try:
            # Navigation commands
            if command.target_page:
                return self._execute_navigation(command)
            
            # Action commands
            elif command.action:
                return command.action(query)
            
            else:
                return CommandExecution(
                    success=False,
                    message="Command not executable",
                    error="Command has no action or target page"
                )
        
        except Exception as e:
            self._logger.error(f"Command execution failed: {e}")
            return CommandExecution(
                success=False,
                message="Command execution failed",
                error=str(e)
            )
    
    def _execute_navigation(self, command: Command) -> CommandExecution:
        """Execute navigation command."""
        try:
            # Normalize page name
            target_page = normalize_page(command.target_page)
            
            # Update session state
            self._set_active_page(target_page)
            
            # Close command palette
            self._close_palette()
            
            # Trigger rerun
            st.rerun()
            
            return CommandExecution(
                success=True,
                message=f"Navigated to {command.title}",
                data={"target_page": target_page}
            )
        
        except Exception as e:
            return CommandExecution(
                success=False,
                message="Navigation failed",
                error=str(e)
            )
    
    def _track_command_usage(self, command_id: str):
        """Track command usage for suggestions."""
        if command_id in self._recent_commands:
            self._recent_commands.remove(command_id)
        self._recent_commands.append(command_id)
        
        # Keep only last 20 commands
        self._recent_commands = self._recent_commands[-20:]
    
    # ── Command Actions ───────────────────────────────────────────────────────────
    
    def _action_clean_missing_values(self, query: str) -> CommandExecution:
        """Action: Clean missing values."""
        try:
            # This would integrate with Data Forge
            st.session_state.active_page = "⚒️ Data Forge"
            self._close_palette()
            st.session_state._auto_clean_missing = True
            
            return CommandExecution(
                success=True,
                message="Opening Data Forge to clean missing values",
                data={"action": "clean_missing"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to clean missing values", error=str(e))
    
    def _action_remove_duplicates(self, query: str) -> CommandExecution:
        """Action: Remove duplicates."""
        try:
            st.session_state.active_page = "⚒️ Data Forge"
            self._close_palette()
            st.session_state._auto_remove_duplicates = True
            
            return CommandExecution(
                success=True,
                message="Opening Data Forge to remove duplicates",
                data={"action": "remove_duplicates"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to remove duplicates", error=str(e))
    
    def _action_sample_dataset(self, query: str) -> CommandExecution:
        """Action: Sample dataset."""
        try:
            st.session_state.active_page = "⚒️ Data Forge"
            self._close_palette()
            st.session_state._auto_sample = True
            
            return CommandExecution(
                success=True,
                message="Opening Data Forge to sample dataset",
                data={"action": "sample_dataset"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to sample dataset", error=str(e))
    
    def _action_describe_dataset(self, query: str) -> CommandExecution:
        """Action: Describe dataset."""
        try:
            st.session_state.active_page = "🔬 Auto Analysis"
            self._close_palette()
            st.session_state._auto_describe = True
            
            return CommandExecution(
                success=True,
                message="Opening Auto Analysis to describe dataset",
                data={"action": "describe_dataset"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to describe dataset", error=str(e))
    
    def _action_auto_analysis(self, query: str) -> CommandExecution:
        """Action: Run auto analysis."""
        try:
            st.session_state.active_page = "🔬 Auto Analysis"
            self._close_palette()
            
            return CommandExecution(
                success=True,
                message="Opening Auto Analysis",
                data={"action": "auto_analysis"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to run auto analysis", error=str(e))
    
    def _action_generate_code(self, query: str) -> CommandExecution:
        """Action: Generate code."""
        try:
            st.session_state.active_page = "🤖 AI Assistant"
            self._close_palette()
            st.session_state._code_generation_prompt = query or "Generate analysis code for my dataset"
            
            return CommandExecution(
                success=True,
                message="Opening AI Assistant to generate code",
                data={"action": "generate_code", "prompt": query}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to generate code", error=str(e))
    
    def _action_explain_dataset(self, query: str) -> CommandExecution:
        """Action: Explain dataset."""
        try:
            st.session_state.active_page = "🤖 AI Assistant"
            self._close_palette()
            st.session_state._explain_prompt = query or "Explain my dataset patterns"
            
            return CommandExecution(
                success=True,
                message="Opening AI Assistant to explain dataset",
                data={"action": "explain_dataset", "prompt": query}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to explain dataset", error=str(e))
    
    def _action_hirax_fft(self, query: str) -> CommandExecution:
        """Action: Run FFT analysis."""
        try:
            st.session_state.active_page = "🔭 HIRAX Inspector"
            self._close_palette()
            st.session_state._hirax_action = "fft"
            
            return CommandExecution(
                success=True,
                message="Opening HIRAX Inspector for FFT analysis",
                data={"action": "hirax_fft"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to run FFT analysis", error=str(e))
    
    def _action_hirax_rfi(self, query: str) -> CommandExecution:
        """Action: Detect RFI."""
        try:
            st.session_state.active_page = "🔭 HIRAX Inspector"
            self._close_palette()
            st.session_state._hirax_action = "rfi"
            
            return CommandExecution(
                success=True,
                message="Opening HIRAX Inspector for RFI detection",
                data={"action": "hirax_rfi"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to detect RFI", error=str(e))
    
    def _action_hirax_spectral(self, query: str) -> CommandExecution:
        """Action: Spectral analysis."""
        try:
            st.session_state.active_page = "🔭 HIRAX Inspector"
            self._close_palette()
            st.session_state._hirax_action = "spectral"
            
            return CommandExecution(
                success=True,
                message="Opening HIRAX Inspector for spectral analysis",
                data={"action": "hirax_spectral"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to run spectral analysis", error=str(e))
    
    def _action_create_pipeline(self, query: str) -> CommandExecution:
        """Action: Create pipeline."""
        try:
            st.session_state.active_page = "🧭 Agent Pipeline"
            self._close_palette()
            
            return CommandExecution(
                success=True,
                message="Opening Agent Pipeline to create workflow",
                data={"action": "create_pipeline"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to create pipeline", error=str(e))
    
    def _action_run_pipeline(self, query: str) -> CommandExecution:
        """Action: Run pipeline."""
        try:
            st.session_state.active_page = "🧭 Agent Pipeline"
            self._close_palette()
            
            return CommandExecution(
                success=True,
                message="Opening Agent Pipeline to run workflow",
                data={"action": "run_pipeline"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to run pipeline", error=str(e))
    
    def _action_generate_report(self, query: str) -> CommandExecution:
        """Action: Generate report."""
        try:
            st.session_state.active_page = "📖 Report Studio"
            self._close_palette()
            
            return CommandExecution(
                success=True,
                message="Opening Report Studio to generate report",
                data={"action": "generate_report"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to generate report", error=str(e))
    
    def _action_export_dashboard(self, query: str) -> CommandExecution:
        """Action: Export dashboard."""
        try:
            st.session_state.active_page = "💾 Export Center"
            self._close_palette()
            
            return CommandExecution(
                success=True,
                message="Opening Export Center to export dashboard",
                data={"action": "export_dashboard"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to export dashboard", error=str(e))
    
    def _action_clear_cache(self, query: str) -> CommandExecution:
        """Action: Clear cache."""
        try:
            # Clear various caches
            if "datasets" in st.session_state:
                # Keep datasets but clear temporary data
                pass
            
            # Clear context engine cache
            if get_context_engine:
                try:
                    context = get_context_engine()
                    context.cleanup_old_context(max_age_hours=0)  # Clear all
                except:
                    pass
            
            return CommandExecution(
                success=True,
                message="Cache cleared successfully",
                data={"action": "clear_cache"}
            )
        except Exception as e:
            return CommandExecution(False, "Failed to clear cache", error=str(e))
    
    # ── Helper Methods ───────────────────────────────────────────────────────────
    
    def _has_active_data(self) -> bool:
        """Check if there's active data."""
        if get_context_engine:
            try:
                return get_context_engine().has_active_data()
            except:
                pass
        
        # Fallback to checking datasets directly
        return bool(st.session_state.get("datasets", {}))
    
    def _has_active_hirax(self) -> bool:
        """Check if there's active HIRAX session."""
        if get_context_engine:
            try:
                return get_context_engine().get_active_hirax_session() is not None
            except:
                pass
        return False
    
    def get_all_commands(self) -> List[Command]:
        """Get all registered commands."""
        return list(self._commands.values())
    
    def get_commands_by_category(self, category: CommandType) -> List[Command]:
        """Get commands by category."""
        return [cmd for cmd in self._commands.values() if cmd.category == category]
    
    def get_recent_commands(self, limit: int = 10) -> List[Command]:
        """Get recently used commands."""
        recent = []
        for cmd_id in self._recent_commands[-limit:]:
            if cmd_id in self._commands:
                recent.append(self._commands[cmd_id])
        return recent


# ── Global Instance ─────────────────────────────────────────────────────────────

_global_command_palette: Optional[UniversalCommandPalette] = None


def get_command_palette() -> UniversalCommandPalette:
    """Get the global command palette instance."""
    global _global_command_palette
    if _global_command_palette is None:
        _global_command_palette = UniversalCommandPalette()
    return _global_command_palette


def init_command_palette():
    """Initialize the command palette (call once at app startup)."""
    return get_command_palette()


# ── UI Rendering ───────────────────────────────────────────────────────────────

def ensure_command_palette_state():
    """Ensure command palette state exists after Streamlit reruns/state resets."""
    try:
        if "command_palette" not in st.session_state or not isinstance(st.session_state.get("command_palette"), dict):
            st.session_state["command_palette"] = {
                "is_open": False,
                "query": "",
                "selected_index": 0,
                "filtered_commands": [],
                "execution_history": [],
                "last_execution": None,
            }
        else:
            state = st.session_state["command_palette"]
            state.setdefault("is_open", False)
            state.setdefault("query", "")
            state.setdefault("selected_index", 0)
            state.setdefault("filtered_commands", [])
            state.setdefault("execution_history", [])
            state.setdefault("last_execution", None)
        return True
    except Exception:
        return False


def render_command_palette():
    """Render the command palette UI."""
    if not ensure_command_palette_state():
        return
    palette = get_command_palette()
    state = st.session_state.get("command_palette", {})
    
    if not state["is_open"]:
        return
    
    # Render floating modal
    st.markdown("""
    <style>
    .command-palette-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 9999;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 10vh;
    }
    
    .command-palette-modal {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        width: 90%;
        max-width: 600px;
        max-height: 70vh;
        overflow: hidden;
    }
    
    .command-palette-header {
        padding: 16px 20px;
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
        color: var(--text-color);
    }
    
    .command-palette-input {
        width: 100%;
        padding: 12px 20px;
        border: none;
        outline: none;
        font-size: 16px;
        background: transparent;
        color: var(--text-color);
    }
    
    .command-palette-results {
        max-height: 400px;
        overflow-y: auto;
    }
    
    .command-palette-item {
        padding: 12px 20px;
        cursor: pointer;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 12px;
        transition: background-color 0.2s;
    }
    
    .command-palette-item:hover,
    .command-palette-item.selected {
        background: var(--secondary-background-color);
    }
    
    .command-palette-icon {
        font-size: 18px;
        width: 24px;
        text-align: center;
    }
    
    .command-palette-content {
        flex: 1;
    }
    
    .command-palette-title {
        font-weight: 500;
        color: var(--text-color);
        margin-bottom: 2px;
    }
    
    .command-palette-description {
        font-size: 12px;
        color: var(--text-color-secondary);
        opacity: 0.7;
    }
    
    .command-palette-shortcut {
        font-size: 11px;
        color: var(--text-color-secondary);
        opacity: 0.5;
        background: var(--secondary-background-color);
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Input field
    query = st.text_input(
        "🔍 Search commands...",
        value=state["query"],
        key="command_palette_input",
        placeholder="Type a command or search...",
        help="Use Ctrl+K to open, Escape to close, ↑↓ to navigate",
        label_visibility="collapsed"
    )
    
    # Update query and filter commands
    if query != state["query"]:
        state["query"] = query
        state["filtered_commands"] = palette.process_query(query)
        state["selected_index"] = 0
    
    # Get filtered commands
    commands = state["filtered_commands"]
    if not commands and state["query"]:
        commands = palette.process_query(state["query"])
        state["filtered_commands"] = commands
    
    # Display results
    if commands:
        for i, command in enumerate(commands):
            is_selected = i == state["selected_index"]
            
            col1, col2, col3 = st.columns([1, 6, 1])
            
            with col1:
                st.markdown(f"**{command.icon}**")
            
            with col2:
                st.markdown(f"**{command.title}**")
                st.caption(command.description)
            
            with col3:
                if command.shortcut:
                    st.markdown(f"`{command.shortcut}`")
            
            # Handle selection
            if st.button("", key=f"cmd_{command.id}", use_container_width=True):
                result = palette.execute_command(command.id, state["query"])
                state["last_execution"] = result
                if result.success:
                    state["is_open"] = False
                st.rerun()
    
    elif state["query"]:
        st.info("No matching commands found. Try different keywords.")
    else:
        # Show default suggestions
        suggestions = palette._get_default_suggestions()
        if suggestions:
            st.markdown("### Popular Commands")
            for command in suggestions:
                if st.button(f"{command.icon} {command.title}", key=f"suggest_{command.id}"):
                    result = palette.execute_command(command.id)
                    state["last_execution"] = result
                    if result.success:
                        state["is_open"] = False
                    st.rerun()
    
    # Keyboard navigation
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        const input = document.querySelector('[data-testid="stTextInput"] input');
        if (!input) return;
        
        if (e.key === 'Escape') {
            // Close palette
            window.parent.postMessage({type: 'close_command_palette'}, '*');
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            // Navigate down
            window.parent.postMessage({type: 'navigate_command_palette', direction: 'down'}, '*');
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            // Navigate up
            window.parent.postMessage({type: 'navigate_command_palette', direction: 'up'}, '*');
        } else if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            // Execute selected command
            window.parent.postMessage({type: 'execute_command_palette'}, '*');
        }
    });
    </script>
    """, unsafe_allow_html=True)


def toggle_command_palette():
    """Toggle command palette open/closed state."""
    if not ensure_command_palette_state():
        return
    state = st.session_state["command_palette"]
    state["is_open"] = not state.get("is_open", False)
    if state["is_open"]:
        state["query"] = ""
        state["selected_index"] = 0
        state["filtered_commands"] = get_command_palette()._get_default_suggestions()


def handle_keyboard_shortcut():
    """Handle Ctrl+K keyboard shortcut."""
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+K or Cmd+K to open command palette
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            window.parent.postMessage({type: 'toggle_command_palette'}, '*');
        }
    });
    </script>
    """, unsafe_allow_html=True)


# ── Convenience Functions ───────────────────────────────────────────────────────

def register_custom_command(command: Command):
    """Register a custom command."""
    palette = get_command_palette()
    palette._register_command(command)


def execute_command_by_title(title: str, query: str = "") -> CommandExecution:
    """Execute command by title (fuzzy matching)."""
    palette = get_command_palette()
    
    # Find exact match
    for cmd in palette._commands.values():
        if cmd.title.lower() == title.lower():
            return palette.execute_command(cmd.id, query)
    
    # Find fuzzy match
    for cmd in palette._commands.values():
        if title.lower() in cmd.title.lower():
            return palette.execute_command(cmd.id, query)
    
    return CommandExecution(False, "Command not found", error=f"No command found for '{title}'")