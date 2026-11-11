"""
UDIP v3 — modules/context_engine.py
Global Context Engine for intelligent workspace awareness.

Transforms UDIP from isolated tools into a connected analytical workspace.
Maintains global context across all pages and modules while preserving
modular architecture and avoiding circular imports.
"""

import streamlit as st
import pandas as pd
import time
import uuid
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging


class ContextEngineState(dict):
    """Dict-compatible session state wrapper with a few legacy helper methods."""

    def get_active_dataset_id(self) -> Optional[str]:
        return self.get("workspace_context", {}).get("active_dataset_id")

    def get_active_dataset_name(self) -> Optional[str]:
        dataset_id = self.get_active_dataset_id()
        datasets = st.session_state.get("datasets", {})
        if dataset_id and dataset_id in datasets:
            return datasets[dataset_id].get("name", dataset_id)
        active_dataset = st.session_state.get("active_dataset")
        if isinstance(active_dataset, str) and active_dataset:
            return active_dataset
        return None

    def get_current_workflow_stage(self) -> "WorkflowStage":
        stage_value = self.get("current_stage", WorkflowStage.DATA_INGESTION.value)
        try:
            return WorkflowStage(stage_value)
        except Exception:
            return WorkflowStage.DATA_INGESTION


# Safe imports to avoid circular dependencies
try:
    from modules.dataset_memory import get_dataset_memory
    from modules.system_observer import get_system_observer
except ImportError:
    get_dataset_memory = None
    get_system_observer = None


class ContextType(Enum):
    """Types of context items tracked by the engine."""
    DATASET = "dataset"
    DATAFRAME = "dataframe"
    SQL_RESULT = "sql_result"
    CHART = "chart"
    ML_MODEL = "ml_model"
    HIRAX_SESSION = "hirax_session"
    EXPERIMENT = "experiment"
    REPORT = "report"
    TRANSFORMATION = "transformation"
    QUERY = "query"
    CODE = "code"
    PIPELINE = "pipeline"


class WorkflowStage(Enum):
    """Workflow stages for smart recommendations."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPARATION = "data_preparation"
    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    MODELING = "modeling"
    REPORTING = "reporting"
    DEPLOYMENT = "deployment"


@dataclass
class ContextItem:
    """Represents a single context item."""
    id: str
    type: ContextType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_active: bool = False
    size_bytes: int = 0
    parent_id: Optional[str] = None


@dataclass
class WorkflowMemory:
    """Tracks workflow history and patterns."""
    action: str
    page: str
    context_type: ContextType
    context_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Smart recommendation for next actions."""
    title: str
    description: str
    target_page: str
    confidence: float
    reasoning: str
    icon: str = "💡"


class GlobalContextEngine:
    """
    Global Context Engine for UDIP.
    
    Maintains workspace awareness across all pages and modules while
    preserving modular architecture and avoiding circular imports.
    """
    
    def __init__(self):
        """Initialize the context engine."""
        self._logger = logging.getLogger(__name__)
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for context engine."""
        existing = st.session_state.get("context_engine")
        if isinstance(existing, ContextEngineState):
            return

        if isinstance(existing, dict):
            st.session_state.context_engine = ContextEngineState(existing)
            return

        st.session_state.context_engine = ContextEngineState({
            "active_items": {},  # context_id -> ContextItem
            "workflow_memory": [],  # List[WorkflowMemory]
            "current_stage": WorkflowStage.DATA_INGESTION.value,
            "session_start": time.time(),
            "last_activity": time.time(),
            "recommendations": [],
            "recommendation_cache": {},
            "show_context_panel": False,
            "workspace_context": {
                "active_dataset_id": None,
                "active_dataframe_id": None,
                "active_sql_result_id": None,
                "active_chart_ids": [],
                "active_ml_model_ids": [],
                "active_hirax_session_id": None,
                "active_experiment_id": None,
                "active_report_ids": [],
            }
        })
    
    # ── Active Workspace Context ────────────────────────────────────────────────
    
    def set_active_dataset(self, dataset_id: str, name: str, metadata: Optional[Dict] = None):
        """Set the active dataset."""
        context_item = ContextItem(
            id=dataset_id,
            type=ContextType.DATASET,
            name=name,
            metadata=metadata or {},
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_dataset_id"] = dataset_id
        self._track_workflow("set_active_dataset", "context_engine", ContextType.DATASET, dataset_id)
    
    def set_active_dataframe(self, df: pd.DataFrame, name: str, metadata: Optional[Dict] = None):
        """Set the active dataframe (metadata only, not the actual dataframe)."""
        df_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=df_id,
            type=ContextType.DATAFRAME,
            name=name,
            metadata={
                **(metadata or {}),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "sample_data": df.head().to_dict() if len(df) > 0 else {}
            },
            is_active=True,
            size_bytes=df.memory_usage(deep=True).sum()
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_dataframe_id"] = df_id
        self._track_workflow("set_active_dataframe", "context_engine", ContextType.DATAFRAME, df_id)
        
        # Update workflow stage based on data state
        self._update_workflow_stage()
    
    def set_active_sql_result(self, result_df: pd.DataFrame, query: str, name: Optional[str] = None):
        """Set active SQL query result."""
        result_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=result_id,
            type=ContextType.SQL_RESULT,
            name=name or f"SQL Query {len(self._get_items_by_type(ContextType.SQL_RESULT)) + 1}",
            metadata={
                "query": query,
                "shape": result_df.shape,
                "columns": list(result_df.columns),
                "execution_time": time.time()
            },
            is_active=True,
            size_bytes=result_df.memory_usage(deep=True).sum()
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_sql_result_id"] = result_id
        self._track_workflow("set_active_sql_result", "context_engine", ContextType.SQL_RESULT, result_id)
    
    def set_active_chart(self, chart_config: Dict, name: str, metadata: Optional[Dict] = None):
        """Set active chart configuration."""
        chart_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=chart_id,
            type=ContextType.CHART,
            name=name,
            metadata={
                **(metadata or {}),
                "chart_config": chart_config,
                "created_at": time.time()
            },
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_chart_ids"].append(chart_id)
        self._track_workflow("set_active_chart", "context_engine", ContextType.CHART, chart_id)
    
    def set_active_ml_model(self, model_info: Dict, name: str, metadata: Optional[Dict] = None):
        """Set active ML model information."""
        model_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=model_id,
            type=ContextType.ML_MODEL,
            name=name,
            metadata={
                **(metadata or {}),
                "model_info": model_info,
                "created_at": time.time()
            },
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_ml_model_ids"].append(model_id)
        self._track_workflow("set_active_ml_model", "context_engine", ContextType.ML_MODEL, model_id)
    
    def toggle_context_panel(self):
        """Toggle the visibility of the context panel."""
        st.session_state.context_engine["show_context_panel"] = not st.session_state.context_engine.get("show_context_panel", False)
    
    def set_active_hirax_session(self, session_info: Dict, name: str):
        """Set active HIRAX session."""
        session_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=session_id,
            type=ContextType.HIRAX_SESSION,
            name=name,
            metadata=session_info,
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_hirax_session_id"] = session_id
        self._track_workflow("set_active_hirax_session", "context_engine", ContextType.HIRAX_SESSION, session_id)
    
    def set_active_experiment(self, experiment_info: Dict, name: str):
        """Set active experiment."""
        experiment_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=experiment_id,
            type=ContextType.EXPERIMENT,
            name=name,
            metadata=experiment_info,
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_experiment_id"] = experiment_id
        self._track_workflow("set_active_experiment", "context_engine", ContextType.EXPERIMENT, experiment_id)
    
    def set_active_report(self, report_info: Dict, name: str):
        """Set active report."""
        report_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=report_id,
            type=ContextType.REPORT,
            name=name,
            metadata=report_info,
            is_active=True
        )
        self._add_context_item(context_item)
        st.session_state.context_engine["workspace_context"]["active_report_ids"].append(report_id)
        self._track_workflow("set_active_report", "context_engine", ContextType.REPORT, report_id)
    
    # ── Cross-Page Awareness ─────────────────────────────────────────────────────
    
    def get_active_dataset(self) -> Optional[ContextItem]:
        """Get the currently active dataset."""
        dataset_id = st.session_state.context_engine["workspace_context"]["active_dataset_id"]
        if dataset_id:
            return self._get_context_item(dataset_id)
        return None
    
    def get_active_dataframe_info(self) -> Optional[ContextItem]:
        """Get information about the active dataframe (metadata only)."""
        df_id = st.session_state.context_engine["workspace_context"]["active_dataframe_id"]
        if df_id:
            return self._get_context_item(df_id)
        return None
    
    def get_active_sql_result(self) -> Optional[ContextItem]:
        """Get the active SQL result."""
        result_id = st.session_state.context_engine["workspace_context"]["active_sql_result_id"]
        if result_id:
            return self._get_context_item(result_id)
        return None
    
    def get_active_charts(self) -> List[ContextItem]:
        """Get all active charts."""
        chart_ids = st.session_state.context_engine["workspace_context"]["active_chart_ids"]
        return [self._get_context_item(chart_id) for chart_id in chart_ids if self._get_context_item(chart_id)]
    
    def get_active_ml_models(self) -> List[ContextItem]:
        """Get all active ML models."""
        model_ids = st.session_state.context_engine["workspace_context"]["active_ml_model_ids"]
        return [self._get_context_item(model_id) for model_id in model_ids if self._get_context_item(model_id)]
    
    def get_active_hirax_session(self) -> Optional[ContextItem]:
        """Get the active HIRAX session."""
        session_id = st.session_state.context_engine["workspace_context"]["active_hirax_session_id"]
        if session_id:
            return self._get_context_item(session_id)
        return None
    
    def get_active_experiment(self) -> Optional[ContextItem]:
        """Get the active experiment."""
        experiment_id = st.session_state.context_engine["workspace_context"]["active_experiment_id"]
        if experiment_id:
            return self._get_context_item(experiment_id)
        return None
    
    def get_active_reports(self) -> List[ContextItem]:
        """Get all active reports."""
        report_ids = st.session_state.context_engine["workspace_context"]["active_report_ids"]
        return [self._get_context_item(report_id) for report_id in report_ids if self._get_context_item(report_id)]
    
    def has_active_data(self) -> bool:
        """Check if there's any active data (dataset or dataframe)."""
        return (
            st.session_state.context_engine["workspace_context"]["active_dataset_id"] is not None or
            st.session_state.context_engine["workspace_context"]["active_dataframe_id"] is not None
        )
    
    def get_current_workflow_stage(self) -> WorkflowStage:
        """Get the current workflow stage."""
        return WorkflowStage(st.session_state.context_engine["current_stage"])
    
    # ── Workflow Memory ─────────────────────────────────────────────────────────
    
    def track_transformation(self, transformation_name: str, input_data: Dict, output_data: Dict):
        """Track a data transformation."""
        transform_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=transform_id,
            type=ContextType.TRANSFORMATION,
            name=transformation_name,
            metadata={
                "input_data": input_data,
                "output_data": output_data,
                "transformation_type": transformation_name
            }
        )
        self._add_context_item(context_item)
        self._track_workflow("apply_transformation", "context_engine", ContextType.TRANSFORMATION, transform_id)
    
    def track_sql_query(self, query: str, result_info: Dict):
        """Track an executed SQL query."""
        query_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=query_id,
            type=ContextType.QUERY,
            name=f"SQL Query {len(self._get_items_by_type(ContextType.QUERY)) + 1}",
            metadata={
                "query": query,
                "result_info": result_info,
                "execution_time": time.time()
            }
        )
        self._add_context_item(context_item)
        self._track_workflow("execute_sql_query", "context_engine", ContextType.QUERY, query_id)
    
    def track_ai_code_generation(self, prompt: str, code: str, language: str = "python"):
        """Track AI-generated code."""
        code_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=code_id,
            type=ContextType.CODE,
            name=f"AI Code {len(self._get_items_by_type(ContextType.CODE)) + 1}",
            metadata={
                "prompt": prompt,
                "code": code,
                "language": language,
                "generated_at": time.time()
            }
        )
        self._add_context_item(context_item)
        self._track_workflow("generate_ai_code", "context_engine", ContextType.CODE, code_id)
    
    def track_pipeline_execution(self, pipeline_info: Dict):
        """Track a pipeline execution."""
        pipeline_id = str(uuid.uuid4())
        context_item = ContextItem(
            id=pipeline_id,
            type=ContextType.PIPELINE,
            name=pipeline_info.get("name", f"Pipeline {len(self._get_items_by_type(ContextType.PIPELINE)) + 1}"),
            metadata=pipeline_info
        )
        self._add_context_item(context_item)
        self._track_workflow("execute_pipeline", "context_engine", ContextType.PIPELINE, pipeline_id)
    
    def get_workflow_history(self, limit: int = 50) -> List[WorkflowMemory]:
        """Get recent workflow history."""
        history = st.session_state.context_engine["workflow_memory"]
        return sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_items_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """Get all items of a specific type."""
        return self._get_items_by_type(context_type)
    
    # ── Smart Recommendations ─────────────────────────────────────────────────────
    
    def get_next_actions(self, current_page: Optional[str] = None) -> List[Recommendation]:
        """Get smart recommendations for next actions."""
        cache_key = f"recommendations_{current_page}_{self.get_current_workflow_stage().value}"
        
        # Check cache first
        if cache_key in st.session_state.context_engine["recommendation_cache"]:
            cache_time, recommendations = st.session_state.context_engine["recommendation_cache"][cache_key]
            if time.time() - cache_time < 30:  # Cache for 30 seconds
                return recommendations
        
        recommendations = []
        workflow_stage = self.get_current_workflow_stage()
        
        # Stage-based recommendations
        if workflow_stage == WorkflowStage.DATA_INGESTION:
            if self.has_active_data():
                recommendations.append(Recommendation(
                    "Clean Your Data",
                    "Start preparing your dataset for analysis",
                    "⚒️ Data Forge",
                    0.9,
                    "You have uploaded data - Data Forge can help clean and validate it",
                    "🧹"
                ))
                recommendations.append(Recommendation(
                    "Explore Data Structure",
                    "Understand your dataset's characteristics",
                    "📊 Visualize",
                    0.8,
                    "Visualize your data to identify patterns and issues",
                    "📈"
                ))
        
        elif workflow_stage == WorkflowStage.DATA_PREPARATION:
            if self.get_active_dataframe_info():
                recommendations.append(Recommendation(
                    "Create Visualizations",
                    "Generate charts to explore your cleaned data",
                    "📊 Visualize",
                    0.9,
                    "Your data is ready for visualization and exploration",
                    "📊"
                ))
                recommendations.append(Recommendation(
                    "Run SQL Queries",
                    "Query your data with SQL for deeper insights",
                    "📝 SQL Editor",
                    0.7,
                    "Use SQL to extract specific insights from your data",
                    "🔍"
                ))
        
        elif workflow_stage == WorkflowStage.EXPLORATION:
            charts = self.get_active_charts()
            if charts:
                recommendations.append(Recommendation(
                    "Build Report",
                    "Compile your findings into a professional report",
                    "📖 Report Studio",
                    0.8,
                    f"You have {len(charts)} charts ready for reporting",
                    "📝"
                ))
            recommendations.append(Recommendation(
                "Apply Machine Learning",
                "Discover patterns with ML algorithms",
                "🧠 Machine Learning",
                0.7,
                "Your explored data is ready for machine learning analysis",
                    "🤖"
            ))
        
        elif workflow_stage == WorkflowStage.ANALYSIS:
            models = self.get_active_ml_models()
            if models:
                recommendations.append(Recommendation(
                    "Document Findings",
                    "Create comprehensive reports of your analysis",
                    "📖 Report Studio",
                    0.9,
                    f"You have {len(models)} models to document",
                    "📋"
                ))
        
        # Context-specific recommendations
        if self.get_active_hirax_session():
            recommendations.append(Recommendation(
                "Analyze Signals",
                "Process radio frequency signals",
                "🛸 Signal Intelligence",
                0.8,
                "HIRAX session active - ready for signal analysis",
                "📡"
            ))
        
        # Sort by confidence and limit
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        recommendations = recommendations[:5]
        
        # Cache recommendations
        st.session_state.context_engine["recommendation_cache"][cache_key] = (time.time(), recommendations)
        
        return recommendations
    
    # ── Internal Helper Methods ───────────────────────────────────────────────────
    
    def _add_context_item(self, item: ContextItem):
        """Add a context item to the active items."""
        st.session_state.context_engine["active_items"][item.id] = item
        st.session_state.context_engine["last_activity"] = time.time()
    
    def _get_context_item(self, item_id: str) -> Optional[ContextItem]:
        """Get a context item by ID."""
        return st.session_state.context_engine["active_items"].get(item_id)
    
    def _get_items_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """Get all items of a specific type."""
        items = st.session_state.context_engine["active_items"]
        return [item for item in items.values() if item.type == context_type]
    
    def _track_workflow(self, action: str, page: str, context_type: ContextType, context_id: str, metadata: Optional[Dict] = None):
        """Track a workflow action."""
        workflow_memory = WorkflowMemory(
            action=action,
            page=page,
            context_type=context_type,
            context_id=context_id,
            metadata=metadata or {}
        )
        st.session_state.context_engine["workflow_memory"].append(workflow_memory)
        st.session_state.context_engine["last_activity"] = time.time()
    
    def _update_workflow_stage(self):
        """Update the workflow stage based on current context."""
        active_items = st.session_state.context_engine["active_items"]
        
        # Check for HIRAX activity first
        if any(item.type == ContextType.HIRAX_SESSION for item in active_items.values()):
            st.session_state.context_engine["current_stage"] = WorkflowStage.ANALYSIS.value
            return
        
        # Check for ML models
        if any(item.type == ContextType.ML_MODEL for item in active_items.values()):
            st.session_state.context_engine["current_stage"] = WorkflowStage.MODELING.value
            return
        
        # Check for charts
        if any(item.type == ContextType.CHART for item in active_items.values()):
            st.session_state.context_engine["current_stage"] = WorkflowStage.EXPLORATION.value
            return
        
        # Check for transformations
        if any(item.type == ContextType.TRANSFORMATION for item in active_items.values()):
            st.session_state.context_engine["current_stage"] = WorkflowStage.DATA_PREPARATION.value
            return
        
        # Check for data
        if any(item.type in [ContextType.DATASET, ContextType.DATAFRAME] for item in active_items.values()):
            st.session_state.context_engine["current_stage"] = WorkflowStage.DATA_INGESTION.value
            return
    
    # ── Integration Methods ───────────────────────────────────────────────────────
    
    def integrate_with_dataset_memory(self):
        """Safely integrate with DatasetMemory module."""
        if get_dataset_memory:
            try:
                memory = get_dataset_memory()
                # Sync dataset information
                for dataset_id, dataset_info in memory.get_all_datasets().items():
                    if dataset_id not in st.session_state.context_engine["active_items"]:
                        self.set_active_dataset(
                            dataset_id,
                            dataset_info.get("name", f"Dataset {dataset_id}"),
                            dataset_info
                        )
            except Exception as e:
                self._logger.warning(f"Failed to integrate with DatasetMemory: {e}")
    
    def integrate_with_system_observer(self):
        """Safely integrate with SystemObserver module."""
        if get_system_observer:
            try:
                observer = get_system_observer()
                # Track system events in workflow memory
                recent_events = observer.get_recent_events(limit=10)
                for event in recent_events:
                    self._track_workflow(
                        f"system_event_{event.get('type', 'unknown')}",
                        "system_observer",
                        ContextType.EXPERIMENT,  # Use experiment for system events
                        str(event.get("id", "unknown")),
                        event
                    )
            except Exception as e:
                self._logger.warning(f"Failed to integrate with SystemObserver: {e}")
    
    def cleanup_old_context(self, max_age_hours: int = 24):
        """Clean up old context items to prevent memory bloat."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        active_items = st.session_state.context_engine["active_items"]
        items_to_remove = []
        
        for item_id, item in active_items.items():
            # Keep active items and recent items
            if not item.is_active and (current_time - item.created_at) > max_age_seconds:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            del active_items[item_id]
            
            # Remove from workspace context if present
            workspace_ctx = st.session_state.context_engine["workspace_context"]
            for key, value in workspace_ctx.items():
                if isinstance(value, list) and item_id in value:
                    value.remove(item_id)
                elif value == item_id:
                    workspace_ctx[key] = None


# ── Global Instance ─────────────────────────────────────────────────────────────

_global_context_engine: Optional[GlobalContextEngine] = None


def get_context_engine() -> GlobalContextEngine:
    """Get the global context engine instance."""
    global _global_context_engine
    if _global_context_engine is None:
        _global_context_engine = GlobalContextEngine()
    return _global_context_engine


def init_context_engine():
    """Initialize the context engine (call once at app startup)."""
    engine = get_context_engine()
    engine.integrate_with_dataset_memory()
    engine.integrate_with_system_observer()
    return engine


# ── Convenience Functions ───────────────────────────────────────────────────────

def track_data_upload(dataset_id: str, name: str, df: pd.DataFrame, metadata: Optional[Dict] = None):
    """Convenience function to track data upload."""
    engine = get_context_engine()
    engine.set_active_dataset(dataset_id, name, metadata)
    engine.set_active_dataframe(df, name, metadata)


def track_chart_creation(chart_config: Dict, name: str, metadata: Optional[Dict] = None):
    """Convenience function to track chart creation."""
    engine = get_context_engine()
    engine.set_active_chart(chart_config, name, metadata)


def track_ml_training(model_info: Dict, name: str, metadata: Optional[Dict] = None):
    """Convenience function to track ML model training."""
    engine = get_context_engine()
    engine.set_active_ml_model(model_info, name, metadata)


def get_workspace_context() -> Dict[str, Any]:
    """Get the current workspace context."""
    engine = get_context_engine()
    return {
        "has_data": engine.has_active_data(),
        "active_dataset": engine.get_active_dataset(),
        "active_dataframe": engine.get_active_dataframe_info(),
        "active_charts": engine.get_active_charts(),
        "active_models": engine.get_active_ml_models(),
        "workflow_stage": engine.get_current_workflow_stage(),
        "recommendations": engine.get_next_actions()
    }


def render_context_aware_sidebar_info():
    """Render context-aware information in the sidebar."""
    engine = get_context_engine()
    context = get_workspace_context()
    
    st.markdown("### 🧠 Workspace Context")
    
    # Current stage
    stage_names = {
        WorkflowStage.DATA_INGESTION: "📥 Data Ingestion",
        WorkflowStage.DATA_PREPARATION: "🧹 Data Preparation", 
        WorkflowStage.EXPLORATION: "🔍 Exploration",
        WorkflowStage.ANALYSIS: "📊 Analysis",
        WorkflowStage.MODELING: "🤖 Modeling",
        WorkflowStage.REPORTING: "📋 Reporting",
        WorkflowStage.DEPLOYMENT: "🚀 Deployment"
    }
    
    stage_name = stage_names.get(context["workflow_stage"], "❓ Unknown")
    st.markdown(f"**Stage:** {stage_name}")
    
    # Active items summary
    if context["has_data"]:
        st.success("✅ Data loaded")
    else:
        st.info("📥 No data loaded")
    
    if context["active_charts"]:
        st.markdown(f"📊 Charts: {len(context['active_charts'])}")
    
    if context["active_models"]:
        st.markdown(f"🤖 Models: {len(context['active_models'])}")
    
    # Smart recommendations
    recommendations = context["recommendations"]
    if recommendations:
        st.markdown("### 💡 Smart Suggestions")
        for rec in recommendations[:3]:  # Show top 3
            if st.button(f"{rec.icon} {rec.title}", key=f"rec_{rec.title}", help=rec.reasoning):
                # Navigate to recommended page
                st.session_state.active_page = rec.target_page
                st.rerun()
