"""
UDIP v3 — modules/pipeline_builder.py
Visual Pipeline Builder for no-code workflow construction.

Allows users to visually construct analytical workflows using drag-and-drop pipeline nodes.
Combines no-code workflows, AI orchestration, HIRAX scientific processing, and execution pipelines.
"""

import streamlit as st
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import asyncio
import threading
from datetime import datetime

from modules.pipeline_runner import run_pipeline


class NodeType(Enum):
    """Pipeline node types."""
    # Data Operations
    UPLOAD = "upload"
    CLEAN = "clean"
    MERGE = "merge"
    SQL_QUERY = "sql_query"
    TRANSFORM = "transform"
    VALIDATION = "validation"
    
    # Analytics
    VISUALIZE = "visualize"
    AUTO_ANALYSIS = "auto_analysis"
    ML_TRAINING = "ml_training"
    FEATURE_ENGINEERING = "feature_engineering"
    
    # AI
    AI_ASSISTANT = "ai_assistant"
    AGENT_PIPELINE = "agent_pipeline"
    CODE_EXECUTION = "code_execution"
    
    # HIRAX
    RFI_DETECTION = "rfi_detection"
    FFT = "fft"
    POWER_SPECTRUM = "power_spectrum"
    SIGNAL_CLASSIFICATION = "signal_classification"
    HIRAX_VALIDATION = "hirax_validation"
    
    # Export
    REPORT_GENERATION = "report_generation"
    DASHBOARD_EXPORT = "dashboard_export"
    CSV_EXPORT = "csv_export"
    PDF_EXPORT = "pdf_export"


class NodeStatus(Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineNode:
    """Individual pipeline node."""
    id: str
    type: NodeType
    name: str
    description: str
    position: Tuple[float, float]  # x, y coordinates
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    error_message: Optional[str] = None
    execution_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class PipelineConnection:
    """Connection between pipeline nodes."""
    id: str
    source_node_id: str
    target_node_id: str
    source_output: str = "output"
    target_input: str = "input"
    created_at: float = field(default_factory=time.time)


@dataclass
class Pipeline:
    """Complete pipeline definition."""
    id: str
    name: str
    description: str
    nodes: Dict[str, PipelineNode] = field(default_factory=dict)
    connections: Dict[str, PipelineConnection] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.IDLE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Node execution result."""
    node_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PipelineBuilder:
    """
    Visual Pipeline Builder for no-code workflow construction.
    
    Provides drag-and-drop interface for building analytical workflows.
    """
    
    def __init__(self):
        """Initialize the pipeline builder."""
        self._logger = logging.getLogger(__name__)
        self._current_pipeline: Optional[Pipeline] = None
        self._execution_history: List[ExecutionResult] = []
        self._node_templates = self._initialize_node_templates()
        self._pipeline_templates = self._initialize_pipeline_templates()
        self._execution_thread: Optional[threading.Thread] = None
        self._execution_lock = threading.Lock()
        
        # Initialize session state for pipeline builder
        if "pipeline_builder" not in st.session_state:
            st.session_state.pipeline_builder = {
                "current_pipeline": None,
                "saved_pipelines": {},
                "execution_status": PipelineStatus.IDLE,
                "selected_node": None,
                "canvas_zoom": 1.0,
                "canvas_offset": {"x": 0, "y": 0},
                "show_grid": True,
                "auto_layout": True,
                "execution_history": []
            }
    
    def _initialize_node_templates(self) -> Dict[NodeType, Dict[str, Any]]:
        """Initialize node templates with configurations."""
        return {
            # Data Operations
            NodeType.UPLOAD: {
                "name": "Upload Data",
                "description": "Upload data from various sources",
                "icon": "📤",
                "category": "Data Operations",
                "config_schema": {
                    "source": {"type": "select", "options": ["file", "database", "api"], "default": "file"},
                    "file_type": {"type": "select", "options": ["csv", "json", "parquet"], "default": "csv"},
                    "dataset_name": {"type": "text", "default": ""}
                },
                "inputs": [],
                "outputs": ["dataset"]
            },
            
            NodeType.CLEAN: {
                "name": "Clean Data",
                "description": "Clean and preprocess data",
                "icon": "🧹",
                "category": "Data Operations",
                "config_schema": {
                    "remove_duplicates": {"type": "boolean", "default": True},
                    "handle_missing": {"type": "select", "options": ["drop", "fill", "interpolate"], "default": "drop"},
                    "remove_outliers": {"type": "boolean", "default": False},
                    "outlier_method": {"type": "select", "options": ["iqr", "zscore", "isolation"], "default": "iqr"}
                },
                "inputs": ["dataset"],
                "outputs": ["cleaned_dataset"]
            },
            
            NodeType.MERGE: {
                "name": "Merge Datasets",
                "description": "Combine multiple datasets",
                "icon": "🔗",
                "category": "Data Operations",
                "config_schema": {
                    "merge_type": {"type": "select", "options": ["inner", "outer", "left", "right"], "default": "inner"},
                    "merge_key": {"type": "text", "default": ""},
                    "suffixes": {"type": "text", "default": "_left,_right"}
                },
                "inputs": ["dataset1", "dataset2"],
                "outputs": ["merged_dataset"]
            },
            
            NodeType.SQL_QUERY: {
                "name": "SQL Query",
                "description": "Execute SQL queries on data",
                "icon": "🗃️",
                "category": "Data Operations",
                "config_schema": {
                    "query": {"type": "textarea", "default": "SELECT * FROM dataset"},
                    "database": {"type": "select", "options": ["sqlite", "postgresql"], "default": "sqlite"}
                },
                "inputs": ["dataset"],
                "outputs": ["query_result"]
            },
            
            NodeType.TRANSFORM: {
                "name": "Transform Data",
                "description": "Transform and reshape data",
                "icon": "🔄",
                "category": "Data Operations",
                "config_schema": {
                    "transform_type": {"type": "select", "options": ["pivot", "melt", "group", "filter"], "default": "group"},
                    "transform_config": {"type": "textarea", "default": "{}"}
                },
                "inputs": ["dataset"],
                "outputs": ["transformed_dataset"]
            },
            
            NodeType.VALIDATION: {
                "name": "Validate Data",
                "description": "Validate data quality and schema",
                "icon": "✅",
                "category": "Data Operations",
                "config_schema": {
                    "validation_rules": {"type": "textarea", "default": "{}"},
                    "strict_mode": {"type": "boolean", "default": False}
                },
                "inputs": ["dataset"],
                "outputs": ["validation_result", "validated_dataset"]
            },
            
            # Analytics
            NodeType.VISUALIZE: {
                "name": "Visualize",
                "description": "Create data visualizations",
                "icon": "📊",
                "category": "Analytics",
                "config_schema": {
                    "chart_type": {"type": "select", "options": ["line", "bar", "scatter", "histogram", "heatmap"], "default": "line"},
                    "x_column": {"type": "text", "default": ""},
                    "y_column": {"type": "text", "default": ""},
                    "title": {"type": "text", "default": ""}
                },
                "inputs": ["dataset"],
                "outputs": ["chart"]
            },
            
            NodeType.AUTO_ANALYSIS: {
                "name": "Auto Analysis",
                "description": "Automated data analysis",
                "icon": "🔍",
                "category": "Analytics",
                "config_schema": {
                    "analysis_type": {"type": "select", "options": ["descriptive", "correlation", "trend", "anomaly"], "default": "descriptive"},
                    "auto_report": {"type": "boolean", "default": True}
                },
                "inputs": ["dataset"],
                "outputs": ["analysis_result", "report"]
            },
            
            NodeType.ML_TRAINING: {
                "name": "ML Training",
                "description": "Train machine learning models",
                "icon": "🤖",
                "category": "Analytics",
                "config_schema": {
                    "model_type": {"type": "select", "options": ["regression", "classification", "clustering"], "default": "regression"},
                    "target_column": {"type": "text", "default": ""},
                    "feature_columns": {"type": "text", "default": ""},
                    "test_size": {"type": "number", "default": 0.2}
                },
                "inputs": ["dataset"],
                "outputs": ["model", "predictions", "metrics"]
            },
            
            NodeType.FEATURE_ENGINEERING: {
                "name": "Feature Engineering",
                "description": "Create and transform features",
                "icon": "🧪",
                "category": "Analytics",
                "config_schema": {
                    "feature_type": {"type": "select", "options": ["polynomial", "interaction", "encoding", "scaling"], "default": "scaling"},
                    "feature_config": {"type": "textarea", "default": "{}"}
                },
                "inputs": ["dataset"],
                "outputs": ["features", "feature_importance"]
            },
            
            # AI
            NodeType.AI_ASSISTANT: {
                "name": "AI Assistant",
                "description": "AI-powered data analysis",
                "icon": "🤖",
                "category": "AI",
                "config_schema": {
                    "query": {"type": "textarea", "default": "Analyze this data"},
                    "analysis_type": {"type": "select", "options": ["summary", "insights", "recommendations"], "default": "summary"}
                },
                "inputs": ["dataset"],
                "outputs": ["ai_response", "insights"]
            },
            
            NodeType.AGENT_PIPELINE: {
                "name": "Agent Pipeline",
                "description": "Execute AI agent workflows",
                "icon": "🧭",
                "category": "AI",
                "config_schema": {
                    "agent_type": {"type": "select", "options": ["data_analyst", "ml_engineer", "reporter"], "default": "data_analyst"},
                    "pipeline_config": {"type": "textarea", "default": "{}"}
                },
                "inputs": ["dataset"],
                "outputs": ["agent_result", "report"]
            },
            
            NodeType.CODE_EXECUTION: {
                "name": "Code Execution",
                "description": "Execute custom Python code",
                "icon": "📝",
                "category": "AI",
                "config_schema": {
                    "code": {"type": "textarea", "default": "# Enter your Python code"},
                    "libraries": {"type": "text", "default": "pandas,numpy"}
                },
                "inputs": ["dataset"],
                "outputs": ["code_result", "output_data"]
            },
            
            # HIRAX
            NodeType.RFI_DETECTION: {
                "name": "RFI Detection",
                "description": "Detect and flag RFI contamination",
                "icon": "📡",
                "category": "HIRAX",
                "config_schema": {
                    "detection_method": {"type": "select", "options": ["threshold", "statistical", "ml"], "default": "threshold"},
                    "threshold": {"type": "number", "default": 3.0},
                    "flag_method": {"type": "select", "options": ["mask", "interpolate", "remove"], "default": "mask"}
                },
                "inputs": ["hirax_data"],
                "outputs": ["rfi_flags", "cleaned_hirax"]
            },
            
            NodeType.FFT: {
                "name": "FFT Analysis",
                "description": "Fast Fourier Transform analysis",
                "icon": "📊",
                "category": "HIRAX",
                "config_schema": {
                    "window_function": {"type": "select", "options": ["hann", "hamming", "blackman", "none"], "default": "hann"},
                    "zero_padding": {"type": "boolean", "default": True},
                    "power_spectrum": {"type": "boolean", "default": True}
                },
                "inputs": ["hirax_data"],
                "outputs": ["fft_result", "frequency_spectrum"]
            },
            
            NodeType.POWER_SPECTRUM: {
                "name": "Power Spectrum",
                "description": "Generate power spectrum analysis",
                "icon": "🌈",
                "category": "HIRAX",
                "config_schema": {
                    "averaging": {"type": "boolean", "default": True},
                    "smoothing": {"type": "boolean", "default": False},
                    "normalize": {"type": "boolean", "default": True}
                },
                "inputs": ["fft_result"],
                "outputs": ["power_spectrum", "spectral_density"]
            },
            
            NodeType.SIGNAL_CLASSIFICATION: {
                "name": "Signal Classification",
                "description": "Classify signals in HIRAX data",
                "icon": "🔍",
                "category": "HIRAX",
                "config_schema": {
                    "classification_method": {"type": "select", "options": ["threshold", "ml", "statistical"], "default": "threshold"},
                    "signal_threshold": {"type": "number", "default": 5.0}
                },
                "inputs": ["power_spectrum"],
                "outputs": ["signal_classifications", "signal_flags"]
            },
            
            NodeType.HIRAX_VALIDATION: {
                "name": "HIRAX Validation",
                "description": "Validate HIRAX data quality",
                "icon": "✅",
                "category": "HIRAX",
                "config_schema": {
                    "validation_type": {"type": "select", "options": ["quality", "completeness", "scientific"], "default": "quality"},
                    "quality_threshold": {"type": "number", "default": 0.8}
                },
                "inputs": ["hirax_data"],
                "outputs": ["validation_result", "quality_metrics"]
            },
            
            # Export
            NodeType.REPORT_GENERATION: {
                "name": "Report Generation",
                "description": "Generate comprehensive reports",
                "icon": "📋",
                "category": "Export",
                "config_schema": {
                    "report_type": {"type": "select", "options": ["summary", "detailed", "executive"], "default": "summary"},
                    "format": {"type": "select", "options": ["html", "pdf", "markdown"], "default": "html"},
                    "include_charts": {"type": "boolean", "default": True}
                },
                "inputs": ["analysis_result"],
                "outputs": ["report"]
            },
            
            NodeType.DASHBOARD_EXPORT: {
                "name": "Dashboard Export",
                "description": "Export interactive dashboards",
                "icon": "📊",
                "category": "Export",
                "config_schema": {
                    "dashboard_type": {"type": "select", "options": ["streamlit", "plotly", "tableau"], "default": "streamlit"},
                    "export_format": {"type": "select", "options": ["html", "json", "py"], "default": "html"}
                },
                "inputs": ["chart", "analysis_result"],
                "outputs": ["dashboard"]
            },
            
            NodeType.CSV_EXPORT: {
                "name": "CSV Export",
                "description": "Export data to CSV format",
                "icon": "📄",
                "category": "Export",
                "config_schema": {
                    "filename": {"type": "text", "default": "output.csv"},
                    "include_index": {"type": "boolean", "default": False}
                },
                "inputs": ["dataset"],
                "outputs": ["csv_file"]
            },
            
            NodeType.PDF_EXPORT: {
                "name": "PDF Export",
                "description": "Export results to PDF format",
                "icon": "📑",
                "category": "Export",
                "config_schema": {
                    "filename": {"type": "text", "default": "output.pdf"},
                    "include_charts": {"type": "boolean", "default": True}
                },
                "inputs": ["report", "chart"],
                "outputs": ["pdf_file"]
            }
        }
    
    def _initialize_pipeline_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pipeline templates."""
        return {
            "hirax_full_analysis": {
                "name": "HIRAX Full Analysis",
                "description": "Complete HIRAX data analysis pipeline",
                "nodes": [
                    {"type": "upload", "name": "Upload HIRAX Data", "position": (100, 100)},
                    {"type": "rfi_detection", "name": "RFI Detection", "position": (300, 100)},
                    {"type": "fft", "name": "FFT Analysis", "position": (500, 100)},
                    {"type": "power_spectrum", "name": "Power Spectrum", "position": (700, 100)},
                    {"type": "signal_classification", "name": "Signal Classification", "position": (900, 100)},
                    {"type": "hirax_validation", "name": "HIRAX Validation", "position": (1100, 100)},
                    {"type": "report_generation", "name": "Generate Report", "position": (1300, 100)}
                ],
                "connections": [
                    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
                ]
            },
            
            "data_cleaning": {
                "name": "Data Cleaning Workflow",
                "description": "Standard data cleaning and validation pipeline",
                "nodes": [
                    {"type": "upload", "name": "Upload Data", "position": (100, 100)},
                    {"type": "validation", "name": "Initial Validation", "position": (300, 100)},
                    {"type": "clean", "name": "Clean Data", "position": (500, 100)},
                    {"type": "validation", "name": "Final Validation", "position": (700, 100)},
                    {"type": "csv_export", "name": "Export Clean Data", "position": (900, 100)}
                ],
                "connections": [
                    (0, 1), (1, 2), (2, 3), (3, 4)
                ]
            },
            
            "ml_training": {
                "name": "ML Training Workflow",
                "description": "Complete machine learning training pipeline",
                "nodes": [
                    {"type": "upload", "name": "Upload Data", "position": (100, 100)},
                    {"type": "feature_engineering", "name": "Feature Engineering", "position": (300, 100)},
                    {"type": "ml_training", "name": "Train Model", "position": (500, 100)},
                    {"type": "visualize", "name": "Visualize Results", "position": (700, 100)},
                    {"type": "report_generation", "name": "Generate Report", "position": (900, 100)}
                ],
                "connections": [
                    (0, 1), (1, 2), (2, 3), (2, 4)
                ]
            },
            
            "auto_reporting": {
                "name": "Auto Reporting Workflow",
                "description": "Automated analysis and reporting pipeline",
                "nodes": [
                    {"type": "upload", "name": "Upload Data", "position": (100, 100)},
                    {"type": "auto_analysis", "name": "Auto Analysis", "position": (300, 100)},
                    {"type": "visualize", "name": "Create Charts", "position": (500, 100)},
                    {"type": "ai_assistant", "name": "AI Insights", "position": (700, 100)},
                    {"type": "report_generation", "name": "Generate Report", "position": (900, 100)},
                    {"type": "dashboard_export", "name": "Export Dashboard", "position": (1100, 100)}
                ],
                "connections": [
                    (0, 1), (1, 2), (1, 3), (1, 4), (2, 5)
                ]
            }
        }
    
    def create_pipeline(self, name: str, description: str = "") -> Pipeline:
        """
        Create a new pipeline.
        
        Args:
            name: Pipeline name
            description: Pipeline description
            
        Returns:
            Created pipeline
        """
        pipeline = Pipeline(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        
        self._current_pipeline = pipeline
        st.session_state.pipeline_builder["current_pipeline"] = pipeline
        
        return pipeline
    
    def add_node(self, node_type: NodeType, position: Tuple[float, float], 
                 config: Optional[Dict[str, Any]] = None) -> PipelineNode:
        """
        Add a node to the current pipeline.
        
        Args:
            node_type: Type of node to add
            position: Position on canvas (x, y)
            config: Node configuration
            
        Returns:
            Created node
        """
        if not self._current_pipeline:
            raise ValueError("No current pipeline")
        
        template = self._node_templates[node_type]
        
        node = PipelineNode(
            id=str(uuid.uuid4()),
            type=node_type,
            name=template["name"],
            description=template["description"],
            position=position,
            config=config or {},
            inputs=template["inputs"].copy(),
            outputs=template["outputs"].copy()
        )
        
        self._current_pipeline.nodes[node.id] = node
        self._current_pipeline.updated_at = time.time()
        
        return node
    
    def add_connection(self, source_node_id: str, target_node_id: str,
                      source_output: str = "output", target_input: str = "input") -> PipelineConnection:
        """
        Add a connection between nodes.
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            source_output: Source output name
            target_input: Target input name
            
        Returns:
            Created connection
        """
        if not self._current_pipeline:
            raise ValueError("No current pipeline")
        
        connection = PipelineConnection(
            id=str(uuid.uuid4()),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_output=source_output,
            target_input=target_input
        )
        
        self._current_pipeline.connections[connection.id] = connection
        self._current_pipeline.updated_at = time.time()
        
        return connection
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the pipeline.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if successful
        """
        if not self._current_pipeline:
            return False
        
        # Remove node
        if node_id in self._current_pipeline.nodes:
            del self._current_pipeline.nodes[node_id]
        
        # Remove connections to/from this node
        connections_to_remove = []
        for conn_id, conn in self._current_pipeline.connections.items():
            if conn.source_node_id == node_id or conn.target_node_id == node_id:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self._current_pipeline.connections[conn_id]
        
        self._current_pipeline.updated_at = time.time()
        return True
    
    def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a connection from the pipeline.
        
        Args:
            connection_id: Connection ID to remove
            
        Returns:
            True if successful
        """
        if not self._current_pipeline:
            return False
        
        if connection_id in self._current_pipeline.connections:
            del self._current_pipeline.connections[connection_id]
            self._current_pipeline.updated_at = time.time()
            return True
        
        return False
    
    def get_pipeline_execution_order(self) -> List[str]:
        """
        Get the execution order for pipeline nodes.
        
        Returns:
            List of node IDs in execution order
        """
        if not self._current_pipeline:
            return []
        
        # Build dependency graph
        dependencies = defaultdict(set)
        for conn in self._current_pipeline.connections.values():
            dependencies[conn.target_node_id].add(conn.source_node_id)
        
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if node_id in visited:
                return
            
            temp_visited.add(node_id)
            for dep_id in dependencies[node_id]:
                visit(dep_id)
            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
        
        for node_id in self._current_pipeline.nodes:
            if node_id not in visited:
                visit(node_id)
        
        return order
    
    def _build_execution_plan(self) -> List[Dict[str, Any]]:
        """Convert the current visual pipeline into a pipeline_runner execution plan."""
        if not self._current_pipeline:
            raise ValueError("No current pipeline to execute.")

        steps: List[Dict[str, Any]] = []
        for node_id in self.get_pipeline_execution_order():
            node = self._current_pipeline.nodes[node_id]
            cfg = node.config or {}

            if node.type == NodeType.UPLOAD:
                file_path = cfg.get("file_path") or cfg.get("dataset_name")
                if not isinstance(file_path, str) or not file_path.strip():
                    raise ValueError(f"Upload node '{node.name}' requires a valid file_path or dataset_name.")
                steps.append({"action": "load_dataset", "file_path": file_path.strip()})

            elif node.type == NodeType.CLEAN:
                method = cfg.get("handle_missing") or cfg.get("transform_type") or "standard"
                steps.append({"action": "clean_data", "method": str(method)})

            elif node.type == NodeType.MERGE:
                merge_key = cfg.get("merge_key") or cfg.get("key")
                if not isinstance(merge_key, str) or not merge_key.strip():
                    raise ValueError(f"Merge node '{node.name}' requires a merge_key.")
                dataset_name = cfg.get("dataset_name") or cfg.get("other_dataset")
                if not isinstance(dataset_name, str) or not dataset_name.strip():
                    raise ValueError(f"Merge node '{node.name}' requires a dataset_name for the second dataset.")
                steps.append({
                    "action": "merge_datasets",
                    "dataset_name": dataset_name.strip(),
                    "key": merge_key.strip(),
                    "how": cfg.get("merge_type", "inner")
                })

            elif node.type == NodeType.SQL_QUERY:
                query = cfg.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"SQL Query node '{node.name}' requires a SQL query.")
                steps.append({"action": "run_sql", "query": query.strip()})

            elif node.type == NodeType.VISUALIZE:
                chart_type = cfg.get("chart_type", "line")
                steps.append({"action": "create_visualization", "type": str(chart_type)})

            elif node.type == NodeType.RFI_DETECTION:
                threshold = cfg.get("threshold", 3.0)
                try:
                    threshold_value = float(threshold)
                except (TypeError, ValueError):
                    threshold_value = 3.0
                steps.append({"action": "detect_rfi", "threshold": threshold_value})

            elif node.type == NodeType.FFT:
                sampling_rate = cfg.get("sampling_rate", 1.0)
                try:
                    sampling_rate_value = float(sampling_rate)
                except (TypeError, ValueError):
                    sampling_rate_value = 1.0
                steps.append({"action": "compute_fft", "sampling_rate": sampling_rate_value})

            elif node.type == NodeType.POWER_SPECTRUM:
                sampling_rate = cfg.get("sampling_rate", 1.0)
                try:
                    sampling_rate_value = float(sampling_rate)
                except (TypeError, ValueError):
                    sampling_rate_value = 1.0
                steps.append({"action": "power_spectrum", "sampling_rate": sampling_rate_value})

            elif node.type == NodeType.SIGNAL_CLASSIFICATION:
                steps.append({"action": "classify_signal"})

            elif node.type == NodeType.HIRAX_VALIDATION:
                steps.append({"action": "detect_schema"})

            elif node.type == NodeType.ML_TRAINING:
                target = cfg.get("target_column") or cfg.get("target")
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(f"ML Training node '{node.name}' requires a target_column.")
                features = cfg.get("feature_columns", "")
                features_list = [f.strip() for f in str(features).split(",") if f.strip()]
                task = "classification" if str(cfg.get("model_type", "classification")).lower() == "classification" else "regression"
                steps.append({
                    "action": "train_model",
                    "target": target.strip(),
                    "features": features_list,
                    "task": task,
                })

            elif node.type == NodeType.FEATURE_ENGINEERING:
                steps.append({"action": "engineer_features", "method": "generate"})

            elif node.type == NodeType.REPORT_GENERATION:
                report_format = cfg.get("format", "html")
                steps.append({"action": "generate_report", "format": str(report_format)})

            elif node.type == NodeType.VALIDATION:
                steps.append({"action": "detect_schema"})

            elif node.type == NodeType.AI_ASSISTANT:
                ds_name = cfg.get("analysis_type", "dataset")
                steps.append({"action": "generate_ai_report", "ds_name": str(ds_name)})

            else:
                raise ValueError(f"Pipeline node type '{node.type.value}' is not supported for execution.")

        return steps

    def execute_pipeline(self) -> bool:
        """
        Execute the current pipeline.
        
        Returns:
            True if execution started successfully
        """
        if not self._current_pipeline:
            return False
        
        with self._execution_lock:
            if self._execution_thread and self._execution_thread.is_alive():
                return False  # Already executing
            
            # Reset node statuses
            for node in self._current_pipeline.nodes.values():
                node.status = NodeStatus.PENDING
                node.error_message = None
                node.execution_time = 0.0
            
            self._current_pipeline.status = PipelineStatus.RUNNING
            st.session_state.pipeline_builder["execution_status"] = PipelineStatus.RUNNING
            
            # Start execution in separate thread
            self._execution_thread = threading.Thread(target=self._execute_pipeline_thread)
            self._execution_thread.start()
            
            return True
    
    def _execute_pipeline_thread(self):
        """Execute pipeline in separate thread."""
        try:
            execution_order = self.get_pipeline_execution_order()
            plan_steps = self._build_execution_plan()
            plan = {"execution_plan": plan_steps}
            result = run_pipeline(plan)

            step_results = result.get("steps_executed", [])
            for idx, node_id in enumerate(execution_order):
                node = self._current_pipeline.nodes[node_id]
                node.status = NodeStatus.COMPLETED
                node.error_message = None
                node.execution_time = 0.0

                if idx < len(step_results):
                    step = step_results[idx]
                    if not step.get("success", False):
                        node.status = NodeStatus.FAILED
                        node.error_message = step.get("error") or step.get("output")
                    node.execution_time = 0.0
                else:
                    node.status = NodeStatus.FAILED
                    node.error_message = "Execution output missing for node."

            if self._current_pipeline.status != PipelineStatus.CANCELLED:
                failed_nodes = [n for n in self._current_pipeline.nodes.values() if n.status == NodeStatus.FAILED]
                self._current_pipeline.status = PipelineStatus.FAILED if failed_nodes else PipelineStatus.COMPLETED

            st.session_state.pipeline_builder["execution_status"] = self._current_pipeline.status
            st.session_state.pipeline_builder["execution_history"].append(result)

        except Exception as e:
            self._logger.error(f"Pipeline execution failed: {e}")
            if self._current_pipeline:
                self._current_pipeline.status = PipelineStatus.FAILED
            st.session_state.pipeline_builder["execution_status"] = PipelineStatus.FAILED
            if self._current_pipeline:
                for node in self._current_pipeline.nodes.values():
                    if node.status == NodeStatus.RUNNING:
                        node.status = NodeStatus.FAILED
                        node.error_message = str(e)
    
    def _execute_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a single pipeline node.
        
        Args:
            node: Node to execute
            input_data: Input data for the node
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        try:
            # Route to appropriate execution handler
            if node.type == NodeType.UPLOAD:
                result = self._execute_upload_node(node, input_data)
            elif node.type == NodeType.CLEAN:
                result = self._execute_clean_node(node, input_data)
            elif node.type == NodeType.VISUALIZE:
                result = self._execute_visualize_node(node, input_data)
            elif node.type == NodeType.RFI_DETECTION:
                result = self._execute_rfi_detection_node(node, input_data)
            elif node.type == NodeType.FFT:
                result = self._execute_fft_node(node, input_data)
            elif node.type == NodeType.AI_ASSISTANT:
                result = self._execute_ai_assistant_node(node, input_data)
            elif node.type == NodeType.REPORT_GENERATION:
                result = self._execute_report_generation_node(node, input_data)
            else:
                result = ExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=f"Node type {node.type} not implemented"
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                node_id=node.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _execute_upload_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute upload node."""
        # Implementation would integrate with data upload systems
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"uploaded": True, "dataset_name": node.config.get("dataset_name", "uploaded_data")}
        )
    
    def _execute_clean_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute data cleaning node."""
        # Implementation would integrate with data cleaning systems
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"cleaned": True, "rows_removed": 0}
        )
    
    def _execute_visualize_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute visualization node."""
        # Implementation would integrate with visualization systems
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"chart_type": node.config.get("chart_type", "line"), "chart_data": {}}
        )
    
    def _execute_rfi_detection_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute RFI detection node."""
        # Implementation would integrate with HIRAX RFI detection
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"rfi_flags": [], "cleaned_data": {}}
        )
    
    def _execute_fft_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute FFT node."""
        # Implementation would integrate with HIRAX FFT analysis
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"fft_result": {}, "frequency_spectrum": {}}
        )
    
    def _execute_ai_assistant_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute AI assistant node."""
        # Implementation would integrate with AI assistant
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"ai_response": "Analysis complete", "insights": []}
        )
    
    def _execute_report_generation_node(self, node: PipelineNode, input_data: Dict[str, Any]) -> ExecutionResult:
        """Execute report generation node."""
        # Implementation would integrate with report generation
        return ExecutionResult(
            node_id=node.id,
            success=True,
            data={"report": "Generated report", "format": node.config.get("format", "html")}
        )
    
    def cancel_execution(self) -> bool:
        """
        Cancel pipeline execution.
        
        Returns:
            True if cancellation successful
        """
        if self._current_pipeline and self._current_pipeline.status == PipelineStatus.RUNNING:
            self._current_pipeline.status = PipelineStatus.CANCELLED
            st.session_state.pipeline_builder["execution_status"] = PipelineStatus.CANCELLED
            return True
        return False
    
    def save_pipeline(self, pipeline_id: Optional[str] = None) -> bool:
        """
        Save the current pipeline.
        
        Args:
            pipeline_id: Optional pipeline ID (uses current if None)
            
        Returns:
            True if successful
        """
        if not self._current_pipeline:
            return False
        
        pipeline_id = pipeline_id or self._current_pipeline.id
        st.session_state.pipeline_builder["saved_pipelines"][pipeline_id] = self._current_pipeline
        
        return True
    
    def load_pipeline(self, pipeline_id: str) -> bool:
        """
        Load a saved pipeline.
        
        Args:
            pipeline_id: Pipeline ID to load
            
        Returns:
            True if successful
        """
        saved_pipelines = st.session_state.pipeline_builder["saved_pipelines"]
        if pipeline_id in saved_pipelines:
            self._current_pipeline = saved_pipelines[pipeline_id]
            st.session_state.pipeline_builder["current_pipeline"] = self._current_pipeline
            return True
        return False
    
    def export_pipeline(self, format: str = "json") -> Optional[str]:
        """
        Export the current pipeline.
        
        Args:
            format: Export format ("json", "yaml")
            
        Returns:
            Exported pipeline string
        """
        if not self._current_pipeline:
            return None
        
        if format == "json":
            return json.dumps({
                "id": self._current_pipeline.id,
                "name": self._current_pipeline.name,
                "description": self._current_pipeline.description,
                "nodes": {
                    node_id: {
                        "id": node.id,
                        "type": node.type.value,
                        "name": node.name,
                        "description": node.description,
                        "position": node.position,
                        "config": node.config,
                        "inputs": node.inputs,
                        "outputs": node.outputs
                    }
                    for node_id, node in self._current_pipeline.nodes.items()
                },
                "connections": {
                    conn_id: {
                        "id": conn.id,
                        "source_node_id": conn.source_node_id,
                        "target_node_id": conn.target_node_id,
                        "source_output": conn.source_output,
                        "target_input": conn.target_input
                    }
                    for conn_id, conn in self._current_pipeline.connections.items()
                }
            }, indent=2)
        
        return None
    
    def generate_pipeline_from_description(self, description: str) -> Optional[Pipeline]:
        """
        Generate a pipeline from natural language description.
        
        Args:
            description: Natural language description
            
        Returns:
            Generated pipeline
        """
        # Simplified AI pipeline generation
        description_lower = description.lower()
        
        if "hirax" in description_lower and "analysis" in description_lower:
            return self._create_hirax_pipeline()
        elif "clean" in description_lower or "cleaning" in description_lower:
            return self._create_cleaning_pipeline()
        elif "ml" in description_lower or "machine learning" in description_lower:
            return self._create_ml_pipeline()
        elif "report" in description_lower or "reporting" in description_lower:
            return self._create_reporting_pipeline()
        else:
            return self._create_general_pipeline()
    
    def _create_hirax_pipeline(self) -> Pipeline:
        """Create a HIRAX analysis pipeline."""
        pipeline = self.create_pipeline("Generated HIRAX Pipeline", "Auto-generated HIRAX analysis pipeline")
        
        # Add nodes
        nodes = [
            self.add_node(NodeType.UPLOAD, (100, 100)),
            self.add_node(NodeType.RFI_DETECTION, (300, 100)),
            self.add_node(NodeType.FFT, (500, 100)),
            self.add_node(NodeType.POWER_SPECTRUM, (700, 100)),
            self.add_node(NodeType.SIGNAL_CLASSIFICATION, (900, 100)),
            self.add_node(NodeType.REPORT_GENERATION, (1100, 100))
        ]
        
        # Add connections
        for i in range(len(nodes) - 1):
            self.add_connection(nodes[i].id, nodes[i + 1].id)
        
        return pipeline
    
    def _create_cleaning_pipeline(self) -> Pipeline:
        """Create a data cleaning pipeline."""
        pipeline = self.create_pipeline("Generated Cleaning Pipeline", "Auto-generated data cleaning pipeline")
        
        nodes = [
            self.add_node(NodeType.UPLOAD, (100, 100)),
            self.add_node(NodeType.VALIDATION, (300, 100)),
            self.add_node(NodeType.CLEAN, (500, 100)),
            self.add_node(NodeType.VALIDATION, (700, 100)),
            self.add_node(NodeType.CSV_EXPORT, (900, 100))
        ]
        
        for i in range(len(nodes) - 1):
            self.add_connection(nodes[i].id, nodes[i + 1].id)
        
        return pipeline
    
    def _create_ml_pipeline(self) -> Pipeline:
        """Create a machine learning pipeline."""
        pipeline = self.create_pipeline("Generated ML Pipeline", "Auto-generated ML training pipeline")
        
        nodes = [
            self.add_node(NodeType.UPLOAD, (100, 100)),
            self.add_node(NodeType.FEATURE_ENGINEERING, (300, 100)),
            self.add_node(NodeType.ML_TRAINING, (500, 100)),
            self.add_node(NodeType.VISUALIZE, (700, 100)),
            self.add_node(NodeType.REPORT_GENERATION, (900, 100))
        ]
        
        for i in range(len(nodes) - 1):
            self.add_connection(nodes[i].id, nodes[i + 1].id)
        
        return pipeline
    
    def _create_reporting_pipeline(self) -> Pipeline:
        """Create a reporting pipeline."""
        pipeline = self.create_pipeline("Generated Reporting Pipeline", "Auto-generated reporting pipeline")
        
        nodes = [
            self.add_node(NodeType.UPLOAD, (100, 100)),
            self.add_node(NodeType.AUTO_ANALYSIS, (300, 100)),
            self.add_node(NodeType.VISUALIZE, (500, 100)),
            self.add_node(NodeType.AI_ASSISTANT, (700, 100)),
            self.add_node(NodeType.REPORT_GENERATION, (900, 100))
        ]
        
        for i in range(len(nodes) - 1):
            self.add_connection(nodes[i].id, nodes[i + 1].id)
        
        return pipeline
    
    def _create_general_pipeline(self) -> Pipeline:
        """Create a general purpose pipeline."""
        pipeline = self.create_pipeline("Generated General Pipeline", "Auto-generated general pipeline")
        
        nodes = [
            self.add_node(NodeType.UPLOAD, (100, 100)),
            self.add_node(NodeType.AUTO_ANALYSIS, (300, 100)),
            self.add_node(NodeType.VISUALIZE, (500, 100))
        ]
        
        for i in range(len(nodes) - 1):
            self.add_connection(nodes[i].id, nodes[i + 1].id)
        
        return pipeline


# Global instance
_global_pipeline_builder: Optional[PipelineBuilder] = None


def get_pipeline_builder() -> PipelineBuilder:
    """Get the global pipeline builder instance."""
    global _global_pipeline_builder
    if _global_pipeline_builder is None:
        _global_pipeline_builder = PipelineBuilder()
    return _global_pipeline_builder


def init_pipeline_builder() -> PipelineBuilder:
    """Initialize the global pipeline builder."""
    global _global_pipeline_builder
    _global_pipeline_builder = PipelineBuilder()
    return _global_pipeline_builder


def render_pipeline_builder_ui():
    """
    Render the complete pipeline builder interface.
    
    Provides drag-and-drop canvas, node palette, and execution controls.
    """
    builder = get_pipeline_builder()
    
    # Apply pipeline builder styling
    _apply_pipeline_builder_styling()
    
    # Header with controls
    _render_pipeline_header()
    
    # Main layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        _render_node_palette()
    
    with col2:
        _render_workflow_canvas()
    
    with col3:
        _render_execution_panel()
    
    # Footer with additional controls
    _render_pipeline_footer()


def _render_pipeline_header():
    """Render pipeline builder header with main controls."""
    builder = get_pipeline_builder()
    
    st.markdown("### 🧩 Visual Pipeline Builder")
    
    # Pipeline name and description
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        pipeline_name = st.text_input(
            "Pipeline Name",
            value=builder._current_pipeline.name if builder._current_pipeline else "New Pipeline",
            key="pipeline_name"
        )
    
    with col2:
        pipeline_desc = st.text_input(
            "Description",
            value=builder._current_pipeline.description if builder._current_pipeline else "",
            key="pipeline_description"
        )
    
    with col3:
        if st.button("🆕 New", help="Create new pipeline"):
            builder.create_pipeline(pipeline_name, pipeline_desc)
            st.rerun()
    
    with col4:
        if st.button("💾 Save", help="Save current pipeline"):
            builder.save_pipeline()
            st.success("Pipeline saved!")
    
    # AI Pipeline Generation
    with st.expander("🤖 AI Pipeline Generation", expanded=False):
        description = st.text_area(
            "Describe your pipeline:",
            placeholder="e.g., 'Build a HIRAX cleaning pipeline with FFT analysis'",
            key="ai_pipeline_description"
        )
        
        if st.button("🚀 Generate Pipeline", key="generate_pipeline"):
            if description:
                generated_pipeline = builder.generate_pipeline_from_description(description)
                if generated_pipeline:
                    st.success("Pipeline generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate pipeline")
    
    # Pipeline Templates
    with st.expander("📋 Pipeline Templates", expanded=False):
        template_cols = st.columns(2)
        
        for i, (template_id, template) in enumerate(builder._pipeline_templates.items()):
            with template_cols[i % 2]:
                if st.button(f"📄 {template['name']}", key=f"template_{template_id}"):
                    _load_pipeline_template(template_id, template)
                    st.rerun()


def _render_node_palette():
    """Render the node palette with draggable node types."""
    builder = get_pipeline_builder()
    
    st.markdown("### 🎯 Node Palette")
    
    # Group nodes by category
    categories = defaultdict(list)
    for node_type, template in builder._node_templates.items():
        categories[template["category"]].append((node_type, template))
    
    for category, nodes in categories.items():
        with st.expander(f"📂 {category}", expanded=False):
            for node_type, template in nodes:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"**{template['icon']}**")
                
                with col2:
                    if st.button(template["name"], key=f"add_node_{node_type.value}"):
                        # Add node to canvas
                        if builder._current_pipeline:
                            position = (100.0, 100.0 + len(builder._current_pipeline.nodes) * 50)
                            builder.add_node(node_type, position)
                            st.rerun()


def _render_workflow_canvas():
    """Render the main workflow canvas."""
    builder = get_pipeline_builder()
    
    st.markdown("### 🎨 Workflow Canvas")
    
    if not builder._current_pipeline:
        st.info("Create a new pipeline to start building workflows")
        return
    
    # Canvas controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔍 Zoom In", key="zoom_in"):
            zoom = st.session_state.pipeline_builder["canvas_zoom"]
            st.session_state.pipeline_builder["canvas_zoom"] = min(zoom * 1.2, 3.0)
            st.rerun()
    
    with col2:
        if st.button("🔍 Zoom Out", key="zoom_out"):
            zoom = st.session_state.pipeline_builder["canvas_zoom"]
            st.session_state.pipeline_builder["canvas_zoom"] = max(zoom / 1.2, 0.5)
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset View", key="reset_view"):
            st.session_state.pipeline_builder["canvas_zoom"] = 1.0
            st.session_state.pipeline_builder["canvas_offset"] = {"x": 0, "y": 0}
            st.rerun()
    
    with col4:
        show_grid = st.checkbox("Grid", value=st.session_state.pipeline_builder["show_grid"])
        st.session_state.pipeline_builder["show_grid"] = show_grid
    
    # Render canvas
    _render_canvas_content()


def _render_canvas_content():
    """Render the actual canvas content with nodes and connections."""
    builder = get_pipeline_builder()
    zoom = st.session_state.pipeline_builder["canvas_zoom"]
    offset = st.session_state.pipeline_builder["canvas_offset"]
    show_grid = st.session_state.pipeline_builder["show_grid"]
    
    # Canvas container
    canvas_html = f"""
    <div class="pipeline-canvas" style="
        width: 100%;
        height: 600px;
        border: 2px solid #ddd;
        border-radius: 8px;
        position: relative;
        overflow: hidden;
        background: {'#f8f9fa' if show_grid else 'white'};
        transform: scale({zoom});
        transform-origin: top left;
    ">
    """
    
    # Add grid if enabled
    if show_grid:
        canvas_html += """
        <div class="grid-background" style="
            position: absolute;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            opacity: 0.3;
        "></div>
        """
    
    # Render connections
    for conn in builder._current_pipeline.connections.values():
        source_node = builder._current_pipeline.nodes.get(conn.source_node_id)
        target_node = builder._current_pipeline.nodes.get(conn.target_node_id)
        
        if source_node and target_node:
            # Calculate connection line
            x1 = source_node.position[0] + 100  # Node width
            y1 = source_node.position[1] + 30   # Node height/2
            x2 = target_node.position[0]
            y2 = target_node.position[1] + 30
            
            canvas_html += f"""
            <svg class="connection" style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            ">
                <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
                    stroke="#3b82f6" stroke-width="2" marker-end="url(#arrowhead)"/>
            </svg>
            """
    
    # Render nodes
    for node in builder._current_pipeline.nodes.values():
        node_template = builder._node_templates[node.type]
        
        # Node styling based on status
        status_colors = {
            NodeStatus.PENDING: "#6b7280",
            NodeStatus.RUNNING: "#3b82f6",
            NodeStatus.COMPLETED: "#10b981",
            NodeStatus.FAILED: "#ef4444",
            NodeStatus.SKIPPED: "#f59e0b"
        }
        
        node_color = status_colors.get(node.status, "#6b7280")
        
        canvas_html += f"""
        <div class="pipeline-node" data-node-id="{node.id}" style="
            position: absolute;
            left: {node.position[0]}px;
            top: {node.position[1]}px;
            width: 200px;
            min-height: 60px;
            background: white;
            border: 2px solid {node_color};
            border-radius: 8px;
            padding: 10px;
            cursor: move;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
        ">
            <div class="node-header" style="
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            ">
                <span style="font-size: 16px; margin-right: 8px;">{node_template['icon']}</span>
                <strong style="font-size: 12px;">{node.name}</strong>
            </div>
            <div class="node-description" style="
                font-size: 10px;
                color: #666;
                margin-bottom: 5px;
            ">
                {node.description}
            </div>
            <div class="node-status" style="
                font-size: 10px;
                color: {node_color};
                font-weight: bold;
            ">
                {node.status.value.upper()}
            </div>
        </div>
        """
    
    canvas_html += """
    </div>
    
    <style>
    .pipeline-node:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        z-index: 10;
    }
    
    .connection {
        z-index: 1;
    }
    </style>
    """
    
    st.markdown(canvas_html, unsafe_allow_html=True)
    
    # Node selection and configuration
    if st.session_state.pipeline_builder["selected_node"]:
        _render_node_configuration()


def _render_node_configuration():
    """Render configuration panel for selected node."""
    builder = get_pipeline_builder()
    selected_node_id = st.session_state.pipeline_builder["selected_node"]
    
    if not builder._current_pipeline or selected_node_id not in builder._current_pipeline.nodes:
        return
    
    node = builder._current_pipeline.nodes[selected_node_id]
    node_template = builder._node_templates[node.type]
    
    st.markdown(f"### ⚙️ Configure {node.name}")
    
    # Node configuration based on schema
    if "config_schema" in node_template:
        for param_name, param_config in node_template["config_schema"].items():
            param_type = param_config.get("type", "text")
            param_default = param_config.get("default")
            param_options = param_config.get("options", [])
            
            if param_type == "text":
                value = st.text_input(
                    param_name.replace("_", " ").title(),
                    value=node.config.get(param_name, param_default),
                    key=f"node_config_{node.id}_{param_name}"
                )
            elif param_type == "textarea":
                value = st.text_area(
                    param_name.replace("_", " ").title(),
                    value=node.config.get(param_name, param_default),
                    key=f"node_config_{node.id}_{param_name}"
                )
            elif param_type == "number":
                value = st.number_input(
                    param_name.replace("_", " ").title(),
                    value=node.config.get(param_name, param_default),
                    key=f"node_config_{node.id}_{param_name}"
                )
            elif param_type == "boolean":
                value = st.checkbox(
                    param_name.replace("_", " ").title(),
                    value=node.config.get(param_name, param_default),
                    key=f"node_config_{node.id}_{param_name}"
                )
            elif param_type == "select":
                value = st.selectbox(
                    param_name.replace("_", " ").title(),
                    options=param_options,
                    index=param_options.index(node.config.get(param_name, param_default)) if node.config.get(param_name, param_default) in param_options else 0,
                    key=f"node_config_{node.id}_{param_name}"
                )
            
            # Update node config
            node.config[param_name] = value
    
    # Node actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete Node", key=f"delete_node_{node.id}"):
            builder.remove_node(node.id)
            st.session_state.pipeline_builder["selected_node"] = None
            st.rerun()
    
    with col2:
        if st.button("📋 Duplicate Node", key=f"duplicate_node_{node.id}"):
            new_position = (node.position[0] + 50, node.position[1] + 50)
            new_node = builder.add_node(node.type, new_position, node.config.copy())
            new_node.name = f"{node.name} (Copy)"
            st.rerun()


def _render_execution_panel():
    """Render the execution control panel."""
    builder = get_pipeline_builder()
    
    st.markdown("### ⚡ Execution Control")
    
    if not builder._current_pipeline:
        st.info("Create a pipeline to enable execution")
        return
    
    # Pipeline status
    status = builder._current_pipeline.status
    status_colors = {
        PipelineStatus.IDLE: "#6b7280",
        PipelineStatus.RUNNING: "#3b82f6",
        PipelineStatus.PAUSED: "#f59e0b",
        PipelineStatus.COMPLETED: "#10b981",
        PipelineStatus.FAILED: "#ef4444",
        PipelineStatus.CANCELLED: "#6b7280"
    }
    
    st.markdown(f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        background: {status_colors.get(status, '#6b7280')};
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    ">
        {status.value.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Execution controls
    if status == PipelineStatus.IDLE:
        if st.button("▶️ Execute Pipeline", key="execute_pipeline"):
            if builder.execute_pipeline():
                st.success("Pipeline execution started!")
                st.rerun()
            else:
                st.error("Failed to start execution")
    
    elif status == PipelineStatus.RUNNING:
        if st.button("⏸️ Pause", key="pause_pipeline"):
            # Pause functionality
            st.info("Pause not implemented yet")
        
        if st.button("⏹️ Cancel", key="cancel_pipeline"):
            if builder.cancel_execution():
                st.success("Pipeline cancelled!")
                st.rerun()
    
    elif status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
        if st.button("🔄 Reset", key="reset_pipeline"):
            builder._current_pipeline.status = PipelineStatus.IDLE
            for node in builder._current_pipeline.nodes.values():
                node.status = NodeStatus.PENDING
            st.rerun()
    
    # Execution progress
    if status == PipelineStatus.RUNNING:
        st.markdown("### 📊 Execution Progress")
        
        total_nodes = len(builder._current_pipeline.nodes)
        completed_nodes = len([n for n in builder._current_pipeline.nodes.values() if n.status == NodeStatus.COMPLETED])
        failed_nodes = len([n for n in builder._current_pipeline.nodes.values() if n.status == NodeStatus.FAILED])
        
        progress = completed_nodes / total_nodes if total_nodes > 0 else 0
        
        st.progress(progress)
        st.markdown(f"**Progress:** {completed_nodes}/{total_nodes} nodes completed")
        
        if failed_nodes > 0:
            st.error(f"**Failed:** {failed_nodes} nodes failed")
    
    # Node status overview
    st.markdown("### 📋 Node Status")
    
    node_statuses = defaultdict(int)
    for node in builder._current_pipeline.nodes.values():
        node_statuses[node.status] += 1
    
    for status, count in node_statuses.items():
        st.write(f"**{status.value.title()}:** {count}")


def _render_pipeline_footer():
    """Render pipeline builder footer with additional controls."""
    builder = get_pipeline_builder()
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("📤 Export", key="export_pipeline"):
            exported = builder.export_pipeline()
            if exported:
                st.download_button(
                    label="💾 Download Pipeline",
                    data=exported,
                    file_name="pipeline.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("📥 Import", key="import_pipeline"):
            # Import functionality
            st.info("Import not implemented yet")
    
    with col3:
        if st.button("📊 Statistics", key="pipeline_stats"):
            _show_pipeline_statistics()
    
    with col4:
        if st.button("🔍 Validate", key="validate_pipeline"):
            _validate_pipeline()
    
    with col5:
        if st.button("🗑️ Clear", key="clear_pipeline"):
            if builder._current_pipeline:
                builder._current_pipeline.nodes.clear()
                builder._current_pipeline.connections.clear()
                st.rerun()


def _load_pipeline_template(template_id: str, template: Dict[str, Any]):
    """Load a pipeline template."""
    builder = get_pipeline_builder()
    
    # Create new pipeline from template
    pipeline = builder.create_pipeline(template["name"], template["description"])
    
    # Add nodes
    node_map = {}
    for i, node_info in enumerate(template["nodes"]):
        node_type = NodeType(node_info["type"])
        position = node_info["position"]
        config = node_info.get("config", {})
        
        node = builder.add_node(node_type, position, config)
        node.name = node_info["name"]
        node_map[i] = node.id
    
    # Add connections
    for source_idx, target_idx in template["connections"]:
        if source_idx in node_map and target_idx in node_map:
            builder.add_connection(node_map[source_idx], node_map[target_idx])


def _show_pipeline_statistics():
    """Show pipeline statistics."""
    builder = get_pipeline_builder()
    
    if not builder._current_pipeline:
        st.info("No pipeline to analyze")
        return
    
    pipeline = builder._current_pipeline
    
    stats = {
        "Total Nodes": len(pipeline.nodes),
        "Total Connections": len(pipeline.connections),
        "Node Types": defaultdict(int),
        "Status Distribution": defaultdict(int)
    }
    
    for node in pipeline.nodes.values():
        stats["Node Types"][node.type.value] += 1
        stats["Status Distribution"][node.status.value] += 1
    
    st.markdown("### 📊 Pipeline Statistics")
    
    for key, value in stats.items():
        if isinstance(value, dict):
            st.markdown(f"**{key}:**")
            for sub_key, sub_value in value.items():
                st.write(f"  • {sub_key}: {sub_value}")
        else:
            st.markdown(f"**{key}:** {value}")


def _validate_pipeline():
    """Validate the current pipeline."""
    builder = get_pipeline_builder()
    
    if not builder._current_pipeline:
        st.info("No pipeline to validate")
        return
    
    pipeline = builder._current_pipeline
    issues = []
    
    # Check for orphaned nodes
    connected_nodes = set()
    for conn in pipeline.connections.values():
        connected_nodes.add(conn.source_node_id)
        connected_nodes.add(conn.target_node_id)
    
    orphaned_nodes = set(pipeline.nodes.keys()) - connected_nodes
    if orphaned_nodes:
        issues.append(f"Orphaned nodes: {len(orphaned_nodes)}")
    
    # Check for circular dependencies
    try:
        builder.get_pipeline_execution_order()
    except ValueError as e:
        issues.append(f"Circular dependency: {e}")
    
    # Check for missing inputs
    for node in pipeline.nodes.values():
        for input_name in node.inputs:
            has_input = any(
                conn.target_node_id == node.id and conn.target_input == input_name
                for conn in pipeline.connections.values()
            )
            if not has_input and input_name != "dataset":  # Allow optional dataset input
                issues.append(f"Missing input '{input_name}' for node '{node.name}'")
    
    if issues:
        st.error("Pipeline validation failed:")
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("Pipeline validation passed!")


def _apply_pipeline_builder_styling():
    """Apply custom styling for the pipeline builder."""
    st.markdown("""
    <style>
    .pipeline-canvas {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .pipeline-node {
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
    }
    
    .pipeline-node.selected {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }
    
    .connection-line {
        stroke: #3b82f6;
        stroke-width: 2;
        fill: none;
    }
    
    .connection-line:hover {
        stroke-width: 3;
    }
    
    .node-palette-item {
        padding: 8px;
        margin: 4px 0;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        cursor: grab;
        transition: all 0.2s;
    }
    
    .node-palette-item:hover {
        border-color: #3b82f6;
        transform: translateX(4px);
    }
    
    .execution-status {
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .grid-background {
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)


def integrate_with_context_engine():
    """Integrate pipeline builder with Context Engine."""
    try:
        if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
            context_engine = st.session_state.context_engine
            
            # Update context with pipeline information
            builder = get_pipeline_builder()
            if builder._current_pipeline:
                # Add pipeline to context
                context_data = {
                    "active_pipeline": {
                        "id": builder._current_pipeline.id,
                        "name": builder._current_pipeline.name,
                        "node_count": len(builder._current_pipeline.nodes),
                        "connection_count": len(builder._current_pipeline.connections),
                        "status": builder._current_pipeline.status.value
                    }
                }
                
                # Update workflow stage based on pipeline activity
                if builder._current_pipeline.status == PipelineStatus.RUNNING:
                    context_engine.set_workflow_stage("pipeline_execution")
                elif builder._current_pipeline.nodes:
                    context_engine.set_workflow_stage("pipeline_construction")
                
                # Store pipeline context
                if hasattr(context_engine, 'set_context_data'):
                    context_engine.set_context_data("pipeline_builder", context_data)
                
    except Exception as e:
        st.error(f"Failed to integrate with Context Engine: {e}")


def integrate_with_workflow_copilot():
    """Integrate pipeline builder with Workflow Copilot."""
    try:
        from modules.workflow_copilot import get_workflow_copilot
        
        copilot = get_workflow_copilot()
        builder = get_pipeline_builder()
        
        if builder._current_pipeline:
            # Update copilot with pipeline context
            copilot.update_workflow_history('pipeline_activity', {
                'pipeline_id': builder._current_pipeline.id,
                'pipeline_name': builder._current_pipeline.name,
                'node_count': len(builder._current_pipeline.nodes),
                'status': builder._current_pipeline.status.value
            })
            
            # Generate pipeline-specific recommendations
            if builder._current_pipeline.status == PipelineStatus.IDLE and len(builder._current_pipeline.nodes) > 0:
                # Suggest execution if pipeline is ready
                execution_order = builder.get_pipeline_execution_order()
                if execution_order:
                    copilot._active_recommendations.append(
                        WorkflowRecommendation(
                            title="Execute Pipeline",
                            description=f"Your pipeline '{builder._current_pipeline.name}' is ready to execute",
                            action_type="execute",
                            target_page="🧭 Agent Pipeline",
                            priority=8,
                            confidence=0.9,
                            reasoning="Pipeline construction complete - execution recommended"
                        )
                    )
            
            elif builder._current_pipeline.status == PipelineStatus.COMPLETED:
                # Suggest next steps after completion
                copilot._active_recommendations.append(
                    WorkflowRecommendation(
                        title="Review Pipeline Results",
                        description="Pipeline execution completed - review results and generate report",
                        action_type="navigate",
                        target_page="📖 Report Studio",
                        priority=7,
                        confidence=0.8,
                        reasoning="Pipeline execution complete - results review recommended"
                    )
                )
                
    except Exception as e:
        st.error(f"Failed to integrate with Workflow Copilot: {e}")


def get_context_aware_pipeline_suggestions() -> List[Dict[str, Any]]:
    """
    Get context-aware pipeline suggestions based on current state.
    
    Returns:
        List of pipeline suggestions
    """
    suggestions = []
    
    try:
        # Get current context
        if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
            context_engine = st.session_state.context_engine
            active_dataset = context_engine.get_active_dataset_name()
            workflow_stage = context_engine.get_current_workflow_stage()
            
            if active_dataset:
                # Suggest pipelines for active dataset
                if 'hirax' in active_dataset.lower():
                    suggestions.append({
                        "name": "HIRAX Analysis Pipeline",
                        "description": "Complete HIRAX data analysis with RFI detection and FFT",
                        "template": "hirax_full_analysis",
                        "priority": "high"
                    })
                else:
                    suggestions.append({
                        "name": "Data Cleaning Pipeline",
                        "description": "Standard data cleaning and validation workflow",
                        "template": "data_cleaning",
                        "priority": "medium"
                    })
                    
                    suggestions.append({
                        "name": "ML Training Pipeline",
                        "description": "Machine learning training and evaluation workflow",
                        "template": "ml_training",
                        "priority": "medium"
                    })
        
        # Get workflow stage suggestions
        if workflow_stage:
            stage = workflow_stage.value if workflow_stage else ""
            
            if "data_ingestion" in stage:
                suggestions.append({
                    "name": "Quick Analysis Pipeline",
                    "description": "Rapid data analysis and visualization",
                    "template": "auto_reporting",
                    "priority": "high"
                })
            elif "analysis" in stage:
                suggestions.append({
                    "name": "Advanced Analytics Pipeline",
                    "description": "Comprehensive analytics with AI insights",
                    "template": "auto_reporting",
                    "priority": "high"
                })
            elif "modeling" in stage:
                suggestions.append({
                    "name": "ML Deployment Pipeline",
                    "description": "Model training, validation, and deployment",
                    "template": "ml_training",
                    "priority": "high"
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
        
    except Exception as e:
        st.error(f"Failed to get context-aware suggestions: {e}")
    
    return suggestions


def render_context_aware_suggestions():
    """Render context-aware pipeline suggestions."""
    suggestions = get_context_aware_pipeline_suggestions()
    
    if not suggestions:
        return
    
    st.markdown("### 💡 Context-Aware Suggestions")
    
    for suggestion in suggestions[:3]:  # Show top 3 suggestions
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{suggestion['name']}**")
            st.markdown(f"*{suggestion['description']}*")
        
        with col2:
            if st.button("🚀 Use", key=f"use_suggestion_{suggestion['template']}"):
                # Load the suggested template
                builder = get_pipeline_builder()
                template = builder._pipeline_templates.get(suggestion['template'])
                if template:
                    _load_pipeline_template(suggestion['template'], template)
                    st.success(f"Loaded {suggestion['name']} template!")
                    st.rerun()


def update_context_from_pipeline_execution():
    """Update context based on pipeline execution results."""
    try:
        builder = get_pipeline_builder()
        
        if builder._current_pipeline and builder._current_pipeline.status == PipelineStatus.COMPLETED:
            # Update context with execution results
            if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
                context_engine = st.session_state.context_engine
                
                # Get execution results
                execution_results = []
                for node in builder._current_pipeline.nodes.values():
                    if node.status == NodeStatus.COMPLETED:
                        execution_results.append({
                            "node_name": node.name,
                            "node_type": node.type.value,
                            "execution_time": node.execution_time
                        })
                
                # Update context with pipeline results
                context_data = {
                    "pipeline_execution": {
                        "pipeline_id": builder._current_pipeline.id,
                        "pipeline_name": builder._current_pipeline.name,
                        "execution_results": execution_results,
                        "total_execution_time": sum(r["execution_time"] for r in execution_results),
                        "completed_at": time.time()
                    }
                }
                
                if hasattr(context_engine, 'set_context_data'):
                    context_engine.set_context_data("pipeline_execution", context_data)
                
                # Update workflow stage
                context_engine.set_workflow_stage("analysis_complete")
                
                # Update workflow copilot
                from modules.workflow_copilot import get_workflow_copilot
                copilot = get_workflow_copilot()
                copilot.update_workflow_history('pipeline_execution_completed', {
                    'pipeline_id': builder._current_pipeline.id,
                    'results_count': len(execution_results)
                })
                
    except Exception as e:
        st.error(f"Failed to update context from pipeline execution: {e}")


def auto_generate_pipeline_from_context():
    """
    Auto-generate pipeline based on current context.
    
    Returns:
        True if pipeline was generated
    """
    try:
        # Get current context
        if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
            context_engine = st.session_state.context_engine
            active_dataset = context_engine.get_active_dataset_name()
            workflow_stage = context_engine.get_current_workflow_stage()
            
            if active_dataset and not get_pipeline_builder()._current_pipeline:
                # Auto-generate pipeline based on dataset and stage
                builder = get_pipeline_builder()
                
                if 'hirax' in active_dataset.lower():
                    # Generate HIRAX pipeline
                    pipeline = builder._create_hirax_pipeline()
                    pipeline.name = f"Auto HIRAX Pipeline for {active_dataset}"
                elif workflow_stage and workflow_stage.value == "data_ingestion":
                    # Generate cleaning pipeline
                    pipeline = builder._create_cleaning_pipeline()
                    pipeline.name = f"Auto Cleaning Pipeline for {active_dataset}"
                else:
                    # Generate general analysis pipeline
                    pipeline = builder._create_general_pipeline()
                    pipeline.name = f"Auto Analysis Pipeline for {active_dataset}"
                
                return True
                
    except Exception as e:
        st.error(f"Failed to auto-generate pipeline: {e}")
    
    return False


def render_pipeline_builder_with_context():
    """
    Render pipeline builder with full context integration.
    
    This is the main rendering function that includes all context features.
    """
    # Integrate with context systems
    integrate_with_context_engine()
    integrate_with_workflow_copilot()
    
    # Update context from execution if needed
    update_context_from_pipeline_execution()
    
    # Auto-generate pipeline if appropriate
    if not get_pipeline_builder()._current_pipeline:
        auto_generated = auto_generate_pipeline_from_context()
        if auto_generated:
            st.info("🤖 Auto-generated pipeline based on current context")
    
    # Render context-aware suggestions
    render_context_aware_suggestions()
    
    # Render main pipeline builder UI
    render_pipeline_builder_ui()
