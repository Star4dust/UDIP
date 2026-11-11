"""
UDIP v3 — modules/workflow_copilot.py
AI Workflow Copilot for intelligent analytical assistance.

Transforms UDIP from reactive analytics to proactive intelligent assistance.
Continuously analyzes workflows and provides contextual guidance.
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import re


class DatasetType(Enum):
    """Dataset types for intelligent analysis."""
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"
    SCIENTIFIC = "scientific"
    HIRAX_VISIBILITY = "hirax_visibility"
    HIRAX_SPECTRAL = "hirax_spectral"
    HIRAX_WATERFALL = "hirax_waterfall"
    UNKNOWN = "unknown"


class WorkflowStage(Enum):
    """Workflow stages for contextual recommendations."""
    DATA_INGESTION = "data_ingestion"
    DATA_PREPARATION = "data_preparation"
    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    MODELING = "modeling"
    REPORTING = "reporting"
    DEPLOYMENT = "deployment"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class DatasetInsight:
    """Intelligent dataset analysis results."""
    dataset_name: str
    dataset_type: DatasetType
    shape: Tuple[int, int]
    columns: List[str]
    missing_data_ratio: float
    outlier_ratio: float
    data_quality_score: float
    ml_suitability: float
    feature_engineering_opportunities: List[str]
    hirax_insights: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    analysis_timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowRecommendation:
    """Contextual workflow recommendation."""
    title: str
    description: str
    action_type: str  # "navigate", "analyze", "transform", "visualize"
    target_page: Optional[str]
    priority: int  # 1-10, higher is more important
    confidence: float  # 0.0-1.0
    reasoning: str
    auto_executable: bool = False
    execution_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligentAlert:
    """Intelligent alert for workflow issues."""
    title: str
    message: str
    level: AlertLevel
    category: str
    actionable: bool
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class WorkflowCopilot:
    """
    AI Workflow Copilot for intelligent analytical assistance.
    
    Continuously analyzes workflows and provides proactive guidance.
    """
    
    def __init__(self):
        """Initialize the Workflow Copilot."""
        self._logger = logging.getLogger(__name__)
        self._dataset_insights: Dict[str, DatasetInsight] = {}
        self._workflow_history: List[Dict[str, Any]] = []
        self._active_recommendations: List[WorkflowRecommendation] = []
        self._active_alerts: List[IntelligentAlert] = []
        self._last_analysis_time = 0.0
        self._analysis_interval = 30.0  # Analyze every 30 seconds
        
        # Initialize session state for copilot
        if "workflow_copilot" not in st.session_state:
            st.session_state.workflow_copilot = {
                "enabled": True,
                "auto_analyze": True,
                "show_alerts": True,
                "show_recommendations": True,
                "panel_collapsed": False,
                "last_interaction": time.time()
            }
    
    def analyze_dataset(self, dataset_name: str, df: pd.DataFrame) -> DatasetInsight:
        """
        Intelligently analyze a dataset and infer characteristics.
        
        Args:
            dataset_name: Name of the dataset
            df: DataFrame to analyze
            
        Returns:
            DatasetInsight with comprehensive analysis
        """
        try:
            # Basic dataset info
            shape = df.shape
            columns = list(df.columns)
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            missing_data_ratio = missing_data.sum() / (df.shape[0] * df.shape[1])
            
            # Outlier detection (simplified)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_counts = []
            
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outlier_counts.append(outliers / len(df[col].dropna()))
            
            outlier_ratio = np.mean(outlier_counts) if outlier_counts else 0.0
            
            # Data quality score
            quality_factors = {
                "completeness": 1.0 - missing_data_ratio,
                "consistency": 1.0 - outlier_ratio,
                "structure": min(1.0, len(columns) / 50.0),  # Penalize too many columns
                "size": min(1.0, np.log10(shape[0]) / 6.0)  # Optimal around 1M rows
            }
            data_quality_score = np.mean(list(quality_factors.values()))
            
            # Dataset type detection
            dataset_type = self._detect_dataset_type(df, dataset_name)
            
            # ML suitability
            ml_suitability = self._calculate_ml_suitability(df, dataset_type)
            
            # Feature engineering opportunities
            feature_opportunities = self._identify_feature_opportunities(df, dataset_type)
            
            # HIRAX-specific analysis
            hirax_insights = {}
            if "hirax" in dataset_name.lower() or dataset_type in [DatasetType.HIRAX_VISIBILITY, DatasetType.HIRAX_SPECTRAL, DatasetType.HIRAX_WATERFALL]:
                hirax_insights = self._analyze_hirax_data(df, dataset_type)
            
            # Generate recommendations
            recommendations = self._generate_dataset_recommendations(
                dataset_type, data_quality_score, missing_data_ratio, outlier_ratio, hirax_insights
            )
            
            insight = DatasetInsight(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                shape=shape,
                columns=columns,
                missing_data_ratio=missing_data_ratio,
                outlier_ratio=outlier_ratio,
                data_quality_score=data_quality_score,
                ml_suitability=ml_suitability,
                feature_engineering_opportunities=feature_opportunities,
                hirax_insights=hirax_insights,
                recommendations=recommendations
            )
            
            # Store insight
            self._dataset_insights[dataset_name] = insight
            
            return insight
            
        except Exception as e:
            self._logger.error(f"Failed to analyze dataset {dataset_name}: {e}")
            return DatasetInsight(
                dataset_name=dataset_name,
                dataset_type=DatasetType.UNKNOWN,
                shape=df.shape,
                columns=list(df.columns),
                missing_data_ratio=0.0,
                outlier_ratio=0.0,
                data_quality_score=0.5,
                ml_suitability=0.5,
                feature_engineering_opportunities=[],
                recommendations=["Analysis failed - please check data format"]
            )
    
    def _detect_dataset_type(self, df: pd.DataFrame, dataset_name: str) -> DatasetType:
        """Detect dataset type based on characteristics."""
        # Check for HIRAX-specific patterns
        if "hirax" in dataset_name.lower():
            if any(col in df.columns for col in ['frequency', 'freq', 'spectral', 'power']):
                return DatasetType.HIRAX_SPECTRAL
            elif any(col in df.columns for col in ['time', 'visibility', 'baseline']):
                return DatasetType.HIRAX_VISIBILITY
            elif df.shape[1] > 100:  # Many frequency channels
                return DatasetType.HIRAX_WATERFALL
        
        # Check for time series
        time_cols = ['time', 'date', 'timestamp', 'datetime']
        if any(col.lower() in time_cols for col in df.columns):
            return DatasetType.TIME_SERIES
        
        # Check for geospatial
        geo_cols = ['lat', 'latitude', 'lon', 'longitude', 'x', 'y', 'geometry']
        if any(col.lower() in geo_cols for col in df.columns):
            return DatasetType.GEOSPATIAL
        
        # Check for scientific data
        sci_patterns = ['temperature', 'pressure', 'wavelength', 'frequency', 'magnitude']
        if any(pattern in ' '.join(df.columns).lower() for pattern in sci_patterns):
            return DatasetType.SCIENTIFIC
        
        # Default to tabular
        return DatasetType.TABULAR
    
    def _calculate_ml_suitability(self, df: pd.DataFrame, dataset_type: DatasetType) -> float:
        """Calculate ML suitability score."""
        factors = []
        
        # Size factor
        size_factor = min(1.0, df.shape[0] / 1000.0)  # Prefer at least 1000 rows
        factors.append(size_factor)
        
        # Feature factor
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / len(df.columns)
        factors.append(numeric_ratio)
        
        # Quality factor
        missing_factor = 1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
        factors.append(missing_factor)
        
        # Type-specific factors
        if dataset_type == DatasetType.TIME_SERIES:
            factors.append(0.8)  # Good for time series ML
        elif dataset_type == DatasetType.HIRAX_SPECTRAL:
            factors.append(0.9)  # Excellent for scientific ML
        elif dataset_type == DatasetType.TABULAR:
            factors.append(0.7)  # Standard tabular ML
        
        return np.mean(factors)
    
    def _identify_feature_opportunities(self, df: pd.DataFrame, dataset_type: DatasetType) -> List[str]:
        """Identify feature engineering opportunities."""
        opportunities = []
        
        # Time series features
        if dataset_type == DatasetType.TIME_SERIES:
            opportunities.extend([
                "Extract temporal features (hour, day, month)",
                "Create lag features",
                "Calculate rolling statistics",
                "Add seasonal decomposition"
            ])
        
        # Numeric transformations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            opportunities.extend([
                "Standardize numeric features",
                "Create polynomial features",
                "Apply log transformations",
                "Generate interaction terms"
            ])
        
        # Categorical features
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            opportunities.extend([
                "One-hot encode categorical variables",
                "Create target encodings",
                "Generate frequency encodings"
            ])
        
        # HIRAX-specific
        if dataset_type in [DatasetType.HIRAX_SPECTRAL, DatasetType.HIRAX_WATERFALL]:
            opportunities.extend([
                "FFT analysis for frequency domain features",
                "Spectral band extraction",
                "RFI mask features",
                "Statistical moment features"
            ])
        
        return opportunities
    
    def _analyze_hirax_data(self, df: pd.DataFrame, dataset_type: DatasetType) -> Dict[str, Any]:
        """Analyze HIRAX-specific data characteristics."""
        insights = {}
        
        if dataset_type == DatasetType.HIRAX_SPECTRAL:
            # Look for frequency structure
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Simulate spectral analysis
                insights['frequency_channels'] = len(numeric_cols)
                insights['spectral_range'] = [df[col].min() for col in numeric_cols[:2]] if len(numeric_cols) >= 2 else []
                insights['power_distribution'] = 'normal' if df[numeric_cols[0]].skew() < 1 else 'skewed'
        
        elif dataset_type == DatasetType.HIRAX_WATERFALL:
            # Waterfall data analysis
            insights['time_resolution'] = df.shape[0]
            insights['frequency_channels'] = df.shape[1]
            insights['data_density'] = df.notna().sum().sum() / (df.shape[0] * df.shape[1])
        
        elif dataset_type == DatasetType.HIRAX_VISIBILITY:
            # Visibility data analysis
            insights['visibility_points'] = df.shape[0]
            insights['baseline_count'] = len([col for col in df.columns if 'baseline' in col.lower()])
        
        return insights
    
    def _generate_dataset_recommendations(self, dataset_type: DatasetType, quality_score: float, 
                                        missing_ratio: float, outlier_ratio: float, 
                                        hirax_insights: Dict[str, Any]) -> List[str]:
        """Generate dataset-specific recommendations."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.7:
            recommendations.append("Data quality issues detected → Open Data Forge for cleaning")
        
        if missing_ratio > 0.1:
            recommendations.append("Missing values detected → Handle missing data in Data Forge")
        
        if outlier_ratio > 0.05:
            recommendations.append("Outliers detected → Investigate in Data Validation")
        
        # Type-specific recommendations
        if dataset_type == DatasetType.TIME_SERIES:
            recommendations.append("Time series detected → Create temporal visualizations")
            recommendations.append("Consider time series modeling in ML module")
        
        elif dataset_type == DatasetType.HIRAX_SPECTRAL:
            recommendations.append("HIRAX spectral data → Run FFT analysis")
            recommendations.append("Check for RFI contamination")
            recommendations.append("Generate power spectrum plots")
        
        elif dataset_type == DatasetType.HIRAX_WATERFALL:
            recommendations.append("HIRAX waterfall data → Process with HIRAX Inspector")
            recommendations.append("Apply RFI detection algorithms")
            recommendations.append("Create waterfall visualizations")
        
        # General recommendations
        if quality_score > 0.8:
            recommendations.append("High quality data → Ready for ML modeling")
        
        return recommendations
    
    def get_workflow_stage(self) -> WorkflowStage:
        """Determine current workflow stage from context."""
        try:
            if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
                stage = st.session_state.context_engine.get_current_workflow_stage()
                if stage:
                    return WorkflowStage(stage.value)
            
            # Fallback: infer from active page
            active_page = st.session_state.get('active_page', '')
            
            if any(page in active_page for page in ['Upload', 'Cloud', 'Connectors']):
                return WorkflowStage.DATA_INGESTION
            elif any(page in active_page for page in ['Data Forge', 'Merge', 'Validation']):
                return WorkflowStage.DATA_PREPARATION
            elif any(page in active_page for page in ['Visualize', 'Auto Analysis', 'Data Pulse']):
                return WorkflowStage.EXPLORATION
            elif any(page in active_page for page in ['Machine Learning', 'Agent']):
                return WorkflowStage.ANALYSIS
            elif any(page in active_page for page in ['Report', 'Dashboard']):
                return WorkflowStage.REPORTING
            else:
                return WorkflowStage.EXPLORATION  # Default
                
        except Exception:
            return WorkflowStage.EXPLORATION
    
    def generate_workflow_recommendations(self) -> List[WorkflowRecommendation]:
        """
        Generate contextual workflow recommendations.
        
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        current_stage = self.get_workflow_stage()
        active_dataset = self._get_active_dataset_info()
        
        # Stage-specific recommendations
        if current_stage == WorkflowStage.DATA_INGESTION:
            if active_dataset and active_dataset.get('name'):
                recommendations.append(WorkflowRecommendation(
                    title="Analyze New Dataset",
                    description=f"Inspect the uploaded dataset '{active_dataset['name']}' for quality and structure",
                    action_type="analyze",
                    target_page="📊 Visualize",
                    priority=8,
                    confidence=0.9,
                    reasoning="New dataset detected - analysis recommended before proceeding"
                ))
        
        elif current_stage == WorkflowStage.DATA_PREPARATION:
            recommendations.append(WorkflowRecommendation(
                title="Validate Data Quality",
                description="Run comprehensive validation checks on prepared data",
                action_type="navigate",
                target_page="✅ Data Validation",
                priority=7,
                confidence=0.8,
                reasoning="Data preparation complete - validation recommended"
            ))
        
        elif current_stage == WorkflowStage.EXPLORATION:
            if active_dataset:
                dataset_name = active_dataset.get('name')
                if dataset_name in self._dataset_insights:
                    insight = self._dataset_insights[dataset_name]
                    
                    if insight.ml_suitability > 0.7:
                        recommendations.append(WorkflowRecommendation(
                            title="Ready for Machine Learning",
                            description="Dataset shows high ML suitability - consider training models",
                            action_type="navigate",
                            target_page="🧠 Machine Learning",
                            priority=8,
                            confidence=insight.ml_suitability,
                            reasoning=f"ML suitability score: {insight.ml_suitability:.2f}"
                        ))
                    
                    if insight.dataset_type in [DatasetType.HIRAX_SPECTRAL, DatasetType.HIRAX_WATERFALL]:
                        recommendations.append(WorkflowRecommendation(
                            title="HIRAX Analysis Recommended",
                            description="Detected HIRAX data structure - run specialized analysis",
                            action_type="navigate",
                            target_page="🔭 HIRAX Inspector",
                            priority=9,
                            confidence=0.95,
                            reasoning="HIRAX scientific data detected"
                        ))
        
        elif current_stage == WorkflowStage.ANALYSIS:
            recommendations.append(WorkflowRecommendation(
                title="Document Findings",
                description="Create comprehensive report of analysis results",
                action_type="navigate",
                target_page="📖 Report Studio",
                priority=6,
                confidence=0.7,
                reasoning="Analysis complete - documentation recommended"
            ))
        
        # Context-aware recommendations
        if active_dataset:
            dataset_name = active_dataset.get('name')
            if dataset_name in self._dataset_insights:
                insight = self._dataset_insights[dataset_name]
                
                # Quality-based recommendations
                if insight.data_quality_score < 0.6:
                    recommendations.append(WorkflowRecommendation(
                        title="Improve Data Quality",
                        description=f"Data quality score is {insight.data_quality_score:.2f} - cleaning recommended",
                        action_type="navigate",
                        target_page="⚒️ Data Forge",
                        priority=9,
                        confidence=0.9,
                        reasoning="Low data quality detected"
                    ))
                
                # Missing data recommendations
                if insight.missing_data_ratio > 0.1:
                    recommendations.append(WorkflowRecommendation(
                        title="Handle Missing Data",
                        description=f"{insight.missing_data_ratio:.1%} missing values detected",
                        action_type="navigate",
                        target_page="⚒️ Data Forge",
                        priority=7,
                        confidence=0.8,
                        reasoning="Significant missing data detected"
                    ))
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x.priority * x.confidence), reverse=True)
        
        # Store recommendations
        self._active_recommendations = recommendations
        
        return recommendations
    
    def get_recommended_next_steps(self) -> List[WorkflowRecommendation]:
        """
        Get smart next-step suggestions based on current context.
        
        Enhanced with sophisticated logic, user behavior learning, and multi-step planning.
        
        Returns:
            List of contextual next steps
        """
        # Generate fresh recommendations
        self.generate_workflow_recommendations()
        
        # Get context-aware recommendations
        contextual_recs = self._get_contextual_next_steps()
        
        # Get behavior-based recommendations
        behavior_recs = self._get_behavior_based_next_steps()
        
        # Get multi-step workflow suggestions
        workflow_recs = self._get_workflow_plan_next_steps()
        
        # Combine and rank recommendations
        all_recs = contextual_recs + behavior_recs + workflow_recs
        
        # Apply dynamic weighting based on user patterns
        weighted_recs = self._apply_recommendation_weighting(all_recs)
        
        # Filter for immediate actions and return top recommendations
        next_steps = [
            rec for rec in weighted_recs 
            if rec.priority >= 6 and rec.confidence >= 0.6
        ]
        
        return next_steps[:5]  # Return top 5 recommendations
    
    def _get_contextual_next_steps(self) -> List[WorkflowRecommendation]:
        """Generate context-aware next steps based on current state."""
        recommendations = []
        current_stage = self.get_workflow_stage()
        active_dataset = self._get_active_dataset_info()
        
        # Stage-specific context recommendations
        if current_stage == WorkflowStage.DATA_INGESTION:
            if active_dataset:
                dataset_name = active_dataset.get('name')
                if dataset_name and dataset_name in self._dataset_insights:
                    insight = self._dataset_insights[dataset_name]
                    
                    # Quality-based recommendations
                    if insight.data_quality_score < 0.7:
                        recommendations.append(WorkflowRecommendation(
                            title="Improve Data Quality First",
                            description=f"Dataset quality is {insight.data_quality_score:.1%} - cleaning recommended before analysis",
                            action_type="navigate",
                            target_page="⚒️ Data Forge",
                            priority=9,
                            confidence=0.9,
                            reasoning="Low data quality will affect all downstream analysis"
                        ))
                    
                    # Dataset type-specific recommendations
                    if insight.dataset_type == DatasetType.TIME_SERIES:
                        recommendations.append(WorkflowRecommendation(
                            title="Prepare Time Series Analysis",
                            description="Time series detected - set up temporal analysis framework",
                            action_type="navigate",
                            target_page="📊 Visualize",
                            priority=8,
                            confidence=0.85,
                            reasoning="Time series requires specialized visualization and analysis"
                        ))
                    
                    elif insight.dataset_type == DatasetType.HIRAX_SPECTRAL:
                        recommendations.append(WorkflowRecommendation(
                            title="Start HIRAX Spectral Analysis",
                            description="HIRAX spectral data detected - begin scientific analysis pipeline",
                            action_type="navigate",
                            target_page="🛸 Signal Intelligence",
                            priority=10,
                            confidence=0.95,
                            reasoning="HIRAX data requires specialized scientific processing"
                        ))
                    
                    elif insight.dataset_type == DatasetType.HIRAX_WATERFALL:
                        recommendations.append(WorkflowRecommendation(
                            title="Process HIRAX Waterfall",
                            description="Waterfall data detected - run RFI detection and calibration",
                            action_type="navigate",
                            target_page="🔭 HIRAX Inspector",
                            priority=10,
                            confidence=0.95,
                            reasoning="Waterfall data needs preprocessing before analysis"
                        ))
        
        elif current_stage == WorkflowStage.DATA_PREPARATION:
            # Check if data is ready for next stage
            if active_dataset:
                dataset_name = active_dataset.get('name')
                if dataset_name in self._dataset_insights:
                    insight = self._dataset_insights[dataset_name]
                    
                    if insight.data_quality_score > 0.8:
                        recommendations.append(WorkflowRecommendation(
                            title="Data Ready for Exploration",
                            description="High quality data - begin exploratory analysis",
                            action_type="navigate",
                            target_page="📊 Visualize",
                            priority=8,
                            confidence=0.8,
                            reasoning="Data quality meets exploration requirements"
                        ))
                    
                    if insight.ml_suitability > 0.7:
                        recommendations.append(WorkflowRecommendation(
                            title="Prepare for Machine Learning",
                            description="Dataset shows high ML suitability - consider feature engineering",
                            action_type="navigate",
                            target_page="🧪 Feature Alchemy",
                            priority=7,
                            confidence=0.75,
                            reasoning=f"ML suitability score: {insight.ml_suitability:.2f}"
                        ))
        
        elif current_stage == WorkflowStage.EXPLORATION:
            # Check for completed visualizations
            if self._has_active_visualizations():
                recommendations.append(WorkflowRecommendation(
                    title="Generate Analysis Report",
                    description="Visualizations complete - document findings",
                    action_type="navigate",
                    target_page="📖 Report Studio",
                    priority=7,
                    confidence=0.7,
                    reasoning="Exploration phase complete - documentation recommended"
                ))
            
            # Check for ML readiness
            if active_dataset:
                dataset_name = active_dataset.get('name')
                if dataset_name in self._dataset_insights:
                    insight = self._dataset_insights[dataset_name]
                    
                    if insight.ml_suitability > 0.8:
                        recommendations.append(WorkflowRecommendation(
                            title="Start Machine Learning",
                            description="Dataset highly suitable for ML models",
                            action_type="navigate",
                            target_page="🧠 Machine Learning",
                            priority=8,
                            confidence=0.85,
                            reasoning="High ML suitability detected"
                        ))
        
        elif current_stage == WorkflowStage.ANALYSIS:
            recommendations.append(WorkflowRecommendation(
                title="Document Analysis Results",
                description="Analysis complete - create comprehensive report",
                action_type="navigate",
                target_page="📖 Report Studio",
                priority=8,
                confidence=0.8,
                reasoning="Analysis results should be documented"
            ))
        
        return recommendations
    
    def _get_behavior_based_next_steps(self) -> List[WorkflowRecommendation]:
        """Generate recommendations based on user behavior patterns."""
        recommendations = []
        
        if len(self._workflow_history) < 3:
            return recommendations  # Not enough history
        
        # Analyze recent behavior patterns
        recent_actions = self._workflow_history[-10:]
        
        # Check for repeated patterns
        action_counts = {}
        for action in recent_actions:
            action_type = action.get('action', '')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Detect user preferences
        if action_counts.get('page_navigation', 0) > 5:
            # User frequently navigates - suggest efficient workflows
            most_visited = self._get_most_visited_pages()
            if most_visited:
                recommendations.append(WorkflowRecommendation(
                    title="Optimize Your Workflow",
                    description=f"You frequently visit {most_visited[0]} - consider workflow automation",
                    action_type="navigate",
                    target_page="🧭 Agent Pipeline",
                    priority=6,
                    confidence=0.6,
                    reasoning="High navigation activity detected"
                ))
        
        # Check for analysis patterns
        analysis_actions = [a for a in recent_actions if 'analysis' in a.get('action', '').lower()]
        if len(analysis_actions) > 3:
            recommendations.append(WorkflowRecommendation(
                title="Create Analysis Pipeline",
                description="You perform frequent analysis - automate your workflow",
                action_type="navigate",
                target_page="🧭 Agent Pipeline",
                priority=7,
                confidence=0.7,
                reasoning="Frequent analysis patterns detected"
            ))
        
        # Check for dataset switching patterns
        dataset_switches = 0
        last_dataset = None
        for action in recent_actions:
            dataset_info = action.get('dataset')
            if dataset_info and dataset_info.get('name') != last_dataset:
                dataset_switches += 1
                last_dataset = dataset_info.get('name')
        
        if dataset_switches > 3:
            recommendations.append(WorkflowRecommendation(
                title="Consolidate Analysis",
                description="You're switching between datasets - consider unified analysis",
                action_type="navigate",
                target_page="🔮 Data Fusion",
                priority=6,
                confidence=0.6,
                reasoning="Multiple dataset analysis detected"
            ))
        
        return recommendations
    
    def _get_workflow_plan_next_steps(self) -> List[WorkflowRecommendation]:
        """Generate multi-step workflow planning recommendations."""
        recommendations = []
        
        # Analyze current workflow progress
        current_stage = self.get_workflow_stage()
        active_dataset = self._get_active_dataset_info()
        
        if not active_dataset:
            return recommendations
        
        dataset_name = active_dataset.get('name')
        if dataset_name not in self._dataset_insights:
            return recommendations
        
        insight = self._dataset_insights[dataset_name]
        
        # Plan complete workflows based on dataset characteristics
        if insight.dataset_type == DatasetType.HIRAX_SPECTRAL:
            # Enhanced HIRAX spectral workflow plan
            workflow_steps = [
                ("HIRAX Spectral Calibration", "🛸 Signal Intelligence", 10, 0.95),
                ("RFI Detection & Mitigation", "📡 RFI Detection", 9, 0.9),
                ("Power Spectrum Analysis", "🌈 Spectral Analysis", 8, 0.85),
                ("Signal Classification", "🛸 Signal Intelligence", 7, 0.8),
                ("Statistical Validation", "🌈 Spectral Analysis", 7, 0.75),
                ("Scientific Report Generation", "📖 Report Studio", 6, 0.7)
            ]
            
            # Add HIRAX-specific contextual recommendations
            hirax_insights = insight.hirax_insights
            
            # Check for RFI indicators
            if hirax_insights.get('power_distribution') == 'skewed':
                recommendations.append(WorkflowRecommendation(
                    title="Priority: RFI Contamination Detected",
                    description="Skewed power distribution suggests RFI contamination - immediate cleaning required",
                    action_type="navigate",
                    target_page="📡 RFI Detection",
                    priority=10,
                    confidence=0.9,
                    reasoning="RFI will compromise all scientific analysis"
                ))
            
            # Check frequency channel coverage
            freq_channels = hirax_insights.get('frequency_channels', 0)
            if freq_channels < 500:
                recommendations.append(WorkflowRecommendation(
                    title="Expand Frequency Coverage",
                    description=f"Only {freq_channels} channels detected - consider full spectrum analysis",
                    action_type="navigate",
                    target_page="🛸 Signal Intelligence",
                    priority=7,
                    confidence=0.7,
                    reasoning="Limited frequency coverage may miss important signals"
                ))
            
            for i, (title, target, priority, confidence) in enumerate(workflow_steps):
                if i == 0 or current_stage != WorkflowStage.ANALYSIS:
                    recommendations.append(WorkflowRecommendation(
                        title=title,
                        description=f"Step {i+1} of HIRAX spectral analysis workflow",
                        action_type="navigate",
                        target_page=target,
                        priority=priority,
                        confidence=confidence,
                        reasoning="Complete HIRAX scientific workflow",
                        auto_executable=i == 0
                    ))
        
        elif insight.dataset_type == DatasetType.HIRAX_WATERFALL:
            # Enhanced HIRAX waterfall workflow plan
            workflow_steps = [
                ("Waterfall Data Calibration", "🔭 HIRAX Inspector", 10, 0.95),
                ("RFI Flagging & Excision", "📡 RFI Detection", 9, 0.9),
                ("Time-Frequency Analysis", "🔭 HIRAX Inspector", 8, 0.85),
                ("Spectral Imaging", "🛸 Signal Intelligence", 7, 0.8),
                ("Waterfall Visualization", "🔭 HIRAX Inspector", 7, 0.75),
                ("Observation Log Analysis", "📖 Report Studio", 6, 0.7)
            ]
            
            # Add HIRAX waterfall-specific recommendations
            hirax_insights = insight.hirax_insights
            time_resolution = hirax_insights.get('time_resolution', 0)
            frequency_channels = hirax_insights.get('frequency_channels', 0)
            data_density = hirax_insights.get('data_density', 0)
            
            # Check data density
            if data_density < 0.8:
                recommendations.append(WorkflowRecommendation(
                    title="Low Data Density Detected",
                    description=f"Only {data_density:.1%} data density - check observation parameters",
                    action_type="navigate",
                    target_page="🔭 HIRAX Inspector",
                    priority=8,
                    confidence=0.8,
                    reasoning="Low density may indicate observation issues"
                ))
            
            # Check time resolution
            if time_resolution < 100:
                recommendations.append(WorkflowRecommendation(
                    title="Improve Time Resolution",
                    description=f"Low time resolution ({time_resolution} points) - consider longer observations",
                    action_type="navigate",
                    target_page="🔭 HIRAX Inspector",
                    priority=7,
                    confidence=0.7,
                    reasoning="Better time resolution improves analysis quality"
                ))
        
        elif insight.dataset_type == DatasetType.HIRAX_VISIBILITY:
            # HIRAX visibility workflow plan
            workflow_steps = [
                ("Visibility Data Calibration", "🔭 HIRAX Inspector", 10, 0.95),
                ("Baseline Analysis", "🔭 HIRAX Inspector", 9, 0.9),
                ("UV Coverage Assessment", "🛸 Signal Intelligence", 8, 0.85),
                ("Imaging Pipeline", "🛸 Signal Intelligence", 7, 0.8),
                ("Source Detection", "🛸 Signal Intelligence", 7, 0.75),
                ("Observation Report", "📖 Report Studio", 6, 0.7)
            ]
            
            # Add HIRAX visibility-specific recommendations
            hirax_insights = insight.hirax_insights
            baseline_count = hirax_insights.get('baseline_count', 0)
            visibility_points = hirax_insights.get('visibility_points', 0)
            
            # Check baseline coverage
            if baseline_count < 100:
                recommendations.append(WorkflowRecommendation(
                    title="Limited Baseline Coverage",
                    description=f"Only {baseline_count} baselines - consider longer observation time",
                    action_type="navigate",
                    target_page="🔭 HIRAX Inspector",
                    priority=7,
                    confidence=0.7,
                    reasoning="More baselines improve UV coverage and imaging"
                ))
            
            # Check visibility data quality
            if visibility_points < 1000:
                recommendations.append(WorkflowRecommendation(
                    title="Increase Visibility Sampling",
                    description=f"Only {visibility_points} visibility points - improve temporal sampling",
                    action_type="navigate",
                    target_page="🔭 HIRAX Inspector",
                    priority=6,
                    confidence=0.6,
                    reasoning="Better sampling improves dynamic range"
                ))
        
        elif insight.dataset_type == DatasetType.TIME_SERIES:
            # Time series workflow plan
            workflow_steps = [
                ("Temporal Visualization", "📊 Visualize", 9, 0.85),
                ("Seasonal Decomposition", "📊 Visualize", 8, 0.8),
                ("Forecasting Models", "🧠 Machine Learning", 8, 0.8),
                ("Anomaly Detection", "🧠 Machine Learning", 7, 0.75),
                ("Time Series Report", "📖 Report Studio", 6, 0.7)
            ]
            
            for i, (title, target, priority, confidence) in enumerate(workflow_steps):
                if i == 0 or current_stage != WorkflowStage.MODELING:
                    recommendations.append(WorkflowRecommendation(
                        title=title,
                        description=f"Step {i+1} of time series analysis workflow",
                        action_type="navigate",
                        target_page=target,
                        priority=priority,
                        confidence=confidence,
                        reasoning="Complete time series analysis workflow"
                    ))
        
        elif insight.ml_suitability > 0.8:
            # ML workflow plan
            workflow_steps = [
                ("Feature Engineering", "🧪 Feature Alchemy", 9, 0.85),
                ("Model Selection", "🧠 Machine Learning", 8, 0.8),
                ("Model Training", "🧠 Machine Learning", 8, 0.8),
                ("Model Evaluation", "🧠 Machine Learning", 7, 0.75),
                ("Model Deployment", "🧠 Machine Learning", 6, 0.7)
            ]
            
            for i, (title, target, priority, confidence) in enumerate(workflow_steps):
                if i == 0 or current_stage != WorkflowStage.MODELING:
                    recommendations.append(WorkflowRecommendation(
                        title=title,
                        description=f"Step {i+1} of machine learning workflow",
                        action_type="navigate",
                        target_page=target,
                        priority=priority,
                        confidence=confidence,
                        reasoning="Complete ML workflow"
                    ))
        
        return recommendations
    
    def _apply_recommendation_weighting(self, recommendations: List[WorkflowRecommendation]) -> List[WorkflowRecommendation]:
        """Apply dynamic weighting based on user behavior and context."""
        weighted_recs = []
        
        for rec in recommendations:
            # Base weight from priority and confidence
            base_weight = rec.priority * rec.confidence
            
            # Context weighting
            current_stage = self.get_workflow_stage()
            
            # Boost recommendations that match current stage
            stage_boost = 0
            if current_stage == WorkflowStage.DATA_INGESTION and "quality" in rec.title.lower():
                stage_boost = 0.2
            elif current_stage == WorkflowStage.DATA_PREPARATION and "explore" in rec.title.lower():
                stage_boost = 0.15
            elif current_stage == WorkflowStage.EXPLORATION and "model" in rec.title.lower():
                stage_boost = 0.1
            
            # User behavior weighting
            behavior_boost = 0
            if len(self._workflow_history) > 5:
                recent_actions = self._workflow_history[-5:]
                
                # Boost recommendations that align with recent behavior
                if any("analysis" in action.get('action', '') for action in recent_actions):
                    if "analysis" in rec.title.lower():
                        behavior_boost = 0.1
                
                if any("visualize" in action.get('action', '') for action in recent_actions):
                    if "visualiz" in rec.title.lower():
                        behavior_boost = 0.1
            
            # Dataset-specific weighting
            dataset_boost = 0
            active_dataset = self._get_active_dataset_info()
            if active_dataset:
                dataset_name = active_dataset.get('name')
                if dataset_name in self._dataset_insights:
                    insight = self._dataset_insights[dataset_name]
                    
                    # Boost HIRAX-specific recommendations for HIRAX data
                    if insight.dataset_type in [DatasetType.HIRAX_SPECTRAL, DatasetType.HIRAX_WATERFALL]:
                        if "hirax" in rec.title.lower() or "spectral" in rec.title.lower():
                            dataset_boost = 0.15
                    
                    # Boost ML recommendations for high ML suitability
                    if insight.ml_suitability > 0.8 and "model" in rec.title.lower():
                        dataset_boost = 0.1
            
            # Apply weights
            final_weight = base_weight + stage_boost + behavior_boost + dataset_boost
            
            # Create weighted recommendation
            weighted_rec = WorkflowRecommendation(
                title=rec.title,
                description=rec.description,
                action_type=rec.action_type,
                target_page=rec.target_page,
                priority=int(rec.priority + (stage_boost + behavior_boost + dataset_boost) * 10),
                confidence=min(1.0, rec.confidence + (behavior_boost + dataset_boost) * 0.1),
                reasoning=rec.reasoning,
                auto_executable=rec.auto_executable,
                execution_params=rec.execution_params
            )
            
            weighted_recs.append(weighted_rec)
        
        # Sort by weighted priority and confidence
        weighted_recs.sort(key=lambda x: (x.priority * x.confidence), reverse=True)
        
        return weighted_recs
    
    def _has_active_visualizations(self) -> bool:
        """Check if there are active visualizations."""
        try:
            # Check for visualization artifacts in session state
            viz_keys = ['charts', 'plots', 'visualizations', 'figures']
            
            for key in viz_keys:
                if key in st.session_state and st.session_state[key]:
                    return True
            
            # Check workflow handoffs for visualization actions
            if '_handoff_data' in st.session_state:
                handoff_data = st.session_state._handoff_data
                for handoff in handoff_data.values():
                    if handoff.get('payload_type') == 'CHART':
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _get_most_visited_pages(self) -> List[str]:
        """Get most frequently visited pages from workflow history."""
        page_counts = {}
        
        for action in self._workflow_history:
            page = action.get('page', '')
            if page:
                page_counts[page] = page_counts.get(page, 0) + 1
        
        # Sort by visit count and return top pages
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)
        return [page for page, count in sorted_pages]
    
    def generate_intelligent_alerts(self) -> List[IntelligentAlert]:
        """
        Generate intelligent alerts about workflow issues.
        
        Returns:
            List of contextual alerts
        """
        alerts = []
        current_time = time.time()
        
        # Memory usage alerts
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > 1000:  # 1GB threshold
                alerts.append(IntelligentAlert(
                    title="High Memory Usage",
                    message=f"Memory usage is {memory_mb:.1f} MB - consider optimization",
                    level=AlertLevel.WARNING,
                    category="performance",
                    actionable=True,
                    suggested_actions=[
                        "Clear unused datasets",
                        "Run memory optimization",
                        "Restart session if needed"
                    ]
                ))
        except ImportError:
            pass
        
        # Dataset quality alerts
        for dataset_name, insight in self._dataset_insights.items():
            if insight.data_quality_score < 0.5:
                alerts.append(IntelligentAlert(
                    title=f"Low Data Quality: {dataset_name}",
                    message=f"Dataset quality score is {insight.data_quality_score:.2f}",
                    level=AlertLevel.WARNING,
                    category="data_quality",
                    actionable=True,
                    suggested_actions=insight.recommendations[:3]
                ))
        
        # Workflow stagnation alerts
        if self._workflow_history:
            last_action = self._workflow_history[-1]
            time_since_last = current_time - last_action.get('timestamp', current_time)
            
            if time_since_last > 300:  # 5 minutes
                alerts.append(IntelligentAlert(
                    title="Workflow Inactivity",
                    message="No recent workflow activity detected",
                    level=AlertLevel.INFO,
                    category="workflow",
                    actionable=True,
                    suggested_actions=[
                        "Review available recommendations",
                        "Explore dataset features",
                        "Try automated workflow"
                    ]
                ))
        
        # HIRAX-specific alerts
        hirax_datasets = [
            (name, insight) for name, insight in self._dataset_insights.items()
            if insight.dataset_type in [DatasetType.HIRAX_SPECTRAL, DatasetType.HIRAX_WATERFALL, DatasetType.HIRAX_VISIBILITY]
        ]
        
        for dataset_name, insight in hirax_datasets:
            if insight.hirax_insights.get('power_distribution') == 'skewed':
                alerts.append(IntelligentAlert(
                    title=f"HIRAX Data Issue: {dataset_name}",
                    message="Skewed power distribution detected - may indicate RFI contamination",
                    level=AlertLevel.WARNING,
                    category="hirax",
                    actionable=True,
                    suggested_actions=[
                        "Run RFI detection",
                        "Apply spectral cleaning",
                        "Check observation parameters"
                    ]
                ))
        
        # Store alerts
        self._active_alerts = alerts
        
        return alerts
    
    def execute_workflow_automation(self, recommendation: WorkflowRecommendation) -> bool:
        """
        Execute automated workflow based on recommendation.
        
        Args:
            recommendation: Recommendation to execute
            
        Returns:
            True if successful
        """
        try:
            if recommendation.action_type == "navigate" and recommendation.target_page:
                # Navigate to recommended page
                st.session_state.active_page = recommendation.target_page
                st.rerun()
                return True
            
            elif recommendation.action_type == "analyze":
                # Trigger analysis
                active_dataset = self._get_active_dataset_info()
                if active_dataset and active_dataset.get('name'):
                    dataset_name = active_dataset['name']
                    if dataset_name in st.session_state.datasets:
                        df = st.session_state.datasets[dataset_name]['df']
                        self.analyze_dataset(dataset_name, df)
                        return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to execute workflow automation: {e}")
            return False
    
    def _get_active_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the active dataset."""
        try:
            if hasattr(st.session_state, 'context_engine') and st.session_state.context_engine:
                dataset_name = st.session_state.context_engine.get_active_dataset_name()
                if dataset_name and dataset_name in st.session_state.datasets:
                    return {
                        'name': dataset_name,
                        'info': st.session_state.datasets[dataset_name]
                    }
            
            # Fallback: check for any dataset
            if 'datasets' in st.session_state and st.session_state.datasets:
                first_dataset = next(iter(st.session_state.datasets.values()))
                return {
                    'name': first_dataset.get('name', 'unknown'),
                    'info': first_dataset
                }
            
            return None
            
        except Exception:
            return None
    
    def update_workflow_history(self, action: str, details: Dict[str, Any] = None):
        """Update workflow history with user action."""
        history_entry = {
            'action': action,
            'timestamp': time.time(),
            'page': st.session_state.get('active_page', 'unknown'),
            'dataset': self._get_active_dataset_info(),
            'details': details or {}
        }
        
        self._workflow_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self._workflow_history) > 100:
            self._workflow_history = self._workflow_history[-100:]
    
    def should_analyze(self) -> bool:
        """Check if copilot should perform analysis."""
        current_time = time.time()
        copilot_settings = st.session_state.workflow_copilot
        
        return (
            copilot_settings.get('enabled', True) and
            copilot_settings.get('auto_analyze', True) and
            (current_time - self._last_analysis_time) > self._analysis_interval
        )
    
    def perform_intelligent_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive intelligent analysis.
        
        Returns:
            Analysis results summary
        """
        if not self.should_analyze():
            return {}
        
        try:
            # Analyze active datasets
            if 'datasets' in st.session_state:
                for dataset_name, dataset_info in st.session_state.datasets.items():
                    if dataset_name not in self._dataset_insights:
                        df = dataset_info['df']
                        self.analyze_dataset(dataset_name, df)
            
            # Generate recommendations
            recommendations = self.generate_workflow_recommendations()
            
            # Generate alerts
            alerts = self.generate_intelligent_alerts()
            
            # Update analysis time
            self._last_analysis_time = time.time()
            
            return {
                'datasets_analyzed': len(self._dataset_insights),
                'recommendations': len(recommendations),
                'alerts': len(alerts),
                'analysis_timestamp': self._last_analysis_time
            }
            
        except Exception as e:
            self._logger.error(f"Failed to perform intelligent analysis: {e}")
            return {'error': str(e)}


# Global instance
_global_workflow_copilot: Optional[WorkflowCopilot] = None


def get_workflow_copilot() -> WorkflowCopilot:
    """Get the global workflow copilot instance."""
    global _global_workflow_copilot
    if _global_workflow_copilot is None:
        _global_workflow_copilot = WorkflowCopilot()
    return _global_workflow_copilot


def init_workflow_copilot() -> WorkflowCopilot:
    """Initialize the global workflow copilot."""
    global _global_workflow_copilot
    _global_workflow_copilot = WorkflowCopilot()
    return _global_workflow_copilot


def render_workflow_copilot_panel():
    """
    Render the floating Workflow Copilot panel.
    
    Provides live recommendations, smart warnings, and next-step buttons.
    """
    copilot = get_workflow_copilot()
    settings = st.session_state.workflow_copilot
    
    # Perform intelligent analysis if needed
    if copilot.should_analyze():
        analysis_results = copilot.perform_intelligent_analysis()
    
    # Get current recommendations and alerts
    recommendations = copilot.get_recommended_next_steps()
    alerts = copilot.generate_intelligent_alerts()
    
    # Panel styling
    st.markdown("""
    <style>
    .copilot-panel {
        position: fixed;
        top: 80px;
        right: 20px;
        width: 350px;
        max-height: 70vh;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 999;
        overflow-y: auto;
        backdrop-filter: blur(10px);
    }
    
    .copilot-panel.dark {
        background: rgba(30, 30, 30, 0.95);
        border: 1px solid #555;
        color: white;
    }
    
    .copilot-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }
    
    .copilot-header.dark {
        border-bottom: 1px solid #555;
    }
    
    .recommendation-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .recommendation-card:hover {
        background: #e9ecef;
        transform: translateX(-2px);
    }
    
    .alert-card {
        padding: 8px;
        margin-bottom: 8px;
        border-radius: 5px;
        border-left: 4px solid;
    }
    
    .alert-info { border-left-color: #17a2b8; background: #d1ecf1; }
    .alert-warning { border-left-color: #ffc107; background: #fff3cd; }
    .alert-error { border-left-color: #dc3545; background: #f8d7da; }
    .alert-success { border-left-color: #28a745; background: #d4edda; }
    
    .workflow-stage {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    
    .copilot-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Toggle button
    with st.container():
        col1, col2 = st.columns([1, 10])
        with col1:
            if st.button("🤖", key="copilot_toggle", help="Toggle Workflow Copilot"):
                settings['panel_collapsed'] = not settings['panel_collapsed']
        with col2:
            st.write("")
    
    # Main panel
    if not settings['panel_collapsed']:
        with st.container():
            st.markdown('<div class="copilot-panel">', unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <div class="copilot-header">
                <div>
                    <h4 style="margin: 0;">🤖 Workflow Copilot</h4>
                    <small style="color: #666;">Intelligent Assistance</small>
                </div>
                <div>
                    <button onclick="toggleCopilot()" style="background: none; border: none; cursor: pointer;">×</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Workflow stage indicator
            current_stage = copilot.get_workflow_stage()
            stage_display = {
                WorkflowStage.DATA_INGESTION: "📥 Data Ingestion",
                WorkflowStage.DATA_PREPARATION: "🔧 Data Preparation", 
                WorkflowStage.EXPLORATION: "🔍 Exploration",
                WorkflowStage.ANALYSIS: "📊 Analysis",
                WorkflowStage.MODELING: "🧠 Modeling",
                WorkflowStage.REPORTING: "📋 Reporting",
                WorkflowStage.DEPLOYMENT: "🚀 Deployment"
            }
            
            st.markdown(f'<div class="workflow-stage">{stage_display.get(current_stage, "❓ Unknown")}</div>', unsafe_allow_html=True)
            
            # Alerts section
            if alerts and settings.get('show_alerts', True):
                st.markdown("### 🚨 Intelligent Alerts")
                
                for alert in alerts[:3]:  # Show top 3 alerts
                    alert_class = f"alert-{alert.level.value}"
                    
                    st.markdown(f"""
                    <div class="alert-card {alert_class}">
                        <strong>{alert.title}</strong><br>
                        <small>{alert.message}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if alert.actionable and alert.suggested_actions:
                        with st.expander("Suggested Actions", expanded=False):
                            for action in alert.suggested_actions:
                                st.write(f"• {action}")
            
            # Recommendations section
            if recommendations and settings.get('show_recommendations', True):
                st.markdown("### 💡 Smart Recommendations")
                
                for i, rec in enumerate(recommendations[:4]):  # Show top 4 recommendations
                    confidence_color = {
                        (0.9, 1.0): "🟢",
                        (0.7, 0.9): "🟡", 
                        (0.5, 0.7): "🟠",
                        (0.0, 0.5): "🔴"
                    }
                    
                    confidence_emoji = "🟢"
                    for (low, high), emoji in confidence_color.items():
                        if low <= rec.confidence < high:
                            confidence_emoji = emoji
                            break
                    
                    # Recommendation card
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="recommendation-card" onclick="executeRecommendation({i})">
                            <strong>{rec.title}</strong><br>
                            <small>{rec.description}</small><br>
                            <small>{confidence_emoji} {rec.confidence:.0%} confidence</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("→", key=f"execute_rec_{i}", help="Execute recommendation"):
                            copilot.execute_workflow_automation(rec)
                            st.rerun()
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Analyze", key="copilot_analyze", help="Run intelligent analysis"):
                    results = copilot.perform_intelligent_analysis()
                    if results:
                        st.success(f"Analysis complete: {results.get('datasets_analyzed', 0)} datasets analyzed")
                    st.rerun()
            
            with col2:
                if st.button("🧹 Clear", key="copilot_clear", help="Clear recommendations"):
                    copilot._active_recommendations.clear()
                    copilot._active_alerts.clear()
                    st.rerun()
            
            # Settings
            with st.expander("⚙️ Settings", expanded=False):
                settings['enabled'] = st.checkbox("Enable Copilot", settings.get('enabled', True))
                settings['auto_analyze'] = st.checkbox("Auto-analyze", settings.get('auto_analyze', True))
                settings['show_alerts'] = st.checkbox("Show Alerts", settings.get('show_alerts', True))
                settings['show_recommendations'] = st.checkbox("Show Recommendations", settings.get('show_recommendations', True))
            
            # Dataset insights
            if copilot._dataset_insights:
                st.markdown("### 📊 Dataset Insights")
                
                for dataset_name, insight in list(copilot._dataset_insights.items())[:2]:  # Show top 2
                    with st.expander(f"📁 {dataset_name}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Type", insight.dataset_type.value.replace('_', ' ').title())
                            st.metric("Quality", f"{insight.data_quality_score:.2f}")
                        
                        with col2:
                            st.metric("ML Suitability", f"{insight.ml_suitability:.2f}")
                            st.metric("Missing", f"{insight.missing_data_ratio:.1%}")
                        
                        if insight.recommendations:
                            st.markdown("**Recommendations:**")
                            for rec in insight.recommendations[:3]:
                                st.write(f"• {rec}")
            
            st.markdown('</div>', unsafe_allow_html=True)


def render_copilot_integration():
    """
    Render copilot integration in main app.
    
    This function should be called in the main app to integrate the copilot.
    """
    copilot = get_workflow_copilot()
    
    # Update workflow history on page changes
    current_page = st.session_state.get('active_page', '')
    if hasattr(copilot, '_last_page') and copilot._last_page != current_page:
        copilot.update_workflow_history('page_navigation', {'page': current_page})
        copilot._last_page = current_page
    
    # Render the copilot panel
    if st.session_state.workflow_copilot.get('enabled', True):
        render_workflow_copilot_panel()


def get_copilot_summary() -> Dict[str, Any]:
    """
    Get copilot summary for dashboard integration.
    
    Returns:
        Summary of copilot status and insights
    """
    copilot = get_workflow_copilot()
    
    return {
        'enabled': st.session_state.workflow_copilot.get('enabled', False),
        'datasets_analyzed': len(copilot._dataset_insights),
        'active_recommendations': len(copilot._active_recommendations),
        'active_alerts': len(copilot._active_alerts),
        'workflow_stage': copilot.get_workflow_stage().value,
        'last_analysis': copilot._last_analysis_time,
        'workflow_actions': len(copilot._workflow_history)
    }
