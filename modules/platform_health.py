"""
UDIP v3 — modules/platform_health.py
Platform health monitoring and diagnostics for UDIP Core Stability Sprint.

Provides comprehensive health monitoring, error tracking, and system diagnostics.
"""

import streamlit as st
import time
import logging
import psutil
import traceback
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class HealthMetric:
    """Represents a health metric."""
    name: str
    value: Any
    status: str  # "healthy", "warning", "critical", "unknown"
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemDiagnostic:
    """Represents a system diagnostic."""
    category: str
    title: str
    description: str
    status: str
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class PlatformHealthMonitor:
    """
    Comprehensive platform health monitoring system.
    
    Tracks system performance, error rates, and provides diagnostics.
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self._logger = logging.getLogger(__name__)
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for health monitoring."""
        if "platform_health" not in st.session_state:
            st.session_state.platform_health = {
                "metrics": {},
                "diagnostics": [],
                "error_log": [],
                "performance_log": [],
                "last_check": time.time(),
                "health_score": 100.0,
                "alerts": []
            }
    
    def collect_health_metrics(self) -> Dict[str, HealthMetric]:
        """Collect current health metrics."""
        metrics = {}
        
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            memory_status = "healthy"
            if memory_percent > 80:
                memory_status = "critical"
            elif memory_percent > 60:
                memory_status = "warning"
            
            metrics["memory"] = HealthMetric(
                name="Memory Usage",
                value=f"{memory_mb:.1f} MB ({memory_percent:.1f}%)",
                status=memory_status,
                details={
                    "rss_mb": memory_mb,
                    "percent": memory_percent,
                    "threshold_warning": 60,
                    "threshold_critical": 80
                }
            )
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            cpu_status = "healthy"
            if cpu_percent > 80:
                cpu_status = "critical"
            elif cpu_percent > 60:
                cpu_status = "warning"
            
            metrics["cpu"] = HealthMetric(
                name="CPU Usage",
                value=f"{cpu_percent:.1f}%",
                status=cpu_status,
                details={
                    "percent": cpu_percent,
                    "threshold_warning": 60,
                    "threshold_critical": 80
                }
            )
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            disk_status = "healthy"
            if disk_percent > 90:
                disk_status = "critical"
            elif disk_percent > 75:
                disk_status = "warning"
            
            metrics["disk"] = HealthMetric(
                name="Disk Usage",
                value=f"{disk_percent:.1f}%",
                status=disk_status,
                details={
                    "percent": disk_percent,
                    "free_gb": disk_usage.free / (1024**3),
                    "threshold_warning": 75,
                    "threshold_critical": 90
                }
            )
            
            # Session state size
            session_size = len(str(st.session_state))
            session_status = "healthy"
            if session_size > 100000:  # 100KB
                session_status = "critical"
            elif session_size > 50000:  # 50KB
                session_status = "warning"
            
            metrics["session_state"] = HealthMetric(
                name="Session State Size",
                value=f"{session_size} bytes",
                status=session_status,
                details={
                    "size_bytes": session_size,
                    "threshold_warning": 50000,
                    "threshold_critical": 100000
                }
            )
            
            # Error rate
            error_log = st.session_state.platform_health.get("error_log", [])
            recent_errors = [e for e in error_log if time.time() - e.get("timestamp", 0) < 3600]  # Last hour
            error_rate = len(recent_errors)
            error_status = "healthy"
            if error_rate > 10:
                error_status = "critical"
            elif error_rate > 5:
                error_status = "warning"
            
            metrics["error_rate"] = HealthMetric(
                name="Error Rate",
                value=f"{error_rate} errors/hour",
                status=error_status,
                details={
                    "recent_errors": error_rate,
                    "threshold_warning": 5,
                    "threshold_critical": 10
                }
            )
            
        except Exception as e:
            self._logger.error(f"Failed to collect health metrics: {e}")
            metrics["collection_error"] = HealthMetric(
                name="Metric Collection",
                value="Failed",
                status="critical",
                details={"error": str(e)}
            )
        
        return metrics
    
    def run_diagnostics(self) -> List[SystemDiagnostic]:
        """Run comprehensive system diagnostics."""
        diagnostics = []
        
        try:
            # Navigation system diagnostic
            from modules.navigation import build_page_map
            page_map = build_page_map()
            nav_status = "healthy"
            nav_issues = []
            
            if len(page_map) < 20:
                nav_status = "warning"
                nav_issues.append("Low number of registered pages")
            
            # Check for broken routes
            broken_routes = []
            for label, render_func in page_map.items():
                try:
                    # Test if render function is callable
                    if not callable(render_func):
                        broken_routes.append(label)
                except:
                    broken_routes.append(label)
            
            if broken_routes:
                nav_status = "critical"
                nav_issues.append(f"Broken routes: {broken_routes}")
            
            diagnostics.append(SystemDiagnostic(
                category="Navigation",
                title="Navigation System",
                description="Centralized routing and page registration",
                status=nav_status,
                recommendations=nav_issues
            ))
            
            # Context engine diagnostic
            context_status = "healthy"
            context_issues = []
            
            try:
                from modules.context_engine import get_context_engine
                context_engine = get_context_engine()
                # Test context engine functionality
                if not hasattr(context_engine, 'has_active_data'):
                    context_status = "critical"
                    context_issues.append("Missing core methods")
            except Exception as e:
                context_status = "critical"
                context_issues.append(f"Initialization error: {e}")
            
            diagnostics.append(SystemDiagnostic(
                category="Context Engine",
                title="Context Engine",
                description="Workspace context and workflow awareness",
                status=context_status,
                recommendations=context_issues
            ))
            
            # Workflow handoff diagnostic
            handoff_status = "healthy"
            handoff_issues = []
            
            try:
                from modules.workflow_handoff import get_workflow_handoff_manager
                handoff_manager = get_workflow_handoff_manager()
                if not hasattr(handoff_manager, '_handoff_actions'):
                    handoff_status = "critical"
                    handoff_issues.append("Missing handoff actions")
            except Exception as e:
                handoff_status = "critical"
                handoff_issues.append(f"Initialization error: {e}")
            
            diagnostics.append(SystemDiagnostic(
                category="Workflow Handoffs",
                title="Workflow Handoff System",
                description="Cross-page workflow transitions",
                status=handoff_status,
                recommendations=handoff_issues
            ))
            
            # Command palette diagnostic
            palette_status = "healthy"
            palette_issues = []
            
            try:
                from modules.command_palette import get_command_palette
                palette = get_command_palette()
                if not hasattr(palette, '_commands'):
                    palette_status = "critical"
                    palette_issues.append("Missing command registry")
            except Exception as e:
                palette_status = "critical"
                palette_issues.append(f"Initialization error: {e}")
            
            diagnostics.append(SystemDiagnostic(
                category="Command Palette",
                title="Command Palette",
                description="Universal command interface",
                status=palette_status,
                recommendations=palette_issues
            ))
            
            # HIRAX systems diagnostic
            hirax_status = "healthy"
            hirax_issues = []
            
            hirax_modules = [
                'modules.hirax_processing',
                'modules.hirax_analysis',
                'modules.hirax_validation'
            ]
            
            for module_path in hirax_modules:
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    if not hasattr(module, 'render'):
                        hirax_status = "warning"
                        hirax_issues.append(f"{module_path} missing render function")
                except ImportError:
                    hirax_status = "critical"
                    hirax_issues.append(f"{module_path} not importable")
            
            diagnostics.append(SystemDiagnostic(
                category="HIRAX Systems",
                title="HIRAX Scientific Workflows",
                description="Radio telescope data analysis systems",
                status=hirax_status,
                recommendations=hirax_issues
            ))
            
        except Exception as e:
            self._logger.error(f"Failed to run diagnostics: {e}")
            diagnostics.append(SystemDiagnostic(
                category="Diagnostics",
                title="System Diagnostics",
                description="Platform diagnostic system",
                status="critical",
                recommendations=[f"Diagnostic error: {e}"]
            ))
        
        return diagnostics
    
    def calculate_health_score(self, metrics: Dict[str, HealthMetric], diagnostics: List[SystemDiagnostic]) -> float:
        """Calculate overall platform health score."""
        score = 100.0
        
        # Weight metrics
        weights = {
            "memory": 20,
            "cpu": 15,
            "disk": 10,
            "session_state": 15,
            "error_rate": 25,
            "navigation": 5,
            "context_engine": 5,
            "workflow_handoffs": 5,
            "command_palette": 5,
            "hirax_systems": 10
        }
        
        # Deduct points for issues
        for metric_name, metric in metrics.items():
            if metric_name in weights:
                weight = weights[metric_name]
                if metric.status == "warning":
                    score -= weight * 0.3
                elif metric.status == "critical":
                    score -= weight * 0.7
        
        for diagnostic in diagnostics:
            if diagnostic.category in weights:
                weight = weights[diagnostic.category]
                if diagnostic.status == "warning":
                    score -= weight * 0.3
                elif diagnostic.status == "critical":
                    score -= weight * 0.7
        
        return max(0.0, min(100.0, score))
    
    def update_health_status(self):
        """Update platform health status."""
        try:
            metrics = self.collect_health_metrics()
            diagnostics = self.run_diagnostics()
            health_score = self.calculate_health_score(metrics, diagnostics)
            
            # Update session state
            st.session_state.platform_health.update({
                "metrics": {name: metric.__dict__ for name, metric in metrics.items()},
                "diagnostics": [diag.__dict__ for diag in diagnostics],
                "last_check": time.time(),
                "health_score": health_score
            })
            
            # Generate alerts for critical issues
            alerts = []
            for metric in metrics.values():
                if metric.status == "critical":
                    alerts.append({
                        "type": "metric",
                        "message": f"Critical issue with {metric.name}: {metric.value}",
                        "timestamp": time.time()
                    })
            
            for diagnostic in diagnostics:
                if diagnostic.status == "critical":
                    alerts.append({
                        "type": "diagnostic",
                        "message": f"Critical diagnostic: {diagnostic.title}",
                        "timestamp": time.time()
                    })
            
            st.session_state.platform_health["alerts"] = alerts
            
        except Exception as e:
            self._logger.error(f"Failed to update health status: {e}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "traceback": traceback.format_exc(),
            "context": context,
            "page": st.session_state.get("active_page", "unknown"),
            "user_agent": st.session_state.get("user_agent", "unknown")
        }
        
        error_log = st.session_state.platform_health.get("error_log", [])
        error_log.append(error_entry)
        
        # Keep only last 100 errors
        if len(error_log) > 100:
            st.session_state.platform_health["error_log"] = error_log[-100:]
        
        self._logger.error(f"Error logged: {error} (context: {context})")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for dashboard."""
        health_data = st.session_state.platform_health
        
        return {
            "health_score": health_data.get("health_score", 100.0),
            "last_check": health_data.get("last_check", time.time()),
            "metrics_count": len(health_data.get("metrics", {})),
            "diagnostics_count": len(health_data.get("diagnostics", [])),
            "error_count": len(health_data.get("error_log", [])),
            "alerts_count": len(health_data.get("alerts", [])),
            "status": self._get_overall_status(health_data.get("health_score", 100.0))
        }
    
    def _get_overall_status(self, health_score: float) -> str:
        """Get overall status from health score."""
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 75:
            return "Good"
        elif health_score >= 60:
            return "Fair"
        elif health_score >= 40:
            return "Poor"
        else:
            return "Critical"


# Global instance
_global_health_monitor: Optional[PlatformHealthMonitor] = None


def get_platform_health_monitor() -> PlatformHealthMonitor:
    """Get the global platform health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = PlatformHealthMonitor()
    return _global_health_monitor


def init_platform_health():
    """Initialize platform health monitoring."""
    monitor = get_platform_health_monitor()
    monitor.update_health_status()
    return monitor


def render_platform_health_dashboard():
    """Render platform health dashboard."""
    monitor = get_platform_health_monitor()
    
    st.markdown('<div class="main-header">🔧 Platform Health</div>', unsafe_allow_html=True)
    
    # Update health status
    monitor.update_health_status()
    
    health_summary = monitor.get_health_summary()
    
    # Overall health score
    score = health_summary["health_score"]
    status = health_summary["status"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Health Score", f"{score:.1f}", delta=None)
        st.caption(f"Status: {status}")
    
    with col2:
        st.metric("Active Issues", health_summary["alerts_count"])
        st.caption("Critical alerts")
    
    with col3:
        last_check = datetime.fromtimestamp(health_summary["last_check"])
        st.metric("Last Check", last_check.strftime("%H:%M:%S"))
        st.caption("System monitoring")
    
    # Health metrics
    st.markdown("### 📊 System Metrics")
    
    metrics = st.session_state.platform_health.get("metrics", {})
    for metric_name, metric_data in metrics.items():
        with st.expander(f"🔍 {metric_data.get('name', metric_name)}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Value:** {metric_data.get('value', 'N/A')}")
                st.markdown(f"**Status:** {metric_data.get('status', 'unknown')}")
            
            with col2:
                details = metric_data.get('details', {})
                if details:
                    st.markdown("**Details:**")
                    for key, value in details.items():
                        st.caption(f"{key}: {value}")
    
    # Diagnostics
    st.markdown("### 🔍 System Diagnostics")
    
    diagnostics = st.session_state.platform_health.get("diagnostics", [])
    for diagnostic in diagnostics:
        status_color = {
            "healthy": "🟢",
            "warning": "🟡", 
            "critical": "🔴"
        }.get(diagnostic.get("status", "unknown"), "⚪")
        
        with st.expander(f"{status_color} {diagnostic.get('title', 'Unknown')}", expanded=False):
            st.markdown(f"**Category:** {diagnostic.get('category', 'Unknown')}")
            st.markdown(f"**Description:** {diagnostic.get('description', 'No description')}")
            st.markdown(f"**Status:** {diagnostic.get('status', 'unknown')}")
            
            recommendations = diagnostic.get('recommendations', [])
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.caption(f"• {rec}")
    
    # Recent errors
    st.markdown("### 🚨 Recent Errors")
    
    error_log = st.session_state.platform_health.get("error_log", [])
    if error_log:
        recent_errors = error_log[-10:]  # Last 10 errors
        for i, error in enumerate(recent_errors):
            with st.expander(f"Error {i+1}: {error.get('error', 'Unknown error')[:50]}...", expanded=False):
                st.code(error.get('traceback', 'No traceback available'))
                st.caption(f"Time: {time.ctime(error.get('timestamp', 0))}")
                st.caption(f"Page: {error.get('page', 'Unknown')}")
                st.caption(f"Context: {error.get('context', 'No context')}")
    else:
        st.success("No recent errors ✓")
    
    # Performance trends
    st.markdown("### 📈 Performance Trends")
    
    performance_log = st.session_state.platform_health.get("performance_log", [])
    if performance_log:
        import pandas as pd
        df = pd.DataFrame(performance_log)
        
        if not df.empty:
            st.line_chart(df.set_index('timestamp')['memory_mb'], title="Memory Usage Trend")
    
    # System recommendations
    st.markdown("### 💡 System Recommendations")
    
    if score < 75:
        st.warning("🔧 Platform health requires attention. See diagnostics above.")
    elif score < 90:
        st.info("ℹ️ Platform health is good but could be optimized.")
    else:
        st.success("✅ Platform health is excellent!")


def get_health_status() -> str:
    """Get current health status."""
    monitor = get_platform_health_monitor()
    summary = monitor.get_health_summary()
    return summary["status"]


def log_platform_error(error: Exception, context: str = ""):
    """Convenience function to log platform errors."""
    monitor = get_platform_health_monitor()
    monitor.log_error(error, context)
