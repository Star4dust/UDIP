"""
UDIP v3 — modules/performance_monitor.py
Performance monitoring and metrics collection.
"""

import streamlit as st
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "system"
    metadata: Dict[str, Any] = None


@dataclass
class Alert:
    """Represents a performance alert."""
    level: str  # "info", "warning", "critical"
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: float
    suggestions: List[str] = None


class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self):
        if "performance_metrics" not in st.session_state:
            st.session_state.performance_metrics = []
        if "performance_alerts" not in st.session_state:
            st.session_state.performance_alerts = []
        if "monitoring_settings" not in st.session_state:
            st.session_state.monitoring_settings = {
                "enable_monitoring": True,
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_usage": 90,
                    "response_time": 5
                },
                "collection_interval": 30,  # seconds
                "max_metrics": 1000,
                "auto_cleanup": True
            }
        if "monitoring_active" not in st.session_state:
            st.session_state.monitoring_active = False
    
    def start_monitoring(self):
        """Start performance monitoring in background."""
        if st.session_state.monitoring_active:
            return
        
        st.session_state.monitoring_active = True
        
        def monitor_loop():
            while st.session_state.monitoring_active:
                try:
                    self._collect_metrics()
                    self._check_alerts()
                    time.sleep(st.session_state.monitoring_settings["collection_interval"])
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(5)  # Wait before retrying
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        st.session_state.monitoring_active = False
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        if not st.session_state.monitoring_settings["enable_monitoring"]:
            return
        
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = [
            PerformanceMetric(
                name="cpu_percent",
                value=cpu_percent,
                unit="%",
                timestamp=current_time,
                category="system"
            ),
            PerformanceMetric(
                name="memory_percent",
                value=memory.percent,
                unit="%",
                timestamp=current_time,
                category="system"
            ),
            PerformanceMetric(
                name="memory_used_gb",
                value=memory.used / (1024**3),
                unit="GB",
                timestamp=current_time,
                category="system"
            ),
            PerformanceMetric(
                name="memory_available_gb",
                value=memory.available / (1024**3),
                unit="GB",
                timestamp=current_time,
                category="system"
            ),
            PerformanceMetric(
                name="disk_usage_percent",
                value=disk.percent,
                unit="%",
                timestamp=current_time,
                category="system"
            ),
        ]
        
        # Application metrics
        if "datasets" in st.session_state:
            datasets_count = len(st.session_state.datasets)
            total_rows = sum(v["df"].shape[0] for v in st.session_state.datasets.values())
            
            metrics.extend([
                PerformanceMetric(
                    name="datasets_loaded",
                    value=datasets_count,
                    unit="count",
                    timestamp=current_time,
                    category="application"
                ),
                PerformanceMetric(
                    name="total_rows",
                    value=total_rows,
                    unit="rows",
                    timestamp=current_time,
                    category="application"
                )
            ])
        
        # Add metrics to history
        st.session_state.performance_metrics.extend(metrics)
        
        # Limit metrics history
        max_metrics = st.session_state.monitoring_settings["max_metrics"]
        if len(st.session_state.performance_metrics) > max_metrics:
            st.session_state.performance_metrics = st.session_state.performance_metrics[-max_metrics:]
    
    def _check_alerts(self):
        """Check for performance alerts."""
        if not st.session_state.monitoring_settings["enable_monitoring"]:
            return
        
        current_time = time.time()
        thresholds = st.session_state.monitoring_settings["alert_thresholds"]
        
        # Get latest metrics
        latest_metrics = {}
        for metric in st.session_state.performance_metrics[-50:]:  # Check last 50 metrics
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = metric
        
        alerts = []
        
        # Check CPU alert
        if "cpu_percent" in latest_metrics:
            cpu_metric = latest_metrics["cpu_percent"]
            if cpu_metric.value > thresholds["cpu_percent"]:
                alerts.append(Alert(
                    level="warning",
                    message=f"High CPU usage: {cpu_metric.value:.1f}%",
                    metric="cpu_percent",
                    threshold=thresholds["cpu_percent"],
                    current_value=cpu_metric.value,
                    timestamp=current_time,
                    suggestions=[
                        "Close unnecessary applications",
                        "Check for background processes",
                        "Consider upgrading CPU if consistently high"
                    ]
                ))
        
        # Check memory alert
        if "memory_percent" in latest_metrics:
            memory_metric = latest_metrics["memory_percent"]
            if memory_metric.value > thresholds["memory_percent"]:
                alerts.append(Alert(
                    level="critical",
                    message=f"High memory usage: {memory_metric.value:.1f}%",
                    metric="memory_percent",
                    threshold=thresholds["memory_percent"],
                    current_value=memory_metric.value,
                    timestamp=current_time,
                    suggestions=[
                        "Close unused applications",
                        "Clear dataset cache if large",
                        "Restart UDIP to free memory",
                        "Consider adding more RAM"
                    ]
                ))
        
        # Check disk alert
        if "disk_usage_percent" in latest_metrics:
            disk_metric = latest_metrics["disk_usage_percent"]
            if disk_metric.value > thresholds["disk_usage"]:
                alerts.append(Alert(
                    level="warning",
                    message=f"High disk usage: {disk_metric.value:.1f}%",
                    metric="disk_usage_percent",
                    threshold=thresholds["disk_usage"],
                    current_value=disk_metric.value,
                    timestamp=current_time,
                    suggestions=[
                        "Clean up temporary files",
                        "Archive old datasets",
                        "Consider disk space optimization"
                    ]
                ))
        
        # Add alerts to history
        if alerts:
            st.session_state.performance_alerts.extend(alerts)
            
            # Limit alerts history
            if len(st.session_state.performance_alerts) > 100:
                st.session_state.performance_alerts = st.session_state.performance_alerts[-100:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory(),
            "disk": psutil.disk_usage('/'),
            "timestamp": time.time()
        }
    
    def get_metrics_history(self, metric_name: Optional[str] = None, limit: int = 100) -> List[PerformanceMetric]:
        """Get metrics history."""
        metrics = st.session_state.performance_metrics
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        return metrics[-limit:] if metrics else []
    
    def get_alerts_history(self, limit: int = 50) -> List[Alert]:
        """Get alerts history."""
        return st.session_state.performance_alerts[-limit:]
    
    def clear_metrics(self):
        """Clear metrics history."""
        st.session_state.performance_metrics.clear()
        st.success("Performance metrics cleared.")
    
    def clear_alerts(self):
        """Clear alerts history."""
        st.session_state.performance_alerts.clear()
        st.success("Performance alerts cleared.")
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export performance metrics."""
        metrics = self.get_metrics_history()
        
        if format_type == "json":
            return json.dumps([
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "category": m.category
                } for m in metrics
            ], indent=2)
        
        elif format_type == "csv":
            import pandas as pd
            df = pd.DataFrame([
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "category": m.category
                } for m in metrics
            ])
            return df.to_csv(index=False)
        
        return ""
    
    def render_monitoring_panel(self):
        """Render performance monitoring panel."""
        st.markdown("### 📊 Performance Monitoring")
        
        # Settings
        settings = st.session_state.monitoring_settings
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings["enable_monitoring"] = st.checkbox(
                "Enable Monitoring",
                value=settings["enable_monitoring"],
                help="Start/stop performance monitoring"
            )
        
        with col2:
            if settings["enable_monitoring"]:
                if st.button("⏹️ Stop Monitoring", key="stop_monitoring"):
                    self.stop_monitoring()
                    st.rerun()
            else:
                if st.button("▶️ Start Monitoring", key="start_monitoring"):
                    self.start_monitoring()
                    st.rerun()
        
        # Alert thresholds
        st.markdown("#### 🚨 Alert Thresholds")
        
        thresholds = settings["alert_thresholds"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            thresholds["cpu_percent"] = st.slider(
                "CPU Alert (%)",
                min_value=50,
                max_value=100,
                value=thresholds["cpu_percent"],
                help="Trigger alert when CPU usage exceeds this percentage"
            )
            
            thresholds["memory_percent"] = st.slider(
                "Memory Alert (%)",
                min_value=70,
                max_value=100,
                value=thresholds["memory_percent"],
                help="Trigger alert when memory usage exceeds this percentage"
            )
        
        with col2:
            thresholds["disk_usage"] = st.slider(
                "Disk Alert (%)",
                min_value=80,
                max_value=100,
                value=thresholds["disk_usage"],
                help="Trigger alert when disk usage exceeds this percentage"
            )
            
            settings["collection_interval"] = st.slider(
                "Collection Interval (seconds)",
                min_value=5,
                max_value=300,
                value=settings["collection_interval"],
                help="How often to collect performance metrics"
            )
        
        st.session_state.monitoring_settings = settings
        
        # Current metrics display
        if settings["enable_monitoring"]:
            current_metrics = self.get_current_metrics()
            
            st.markdown("#### 📈 Current Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "CPU Usage",
                    f"{current_metrics['cpu_percent']:.1f}%",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Memory Usage",
                    f"{current_metrics['memory'].percent:.1f}%",
                    delta=None
                )
                st.caption(f"{current_metrics['memory'].used / (1024**3):.1f}GB / {current_metrics['memory'].total / (1024**3):.1f}GB")
            
            with col3:
                st.metric(
                    "Disk Usage",
                    f"{current_metrics['disk'].percent:.1f}%",
                    delta=None
                )
        
        # Recent alerts
        recent_alerts = self.get_alerts_history(limit=5)
        if recent_alerts:
            st.markdown("#### 🚨 Recent Alerts")
            
            for alert in recent_alerts:
                alert_color = {
                    "info": "blue",
                    "warning": "orange", 
                    "critical": "red"
                }.get(alert.level, "gray")
                
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {alert_color};
                    padding: 10px;
                    margin: 5px 0;
                    background: #f8f9fa;
                    border-radius: 4px;
                ">
                    <div style="font-weight: 600; color: {alert_color};">
                        {alert.level.upper()}: {alert.message}
                    </div>
                    <div style="font-size: 0.9em; color: #666; margin-top: 4px;">
                        {time.strftime('%H:%M:%S', time.localtime(alert.timestamp))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("#### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Metrics (JSON)"):
                json_data = self.export_metrics("json")
                st.download_button(
                    label="Download metrics.json",
                    data=json_data,
                    file_name="performance_metrics.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🧹 Clear History"):
                self.clear_metrics()
                self.clear_alerts()


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def start_monitoring():
    """Start performance monitoring."""
    _performance_monitor.start_monitoring()


def stop_monitoring():
    """Stop performance monitoring."""
    _performance_monitor.stop_monitoring()


def render_performance_monitoring():
    """Render performance monitoring interface."""
    _performance_monitor.render_monitoring_panel()
