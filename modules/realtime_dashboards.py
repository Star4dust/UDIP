"""
UDIP v3 — modules/realtime_dashboards.py
Real-time dashboards with live data streaming capabilities.
"""

import streamlit as st
import time
import threading
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


@dataclass
class DashboardWidget:
    """Represents a dashboard widget."""
    widget_id: str
    title: str
    widget_type: str  # "metric", "chart", "table", "text"
    data_source: str
    refresh_interval: int  # seconds
    config: Dict[str, Any]
    last_update: float = 0


@dataclass
class DataSource:
    """Represents a data source for real-time updates."""
    source_id: str
    name: str
    type: str  # "dataset", "api", "file", "stream"
    connection_config: Dict[str, Any]
    last_update: float = 0


class RealtimeDashboardManager:
    """Manages real-time dashboards and data streaming."""
    
    def __init__(self):
        if "dashboard_widgets" not in st.session_state:
            st.session_state.dashboard_widgets = {}
        if "dashboard_data_sources" not in st.session_state:
            st.session_state.dashboard_data_sources = {}
        if "dashboard_settings" not in st.session_state:
            st.session_state.dashboard_settings = {
                "auto_refresh": True,
                "refresh_interval": 30,
                "max_data_points": 1000,
                "enable_animations": True,
                "theme": "light"
            }
        if "realtime_threads" not in st.session_state:
            st.session_state.realtime_threads = {}
    
    def create_widget(
        self,
        widget_id: str,
        title: str,
        widget_type: str,
        data_source: str,
        refresh_interval: int = 30,
        config: Optional[Dict[str, Any]] = None
    ):
        """Create a new dashboard widget."""
        widget = DashboardWidget(
            widget_id=widget_id,
            title=title,
            widget_type=widget_type,
            data_source=data_source,
            refresh_interval=refresh_interval,
            config=config or {},
            last_update=time.time()
        )
        
        st.session_state.dashboard_widgets[widget_id] = widget
        return widget
    
    def create_data_source(
        self,
        source_id: str,
        name: str,
        source_type: str,
        connection_config: Dict[str, Any]
    ):
        """Create a new data source."""
        data_source = DataSource(
            source_id=source_id,
            name=name,
            type=source_type,
            connection_config=connection_config,
            last_update=time.time()
        )
        
        st.session_state.dashboard_data_sources[source_id] = data_source
        return data_source
    
    def get_widget_data(self, widget: DashboardWidget) -> Any:
        """Get data for a widget from its data source."""
        source = st.session_state.dashboard_data_sources.get(widget.data_source)
        if not source:
            return None
        
        if source.type == "dataset":
            # Get live data from dataset
            datasets = st.session_state.get("datasets", {})
            if source.source_id in datasets:
                df = datasets[source.source_id]["df"]
                
                # Apply sampling for large datasets
                if len(df) > 10000:
                    df = df.sample(n=1000, random_state=42)
                
                return df
        
        elif source.type == "api":
            # Get data from API endpoint
            return self._fetch_api_data(source)
        
        elif source.type == "file":
            # Get data from file
            return self._read_file_data(source)
        
        elif source.type == "stream":
            # Get data from streaming source
            return self._get_stream_data(source)
        
        return None
    
    def _fetch_api_data(self, source: DataSource) -> Any:
        """Fetch data from API endpoint."""
        try:
            import requests
            
            url = source.connection_config.get("url")
            headers = source.connection_config.get("headers", {})
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Update last update time
            source.last_update = time.time()
            
            return response.json()
            
        except Exception as e:
            st.error(f"Failed to fetch data from {source.name}: {e}")
            return None
    
    def _read_file_data(self, source: DataSource) -> Any:
        """Read data from file."""
        try:
            file_path = source.connection_config.get("path")
            file_type = source.connection_config.get("type", "csv")
            
            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type == "json":
                df = pd.read_json(file_path)
            elif file_type == "parquet":
                df = pd.read_parquet(file_path)
            else:
                return None
            
            # Update last update time
            source.last_update = time.time()
            
            return df
            
        except Exception as e:
            st.error(f"Failed to read data from {source.name}: {e}")
            return None
    
    def _get_stream_data(self, source: DataSource) -> Any:
        """Get data from streaming source."""
        # This would integrate with WebSocket or similar for real-time streaming
        # For now, return cached data
        return st.session_state.get(f"stream_data_{source.source_id}")
    
    def start_realtime_updates(self):
        """Start background threads for real-time updates."""
        settings = st.session_state.dashboard_settings
        
        if not settings["auto_refresh"]:
            return
        
        for widget_id, widget in st.session_state.dashboard_widgets.items():
            if widget_id not in st.session_state.realtime_threads:
                # Start update thread for this widget
                thread = threading.Thread(
                    target=self._update_widget_loop,
                    args=(widget,),
                    daemon=True
                )
                thread.start()
                st.session_state.realtime_threads[widget_id] = thread
    
    def _update_widget_loop(self, widget: DashboardWidget):
        """Background loop to update widget data."""
        while st.session_state.dashboard_settings.get("auto_refresh", False):
            try:
                # Get fresh data
                data = self.get_widget_data(widget)
                if data is not None:
                    # Store in session state
                    st.session_state[f"widget_data_{widget.widget_id}"] = data
                    widget.last_update = time.time()
                
                # Wait for next update
                time.sleep(widget.refresh_interval)
                
            except Exception as e:
                print(f"Widget update error for {widget.widget_id}: {e}")
                time.sleep(5)  # Wait before retrying
    
    def stop_realtime_updates(self):
        """Stop all real-time update threads."""
        st.session_state.dashboard_settings["auto_refresh"] = False
        
        # Wait for threads to finish
        for thread in st.session_state.realtime_threads.values():
            if thread.is_alive():
                thread.join(timeout=1)
        
        st.session_state.realtime_threads.clear()
    
    def render_metric_widget(self, widget: DashboardWidget):
        """Render a metric widget."""
        data = st.session_state.get(f"widget_data_{widget.widget_id}")
        if data is None:
            st.error(f"No data available for {widget.title}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                widget.config.get("title", widget.title),
                value=data.get("value", 0),
                delta=data.get("delta", 0)
            )
        
        with col2:
            # Show last update time
            if widget.last_update > 0:
                last_update = datetime.fromtimestamp(widget.last_update).strftime("%H:%M:%S")
                st.caption(f"Last updated: {last_update}")
    
    def render_chart_widget(self, widget: DashboardWidget):
        """Render a chart widget."""
        data = st.session_state.get(f"widget_data_{widget.widget_id}")
        if data is None:
            st.error(f"No data available for {widget.title}")
            return
        
        chart_type = widget.config.get("chart_type", "line")
        
        if isinstance(data, pd.DataFrame):
            if chart_type == "line":
                x_col = widget.config.get("x_column")
                y_col = widget.config.get("y_column")
                
                if x_col and y_col and x_col in data.columns and y_col in data.columns:
                    fig = px.line(data, x=x_col, y=y_col, title=widget.title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Invalid column configuration for line chart")
            
            elif chart_type == "bar":
                x_col = widget.config.get("x_column")
                y_col = widget.config.get("y_column")
                
                if x_col and y_col and x_col in data.columns and y_col in data.columns:
                    fig = px.bar(data, x=x_col, y=y_col, title=widget.title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Invalid column configuration for bar chart")
            
            elif chart_type == "scatter":
                x_col = widget.config.get("x_column")
                y_col = widget.config.get("y_column")
                
                if x_col and y_col and x_col in data.columns and y_col in data.columns:
                    fig = px.scatter(data, x=x_col, y=y_col, title=widget.title)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Invalid column configuration for scatter chart")
            
            elif chart_type == "pie":
                value_col = widget.config.get("value_column")
                name_col = widget.config.get("name_column")
                
                if value_col and value_col in data.columns:
                    fig = px.pie(
                        data, 
                        values=value_col, 
                        names=name_col if name_col and name_col in data.columns else value_col,
                        title=widget.title
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Invalid column configuration for pie chart")
        
        elif isinstance(data, dict):
            if chart_type == "gauge":
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=data.get("value", 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': widget.title},
                    delta={'reference': data.get("reference", 0)},
                    gauge={'axis': {'range': [None, data.get("max", 100)]}}
                ))
                st.plotly_chart(fig, use_container_width=True)
    
    def render_table_widget(self, widget: DashboardWidget):
        """Render a table widget."""
        data = st.session_state.get(f"widget_data_{widget.widget_id}")
        if data is None:
            st.error(f"No data available for {widget.title}")
            return
        
        if isinstance(data, pd.DataFrame):
            # Apply row limit
            max_rows = widget.config.get("max_rows", 100)
            display_data = data.head(max_rows)
            
            st.dataframe(display_data, use_container_width=True)
            
            if len(data) > max_rows:
                st.caption(f"Showing {max_rows} of {len(data)} rows")
    
    def render_text_widget(self, widget: DashboardWidget):
        """Render a text widget."""
        data = st.session_state.get(f"widget_data_{widget.widget_id}")
        if data is None:
            st.error(f"No data available for {widget.title}")
            return
        
        if isinstance(data, str):
            st.markdown(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                st.markdown(f"**{key}:** {value}")
    
    def render_widget(self, widget: DashboardWidget):
        """Render a widget based on its type."""
        # Show last update time
        if widget.last_update > 0:
            last_update = datetime.fromtimestamp(widget.last_update).strftime("%H:%M:%S")
            st.caption(f"🔄 Last updated: {last_update}")
        
        if widget.widget_type == "metric":
            self.render_metric_widget(widget)
        elif widget.widget_type == "chart":
            self.render_chart_widget(widget)
        elif widget.widget_type == "table":
            self.render_table_widget(widget)
        elif widget.widget_type == "text":
            self.render_text_widget(widget)
        else:
            st.error(f"Unknown widget type: {widget.widget_type}")
    
    def render_dashboard_editor(self):
        """Render dashboard configuration interface."""
        st.markdown("### 📊 Real-Time Dashboard Configuration")
        
        # Settings
        settings = st.session_state.dashboard_settings
        
        col1, col2 = st.columns(2)
        with col1:
            settings["auto_refresh"] = st.checkbox(
                "Auto-Refresh",
                value=settings["auto_refresh"],
                help="Automatically update widget data"
            )
        
        with col2:
            settings["refresh_interval"] = st.number_input(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=300,
                value=settings["refresh_interval"],
                help="How often to refresh widget data"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            settings["max_data_points"] = st.number_input(
                "Max Data Points",
                min_value=100,
                max_value=10000,
                value=settings["max_data_points"],
                help="Maximum data points to display"
            )
        
        with col2:
            settings["enable_animations"] = st.checkbox(
                "Enable Animations",
                value=settings["enable_animations"],
                help="Show smooth transitions in charts"
            )
        
        st.session_state.dashboard_settings = settings
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Real-Time Updates", key="start_realtime"):
                self.start_realtime_updates()
                st.success("Real-time updates started!")
        
        with col2:
            if st.button("⏹️ Stop Real-Time Updates", key="stop_realtime"):
                self.stop_realtime_updates()
                st.success("Real-time updates stopped!")
        
        # Widget management
        st.markdown("#### 🎛 Widget Management")
        
        # Add new widget
        with st.expander("Add New Widget", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                widget_id = st.text_input("Widget ID", key="new_widget_id")
                widget_title = st.text_input("Widget Title", key="new_widget_title")
                widget_type = st.selectbox(
                    "Widget Type",
                    options=["metric", "chart", "table", "text"],
                    key="new_widget_type"
                )
            
            with col2:
                # Data source selection
                datasets = st.session_state.get("datasets", {})
                data_sources = ["manual"] + list(datasets.keys())
                
                data_source = st.selectbox(
                    "Data Source",
                    options=data_sources,
                    key="new_widget_data_source"
                )
                
                refresh_interval = st.number_input(
                    "Refresh Interval (seconds)",
                    min_value=5,
                    max_value=300,
                    value=30,
                    key="new_widget_refresh_interval"
                )
            
            if st.button("➕ Add Widget", key="add_widget"):
                if widget_id and widget_title:
                    self.create_widget(
                        widget_id=widget_id,
                        title=widget_title,
                        widget_type=widget_type,
                        data_source=data_source,
                        refresh_interval=refresh_interval
                    )
                    st.success(f"Widget '{widget_title}' added!")
                    st.rerun()
        
        # Display existing widgets
        if st.session_state.dashboard_widgets:
            st.markdown("#### 📈 Active Widgets")
            
            for widget_id, widget in st.session_state.dashboard_widgets.items():
                with st.expander(f"{widget.title} ({widget.widget_type})", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Widget configuration
                        if widget.widget_type == "chart":
                            chart_type = st.selectbox(
                                "Chart Type",
                                options=["line", "bar", "scatter", "pie"],
                                index=["line", "bar", "scatter", "pie"].index(
                                    widget.config.get("chart_type", "line")
                                ),
                                key=f"chart_type_{widget_id}"
                            )
                            widget.config["chart_type"] = chart_type
                            
                            # Column selection for charts
                            if data_source in st.session_state.get("datasets", {}):
                                df = st.session_state.datasets[data_source]["df"]
                                columns = list(df.columns)
                                
                                x_col = st.selectbox(
                                    "X Column",
                                    options=columns,
                                    index=columns.index(widget.config.get("x_column", columns[0]))
                                    if columns else 0,
                                    key=f"x_col_{widget_id}"
                                )
                                y_col = st.selectbox(
                                    "Y Column",
                                    options=columns,
                                    index=columns.index(widget.config.get("y_column", columns[1]))
                                    if len(columns) > 1 else 0,
                                    key=f"y_col_{widget_id}"
                                )
                                
                                widget.config["x_column"] = x_col
                                widget.config["y_column"] = y_col
                    
                        # Refresh interval
                        widget.refresh_interval = st.number_input(
                            "Refresh Interval (seconds)",
                            min_value=5,
                            max_value=300,
                            value=widget.refresh_interval,
                            key=f"refresh_{widget_id}"
                        )
                    
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_{widget_id}"):
                            if widget_id in st.session_state.dashboard_widgets:
                                del st.session_state.dashboard_widgets[widget_id]
                                st.success(f"Widget '{widget.title}' removed!")
                                st.rerun()
        
        else:
            st.info("No widgets configured. Add your first widget above!")
    
    def render_dashboard(self):
        """Render the real-time dashboard."""
        if not st.session_state.dashboard_widgets:
            st.info("No widgets configured. Go to Dashboard Configuration to add widgets.")
            return
        
        st.markdown("### 📊 Real-Time Dashboard")
        
        # Auto-refresh indicator
        if st.session_state.dashboard_settings.get("auto_refresh", False):
            st.success("🔄 Auto-refresh is active")
        
        # Render widgets in a grid
        widgets = list(st.session_state.dashboard_widgets.values())
        
        # Arrange widgets in columns (max 3 per row)
        for i in range(0, len(widgets), 3):
            cols = st.columns(3)
            for j in range(3):
                widget_index = i + j
                if widget_index < len(widgets):
                    with cols[j]:
                        self.render_widget(widgets[widget_index])
                else:
                    with cols[j]:
                        st.empty()


# Global real-time dashboard manager
_realtime_dashboard_manager = RealtimeDashboardManager()


def get_realtime_dashboard_manager() -> RealtimeDashboardManager:
    """Get the global real-time dashboard manager."""
    return _realtime_dashboard_manager


def render_realtime_dashboard():
    """Render the real-time dashboard interface."""
    manager = get_realtime_dashboard_manager()
    
    # Tab interface
    tab1, tab2 = st.tabs(["📊 Dashboard", "⚙️ Configuration"])
    
    with tab1:
        manager.render_dashboard()
    
    with tab2:
        manager.render_dashboard_editor()
