"""
UDIP v3 — modules/ui_stability_optimizer.py
UI stability and responsiveness optimization utilities.

This module provides tools to eliminate excessive reruns, optimize widget stability,
and improve Streamlit UI performance.
"""

import streamlit as st
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RerunOptimizer:
    """Optimizes Streamlit rerun behavior to prevent excessive reruns."""
    
    def __init__(self):
        self.rerun_count = 0
        self.rerun_threshold = 3  # Maximum reruns per session
        self.last_rerun_time = 0
        self.rerun_cooldown = 1.0  # Seconds between reruns
    
    def should_rerun(self, reason: str = "") -> bool:
        """Determine if rerun should be allowed."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_rerun_time < self.rerun_cooldown:
            logger.warning(f"Rerun blocked by cooldown: {reason}")
            return False
        
        # Check threshold
        if self.rerun_count >= self.rerun_threshold:
            logger.warning(f"Rerun blocked by threshold: {reason}")
            return False
        
        # Allow rerun
        self.rerun_count += 1
        self.last_rerun_time = current_time
        logger.info(f"Rerun allowed: {reason} (count: {self.rerun_count})")
        return True
    
    def reset_counter(self):
        """Reset rerun counter."""
        self.rerun_count = 0
        logger.info("Rerun counter reset")


class WidgetStabilizer:
    """Stabilizes widget behavior and prevents UI flickering."""
    
    def __init__(self):
        self.widget_states = {}
        self.state_snapshots = {}
    
    def capture_widget_state(self, widget_key: str, widget_func: Callable) -> Any:
        """Capture widget state before rendering."""
        # Store current state
        if widget_key in st.session_state:
            self.widget_states[widget_key] = st.session_state[widget_key]
        
        # Create snapshot for comparison
        snapshot_key = f"snapshot_{widget_key}_{int(time.time())}"
        self.state_snapshots[snapshot_key] = {
            "state": self.widget_states.get(widget_key),
            "timestamp": time.time()
        }
        
        # Render widget with stabilization
        try:
            result = widget_func()
            
            # Check if state changed significantly
            if widget_key in st.session_state:
                current_state = st.session_state[widget_key]
                previous_state = self.widget_states.get(widget_key)
                
                # Only trigger rerun if state actually changed meaningfully
                if self._is_meaningful_change(previous_state, current_state):
                    logger.info(f"Widget state changed meaningfully: {widget_key}")
                    return result
            else:
                logger.debug(f"Widget state stable: {widget_key}")
                return result
                
        except Exception as e:
            logger.error(f"Widget stabilization error for {widget_key}: {e}")
            return None
    
    def _is_meaningful_change(self, previous: Any, current: Any) -> bool:
        """Determine if state change is meaningful."""
        # Simple heuristic: ignore minor changes
        if previous is None or current is None:
            return True
        
        # For strings, check if actually different
        if isinstance(previous, str) and isinstance(current, str):
            # Ignore whitespace changes
            if previous.strip() != current.strip():
                return True
        
        # For numbers, check if difference is significant
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            if abs(previous - current) > 0.01:  # Small threshold
                return True
        
        return False


class LayoutOptimizer:
    """Optimizes layout rendering for better performance."""
    
    def __init__(self):
        self.layout_cache = {}
        self.render_times = {}
    
    def optimized_columns(self, column_specs: List[Dict[str, Any]]) -> Any:
        """Create optimized columns layout."""
        # Use caching to avoid re-creating columns
        cache_key = f"columns_{hash(str(column_specs))}"
        
        if cache_key in self.layout_cache:
            logger.debug(f"Using cached layout: {cache_key}")
            return self.layout_cache[cache_key]
        
        start_time = time.time()
        
        # Create columns with optimization
        try:
            if len(column_specs) == 1:
                result = st.container()
            elif len(column_specs) == 2:
                result = st.columns([spec.get("width", 1) for spec in column_specs])
            elif len(column_specs) == 3:
                result = st.columns([spec.get("width", 1) for spec in column_specs])
            else:
                # For more columns, use nested layout
                result = st.container()
                with result:
                    inner_cols = st.columns([spec.get("width", 1) for spec in column_specs])
                    for i, (col, spec) in enumerate(zip(inner_cols, column_specs)):
                        with col:
                            spec["content"]()
            
            # Cache the result
            self.layout_cache[cache_key] = result
            
            render_time = time.time() - start_time
            self.render_times[cache_key] = render_time
            
            logger.debug(f"Layout rendered in {render_time:.3f}s: {cache_key}")
            return result
            
        except Exception as e:
            logger.error(f"Layout optimization error: {e}")
            return st.container()
    
    def optimized_expander(self, title: str, content_func: Callable, expanded: bool = False) -> Any:
        """Create optimized expander."""
        cache_key = f"expander_{hash(title)}"
        
        if cache_key in self.layout_cache:
            logger.debug(f"Using cached expander: {cache_key}")
            return self.layout_cache[cache_key]
        
        start_time = time.time()
        
        try:
            with st.expander(title, expanded=expanded):
                result = content_func()
            
            self.layout_cache[cache_key] = result
            render_time = time.time() - start_time
            self.render_times[cache_key] = render_time
            
            logger.debug(f"Expander rendered in {render_time:.3f}s: {title}")
            return result
            
        except Exception as e:
            logger.error(f"Expander optimization error: {e}")
            return st.expander(title, expanded=expanded)(content_func())


class StateTransitionManager:
    """Manages controlled state transitions to prevent excessive reruns."""
    
    def __init__(self):
        self.transitions = {}
        self.transition_history = []
    
    def controlled_transition(self, from_state: str, to_state: str, transition_func: Callable, reason: str = "") -> bool:
        """Execute controlled state transition."""
        transition_key = f"{from_state}_to_{to_state}"
        
        # Check if already transitioning
        if transition_key in self.transitions:
            logger.warning(f"Transition already in progress: {transition_key}")
            return False
        
        # Execute transition
        try:
            self.transitions[transition_key] = {
                "start_time": time.time(),
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason
            }
            
            result = transition_func()
            
            # Record successful transition
            self.transitions.pop(transition_key, None)
            self.transition_history.append({
                "from_state": from_state,
                "to_state": to_state,
                "timestamp": time.time(),
                "duration": time.time() - self.transitions[transition_key]["start_time"],
                "success": True
            })
            
            logger.info(f"State transition successful: {from_state} -> {to_state}")
            return True
            
        except Exception as e:
            # Record failed transition
            if transition_key in self.transitions:
                self.transitions[transition_key]["success"] = False
                self.transitions[transition_key]["error"] = str(e)
            
            self.transition_history.append({
                "from_state": from_state,
                "to_state": to_state,
                "timestamp": time.time(),
                "success": False,
                "error": str(e)
            })
            
            logger.error(f"State transition failed: {from_state} -> {to_state}: {e}")
            return False
    
    def get_transition_history(self) -> List[Dict[str, Any]]:
        """Get transition history."""
        return self.transition_history
    
    def clear_history(self):
        """Clear transition history."""
        self.transition_history.clear()
        logger.info("Transition history cleared")


class PerformanceMonitor:
    """Monitors UI performance and identifies bottlenecks."""
    
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            "render_time": 0.5,  # seconds
            "memory_growth": 10,  # MB
            "widget_count": 100
        }
    
    def start_render_timer(self, component: str) -> str:
        """Start timing component render."""
        timer_id = f"{component}_{int(time.time() * 1000)}"
        self.metrics[timer_id] = {"start": time.time()}
        return timer_id
    
    def end_render_timer(self, timer_id: str) -> float:
        """End timing component render."""
        if timer_id in self.metrics:
            duration = time.time() - self.metrics[timer_id]["start"]
            self.metrics[timer_id]["duration"] = duration
            self.metrics[timer_id]["end"] = time.time()
            
            # Check threshold
            if duration > self.thresholds["render_time"]:
                logger.warning(f"Slow render detected: {timer_id} took {duration:.3f}s")
            
            return duration
        return 0.0
    
    def monitor_memory_growth(self) -> Dict[str, Any]:
        """Monitor session state memory growth."""
        try:
            import sys
            session_size = len(str(st.session_state))
            
            if session_size > self.thresholds["memory_growth"]:
                logger.warning(f"Large session state detected: {session_size} characters")
            
            return {
                "session_size": session_size,
                "warning": session_size > self.thresholds["memory_growth"]
            }
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return {
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "memory_status": self.monitor_memory_growth()
        }


# Global instances
rerun_optimizer = RerunOptimizer()
widget_stabilizer = WidgetStabilizer()
layout_optimizer = LayoutOptimizer()
state_transition_manager = StateTransitionManager()
performance_monitor = PerformanceMonitor()


def get_rerun_optimizer() -> RerunOptimizer:
    """Get global rerun optimizer."""
    return rerun_optimizer


def get_widget_stabilizer() -> WidgetStabilizer:
    """Get global widget stabilizer."""
    return widget_stabilizer


def get_layout_optimizer() -> LayoutOptimizer:
    """Get global layout optimizer."""
    return layout_optimizer


def get_state_transition_manager() -> StateTransitionManager:
    """Get global state transition manager."""
    return state_transition_manager


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return performance_monitor


def optimize_ui_rendering(component_name: str, render_func: Callable) -> Any:
    """
    Optimize UI rendering with performance monitoring.
    
    Args:
        component_name: Name of the component being rendered
        render_func: Function that renders the component
        
    Returns:
        Result of render function with optimization applied
    """
    monitor = get_performance_monitor()
    
    # Start timing
    timer_id = monitor.start_render_timer(component_name)
    
    try:
        # Execute render function
        result = render_func()
        
        # End timing
        duration = monitor.end_render_timer(timer_id)
        
        # Log performance
        if duration > 0.1:  # Only log if significant
            logger.info(f"Component {component_name} rendered in {duration:.3f}s")
        
        return result
        
    except Exception as e:
        monitor.end_render_timer(timer_id)
        logger.error(f"Component {component_name} render failed: {e}")
        return None


def safe_rerun(reason: str = "") -> bool:
    """
    Safely trigger rerun with optimization.
    
    Args:
        reason: Reason for rerun
        
    Returns:
        True if rerun was triggered, False otherwise
    """
    optimizer = get_rerun_optimizer()
    return optimizer.should_rerun(reason)


def stabilize_widget(widget_key: str, widget_func: Callable) -> Any:
    """
    Stabilize widget rendering.
    
    Args:
        widget_key: Key for the widget in session state
        widget_func: Function that renders the widget
        
    Returns:
        Stabilized widget result
    """
    stabilizer = get_widget_stabilizer()
    return stabilizer.capture_widget_state(widget_key, widget_func)


def create_optimized_layout(column_specs: List[Dict[str, Any]]) -> Any:
    """
    Create optimized layout with caching.
    
    Args:
        column_specs: List of column specifications
        
    Returns:
        Optimized layout object
    """
    optimizer = get_layout_optimizer()
    return optimizer.optimized_columns(column_specs)


def create_optimized_expander(title: str, content_func: Callable, expanded: bool = False) -> Any:
    """
    Create optimized expander with caching.
    
    Args:
        title: Expander title
        content_func: Function that renders content
        expanded: Whether expander should be expanded
        
    Returns:
        Optimized expander object
    """
    optimizer = get_layout_optimizer()
    return optimizer.optimized_expander(title, content_func, expanded)
