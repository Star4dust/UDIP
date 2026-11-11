"""
UDIP v3 — modules/scalability_manager.py
Scalability management with caching, lazy loading, and resource optimization.

Enhances platform scalability for large datasets and high-performance operations.
"""

import streamlit as st
import time
import hashlib
import pickle
import os
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps, lru_cache
from dataclasses import dataclass, field
import threading
import weakref
import gc
import psutil
import pandas as pd
import numpy as np


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: float = field(default=3600.0)  # 1 hour default


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    memory_mb: float
    cpu_percent: float
    cache_hit_rate: float
    active_caches: int
    total_cached_items: int


class ScalabilityManager:
    """
    Manages platform scalability through intelligent caching, lazy loading,
    and resource optimization.
    """
    
    def __init__(self, max_cache_size_mb: int = 500, max_items: int = 1000):
        """Initialize the scalability manager."""
        self.max_cache_size_mb = max_cache_size_mb
        self.max_items = max_items
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()
        self._lazy_loaders: Dict[str, Callable] = {}
        self._resource_monitors: List[Callable] = []
        
        # Initialize session state for scalability
        if "scalability_metrics" not in st.session_state:
            st.session_state.scalability_metrics = {
                "cache_operations": [],
                "memory_usage": [],
                "performance_benchmarks": {}
            }
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key for function arguments."""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object."""
        try:
            return len(pickle.dumps(obj))
        except:
            return len(str(obj).encode())
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
    
    def _enforce_cache_limits(self):
        """Enforce cache size and item limits."""
        # Remove expired items first
        self._cleanup_expired_cache()
        
        # If still over limits, remove least recently used items
        while len(self._cache) > self.max_items:
            # Find least recently used item (lowest access count, oldest)
            lru_key = min(self._cache.keys(), 
                         key=lambda k: (self._cache[k].access_count, self._cache[k].timestamp))
            del self._cache[lru_key]
        
        # Check memory usage
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        max_size_bytes = self.max_cache_size_mb * 1024 * 1024
        
        while total_size > max_size_bytes and self._cache:
            # Remove largest items first
            largest_key = max(self._cache.keys(), 
                           key=lambda k: self._cache[k].size_bytes)
            total_size -= self._cache[largest_key].size_bytes
            del self._cache[largest_key]
    
    def cached(self, ttl: float = 3600.0, key_prefix: str = ""):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache key
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{key_prefix}{self._generate_cache_key(func.__name__, args, kwargs)}"
                
                with self._lock:
                    # Check cache
                    if cache_key in self._cache:
                        entry = self._cache[cache_key]
                        if time.time() - entry.timestamp < entry.ttl:
                            entry.access_count += 1
                            self._cache_hits += 1
                            
                            # Log cache hit
                            st.session_state.scalability_metrics["cache_operations"].append({
                                "timestamp": time.time(),
                                "operation": "cache_hit",
                                "function": func.__name__,
                                "key": cache_key[:16] + "..."
                            })
                            
                            return entry.data
                
                # Cache miss - compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time
                
                # Cache the result
                with self._lock:
                    entry = CacheEntry(
                        data=result,
                        timestamp=time.time(),
                        access_count=1,
                        size_bytes=self._estimate_size(result),
                        ttl=ttl
                    )
                    self._cache[cache_key] = entry
                    self._cache_misses += 1
                    
                    # Enforce limits
                    self._enforce_cache_limits()
                    
                    # Log cache miss
                    st.session_state.scalability_metrics["cache_operations"].append({
                        "timestamp": time.time(),
                        "operation": "cache_miss",
                        "function": func.__name__,
                        "computation_time": computation_time,
                        "cache_size": len(self._cache)
                    })
                
                return result
            
            return wrapper
        return decorator
    
    def lazy_load(self, loader_func: Callable, cache_key: str = None) -> Callable:
        """
        Create a lazy-loaded function that caches its result on first call.
        
        Args:
            loader_func: Function to load data
            cache_key: Optional cache key
        """
        if cache_key is None:
            cache_key = f"lazy_{loader_func.__name__}"
        
        def wrapper(*args, **kwargs):
            with self._lock:
                if cache_key not in self._cache:
                    # Load data on first access
                    start_time = time.time()
                    data = loader_func(*args, **kwargs)
                    load_time = time.time() - start_time
                    
                    entry = CacheEntry(
                        data=data,
                        timestamp=time.time(),
                        access_count=1,
                        size_bytes=self._estimate_size(data),
                        ttl=float('inf')  # Lazy loaded data persists
                    )
                    self._cache[cache_key] = entry
                    
                    # Log lazy load
                    st.session_state.scalability_metrics["cache_operations"].append({
                        "timestamp": time.time(),
                        "operation": "lazy_load",
                        "function": loader_func.__name__,
                        "load_time": load_time,
                        "size_bytes": entry.size_bytes
                    })
                
                return self._cache[cache_key].data
        
        return wrapper
    
    def invalidate_cache(self, pattern: str = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match cache keys
        """
        with self._lock:
            if pattern is None:
                # Clear all cache
                self._cache.clear()
            else:
                # Clear entries matching pattern
                keys_to_remove = [key for key in self._cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del self._cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_requests) if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": hit_rate,
                "total_items": len(self._cache),
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_cache_size_mb,
                "utilization_percent": (total_size / (self.max_cache_size_mb * 1024 * 1024)) * 100
            }
    
    def monitor_resources(self) -> ResourceMetrics:
        """Monitor current resource usage."""
        process = psutil.Process()
        
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        cache_stats = self.get_cache_stats()
        
        return ResourceMetrics(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            cache_hit_rate=cache_stats["hit_rate"],
            active_caches=1,  # This manager instance
            total_cached_items=cache_stats["total_items"]
        )
    
    def optimize_memory(self, aggressive: bool = False):
        """
        Optimize memory usage.
        
        Args:
            aggressive: If True, perform aggressive cleanup
        """
        # Clear expired cache entries
        self._cleanup_expired_cache()
        
        # Clear least recently used items if memory is high
        current_memory = self.monitor_resources().memory_mb
        if current_memory > 500 or aggressive:  # 500MB threshold
            # Remove entries with low access counts
            with self._lock:
                items_to_remove = []
                for key, entry in self._cache.items():
                    if entry.access_count < 2 or (aggressive and entry.access_count < 5):
                        items_to_remove.append(key)
                
                for key in items_to_remove:
                    del self._cache[key]
        
        # Force garbage collection
        gc.collect()
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function's performance.
        
        Returns performance metrics.
        """
        # Clear cache to get fresh measurement
        cache_key = f"benchmark_{func.__name__}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Measure performance
        start_time = time.time()
        start_memory = self.monitor_resources().memory_mb
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self.monitor_resources().memory_mb
        
        metrics = {
            "function": func.__name__,
            "execution_time": end_time - start_time,
            "memory_delta_mb": end_memory - start_memory,
            "result_size_bytes": self._estimate_size(result),
            "timestamp": time.time()
        }
        
        # Store benchmark results
        func_name = func.__name__
        if func_name not in st.session_state.scalability_metrics["performance_benchmarks"]:
            st.session_state.scalability_metrics["performance_benchmarks"][func_name] = []
        
        st.session_state.scalability_metrics["performance_benchmarks"][func_name].append(metrics)
        
        # Keep only last 10 benchmarks per function
        benchmarks = st.session_state.scalability_metrics["performance_benchmarks"][func_name]
        if len(benchmarks) > 10:
            st.session_state.scalability_metrics["performance_benchmarks"][func_name] = benchmarks[-10:]
        
        return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.get_cache_stats()
        resource_metrics = self.monitor_resources()
        
        # Analyze recent cache operations
        recent_operations = st.session_state.scalability_metrics["cache_operations"][-100:]
        
        # Calculate average computation times
        cache_miss_times = [op["computation_time"] for op in recent_operations 
                           if op["operation"] == "cache_miss" and "computation_time" in op]
        
        avg_computation_time = np.mean(cache_miss_times) if cache_miss_times else 0.0
        
        # Analyze performance benchmarks
        benchmarks = st.session_state.scalability_metrics["performance_benchmarks"]
        benchmark_summary = {}
        
        for func_name, func_benchmarks in benchmarks.items():
            if func_benchmarks:
                execution_times = [b["execution_time"] for b in func_benchmarks]
                memory_deltas = [b["memory_delta_mb"] for b in func_benchmarks]
                
                benchmark_summary[func_name] = {
                    "avg_execution_time": np.mean(execution_times),
                    "max_execution_time": np.max(execution_times),
                    "avg_memory_delta": np.mean(memory_deltas),
                    "call_count": len(func_benchmarks)
                }
        
        return {
            "cache_performance": cache_stats,
            "resource_usage": {
                "memory_mb": resource_metrics.memory_mb,
                "cpu_percent": resource_metrics.cpu_percent,
                "cached_items": resource_metrics.total_cached_items
            },
            "performance_metrics": {
                "avg_computation_time": avg_computation_time,
                "cache_hit_rate": cache_stats["hit_rate"],
                "total_operations": len(recent_operations)
            },
            "function_benchmarks": benchmark_summary,
            "recommendations": self._generate_recommendations(cache_stats, resource_metrics)
        }
    
    def _generate_recommendations(self, cache_stats: Dict, resource_metrics: ResourceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Cache performance recommendations
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Low cache hit rate. Consider caching more frequently used functions.")
        
        if cache_stats["utilization_percent"] > 80:
            recommendations.append("Cache utilization high. Consider increasing cache size or reducing TTL.")
        
        # Memory usage recommendations
        if resource_metrics.memory_mb > 1000:
            recommendations.append("High memory usage. Consider optimizing data structures or increasing cache cleanup.")
        
        if resource_metrics.cpu_percent > 80:
            recommendations.append("High CPU usage. Consider optimizing algorithms or adding more caching.")
        
        # General recommendations
        if cache_stats["total_items"] > cache_stats["max_items"] * 0.8:
            recommendations.append("Cache approaching item limit. Consider increasing max_items or reducing cache scope.")
        
        return recommendations


# Global instance
_global_scalability_manager: Optional[ScalabilityManager] = None


def get_scalability_manager() -> ScalabilityManager:
    """Get the global scalability manager instance."""
    global _global_scalability_manager
    if _global_scalability_manager is None:
        _global_scalability_manager = ScalabilityManager()
    return _global_scalability_manager


def init_scalability_manager(max_cache_size_mb: int = 500, max_items: int = 1000) -> ScalabilityManager:
    """Initialize the global scalability manager."""
    global _global_scalability_manager
    _global_scalability_manager = ScalabilityManager(max_cache_size_mb, max_items)
    return _global_scalability_manager


# Convenience decorators
def cached(ttl: float = 3600.0, key_prefix: str = ""):
    """Convenience decorator for caching."""
    manager = get_scalability_manager()
    return manager.cached(ttl, key_prefix)


def lazy_load(loader_func: Callable = None, cache_key: str = None):
    """Convenience decorator for lazy loading."""
    manager = get_scalability_manager()
    
    if loader_func is None:
        # Used as @lazy_load(cache_key="...")
        def decorator(func):
            return manager.lazy_load(func, cache_key)
        return decorator
    else:
        # Used as @lazy_load
        return manager.lazy_load(loader_func, cache_key)


def render_scalability_dashboard():
    """Render scalability monitoring dashboard."""
    manager = get_scalability_manager()
    
    st.markdown('<div class="main-header">⚡ Scalability Monitor</div>', unsafe_allow_html=True)
    
    # Get performance report
    report = manager.get_performance_report()
    
    # Key metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cache Hit Rate", f"{report['cache_performance']['hit_rate']:.1%}")
    
    with col2:
        st.metric("Memory Usage", f"{report['resource_usage']['memory_mb']:.1f} MB")
    
    with col3:
        st.metric("Cached Items", report['resource_usage']['cached_items'])
    
    with col4:
        st.metric("Cache Size", f"{report['cache_performance']['total_size_mb']:.1f} MB")
    
    # Cache performance details
    st.markdown("### 🗄️ Cache Performance")
    
    cache_df = pd.DataFrame([report['cache_performance']]).T
    cache_df.columns = ['Value']
    st.dataframe(cache_df, use_container_width=True)
    
    # Function benchmarks
    if report['function_benchmarks']:
        st.markdown("### ⚡ Function Performance")
        
        benchmark_data = []
        for func_name, metrics in report['function_benchmarks'].items():
            benchmark_data.append({
                'Function': func_name,
                'Avg Time (s)': f"{metrics['avg_execution_time']:.3f}",
                'Max Time (s)': f"{metrics['max_execution_time']:.3f}",
                'Avg Memory (MB)': f"{metrics['avg_memory_delta']:.2f}",
                'Calls': metrics['call_count']
            })
        
        df_benchmarks = pd.DataFrame(benchmark_data)
        st.dataframe(df_benchmarks, use_container_width=True)
    
    # Recommendations
    if report['recommendations']:
        st.markdown("### 💡 Performance Recommendations")
        
        for recommendation in report['recommendations']:
            st.warning(recommendation)
    else:
        st.success("✅ Performance is optimal!")
    
    # Cache management controls
    st.markdown("### 🔧 Cache Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear Expired Cache", key="clear_expired"):
            manager._cleanup_expired_cache()
            st.success("Expired cache entries cleared")
    
    with col2:
        if st.button("🗑️ Clear All Cache", key="clear_all"):
            manager.invalidate_cache()
            st.success("All cache entries cleared")
    
    with col3:
        if st.button("⚡ Optimize Memory", key="optimize_memory"):
            manager.optimize_memory()
            st.success("Memory optimization completed")
    
    # Recent operations
    st.markdown("### 📈 Recent Operations")
    
    recent_ops = st.session_state.scalability_metrics["cache_operations"][-20:]
    if recent_ops:
        ops_data = []
        for op in recent_ops:
            ops_data.append({
                'Time': time.ctime(op['timestamp']),
                'Operation': op['operation'],
                'Function': op.get('function', 'N/A'),
                'Details': f"{op.get('computation_time', 0):.3f}s" if 'computation_time' in op else f"{op.get('cache_size', 0)} items"
            })
        
        df_ops = pd.DataFrame(ops_data)
        st.dataframe(df_ops, use_container_width=True)
