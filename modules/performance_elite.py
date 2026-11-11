"""
UDIP v3 — modules/performance_elite.py
Elite performance optimization for maximum responsiveness.

This module provides advanced performance optimization with lazy loading,
adaptive caching, memory management, and benchmarking.
"""

import time
import logging
import functools
import threading
import weakref
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None


class AdaptiveCache:
    """Adaptive caching system with multiple policies."""
    
    def __init__(self, max_size: int = 1000, default_policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.default_policy = default_policy
        self.cache = {}
        self.access_order = []
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self.cache_stats["total_requests"] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl_seconds and (time.time() - entry.created_at) > entry.ttl_seconds:
                    self._evict_key(key)
                    return None
                
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Move to end of access order for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.cache_stats["hits"] += 1
                return entry.value
            else:
                self.cache_stats["misses"] += 1
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """Put value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))
            
            # Check if eviction needed
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )
            
            self.cache[key] = entry
            
            # Update access order
            if key not in self.access_order:
                self.access_order.append(key)
            else:
                self.access_order.remove(key)
                self.access_order.append(key)
    
    def _evict_key(self, key: str):
        """Evict specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.cache_stats["evictions"] += 1
            
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order[0]
            self._evict_key(lru_key)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.cache_stats["evictions"] += len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["total_requests"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "evictions": self.cache_stats["evictions"],
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class LazyLoader:
    """Lazy loading system for expensive operations."""
    
    def __init__(self):
        self.loaders = {}
        self.loaded_objects = {}
        self.load_stats = {}
    
    def register_loader(self, key: str, loader_func: Callable, dependencies: List[str] = None):
        """Register a lazy loader."""
        self.loaders[key] = {
            "loader": loader_func,
            "dependencies": dependencies or [],
            "loaded": False,
            "load_time": 0,
            "access_count": 0
        }
        logger.info(f"Registered lazy loader: {key}")
    
    def get(self, key: str) -> Any:
        """Get object using lazy loading."""
        if key not in self.loaders:
            logger.error(f"Lazy loader not found: {key}")
            return None
        
        loader_info = self.loaders[key]
        
        # Load if not already loaded
        if not loader_info["loaded"]:
            start_time = time.time()
            
            # Check dependencies
            for dep in loader_info["dependencies"]:
                if dep not in self.loaded_objects:
                    self.get(dep)  # Load dependency
            
            # Load the object
            try:
                obj = loader_info["loader"]()
                self.loaded_objects[key] = obj
                loader_info["loaded"] = True
                loader_info["load_time"] = time.time() - start_time
                
                logger.info(f"Lazy loaded: {key} in {loader_info['load_time']:.3f}s")
                
            except Exception as e:
                logger.error(f"Lazy loading failed for {key}: {e}")
                return None
        
        # Update access stats
        loader_info["access_count"] += 1
        
        return self.loaded_objects.get(key)
    
    def preload(self, keys: List[str]):
        """Preload specified keys."""
        logger.info(f"Preloading lazy objects: {keys}")
        
        for key in keys:
            self.get(key)
    
    def unload(self, key: str):
        """Unload a specific object."""
        if key in self.loaded_objects:
            del self.loaded_objects[key]
        
        if key in self.loaders:
            self.loaders[key]["loaded"] = False
        
        logger.info(f"Unloaded: {key}")
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get lazy loading statistics."""
        return {
            "registered_loaders": len(self.loaders),
            "loaded_objects": len(self.loaded_objects),
            "loaders": {
                key: {
                    "loaded": info["loaded"],
                    "load_time": info["load_time"],
                    "access_count": info["access_count"]
                }
                for key, info in self.loaders.items()
            }
        }


class MemoryOptimizer:
    """Advanced memory optimization and monitoring."""
    
    def __init__(self):
        self.memory_pools = {}
        self.allocation_stats = {}
        self.gc_stats = {
            "collections": 0,
            "collection_time": 0.0,
            "memory_freed": 0
        }
        self.memory_thresholds = {
            "warning": 100 * 1024 * 1024,  # 100MB
            "critical": 500 * 1024 * 1024   # 500MB
        }
    
    def allocate_pool(self, pool_name: str, size_bytes: int) -> bool:
        """Allocate memory pool."""
        if pool_name in self.memory_pools:
            logger.warning(f"Memory pool already exists: {pool_name}")
            return False
        
        try:
            # Create memory pool
            pool = bytearray(size_bytes)
            self.memory_pools[pool_name] = {
                "pool": pool,
                "size": size_bytes,
                "allocated_at": time.time(),
                "in_use": True
            }
            
            logger.info(f"Allocated memory pool: {pool_name} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to allocate memory pool {pool_name}: {e}")
            return False
    
    def deallocate_pool(self, pool_name: str):
        """Deallocate memory pool."""
        if pool_name in self.memory_pools:
            pool_info = self.memory_pools[pool_name]
            pool_info["in_use"] = False
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            
            self.gc_stats["collections"] += 1
            self.gc_stats["collection_time"] += time.time()
            self.gc_stats["memory_freed"] = pool_info["size"]
            
            del self.memory_pools[pool_name]
            
            logger.info(f"Deallocated memory pool: {pool_name}, collected {collected} objects")
    
    def optimize_dataframe_memory(self, df) -> Dict[str, Any]:
        """Optimize DataFrame memory usage."""
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return {"error": "Object is not a DataFrame"}
            
            memory_usage = df.memory_usage(deep=True).sum()
            
            optimizations = []
            
            # Optimize data types
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to category if low cardinality
                    unique_count = df[col].nunique()
                    if unique_count < len(df) * 0.5:  # Less than 50% unique
                        df[col] = df[col].astype('category')
                        optimizations.append(f"Converted {col} to category")
            
            # Optimize numeric dtypes
            for col in df.select_dtypes(include=['int64']).columns:
                if df[col].min() >= 0 and df[col].max() < 255:
                    df[col] = df[col].astype('int32')
                    optimizations.append(f"Converted {col} to int32")
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
                optimizations.append(f"Downcast {col} to float32")
            
            new_memory_usage = df.memory_usage(deep=True).sum()
            memory_saved = memory_usage - new_memory_usage
            
            return {
                "original_memory_mb": memory_usage / 1024 / 1024,
                "optimized_memory_mb": new_memory_usage / 1024 / 1024,
                "memory_saved_mb": memory_saved / 1024 / 1024,
                "optimizations_applied": optimizations,
                "memory_reduction_percent": (memory_saved / memory_usage) * 100 if memory_usage > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"DataFrame optimization failed: {e}")
            return {"error": str(e)}
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "pools_allocated": len(self.memory_pools),
                "gc_collections": self.gc_stats["collections"],
                "gc_memory_freed_mb": self.gc_stats["memory_freed"] / 1024 / 1024,
                "thresholds": self.memory_thresholds
            }
            
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}


class PerformanceProfiler:
    """Advanced performance profiling and benchmarking."""
    
    def __init__(self):
        self.profiles = {}
        self.benchmarks = {}
        self.active_profiles = {}
        
    def start_profile(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.active_profiles[profile_id] = {
            "operation": operation_name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": self._get_cpu_usage()
        }
        
        logger.debug(f"Started profiling: {profile_id}")
        return profile_id
    
    def end_profile(self, profile_id: str, success: bool = True, error_message: str = None) -> PerformanceMetrics:
        """End profiling and collect metrics."""
        if profile_id not in self.active_profiles:
            logger.error(f"Profile not found: {profile_id}")
            return PerformanceMetrics(
                operation_name="unknown",
                start_time=0,
                end_time=0,
                duration=0,
                memory_before=0,
                memory_after=0,
                cpu_usage=0,
                success=False,
                error_message="Profile not found"
            )
        
        profile_data = self.active_profiles[profile_id]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()
        
        metrics = PerformanceMetrics(
            operation_name=profile_data["operation"],
            start_time=profile_data["start_time"],
            end_time=end_time,
            duration=end_time - profile_data["start_time"],
            memory_before=profile_data["start_memory"],
            memory_after=end_memory,
            cpu_usage=end_cpu,
            success=success,
            error_message=error_message
        )
        
        # Store profile
        self.profiles[profile_id] = metrics
        
        # Update benchmarks
        operation = profile_data["operation"]
        if operation not in self.benchmarks:
            self.benchmarks[operation] = []
        
        self.benchmarks[operation].append(metrics)
        
        # Remove from active profiles
        del self.active_profiles[profile_id]
        
        logger.info(f"Completed profiling: {profile_id} in {metrics.duration:.3f}s")
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and recommendations."""
        summary = {
            "total_profiles": len(self.profiles),
            "operations_benchmarked": len(self.benchmarks),
            "average_duration": 0.0,
            "slowest_operations": [],
            "memory_intensive_operations": [],
            "recommendations": []
        }
        
        if not self.profiles:
            return summary
        
        # Calculate statistics
        durations = [p.duration for p in self.profiles.values()]
        memory_deltas = [p.memory_after - p.memory_before for p in self.profiles.values()]
        
        summary["average_duration"] = sum(durations) / len(durations)
        
        # Find slowest operations
        sorted_profiles = sorted(self.profiles.values(), key=lambda x: x.duration, reverse=True)
        summary["slowest_operations"] = [
            {"operation": p.operation_name, "duration": p.duration} 
            for p in sorted_profiles[:5]
        ]
        
        # Find memory intensive operations
        memory_intensive = sorted(self.profiles.values(), 
                               key=lambda x: x.memory_after - x.memory_before, reverse=True)
        summary["memory_intensive_operations"] = [
            {"operation": p.operation_name, "memory_delta": p.memory_after - p.memory_before}
            for p in memory_intensive[:5]
        ]
        
        # Generate recommendations
        if summary["average_duration"] > 1.0:
            summary["recommendations"].append("Consider optimizing slow operations")
        
        if max(memory_deltas) > 50:  # 50MB threshold
            summary["recommendations"].append("Consider memory optimization for intensive operations")
        
        return summary
    
    def get_operation_benchmark(self, operation_name: str) -> Dict[str, Any]:
        """Get benchmark data for specific operation."""
        if operation_name not in self.benchmarks:
            return {"error": f"No benchmark data for {operation_name}"}
        
        profiles = self.benchmarks[operation_name]
        
        if not profiles:
            return {"error": f"No profiles found for {operation_name}"}
        
        durations = [p.duration for p in profiles]
        memory_deltas = [p.memory_after - p.memory_before for p in profiles]
        
        return {
            "operation": operation_name,
            "sample_count": len(profiles),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "average_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
            "success_rate": sum(1 for p in profiles if p.success) / len(profiles)
        }


# Global instances
adaptive_cache = AdaptiveCache()
lazy_loader = LazyLoader()
memory_optimizer = MemoryOptimizer()
performance_profiler = PerformanceProfiler()


def get_adaptive_cache() -> AdaptiveCache:
    """Get global adaptive cache."""
    return adaptive_cache


def get_lazy_loader() -> LazyLoader:
    """Get global lazy loader."""
    return lazy_loader


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer."""
    return memory_optimizer


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    return performance_profiler


def performance_monitor(operation_name: str):
    """Decorator for performance monitoring."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            profile_id = profiler.start_profile(operation_name)
            
            try:
                result = func(*args, **kwargs)
                profiler.end_profile(profile_id, success=True)
                return result
            except Exception as e:
                profiler.end_profile(profile_id, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator


def cached(ttl_seconds: Optional[float] = None, cache_key_func: Optional[Callable] = None):
    """Decorator for adaptive caching."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_adaptive_cache()
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def lazy_load(key: str, dependencies: List[str] = None):
    """Decorator for lazy loading."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            loader = get_lazy_loader()
            return loader.get(key)
        
        # Register the loader
        loader = get_lazy_loader()
        loader.register_loader(key, func, dependencies)
        
        return wrapper
    return decorator


def initialize_performance_elite():
    """Initialize elite performance optimization system."""
    logger.info("Initializing elite performance optimization system")
    
    # Preload critical components
    cache = get_adaptive_cache()
    loader = get_lazy_loader()
    
    # Preload common data structures
    critical_keys = ["config", "user_preferences", "active_workspace"]
    loader.preload(critical_keys)
    
    logger.info("Elite performance optimization system initialized")
