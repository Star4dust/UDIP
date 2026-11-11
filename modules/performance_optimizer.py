"""
UDIP v3 — modules/performance_optimizer.py
Large dataset handling and performance optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import psutil


@dataclass
class DatasetConfig:
    """Configuration for dataset optimization."""
    sample_size: Optional[int] = None
    use_polars: bool = False
    chunk_size: int = 10000
    enable_lazy_loading: bool = True
    memory_limit_mb: int = 1024


class PerformanceOptimizer:
    """Optimizes dataset handling for large datasets."""
    
    def __init__(self):
        if "performance_config" not in st.session_state:
            st.session_state.performance_config = DatasetConfig()
        if "dataset_cache" not in st.session_state:
            st.session_state.dataset_cache = {}
    
    def optimize_dataset(self, df: pd.DataFrame, config: Optional[DatasetConfig] = None) -> pd.DataFrame:
        """Optimize a dataset based on size and configuration."""
        config = config or st.session_state.performance_config
        
        # Get dataset size
        memory_mb = self._estimate_memory_usage(df)
        
        # Apply optimizations based on size
        if memory_mb > config.memory_limit_mb:
            return self._apply_sampling(df, config)
        elif config.use_polars and len(df) > 50000:
            return self._apply_polars_optimization(df, config)
        elif len(df) > 100000:
            return self._apply_chunking(df, config)
        else:
            return df
    
    def _estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of DataFrame in MB."""
        memory_bytes = df.memory_usage(deep=True).sum()
        return memory_bytes / (1024 * 1024)
    
    def _apply_sampling(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Apply sampling to reduce dataset size."""
        if config.sample_size and len(df) > config.sample_size:
            # Stratified sampling for better representation
            if len(df.columns) > 0:
                try:
                    # Try stratified sampling on first string column
                    str_cols = df.select_dtypes(include=['object']).columns
                    if len(str_cols) > 0:
                        return df.groupby(str_cols[0]).apply(
                            lambda x: x.sample(min(len(x), config.sample_size // len(df[str_cols[0]].unique())))
                        ).reset_index(drop=True)
                except:
                    pass
            
            # Fallback to random sampling
            return df.sample(n=config.sample_size, random_state=42)
        
        return df
    
    def _apply_polars_optimization(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Apply Polars optimization for large datasets."""
        try:
            import polars as pl
            
            # Convert to Polars for operations
            pl_df = pl.from_pandas(df)
            
            # Optimize data types
            for col_name in pl_df.columns:
                col_type = pl_df[col_name].dtype
                
                # Convert string columns to categorical if low cardinality
                if col_type == pl.Utf8:
                    unique_count = pl_df[col_name].n_unique()
                    if unique_count < 100:  # Threshold for categorical
                        pl_df = pl_df.with_columns([
                            pl.col(col_name).cast(pl.Categorical)
                        ])
            
            # Convert back to pandas with optimizations
            optimized_df = pl_df.to_pandas()
            
            # Apply additional pandas optimizations
            optimized_df = self._optimize_dtypes(optimized_df)
            
            return optimized_df
            
        except ImportError:
            st.warning("Polars not available. Using pandas optimization.")
            return self._optimize_dtypes(df)
    
    def _apply_chunking(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Apply chunking for very large datasets."""
        if config.enable_lazy_loading:
            # Store chunked version info
            st.session_state.current_chunk = 0
            st.session_state.total_chunks = (len(df) - 1) // config.chunk_size + 1
            st.session_state.chunk_size = config.chunk_size
            st.session_state.original_df = df
            
            # Return first chunk
            start_idx = 0
            end_idx = min(config.chunk_size, len(df))
            return df.iloc[start_idx:end_idx].copy()
        
        return df
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas data types for memory efficiency."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # Convert object columns to more efficient types
            if col_type == 'object':
                unique_vals = optimized_df[col].nunique()
                
                # Convert to category if low cardinality
                if unique_vals / len(optimized_df) < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')
                
                # Try datetime conversion
                try:
                    optimized_df[col] = pd.to_datetime(optimized_df[col], errors='ignore')
                except:
                    pass
            
            # Convert int64 to int32 where possible
            elif col_type == 'int64':
                if optimized_df[col].min() >= -2147483648 and optimized_df[col].max() <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
            
            # Convert float64 to float32 where precision allows
            elif col_type == 'float64':
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def get_next_chunk(self) -> Optional[pd.DataFrame]:
        """Get next chunk if using chunked loading."""
        if not hasattr(st.session_state, 'original_df'):
            return None
        
        current_chunk = st.session_state.get('current_chunk', 0)
        total_chunks = st.session_state.get('total_chunks', 1)
        chunk_size = st.session_state.get('chunk_size', 10000)
        
        if current_chunk < total_chunks - 1:
            st.session_state.current_chunk = current_chunk + 1
            start_idx = (current_chunk + 1) * chunk_size
            end_idx = min(start_idx + chunk_size, len(st.session_state.original_df))
            return st.session_state.original_df.iloc[start_idx:end_idx].copy()
        
        return None
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get current chunk information."""
        return {
            "current_chunk": st.session_state.get('current_chunk', 0) + 1,
            "total_chunks": st.session_state.get('total_chunks', 1),
            "chunk_size": st.session_state.get('chunk_size', 10000),
            "total_rows": len(st.session_state.get('original_df', pd.DataFrame())),
            "showing_rows": min(
                st.session_state.get('chunk_size', 10000),
                len(st.session_state.get('original_df', pd.DataFrame())) - st.session_state.get('current_chunk', 0) * st.session_state.get('chunk_size', 10000)
            )
        }
    
    def render_performance_controls(self):
        """Render performance optimization controls."""
        config = st.session_state.performance_config
        
        st.markdown("### ⚡ Performance Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            config.sample_size = st.number_input(
                "Sample Size (0 = no sampling)",
                min_value=0,
                max_value=1000000,
                value=config.sample_size or 0,
                help="Limit dataset size for better performance"
            )
            
            config.chunk_size = st.number_input(
                "Chunk Size",
                min_value=1000,
                max_value=100000,
                value=config.chunk_size,
                help="Size of chunks for large datasets"
            )
        
        with col2:
            config.use_polars = st.checkbox(
                "Use Polars Engine",
                value=config.use_polars,
                help="Use Polars for faster data processing"
            )
            
            config.enable_lazy_loading = st.checkbox(
                "Enable Lazy Loading",
                value=config.enable_lazy_loading,
                help="Load data in chunks for very large datasets"
            )
        
        config.memory_limit_mb = st.slider(
            "Memory Limit (MB)",
            min_value=256,
            max_value=8192,
            value=config.memory_limit_mb,
            help="Maximum memory usage before optimizations kick in"
        )
        
        st.session_state.performance_config = config
    
    def render_chunk_navigation(self):
        """Render chunk navigation controls."""
        if not hasattr(st.session_state, 'original_df'):
            return
        
        chunk_info = self.get_chunk_info()
        
        st.markdown(f"""
        ### 📊 Chunk Navigation
        
        **Showing:** {chunk_info['showing_rows']:,} of {chunk_info['total_rows']:,} rows  
        **Chunk:** {chunk_info['current_chunk']} of {chunk_info['total_chunks']}
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Previous Chunk", disabled=chunk_info['current_chunk'] <= 1):
                prev_chunk = self.get_previous_chunk()
                if prev_chunk is not None:
                    st.session_state.current_page_data = prev_chunk
                    st.rerun()
        
        with col2:
            if st.button("📋 Load Full Dataset"):
                st.session_state.current_page_data = st.session_state.original_df
                # Clear chunking state
                for key in ['current_chunk', 'total_chunks', 'chunk_size', 'original_df']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("Next Chunk ➡️", disabled=chunk_info['current_chunk'] >= chunk_info['total_chunks']):
                next_chunk = self.get_next_chunk()
                if next_chunk is not None:
                    st.session_state.current_page_data = next_chunk
                    st.rerun()
    
    def get_previous_chunk(self) -> Optional[pd.DataFrame]:
        """Get previous chunk."""
        if not hasattr(st.session_state, 'original_df'):
            return None
        
        current_chunk = st.session_state.get('current_chunk', 0)
        chunk_size = st.session_state.get('chunk_size', 10000)
        
        if current_chunk > 0:
            st.session_state.current_chunk = current_chunk - 1
            start_idx = (current_chunk - 1) * chunk_size
            end_idx = min(start_idx + chunk_size, len(st.session_state.original_df))
            return st.session_state.original_df.iloc[start_idx:end_idx].copy()
        
        return None
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        }


# Global performance optimizer instance
_performance_optimizer = PerformanceOptimizer()


def optimize_dataframe(df: pd.DataFrame, config: Optional[DatasetConfig] = None) -> pd.DataFrame:
    """Optimize a DataFrame for better performance."""
    return _performance_optimizer.optimize_dataset(df, config)


def render_performance_panel():
    """Render performance optimization panel."""
    _performance_optimizer.render_performance_controls()


def render_chunk_navigation():
    """Render chunk navigation if chunking is active."""
    _performance_optimizer.render_chunk_navigation()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return _performance_optimizer


def get_system_metrics() -> Dict[str, Any]:
    """Get current system performance metrics."""
    return _performance_optimizer.get_system_performance()


def auto_optimize_settings(df: pd.DataFrame):
    """Automatically configure optimal settings based on dataset size."""
    memory_mb = _performance_optimizer._estimate_memory_usage(df)
    config = st.session_state.performance_config
    
    # Auto-configure based on dataset size
    if memory_mb > 2048:  # > 2GB
        config.sample_size = min(100000, len(df) // 10)
        config.enable_lazy_loading = True
        config.use_polars = True
    elif memory_mb > 1024:  # > 1GB
        config.sample_size = min(500000, len(df) // 5)
        config.use_polars = len(df) > 50000
    elif len(df) > 100000:
        config.chunk_size = 50000
        config.use_polars = True
    
    st.session_state.performance_config = config
