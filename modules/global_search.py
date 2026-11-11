"""
UDIP v3 — modules/global_search.py
Global search functionality for datasets, features, and pages.
"""

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class SearchResult:
    """Represents a search result."""
    title: str
    description: str
    type: str  # "dataset", "page", "feature", "chart"
    action: str  # What happens when clicked
    metadata: Dict[str, Any] = None


class GlobalSearchManager:
    """Manages global search functionality."""
    
    def __init__(self):
        if "search_history" not in st.session_state:
            st.session_state.search_history = []
        if "search_index" not in st.session_state:
            st.session_state.search_index = {}
    
    def index_datasets(self):
        """Index all loaded datasets for search."""
        datasets_index = []
        
        for ds_id, ds in st.session_state.datasets.items():
            df = ds["df"]
            
            # Index dataset metadata
            datasets_index.append({
                "id": ds_id,
                "name": ds["name"],
                "type": "dataset",
                "description": f"{ds['name']} ({len(df):,} rows × {len(df.columns)} columns)",
                "action": f"navigate_dataset_{ds_id}",
                "keywords": [ds["name"].lower()] + list(df.columns.str.lower()),
                "metadata": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "format": ds.get("format", "unknown"),
                    "source": ds.get("source", "unknown")
                }
            })
            
            # Index columns
            for col in df.columns:
                col_type = str(df[col].dtype)
                datasets_index.append({
                    "id": f"{ds_id}_{col}",
                    "name": f"{ds['name']} - {col}",
                    "type": "column",
                    "description": f"Column '{col}' ({col_type}) in {ds['name']}",
                    "action": f"explore_column_{ds_id}_{col}",
                    "keywords": [col.lower(), ds["name"].lower(), col_type.lower()],
                    "metadata": {
                        "dataset": ds_id,
                        "column": col,
                        "dtype": col_type,
                        "dataset_name": ds["name"]
                    }
                })
        
        return datasets_index
    
    def index_pages(self):
        """Index all available pages for search."""
        from modules.navigation import NAV_SECTIONS
        
        pages_index = []
        for section_name, routes in NAV_SECTIONS.items():
            for route in routes:
                pages_index.append({
                    "id": route.label,
                    "name": route.label,
                    "type": "page",
                    "description": f"{route.label} - {section_name}",
                    "action": f"navigate_page_{route.label}",
                    "keywords": [route.label.lower(), section_name.lower()],
                    "metadata": {
                        "section": section_name,
                        "module": route.module
                    }
                })
        
        return pages_index
    
    def index_features(self):
        """Index available features and tools."""
        features_index = [
            {
                "id": "ai_assistant",
                "name": "AI Assistant",
                "type": "feature",
                "description": "Conversational AI for data analysis and code generation",
                "action": "navigate_page_🤖 AI Assistant",
                "keywords": ["ai", "assistant", "chat", "llm", "groq"],
                "metadata": {"category": "AI Systems"}
            },
            {
                "id": "data_forge",
                "name": "Data Forge",
                "type": "feature", 
                "description": "AI-powered data cleaning and transformation tools",
                "action": "navigate_page_⚒️ Data Lab",
                "keywords": ["clean", "transform", "forge", "prepare"],
                "metadata": {"category": "Platform"}
            },
            {
                "id": "visualize",
                "name": "Visualize",
                "type": "feature",
                "description": "50+ chart types with professional styling",
                "action": "navigate_page_📊 Visualize", 
                "keywords": ["chart", "graph", "plot", "visualize", "dashboard"],
                "metadata": {"category": "Platform"}
            },
            {
                "id": "merge",
                "name": "Merge Wizard",
                "type": "feature",
                "description": "Visual multi-step dataset joining tool",
                "action": "navigate_page_🔗 Merge",
                "keywords": ["merge", "join", "combine", "union"],
                "metadata": {"category": "Platform"}
            },
            {
                "id": "sql_editor",
                "name": "SQL Editor",
                "type": "feature",
                "description": "Query datasets with standard SQL using DuckDB",
                "action": "navigate_page_SQL Editor",
                "keywords": ["sql", "query", "database", "duckdb"],
                "metadata": {"category": "Development"}
            },
            {
                "id": "code_editor",
                "name": "Code Editor",
                "type": "feature",
                "description": "Python IDE with syntax highlighting and execution",
                "action": "navigate_page_🐍 Code Editor",
                "keywords": ["code", "python", "editor", "ide", "script"],
                "metadata": {"category": "Development"}
            },
            {
                "id": "hirax_inspector",
                "name": "HIRAX Inspector",
                "type": "feature",
                "description": "Specialized HDF5 data exploration for radio astronomy",
                "action": "navigate_page_🔭 HIRAX Inspector",
                "keywords": ["hirax", "hdf5", "radio", "astronomy", "cosmic"],
                "metadata": {"category": "Cosmic Signals Lab"}
            }
        ]
        
        return features_index
    
    def build_search_index(self):
        """Build the complete search index."""
        if time.time() - st.session_state.get("last_index_time", 0) > 300:  # Rebuild every 5 minutes
            datasets = self.index_datasets()
            pages = self.index_pages()
            features = self.index_features()
            
            st.session_state.search_index = {
                "datasets": datasets,
                "pages": pages,
                "features": features
            }
            st.session_state.last_index_time = time.time()
    
    def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """Perform global search across all indexed content."""
        if not query.strip():
            return []
        
        self.build_search_index()
        
        query_lower = query.lower()
        results = []
        
        # Search all categories
        for category, items in st.session_state.search_index.items():
            for item in items:
                # Calculate relevance score
                score = 0
                
                # Exact name match
                if query_lower == item["name"].lower():
                    score += 100
                # Name contains query
                elif query_lower in item["name"].lower():
                    score += 50
                
                # Keywords match
                for keyword in item["keywords"]:
                    if query_lower == keyword:
                        score += 30
                    elif query_lower in keyword:
                        score += 15
                
                # Description match
                if query_lower in item["description"].lower():
                    score += 10
                
                if score > 0:
                    results.append(SearchResult(
                        title=item["name"],
                        description=item["description"],
                        type=item["type"],
                        action=item["action"],
                        metadata=item["metadata"]
                    ))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: -self._calculate_relevance(x, query_lower))
        
        # Add to search history
        if query not in [h["query"] for h in st.session_state.search_history[-10:]]:
            st.session_state.search_history.append({
                "query": query,
                "timestamp": time.time()
            })
        
        return results[:max_results]
    
    def _calculate_relevance(self, result: SearchResult, query: str) -> int:
        """Calculate relevance score for a result."""
        score = 0
        
        if query in result.title.lower():
            score += 50
        if query in result.description.lower():
            score += 20
        
        # Boost certain types
        if result.type == "dataset":
            score += 10
        elif result.type == "page":
            score += 5
        
        return score
    
    def execute_action(self, action: str):
        """Execute a search result action."""
        if action.startswith("navigate_page_"):
            page_name = action.replace("navigate_page_", "")
            st.session_state.active_page = page_name
            st.rerun()
        
        elif action.startswith("navigate_dataset_"):
            ds_id = action.replace("navigate_dataset_", "")
            st.session_state._preview_ds = ds_id
            st.session_state.active_page = "📁 Upload Data"
            st.rerun()
        
        elif action.startswith("explore_column_"):
            parts = action.replace("explore_column_", "").split("_")
            ds_id, column = parts[0], "_".join(parts[1:])
            st.session_state._explore_ds = ds_id
            st.session_state._explore_col = column
            st.session_state.active_page = "⚒️ Data Lab"
            st.rerun()


# Global search manager instance
_search_manager = GlobalSearchManager()


def render_global_search():
    """Render the global search interface."""
    # Toggle search visibility
    if "search_visible" not in st.session_state:
        st.session_state.search_visible = False
    
    if not st.session_state.search_visible:
        return
    
    st.markdown("### 🔍 Global Search")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search datasets, pages, features...",
            key="global_search_input",
            placeholder="Type to search...",
            help="Use Ctrl+/ to toggle search"
        )
    
    with col2:
        if st.button("✕", key="close_search", help="Close search"):
            st.session_state.search_visible = False
            st.rerun()
    
    # Perform search
    if search_query:
        with st.spinner("Searching..."):
            results = _search_manager.search(search_query)
        
        if results:
            st.markdown(f"**Found {len(results)} results for '{search_query}'**")
            
            # Group results by type
            by_type = {}
            for result in results:
                if result.type not in by_type:
                    by_type[result.type] = []
                by_type[result.type].append(result)
            
            # Display results
            for result_type, type_results in by_type.items():
                st.markdown(f"#### {result_type.title()}s")
                
                for result in type_results:
                    col_result, col_action = st.columns([4, 1])
                    
                    with col_result:
                        st.markdown(f"""
                        <div style="
                            padding: 12px;
                            border: 1px solid #e1e5e9;
                            border-radius: 8px;
                            margin: 4px 0;
                            background: #f8f9fa;
                        ">
                            <div style="font-weight: 600; color: #0f766e;">{result.title}</div>
                            <div style="font-size: 0.9em; color: #6b7280; margin-top: 4px;">{result.description}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_action:
                        if st.button("→", key=f"action_{result.action}", help="Open"):
                            _search_manager.execute_action(result.action)
            
        else:
            st.info(f"No results found for '{search_query}'")
            
            # Show suggestions
            st.markdown("**Suggestions:**")
            st.markdown("- Try different keywords")
            st.markdown("- Check spelling") 
            st.markdown("- Use broader terms")
    
    # Search history
    if st.session_state.search_history:
        with st.expander("Recent Searches", expanded=False):
            for search_item in reversed(st.session_state.search_history[-5:]):
                if st.button(search_item["query"], key=f"history_{search_item['timestamp']}"):
                    st.session_state.global_search_input = search_item["query"]
                    st.rerun()


def get_search_manager() -> GlobalSearchManager:
    """Get the global search manager instance."""
    return _search_manager
