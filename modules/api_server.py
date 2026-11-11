"""
UDIP v3 — modules/api_server.py
REST API server for external integrations.
"""

import streamlit as st
import json
import time
import hashlib
import secrets
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io


@dataclass
class APIKey:
    """API key for external access."""
    key_id: str
    name: str
    permissions: List[str]
    created_at: float
    last_used: float
    usage_count: int = 0
    rate_limit: int = 1000  # requests per hour
    is_active: bool = True


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    path: str
    method: str
    description: str
    permissions_required: List[str]
    rate_limit: int = 100


class APIServer:
    """REST API server for UDIP integrations."""
    
    def __init__(self):
        if "api_keys" not in st.session_state:
            st.session_state.api_keys = {}
        if "api_settings" not in st.session_state:
            st.session_state.api_settings = {
                "enable_api_server": False,
                "port": 8000,
                "host": "localhost",
                "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
                "rate_limiting": True,
                "require_auth": True,
                "log_requests": True,
                "max_request_size": 10485760  # 10MB
            }
        if "api_endpoints" not in st.session_state:
            st.session_state.api_endpoints = self._init_endpoints()
        if "api_logs" not in st.session_state:
            st.session_state.api_logs = []
    
    def _init_endpoints(self) -> List[APIEndpoint]:
        """Initialize available API endpoints."""
        return [
            APIEndpoint(
                path="/datasets",
                method="GET",
                description="List all datasets",
                permissions_required=["read_data"],
                rate_limit=100
            ),
            APIEndpoint(
                path="/datasets/{dataset_id}",
                method="GET",
                description="Get specific dataset",
                permissions_required=["read_data"],
                rate_limit=200
            ),
            APIEndpoint(
                path="/datasets/{dataset_id}",
                method="POST",
                description="Create/update dataset",
                permissions_required=["write_data"],
                rate_limit=50
            ),
            APIEndpoint(
                path="/datasets/{dataset_id}",
                method="DELETE",
                description="Delete dataset",
                permissions_required=["delete_data"],
                rate_limit=20
            ),
            APIEndpoint(
                path="/query",
                method="POST",
                description="Execute SQL query on datasets",
                permissions_required=["execute_code"],
                rate_limit=30
            ),
            APIEndpoint(
                path="/export",
                method="POST",
                description="Export dataset in various formats",
                permissions_required=["read_data"],
                rate_limit=25
            ),
            APIEndpoint(
                path="/charts",
                method="POST",
                description="Generate charts from data",
                permissions_required=["read_data"],
                rate_limit=40
            ),
            APIEndpoint(
                path="/health",
                method="GET",
                description="Get system health status",
                permissions_required=[],
                rate_limit=1000
            )
        ]
    
    def create_api_key(self, name: str, permissions: List[str]) -> str:
        """Create a new API key."""
        key_id = f"key_{int(time.time() * 1000)}"
        api_key = secrets.token_urlsafe(32)
        
        api_key_obj = APIKey(
            key_id=key_id,
            name=name,
            permissions=permissions,
            created_at=time.time(),
            last_used=0,
            usage_count=0,
            is_active=True
        )
        
        st.session_state.api_keys[key_id] = api_key_obj
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key."""
        for key_obj in st.session_state.api_keys.values():
            if key_obj.is_active and secrets.compare_digest(api_key, key_obj.key_id):
                # Update usage
                key_obj.last_used = time.time()
                key_obj.usage_count += 1
                return key_obj
        
        return None
    
    def check_rate_limit(self, api_key_obj: APIKey, endpoint: APIEndpoint) -> bool:
        """Check if API key is within rate limits."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Count requests in last hour
        recent_requests = sum(
            1 for log in st.session_state.api_logs
            if log["api_key_id"] == api_key_obj.key_id and 
               log["timestamp"] > hour_ago
        )
        
        return recent_requests < endpoint.rate_limit
    
    def log_api_request(
        self,
        api_key_id: str,
        endpoint_path: str,
        method: str,
        status_code: int,
        response_size: int = 0,
        duration: float = 0,
        error: Optional[str] = None
    ):
        """Log an API request."""
        log_entry = {
            "timestamp": time.time(),
            "api_key_id": api_key_id,
            "endpoint_path": endpoint_path,
            "method": method,
            "status_code": status_code,
            "response_size": response_size,
            "duration": duration,
            "error": error,
            "ip_address": "api_server"
        }
        
        st.session_state.api_logs.append(log_entry)
        
        # Limit log size
        if len(st.session_state.api_logs) > 10000:
            st.session_state.api_logs = st.session_state.api_logs[-10000:]
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="UDIP API",
            description="REST API for UDIP data integration",
            version="1.0.0"
        )
        
        # Add CORS middleware
        settings = st.session_state.api_settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    def render_api_server_interface(self):
        """Render API server management interface."""
        st.markdown("### 🌐 API Server Configuration")
        
        settings = st.session_state.api_settings
        
        # Server settings
        col1, col2 = st.columns(2)
        with col1:
            settings["enable_api_server"] = st.checkbox(
                "Enable API Server",
                value=settings["enable_api_server"],
                help="Start/stop the REST API server"
            )
            
            settings["port"] = st.number_input(
                "Port",
                min_value=1000,
                max_value=65535,
                value=settings["port"],
                help="Port for API server"
            )
        
        with col2:
            settings["host"] = st.text_input(
                "Host",
                value=settings["host"],
                help="Host address for API server"
            )
            
            settings["require_auth"] = st.checkbox(
                "Require Authentication",
                value=settings["require_auth"],
                help="Require API key for access"
            )
        
        st.session_state.api_settings = settings
        
        # API Key management
        if settings["enable_api_server"] and settings["require_auth"]:
            st.markdown("#### 🔑 API Key Management")
            
            # Create new API key
            with st.expander("Create New API Key", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    key_name = st.text_input("Key Name", key="new_key_name")
                    key_permissions = st.multiselect(
                        "Permissions",
                        options=["read_data", "write_data", "delete_data", "execute_code"],
                        key="new_key_permissions"
                    )
                
                with col2:
                    if st.button("🔑 Create Key", key="create_api_key"):
                        if key_name and key_permissions:
                            api_key = self.create_api_key(key_name, key_permissions)
                            st.success(f"API key '{key_name}' created!")
                            st.code(api_key, language="text")
                            st.info("⚠️ Save this key securely! It won't be shown again.")
            
            # List existing keys
            st.markdown("#### 📋 Existing API Keys")
            
            for key_id, key_obj in st.session_state.api_keys.items():
                with st.expander(f"{key_obj.name} ({'✅' if key_obj.is_active else '⏸️'})", expanded=False):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Name:** {key_obj.name}")
                        st.markdown(f"**Created:** {datetime.fromtimestamp(key_obj.created_at).strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**Usage:** {key_obj.usage_count:,} requests")
                        st.markdown(f"**Last Used:** {datetime.fromtimestamp(key_obj.last_used).strftime('%Y-%m-%d %H:%M') if key_obj.last_used > 0 else 'Never'}")
                    
                    with col2:
                        if st.button("🔄 Regenerate", key=f"regen_{key_id}"):
                            # In a real implementation, this would regenerate the key
                            st.info("Key regeneration not implemented yet")
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{key_id}"):
                            if st.checkbox(f"Confirm delete {key_obj.name}", key=f"confirm_delete_{key_id}"):
                                key_obj.is_active = False
                                st.success(f"API key '{key_obj.name}' deleted!")
                                st.rerun()
        
        # Server control
        if settings["enable_api_server"]:
            st.markdown("#### 🚀 Server Control")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start Server", key="start_api_server"):
                    st.success("API server starting...")
                    # In a real implementation, this would start the server
                    st.info("Server would start on http://{}:{}".format(settings["host"], settings["port"]))
            
            with col2:
                if st.button("⏹️ Stop Server", key="stop_api_server"):
                    st.success("API server stopping...")
                    st.info("Server would stop")
        
        # API documentation
        st.markdown("#### 📚 API Documentation")
        
        endpoints = st.session_state.api_endpoints
        
        st.markdown("**Available Endpoints:**")
        
        for endpoint in endpoints:
            with st.expander(f"{endpoint.method} {endpoint.path}", expanded=False):
                st.markdown(f"**Description:** {endpoint.description}")
                st.markdown(f"**Permissions Required:** {', '.join(endpoint.permissions_required)}")
                st.markdown(f"**Rate Limit:** {endpoint.rate_limit} requests/hour")
                
                # Example request
                if endpoint.method == "GET":
                    st.code(f"GET {endpoint.path}", language="http")
                elif endpoint.method == "POST":
                    st.code(f"POST {endpoint.path}", language="http")
                    st.markdown("```json")
                    st.json({
                        "data": "example payload"
                    })
                    st.markdown("```")
                elif endpoint.method == "DELETE":
                    st.code(f"DELETE {endpoint.path}", language="http")


# Global API server instance
_api_server = APIServer()


def get_api_server() -> APIServer:
    """Get the global API server instance."""
    return _api_server


def render_api_server():
    """Render API server management interface."""
    server = get_api_server()
    server.render_api_server_interface()
