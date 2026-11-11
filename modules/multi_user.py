"""
UDIP v3 — modules/multi_user.py
Multi-user architecture with role-based access control.
"""

import streamlit as st
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    DEVELOPER = "developer"


class Permission(Enum):
    """System permissions."""
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    MANAGE_USERS = "manage_users"
    MANAGE_PLUGINS = "manage_plugins"
    SYSTEM_CONFIG = "system_config"
    EXECUTE_CODE = "execute_code"
    VIEW_DASHBOARDS = "view_dashboards"
    MANAGE_DASHBOARDS = "manage_dashboards"


@dataclass
class User:
    """User account information."""
    id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission]
    created_at: float
    last_login: float
    is_active: bool
    workspace_access: List[str] = None
    preferences: Dict[str, Any] = None


@dataclass
class Workspace:
    """Workspace information."""
    id: str
    name: str
    description: str
    owner_id: str
    created_at: float
    is_active: bool
    member_ids: List[str] = None
    settings: Dict[str, Any] = None


@dataclass
class AuditLog:
    """Audit log entry."""
    id: str
    user_id: str
    action: str
    resource: str
    timestamp: float
    details: Dict[str, Any] = None
    ip_address: str = None


class MultiUserManager:
    """Manages multi-user architecture and role-based access."""
    
    def __init__(self):
        if "users" not in st.session_state:
            st.session_state.users = self._init_default_users()
        if "workspaces" not in st.session_state:
            st.session_state.workspaces = self._init_default_workspaces()
        if "current_user" not in st.session_state:
            st.session_state.current_user = None
        if "current_workspace" not in st.session_state:
            st.session_state.current_workspace = None
        if "audit_logs" not in st.session_state:
            st.session_state.audit_logs = []
        if "multi_user_settings" not in st.session_state:
            st.session_state.multi_user_settings = {
                "enable_multi_user": False,
                "allow_registration": False,
                "require_email_verification": False,
                "session_timeout": 3600,  # 1 hour
                "max_workspaces_per_user": 5,
                "audit_log_retention": 90  # days
            }
    
    def _init_default_users(self) -> Dict[str, User]:
        """Initialize default admin user."""
        admin_user = User(
            id="admin_001",
            username="admin",
            email="admin@udip.local",
            role=UserRole.ADMIN,
            permissions=set(Permission),
            created_at=time.time(),
            last_login=time.time(),
            is_active=True,
            workspace_access=["default"],
            preferences={"theme": "light", "notifications": True}
        )
        
        return {"admin_001": admin_user}
    
    def _init_default_workspaces(self) -> Dict[str, Workspace]:
        """Initialize default workspace."""
        default_workspace = Workspace(
            id="ws_default",
            name="Default Workspace",
            description="Default UDIP workspace for all users",
            owner_id="admin_001",
            created_at=time.time(),
            is_active=True,
            member_ids=["admin_001"],
            settings={"allow_guest_access": False}
        )
        
        return {"ws_default": default_workspace}
    
    def get_current_user(self) -> Optional[User]:
        """Get currently authenticated user."""
        return st.session_state.get("current_user")
    
    def get_current_workspace(self) -> Optional[Workspace]:
        """Get currently active workspace."""
        return st.session_state.get("current_workspace")
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        users = st.session_state.users
        
        for user in users.values():
            if user.username == username and self._verify_password(password, user.id):
                user.last_login = time.time()
                return user
        
        return None
    
    def _verify_password(self, password: str, user_id: str) -> bool:
        """Verify password against stored hash."""
        # In a real implementation, this would use proper password hashing
        # For demo purposes, using simple verification
        stored_passwords = {
            "admin_001": "admin123"  # In production, use bcrypt/argon2
        }
        
        return stored_passwords.get(user_id) == password
    
    def has_permission(self, permission: Permission, user: Optional[User] = None) -> bool:
        """Check if user has a specific permission."""
        if not user:
            user = self.get_current_user()
        
        if not user:
            return False
        
        return permission in user.permissions
    
    def can_access_workspace(self, workspace_id: str, user: Optional[User] = None) -> bool:
        """Check if user can access a workspace."""
        if not user:
            user = self.get_current_user()
        
        if not user:
            return False
        
        # Admin can access all workspaces
        if user.role == UserRole.ADMIN:
            return True
        
        # Check workspace membership
        return workspace_id in (user.workspace_access or [])
    
    def log_audit_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        audit_log = AuditLog(
            id=f"audit_{int(time.time() * 1000)}",
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=time.time(),
            details=details or {},
            ip_address=st.session_state.get("client_ip", "unknown")
        )
        
        st.session_state.audit_logs.append(audit_log)
        
        # Limit audit log size
        max_logs = st.session_state.multi_user_settings["audit_log_retention"] * 24  # per day
        if len(st.session_state.audit_logs) > max_logs:
            st.session_state.audit_logs = st.session_state.audit_logs[-max_logs:]
    
    def create_user(
        self,
        username: str,
        email: str,
        role: UserRole,
        permissions: Optional[Set[Permission]] = None
    ) -> User:
        """Create a new user."""
        user_id = f"user_{int(time.time() * 1000)}"
        
        # Default permissions based on role
        if permissions is None:
            if role == UserRole.ADMIN:
                permissions = set(Permission)
            elif role == UserRole.ANALYST:
                permissions = {
                    Permission.READ_DATA,
                    Permission.WRITE_DATA,
                    Permission.VIEW_DASHBOARDS,
                    Permission.EXECUTE_CODE
                }
            elif role == UserRole.DEVELOPER:
                permissions = {
                    Permission.READ_DATA,
                    Permission.WRITE_DATA,
                    Permission.EXECUTE_CODE,
                    Permission.MANAGE_PLUGINS
                }
            else:  # VIEWER
                permissions = {
                    Permission.READ_DATA,
                    Permission.VIEW_DASHBOARDS
                }
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=time.time(),
            last_login=0,
            is_active=True,
            workspace_access=["default"],
            preferences={}
        )
        
        st.session_state.users[user_id] = user
        
        # Log audit event
        self.log_audit_event(
            user_id=st.session_state.get("current_user", {}).get("id", "system"),
            action="user_created",
            resource=f"user_{user_id}",
            details={"username": username, "role": role.value}
        )
        
        return user
    
    def create_workspace(
        self,
        name: str,
        description: str,
        owner_id: str
    ) -> Workspace:
        """Create a new workspace."""
        workspace_id = f"ws_{int(time.time() * 1000)}"
        
        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id,
            created_at=time.time(),
            is_active=True,
            member_ids=[owner_id],
            settings={}
        )
        
        st.session_state.workspaces[workspace_id] = workspace
        
        # Log audit event
        self.log_audit_event(
            user_id=st.session_state.get("current_user", {}).get("id", "system"),
            action="workspace_created",
            resource=f"workspace_{workspace_id}",
            details={"name": name, "owner": owner_id}
        )
        
        return workspace
    
    def render_user_management(self):
        """Render user management interface."""
        st.markdown("### 👥 User Management")
        
        if not self.has_permission(Permission.MANAGE_USERS):
            st.error("You don't have permission to manage users.")
            return
        
        # User creation
        with st.expander("Create New User", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username", key="new_user_username")
                new_email = st.text_input("Email", key="new_user_email")
                new_role = st.selectbox(
                    "Role",
                    options=[role.value for role in UserRole],
                    key="new_user_role"
                )
            
            with col2:
                st.markdown("**Permissions**")
                selected_permissions = []
                
                if new_role == UserRole.ADMIN.value:
                    selected_permissions = list(Permission)
                elif new_role == UserRole.ANALYST.value:
                    selected_permissions = [
                        Permission.READ_DATA.value,
                        Permission.WRITE_DATA.value,
                        Permission.VIEW_DASHBOARDS.value,
                        Permission.EXECUTE_CODE.value
                    ]
                elif new_role == UserRole.DEVELOPER.value:
                    selected_permissions = [
                        Permission.READ_DATA.value,
                        Permission.WRITE_DATA.value,
                        Permission.EXECUTE_CODE.value,
                        Permission.MANAGE_PLUGINS.value
                    ]
                else:  # VIEWER
                    selected_permissions = [
                        Permission.READ_DATA.value,
                        Permission.VIEW_DASHBOARDS.value
                    ]
                
                for permission in selected_permissions:
                    st.checkbox(permission.value, value=True, disabled=True)
            
            if st.button("Create User", key="create_user"):
                if new_username and new_email:
                    self.create_user(new_username, new_email, UserRole(new_role))
                    st.success(f"User '{new_username}' created successfully!")
                    st.rerun()
        
        # User list
        st.markdown("#### Current Users")
        
        users = list(st.session_state.users.values())
        if users:
            for user in users:
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**{user.username}**")
                    st.caption(f"Role: {user.role.value}")
                
                with col2:
                    st.markdown(f"**{user.email}**")
                    st.caption(f"Created: {time.strftime('%Y-%m-%d', time.localtime(user.created_at))}")
                
                with col3:
                    status = "✅ Active" if user.is_active else "⏸️ Inactive"
                    st.markdown(status)
                
                with col4:
                    if user.id != "admin_001":  # Don't allow deleting admin
                        if st.button("🗑️", key=f"delete_user_{user.id}"):
                            if st.checkbox(f"Confirm delete {user.username}", key=f"confirm_delete_{user.id}"):
                                del st.session_state.users[user.id]
                                self.log_audit_event(
                                    user_id=st.session_state.get("current_user", {}).get("id", "system"),
                                    action="user_deleted",
                                    resource=f"user_{user.id}",
                                    details={"username": user.username}
                                )
                                st.success(f"User '{user.username}' deleted!")
                                st.rerun()
        else:
            st.info("No users found.")
    
    def render_workspace_management(self):
        """Render workspace management interface."""
        st.markdown("### 🏢 Workspace Management")
        
        if not self.has_permission(Permission.MANAGE_DASHBOARDS):
            st.error("You don't have permission to manage workspaces.")
            return
        
        # Workspace creation
        with st.expander("Create New Workspace", expanded=False):
            new_name = st.text_input("Workspace Name", key="new_workspace_name")
            new_description = st.text_area("Description", key="new_workspace_description")
            
            if st.button("Create Workspace", key="create_workspace"):
                if new_name:
                    current_user = self.get_current_user()
                    if current_user:
                        self.create_workspace(new_name, new_description, current_user.id)
                        st.success(f"Workspace '{new_name}' created successfully!")
                        st.rerun()
        
        # Workspace list
        st.markdown("#### Current Workspaces")
        
        workspaces = list(st.session_state.workspaces.values())
        if workspaces:
            for workspace in workspaces:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{workspace.name}**")
                    st.caption(f"Owner: {workspace.owner_id}")
                
                with col2:
                    st.markdown(f"**{workspace.description[:100]}{'...' if len(workspace.description) > 100 else ''}**")
                    st.caption(f"Created: {time.strftime('%Y-%m-%d', time.localtime(workspace.created_at))}")
                
                with col3:
                    status = "✅ Active" if workspace.is_active else "⏸️ Inactive"
                    st.markdown(status)
        else:
            st.info("No workspaces found.")
    
    def render_audit_logs(self):
        """Render audit logs interface."""
        st.markdown("### 📋 Audit Logs")
        
        if not self.has_permission(Permission.MANAGE_USERS):
            st.error("You don't have permission to view audit logs.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            action_filter = st.selectbox(
                "Filter by Action",
                options=["All", "user_created", "user_deleted", "workspace_created", "login", "logout"],
                key="audit_action_filter"
            )
        
        with col2:
            days_filter = st.number_input(
                "Last N Days",
                min_value=1,
                max_value=90,
                value=7,
                key="audit_days_filter"
            )
        
        # Filter logs
        logs = st.session_state.audit_logs
        if action_filter != "All":
            logs = [log for log in logs if log.action == action_filter]
        
        cutoff_time = time.time() - (days_filter * 24 * 3600)
        logs = [log for log in logs if log.timestamp > cutoff_time]
        
        # Display logs
        if logs:
            for log in reversed(logs[-50:]):  # Show last 50
                with st.expander(
                    f"{log.action} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log.timestamp))}",
                    expanded=False
                ):
                    st.markdown(f"**User ID:** {log.user_id}")
                    st.markdown(f"**Action:** {log.action}")
                    st.markdown(f"**Resource:** {log.resource}")
                    st.markdown(f"**IP Address:** {log.ip_address}")
                    
                    if log.details:
                        st.markdown("**Details:**")
                        st.json(log.details)
        else:
            st.info("No audit logs found for the selected criteria.")
    
    def render_multi_user_settings(self):
        """Render multi-user system settings."""
        st.markdown("### ⚙️ Multi-User Settings")
        
        if not self.has_permission(Permission.SYSTEM_CONFIG):
            st.error("You don't have permission to change system settings.")
            return
        
        settings = st.session_state.multi_user_settings
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings["enable_multi_user"] = st.checkbox(
                "Enable Multi-User Mode",
                value=settings["enable_multi_user"],
                help="Enable multi-user architecture with role-based access"
            )
            
            settings["allow_registration"] = st.checkbox(
                "Allow User Registration",
                value=settings["allow_registration"],
                help="Allow users to self-register"
            )
        
        with col2:
            settings["require_email_verification"] = st.checkbox(
                "Require Email Verification",
                value=settings["require_email_verification"],
                help="Require email verification for new accounts"
            )
            
            settings["session_timeout"] = st.number_input(
                "Session Timeout (minutes)",
                min_value=5,
                max_value=1440,
                value=settings["session_timeout"],
                help="Automatically log out users after inactivity"
            )
        
        st.session_state.multi_user_settings = settings
        
        if st.button("Save Settings", key="save_multi_user_settings"):
            st.success("Multi-user settings saved!")
            st.rerun()


# Global multi-user manager instance
_multi_user_manager = MultiUserManager()


def get_multi_user_manager() -> MultiUserManager:
    """Get the global multi-user manager instance."""
    return _multi_user_manager


def render_multi_user_interface():
    """Render the complete multi-user interface."""
    manager = get_multi_user_manager()
    
    # Only show multi-user features if enabled
    if not st.session_state.multi_user_settings.get("enable_multi_user", False):
        st.info("Multi-user mode is disabled. Enable it in Settings to use these features.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "🏢 Workspaces", "📋 Audit Logs", "⚙️ Settings"])
    
    with tab1:
        manager.render_user_management()
    
    with tab2:
        manager.render_workspace_management()
    
    with tab3:
        manager.render_audit_logs()
    
    with tab4:
        manager.render_multi_user_settings()
