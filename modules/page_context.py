"""
UDIP v3 — modules/page_context.py
Page context management for context-aware rendering.
"""

import streamlit as st
from typing import Dict, List, Callable, Optional


def get_current_workflow() -> str:
    """
    Determine the current workflow based on active page.
    
    Returns:
        str: Current workflow category
    """
    current_page = st.session_state.get('active_page', '')
    
    # Workflow classification
    if 'Upload' in current_page or 'Data Forge' in current_page:
        return 'upload'
    elif 'HIRAX' in current_page or 'Cosmic' in current_page or 'Signals' in current_page:
        return 'hirax'
    elif 'AI Assistant' in current_page or 'Copilot' in current_page or 'Agent' in current_page:
        return 'ai_workspace'
    elif 'Report' in current_page or 'Export' in current_page or 'Story' in current_page:
        return 'reporting'
    elif 'Dashboard' in current_page:
        return 'dashboard'
    elif 'Auto Analysis' in current_page or 'Explore' in current_page:
        return 'analysis'
    else:
        return 'general'


def should_render_command_center() -> bool:
    """
    Determine if command center header should render.
    
    Returns:
        bool: True if command center should render
    """
    workflow = get_current_workflow()
    
    # Command center is global but can be suppressed for specific workflows
    # Keep it for now as it provides useful global stats
    return True


def should_render_copilot() -> bool:
    """
    Determine if workflow copilot should render.
    
    Returns:
        bool: True if copilot should render
    """
    workflow = get_current_workflow()
    
    # Copilot is most useful in AI workspace and general workflows
    # Less relevant for focused scientific workflows like HIRAX
    if workflow in ['ai_workspace', 'dashboard', 'analysis', 'general']:
        return True
    else:
        return False


def should_render_adaptive_workspace() -> bool:
    """
    Determine if adaptive workspace should render.
    
    Returns:
        bool: True if adaptive workspace should render
    """
    workflow = get_current_workflow()
    
    # Adaptive workspace is most useful in general workflows
    # Less relevant for focused scientific workflows
    if workflow in ['dashboard', 'analysis', 'general']:
        return True
    else:
        return False


def should_render_dataset_selector() -> bool:
    """
    Determine if dataset selector should render.
    
    Returns:
        bool: True if dataset selector should render
    """
    workflow = get_current_workflow()
    
    # Dataset selector is useful in analysis, HIRAX, and reporting workflows
    if workflow in ['hirax', 'analysis', 'reporting', 'dashboard']:
        return True
    else:
        return False


def get_page_context() -> Dict[str, bool]:
    """
    Get complete page context for rendering decisions.
    
    Returns:
        dict: Context flags for various components
    """
    workflow = get_current_workflow()
    
    return {
        'workflow': workflow,
        'command_center': should_render_command_center(),
        'copilot': should_render_copilot(),
        'adaptive_workspace': should_render_adaptive_workspace(),
        'dataset_selector': should_render_dataset_selector(),
    }
