"""
Central navigation registry for UDIP.

The app keeps pages lazy-loaded, but the route labels and sidebar grouping live
here so the sidebar and router cannot drift apart.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable


from modules.core import handle_page_errors

DEFAULT_PAGE = "📌 Dashboard"


@dataclass(frozen=True)
class PageRoute:
    label: str
    module: str


NAV_SECTIONS: "OrderedDict[str, list[PageRoute]]" = OrderedDict(
    {
        "🚀 START HERE": [
            PageRoute("📌 Dashboard", "udip_pages.dashboard"),
            PageRoute("📁 Upload Data", "pages.upload"),
            PageRoute("☁️ Cloud Sources", "udip_pages.cloud_sources"),
            PageRoute("🗄️ Database Connectors", "udip_pages.connectors"),
        ],
        "🧹 PREPARE DATA": [
            PageRoute("⚒️ Data Forge", "udip_pages.data_forge"),
            PageRoute("🔗 Merge Wizard", "udip_pages.merge_wizard"),
            PageRoute("📝 SQL Editor", "udip_pages.sql_editor"),
            PageRoute("✅ Data Validation", "udip_pages.validation"),
            PageRoute("🧪 Feature Alchemy", "udip_pages.feature_alchemy"),
        ],
        "📊 EXPLORE & ANALYZE": [
            PageRoute("📊 Visualize", "pages.visualize"),
            PageRoute("💓 Data Pulse", "udip_pages.data_pulse"),
        ],
        "🤖 AI WORKSPACE": [
            PageRoute("🤖 AI Assistant", "udip_pages.ai_assistant"),
            PageRoute("💻 Code Editor", "pages.code_editor"),
        ],
        "🛰️ COSMIC SIGNALS LAB": [
            PageRoute("🔭 HIRAX Inspector", "pages.hirax_inspector"),
            PageRoute("📡 RFI Detection", "modules.hirax_processing"),
            PageRoute("🛸 Signal Intelligence", "modules.hirax_analysis"),
            PageRoute("🌈 Spectral Analysis", "modules.hirax_validation"),
            PageRoute("🌌 21cm Cosmology", "modules.cosmology_21cm"),
            PageRoute("✅ Validation Workbench", "udip_pages.validation"),
        ],
        "📝 REPORT & EXPORT": [
            PageRoute("📝 Report Studio", "udip_pages.data_story"),
            PageRoute("🏗️ Dashboard Builder", "udip_pages.dashboard"),
            PageRoute("📦 Export Center", "pages.export"),
            PageRoute("⏰ Scheduler", "udip_pages.scheduler_page"),
        ],
        "⚙️ SYSTEM & DEVELOPMENT": [
            PageRoute("⚙️ Settings", "pages.settings"),
            PageRoute("🌐 API Server", "udip_pages.api_server"),
        ],
    }
)


ROUTE_ALIASES = {
    "Execution UI": "⚡ Execution UI",
    "Merge Datasets": "🔗 Merge Wizard",
    "Data Lab": "⚒️ Data Forge",
    "📊 Auto Analysis": "📊 Visualize",
    "🔬 Auto Analysis": "📊 Visualize",
    "📈 Statistical Analysis": "📊 Visualize",
    "🧠 Machine Learning": "📊 Visualize",
    " Signal Intelligence": "🛸 Signal Intelligence",
}


def iter_routes() -> Iterable[PageRoute]:
    for routes in NAV_SECTIONS.values():
        yield from routes


@handle_page_errors
def _render_module(module_path: str):
    module = import_module(module_path)
    if hasattr(module, "render"):
        return module.render()
    else:
        import streamlit as st
        st.error(f"Module {module_path} has no render() function.")


def build_page_map(custom_plugins: dict | None = None) -> dict[str, Callable]:
    """Build a lazy page map for core pages plus discovered plugins."""
    page_map: dict[str, Callable] = {
        route.label: (lambda module_path=route.module: _render_module(module_path))
        for route in iter_routes()
    }

    for plugin_id, plugin_meta in (custom_plugins or {}).items():
        label = plugin_meta.get("label") or plugin_id
        icon = plugin_meta.get("icon") or "🔌"
        route_label = f"{icon} {label}" if icon not in label else label
        page_map[route_label] = plugin_meta["render"]

    return page_map


def normalize_page(page: str | None, page_map: dict[str, Callable] | None = None) -> str:
    """Return a known route label, falling back to the default page."""
    candidate = ROUTE_ALIASES.get(page or "", page or DEFAULT_PAGE)
    if page_map and candidate not in page_map:
        return DEFAULT_PAGE
    return candidate
