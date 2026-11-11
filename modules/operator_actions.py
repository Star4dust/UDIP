"""
UDIP v3 - modules/operator_actions.py
Safe handoff actions that move agent output into other UDIP surfaces.
"""

from __future__ import annotations

import re
import uuid

import streamlit as st

from modules.shared_utilities import get_shared_logger
from modules.circular_import_fixes import get_dependency_injector


def extract_code_block(text: str, language: str) -> str | None:
    """Extract the first fenced code block for a given language."""
    if not text:
        return None
    pattern = rf"```{re.escape(language)}\s*\n([\s\S]*?)```"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_python_code(text: str) -> str | None:
    return extract_code_block(text, "python")


def extract_sql_code(text: str) -> str | None:
    return extract_code_block(text, "sql")


def is_read_only_sql(query: str) -> tuple[bool, str]:
    """Allow only read-style SQL statements for direct execution."""
    if not query or not query.strip():
        return False, "No SQL query was found."

    stripped = re.sub(r"--.*?$", "", query, flags=re.MULTILINE).strip()
    if not stripped:
        return False, "The SQL query is empty after removing comments."

    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "Direct SQL execution only supports read-only SELECT or WITH queries."

    blocked = [
        "insert ", "update ", "delete ", "drop ", "alter ", "create ", "copy ",
        "attach ", "detach ", "truncate ", "replace ", "grant ", "revoke ",
        "export ", "call ",
    ]
    if any(token in lowered for token in blocked):
        return False, "The SQL query contains write or schema-changing commands, so it was blocked."

    return True, ""


def set_operator_handoff(target_page: str, note: str):
    """Store a short note to show on the destination page after navigation."""
    st.session_state["_operator_handoff"] = {
        "page": target_page,
        "note": note,
    }


def render_operator_handoff(target_page: str):
    """Render and clear a handoff note if this page is the destination."""
    handoff = st.session_state.get("_operator_handoff")
    if handoff and handoff.get("page") == target_page:
        st.info(handoff.get("note", "Agent handoff ready."))
        st.session_state.pop("_operator_handoff", None)


def _infer_visual_target(run: dict) -> tuple[str, str]:
    text = " ".join(
        str(run.get(key, "") or "")
        for key in ("blueprint", "task_brief", "deliverable", "response")
    ).lower()

    if any(word in text for word in ["trend", "time series", "timeline", "over time", "forecast"]):
        return "📈 Trends & Time Series", "Line chart"
    if any(word in text for word in ["correlation", "relationship", "scatter", "regression"]):
        return "🔗 Relationships & Correlations", "Scatter plot"
    if any(word in text for word in ["share", "composition", "part-to-whole", "proportion"]):
        return "🌳 Hierarchical & Part-to-Whole", "Treemap"
    return "📊 Distribution & Comparison", "Bar chart"


def build_operator_actions(run: dict) -> list[dict]:
    """Build a safe, deterministic set of next actions for an agent run."""
    actions: list[dict] = []
    python_code = run.get("code") or extract_python_code(run.get("response", ""))
    sql_code = extract_sql_code(run.get("response", ""))
    has_dataset = bool(run.get("dataset_id"))

    if python_code:
        actions.append(
            {
                "id": "code_editor",
                "label": "Load into Code Editor",
                "description": "Move the returned Python into the live editor so you can run or refine it.",
            }
        )

    if sql_code or has_dataset:
        actions.append(
            {
                "id": "sql_editor",
                "label": "Open SQL Editor",
                "description": "Prepare a query workspace from the result or from the selected dataset.",
            }
        )
    if sql_code:
        actions.append(
            {
                "id": "run_sql_preview",
                "label": "Run SQL Preview",
                "description": "Execute the returned SQL as a read-only preview directly inside Agent Lab.",
            }
        )

    if has_dataset:
        actions.append(
            {
                "id": "data_forge",
                "label": "Open Data Forge",
                "description": "Send the task brief into prompt-driven restructuring and cleaning.",
            }
        )
        actions.append(
            {
                "id": "visualize",
                "label": "Open Visualize",
                "description": "Open the selected dataset in Visualization Studio with a suggested chart direction.",
            }
        )
        actions.append(
            {
                "id": "preview_forge_transform",
                "label": "Preview Data Forge Transform",
                "description": "Run a prompt-driven restructuring preview on the selected dataset without saving yet.",
            }
        )

    actions.append(
        {
            "id": "ai_assistant",
            "label": "Continue in AI Assistant",
            "description": "Carry the result into the main conversational assistant for follow-up work.",
        }
    )
    return actions


def _build_forge_prompt(run: dict) -> str:
    brief = (run.get("task_brief") or "").strip()
    constraints = (run.get("constraints") or "").strip()
    deliverable = (run.get("deliverable") or "").strip()
    response = (run.get("response") or "").strip()

    sections = []
    if brief:
        sections.append(f"Goal:\n{brief}")
    if deliverable:
        sections.append(f"Desired result:\n{deliverable}")
    if constraints:
        sections.append(f"Constraints:\n{constraints}")
    if response:
        sections.append(f"Agent notes:\n{response[:1200]}")
    return "\n\n".join(sections).strip()


def execute_operator_action(action_id: str, run: dict) -> tuple[bool, str]:
    """Apply a safe action and prepare the destination page."""
    selected_ds = run.get("dataset_id")
    datasets = st.session_state.get("datasets", {})
    has_live_dataset = bool(selected_ds and selected_ds in datasets)
    dataset_name = run.get("dataset_name") or (datasets[selected_ds]["name"] if has_live_dataset else selected_ds) or "current dataset"
    response = run.get("response", "")
    python_code = run.get("code") or extract_python_code(response)
    sql_code = extract_sql_code(response)
    state_mgr = get_state_manager()

    if action_id == "code_editor":
        if not python_code:
            return False, "No Python code was returned for this run."
        st.session_state.editor_code = python_code
        save_code(python_code)
        set_operator_handoff("🐍 Code Editor", f"Loaded Python from Agent Lab into Code Editor for {run.get('blueprint', 'this run')}.")
        st.session_state.active_page = "🐍 Code Editor"
        return True, "Opened Code Editor."

    if action_id == "sql_editor":
        query = sql_code
        if not query and has_live_dataset:
            query = f"SELECT *\nFROM {selected_ds}\nLIMIT 100;"
        if not query:
            return False, "No SQL was available and no dataset was selected to build a starter query."
        st.session_state.sql_query = query
        if has_live_dataset:
            st.session_state.ai_selected_dataset = selected_ds
        set_operator_handoff("🗄️ SQL Editor", f"Prepared SQL Editor for {dataset_name} with a starter query.")
        st.session_state.active_page = "🗄️ SQL Editor"
        return True, "Opened SQL Editor."

    if action_id == "data_forge":
        if not has_live_dataset:
            return False, "Data Forge needs a selected dataset."
        st.session_state.forge_ds = selected_ds
        st.session_state._forge_prompt = _build_forge_prompt(run)
        for key in ("_forge_l2_result", "_forge_l2_explain", "_forge_l2_changes"):
            st.session_state.pop(key, None)
        set_operator_handoff("⚒️ Data Forge", f"Prepared Data Forge for {dataset_name} with the agent brief preloaded.")
        st.session_state.active_page = "⚒️ Data Forge"
        return True, "Opened Data Forge."

    if action_id == "visualize":
        if not has_live_dataset:
            return False, "Visualization handoff needs a selected dataset."
        cat_choice, chart_type = _infer_visual_target(run)
        viz_state = state_mgr.get_page_state("visualize") or {"ds_id": None, "cat_choice": cat_choice, "chart_type": chart_type, "selections": {}}
        viz_state["ds_id"] = selected_ds
        viz_state["cat_choice"] = cat_choice
        viz_state["chart_type"] = chart_type
        state_mgr.save_page_state("visualize", viz_state)
        st.session_state.viz_ds_select = selected_ds
        st.session_state.viz_cat = cat_choice
        st.session_state.viz_chart_type = chart_type
        set_operator_handoff("📊 Visualize", f"Opened Visualization Studio for {dataset_name} with a suggested starting chart.")
        st.session_state.active_page = "📊 Visualize"
        return True, "Opened Visualization Studio."

    if action_id == "ai_assistant":
        if "chat_history" not in st.session_state or st.session_state.chat_history is None:
            st.session_state.chat_history = []
        handoff_prompt = (
            "Continue this Agent Lab run.\n\n"
            f"Blueprint: {run.get('blueprint', 'Agent Lab')}\n"
            f"Task brief:\n{run.get('task_brief', '')}\n\n"
            "Please continue from the existing outcome and refine it if needed."
        )
        st.session_state.chat_history.append({"role": "user", "content": handoff_prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        save_chat(st.session_state.chat_history)
        set_operator_handoff("🤖 AI Assistant", "Moved the Agent Lab run into AI Assistant so you can continue conversationally.")
        st.session_state.active_page = "🤖 AI Assistant"
        return True, "Opened AI Assistant."

    return False, f"Unknown operator action: {action_id}"


def run_sql_preview_from_run(run: dict) -> tuple[bool, str]:
    """Execute read-only SQL from an agent run and store the preview."""
    from modules.sql_engine import run_sql

    response = run.get("response", "")
    sql_code = extract_sql_code(response)
    if not sql_code:
        return False, "No SQL block was found in this run."

    allowed, reason = is_read_only_sql(sql_code)
    if not allowed:
        return False, reason

    result, error = run_sql(sql_code)
    if error:
        return False, error

    st.session_state["agent_lab_sql_preview"] = {
        "query": sql_code,
        "result": result,
        "rows": len(result),
        "columns": len(result.columns),
        "run_blueprint": run.get("blueprint", "Agent Lab"),
    }
    return True, f"SQL preview ready with {len(result):,} rows."


def preview_forge_transform_from_run(run: dict, groq_key: str) -> tuple[bool, str]:
    """Generate a preview-only Data Forge transform from the agent run."""
    from udip_pages.data_forge import apply_prompt_to_df, validate_result

    selected_ds = run.get("dataset_id")
    datasets = st.session_state.get("datasets", {})
    if not selected_ds or selected_ds not in datasets:
        return False, "A live dataset is required for a Data Forge preview."
    if not groq_key:
        return False, "Groq API key not configured."

    source_df = datasets[selected_ds]["df"].copy()
    prompt_parts = [
        run.get("task_brief", "").strip(),
        run.get("deliverable", "").strip(),
        run.get("constraints", "").strip(),
    ]
    prompt = "\n\n".join(part for part in prompt_parts if part)
    if not prompt:
        prompt = "Restructure and clean this dataset conservatively based on the available context."

    result_df, explanation, changes = apply_prompt_to_df(source_df, prompt, groq_key)
    passed, issues = validate_result(result_df, source_df)

    st.session_state["agent_lab_forge_preview"] = {
        "dataset_id": selected_ds,
        "dataset_name": datasets[selected_ds]["name"],
        "prompt": prompt,
        "result_df": result_df,
        "explanation": explanation,
        "changes": changes,
        "passed": passed,
        "issues": issues,
        "source_rows": len(source_df),
        "result_rows": len(result_df),
    }
    return True, "Data Forge preview generated."


def save_sql_preview_as_dataset() -> tuple[bool, str]:
    """Save the current SQL preview as a new UDIP dataset."""
    preview = st.session_state.get("agent_lab_sql_preview")
    if not preview or preview.get("result") is None:
        return False, "No SQL preview is available to save."

    result = preview["result"]
    ds_id = str(uuid.uuid4())[:8]
    name = "agent_lab_sql_preview.parquet"
    entry = {
        "name": name,
        "df": result,
        "format": "parquet",
        "source": "agent_lab_sql_preview",
        "varname": sanitize_varname("agent_lab_sql_preview"),
    }
    ok, err = save_dataset(ds_id, entry)
    if not ok:
        return False, err

    st.session_state.datasets[ds_id] = entry
    add_history("Agent SQL", f"Saved SQL preview as dataset ({len(result):,} rows)")
    return True, f"Saved SQL preview as dataset `{name}`."


def save_forge_preview_as_dataset() -> tuple[bool, str]:
    """Save the current Data Forge preview as a new UDIP dataset."""
    preview = st.session_state.get("agent_lab_forge_preview")
    if not preview or preview.get("result_df") is None:
        return False, "No Data Forge preview is available to save."

    result_df = preview["result_df"]
    source_name = preview.get("dataset_name", "dataset")
    clean_name = sanitize_varname(source_name.rsplit(".", 1)[0] or "dataset")
    file_name = f"{clean_name}_agent_lab_preview.parquet"
    ds_id = str(uuid.uuid4())[:8]
    entry = {
        "name": file_name,
        "df": result_df,
        "format": "parquet",
        "source": "agent_lab_forge_preview",
        "varname": sanitize_varname(file_name.rsplit(".", 1)[0]),
    }
    ok, err = save_dataset(ds_id, entry)
    if not ok:
        return False, err

    st.session_state.datasets[ds_id] = entry
    add_history("Agent Forge", f"Saved Data Forge preview from {source_name}")
    return True, f"Saved preview dataset as `{file_name}`."
