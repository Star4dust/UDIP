"""
UDIP v3 - udip_pages/agent_lab.py
Dedicated workbench for guided agent workflows.
"""

import re
import time

import streamlit as st
import pandas as pd

from modules.ai_agents import collaborative_response, get_agent_for_task, get_all_agents
from modules.ai_engine import (
    build_memory_context,
    extract_search_query,
    search_web,
    should_search_web,
)
from modules.ai_logger import log_ai_interaction
from modules.core import add_history, load_config, pin_to_report, save_code
from modules.operator_actions import (
    build_operator_actions,
    execute_operator_action,
    preview_forge_transform_from_run,
    run_sql_preview_from_run,
    save_forge_preview_as_dataset,
    save_sql_preview_as_dataset,
)
from modules.state_manager import get_state_manager


BLUEPRINTS = {
    "Plan Mission": {
        "icon": "🧭",
        "mode": "Plan",
        "style": "Balanced",
        "agent": "udip_orchestrator",
        "focus": "Break the work into phases, surface risks, and give a practical execution path.",
        "intent": "Turn a broad request into an ordered, low-risk action plan.",
    },
    "Build Solution": {
        "icon": "🛠️",
        "mode": "Build",
        "style": "Balanced",
        "agent": "udip_orchestrator",
        "focus": "Produce a concrete implementation, runnable code, or a polished deliverable.",
        "intent": "Create the artifact the user can use immediately inside UDIP.",
    },
    "Debug Failure": {
        "icon": "🩺",
        "mode": "Debug",
        "style": "Balanced",
        "agent": "code_architect",
        "focus": "Find the root cause, propose the fix, and explain how to verify it safely.",
        "intent": "Diagnose a failure quickly and return the most reliable fix path.",
    },
    "Research Brief": {
        "icon": "🔎",
        "mode": "Research",
        "style": "Deep",
        "agent": "udip_orchestrator",
        "focus": "Synthesize current knowledge cleanly and separate facts from recommendations.",
        "intent": "Gather outside context and translate it into a strong decision-ready brief.",
    },
    "Review Artifact": {
        "icon": "🛡️",
        "mode": "Review",
        "style": "Concise",
        "agent": "qa_sentinel",
        "focus": "Audit for correctness, risk, missing assumptions, and brittle logic.",
        "intent": "Inspect an existing artifact and call out what needs tightening.",
    },
}


def _init_shared_ai_state(state_mgr):
    ai_state = state_mgr.get_page_state("ai") or {}
    ai_state.setdefault("selected_agent", "udip_orchestrator")
    ai_state.setdefault("task_mode", "auto")
    ai_state.setdefault("response_style", "balanced")
    ai_state.setdefault("focus_notes", "")

    if "selected_agent" not in st.session_state or st.session_state.get("selected_agent") not in get_all_agents():
        st.session_state.selected_agent = ai_state["selected_agent"]
    if "ai_task_mode" not in st.session_state:
        st.session_state.ai_task_mode = ai_state["task_mode"].title()
    if "ai_response_style" not in st.session_state:
        st.session_state.ai_response_style = ai_state["response_style"].title()
    if "ai_focus_notes" not in st.session_state:
        st.session_state.ai_focus_notes = ai_state["focus_notes"]
    if "ai_quality_review" not in st.session_state:
        st.session_state.ai_quality_review = True


def _init_lab_state(state_mgr):
    lab_state = state_mgr.get_page_state("agent_lab") or {}
    lab_state.setdefault("blueprint", "Plan Mission")
    lab_state.setdefault("task_brief", "")
    lab_state.setdefault("constraints", "")
    lab_state.setdefault("deliverable", "")
    lab_state.setdefault("artifact_text", "")

    if "agent_lab_blueprint" not in st.session_state:
        st.session_state.agent_lab_blueprint = lab_state["blueprint"]
    if "agent_lab_task_brief" not in st.session_state:
        st.session_state.agent_lab_task_brief = lab_state["task_brief"]
    if "agent_lab_constraints" not in st.session_state:
        st.session_state.agent_lab_constraints = lab_state["constraints"]
    if "agent_lab_deliverable" not in st.session_state:
        st.session_state.agent_lab_deliverable = lab_state["deliverable"]
    if "agent_lab_artifact" not in st.session_state:
        st.session_state.agent_lab_artifact = lab_state["artifact_text"]
    if "agent_lab_last_run" not in st.session_state:
        st.session_state.agent_lab_last_run = None


def _save_states(state_mgr):
    state_mgr.save_page_state(
        "ai",
        {
            "selected_agent": st.session_state.get("selected_agent", "udip_orchestrator"),
            "task_mode": st.session_state.get("ai_task_mode", "Auto").lower(),
            "response_style": st.session_state.get("ai_response_style", "Balanced").lower(),
            "focus_notes": st.session_state.get("ai_focus_notes", ""),
            "last_query": st.session_state.get("agent_lab_task_brief", ""),
        },
    )
    state_mgr.save_page_state(
        "agent_lab",
        {
            "blueprint": st.session_state.get("agent_lab_blueprint", "Plan Mission"),
            "task_brief": st.session_state.get("agent_lab_task_brief", ""),
            "constraints": st.session_state.get("agent_lab_constraints", ""),
            "deliverable": st.session_state.get("agent_lab_deliverable", ""),
            "artifact_text": st.session_state.get("agent_lab_artifact", ""),
        },
    )


def _apply_blueprint(blueprint_name):
    blueprint = BLUEPRINTS[blueprint_name]
    st.session_state.agent_lab_blueprint = blueprint_name
    st.session_state.ai_task_mode = blueprint["mode"]
    st.session_state.ai_response_style = blueprint["style"]
    st.session_state.selected_agent = blueprint["agent"]
    st.session_state.ai_focus_notes = blueprint["focus"]


def _build_prompt(blueprint_name):
    blueprint = BLUEPRINTS[blueprint_name]
    task_brief = st.session_state.get("agent_lab_task_brief", "").strip()
    constraints = st.session_state.get("agent_lab_constraints", "").strip()
    deliverable = st.session_state.get("agent_lab_deliverable", "").strip()
    artifact_text = st.session_state.get("agent_lab_artifact", "").strip()

    sections = [
        f"AGENT LAB BLUEPRINT: {blueprint_name}",
        f"BLUEPRINT INTENT: {blueprint['intent']}",
        f"TASK BRIEF:\n{task_brief or 'No task brief provided.'}",
    ]

    if deliverable:
        sections.append(f"DESIRED DELIVERABLE:\n{deliverable}")
    if constraints:
        sections.append(f"CONSTRAINTS:\n{constraints}")
    if artifact_text:
        artifact_title = "ERROR / CURRENT ARTIFACT" if blueprint["mode"].lower() in {"debug", "review"} else "REFERENCE MATERIAL"
        sections.append(f"{artifact_title}:\n{artifact_text}")

    sections.append(
        "RESPONSE REQUIREMENTS:\n"
        "- Keep the answer actionable inside UDIP.\n"
        "- Use the selected dataset when relevant.\n"
        "- State assumptions clearly if the brief is incomplete.\n"
        "- If code is needed, return a runnable Python block."
    )
    return "\n\n".join(sections)

def render():
    st.markdown('<div class="main-header">🧪 Agent Lab</div>', unsafe_allow_html=True)
    st.markdown("Run focused agent workflows for planning, building, debugging, research, and review without losing UDIP context.")

    state_mgr = get_state_manager()
    _init_shared_ai_state(state_mgr)
    _init_lab_state(state_mgr)

    st.markdown(
        """
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Agent Workbench</div>
            <div class="udip-panel__title">Blueprint-driven execution</div>
            <div class="udip-panel__text">
                Agent Lab turns the new UDIP agent stack into a guided workspace. Pick a blueprint, feed it the right
                context, and get a response that is easier to trust, reuse, and hand off across the platform.
            </div>
            <div class="udip-pill-row">
                <div class="udip-pill udip-pill--accent">Plan</div>
                <div class="udip-pill">Build</div>
                <div class="udip-pill">Debug</div>
                <div class="udip-pill">Research</div>
                <div class="udip-pill udip-pill--warm">Review</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("datasets"):
        if "ai_selected_dataset" not in st.session_state:
            st.session_state.ai_selected_dataset = list(st.session_state.datasets.keys())[0]

        ds_keys = list(st.session_state.datasets.keys())
        st.selectbox(
            "Primary dataset",
            options=ds_keys,
            format_func=lambda k: st.session_state.datasets[k]["name"],
            key="ai_selected_dataset",
            help="Agent Lab will prioritize this dataset when the task is data-aware.",
        )

        selected_ds = st.session_state.get("ai_selected_dataset")
        if selected_ds and selected_ds in st.session_state.datasets:
            ds = st.session_state.datasets[selected_ds]
            st.caption(
                f"Using `{selected_ds}` • {ds['df'].shape[0]:,} rows • {ds['df'].shape[1]} columns • "
                f"{', '.join(str(c) for c in ds['df'].columns[:6])}"
                f"{'...' if len(ds['df'].columns) > 6 else ''}"
            )
    else:
        st.info("No dataset is selected yet. Agent Lab still works for architecture, product, and code tasks.")

    blueprint_cols = st.columns(len(BLUEPRINTS))
    for col, (name, cfg) in zip(blueprint_cols, BLUEPRINTS.items()):
        is_active = st.session_state.get("agent_lab_blueprint") == name
        label = f"{cfg['icon']} {name}"
        if col.button(label, key=f"agent_lab_bp_{name}", width="stretch", type="primary" if is_active else "secondary"):
            _apply_blueprint(name)
            _save_states(state_mgr)
            st.rerun()

    active_blueprint = st.session_state.get("agent_lab_blueprint", "Plan Mission")
    active_cfg = BLUEPRINTS[active_blueprint]
    recommended_agent = get_agent_for_task(st.session_state.get("agent_lab_task_brief", "") or active_blueprint)
    recommended_label = get_all_agents()[recommended_agent].name if recommended_agent in get_all_agents() else "UDIP Orchestrator"

    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Active Blueprint</div>
            <div class="udip-panel__title">{active_cfg['icon']} {active_blueprint}</div>
            <div class="udip-panel__text">{active_cfg['intent']}</div>
            <div class="udip-pill-row">
                <div class="udip-pill udip-pill--accent">Mode: {active_cfg['mode']}</div>
                <div class="udip-pill">Style: {active_cfg['style']}</div>
                <div class="udip-pill">Suggested specialist: {recommended_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 0.9], gap="large")
    with left:
        st.text_area(
            "Task brief",
            key="agent_lab_task_brief",
            height=170,
            placeholder="Describe what you want the agent to achieve, fix, or investigate.",
        )
        st.text_area(
            "Constraints and non-negotiables",
            key="agent_lab_constraints",
            height=120,
            placeholder="Performance limits, style rules, deadlines, dependencies, or things the agent must avoid.",
        )
        st.text_area(
            "Desired deliverable",
            key="agent_lab_deliverable",
            height=110,
            placeholder="What should the final answer look like: a plan, code patch, review notes, research brief, etc.",
        )
        st.text_area(
            "Reference material, code, or error trace",
            key="agent_lab_artifact",
            height=180,
            placeholder="Paste code, stack traces, draft text, SQL, design notes, or any artifact the agent should inspect.",
        )

    with right:
        agent_options = get_all_agents()
        st.selectbox(
            "Primary specialist",
            options=list(agent_options.keys()),
            format_func=lambda key: f"{agent_options[key].name} ({agent_options[key].role})",
            key="selected_agent",
            help="You can override the recommended specialist when you want a narrower point of view.",
        )
        st.selectbox("Task mode", ["Auto", "Plan", "Build", "Debug", "Research", "Rewrite", "Explain", "Review"], key="ai_task_mode")
        st.selectbox("Answer style", ["Balanced", "Concise", "Deep"], key="ai_response_style")
        st.text_input(
            "Focus notes",
            key="ai_focus_notes",
            placeholder="Correctness, speed, citations, code quality, clarity...",
        )
        st.toggle("Deep quality review", key="ai_quality_review", value=st.session_state.get("ai_quality_review", True))

        memory_summary = build_memory_context().strip()
        with st.expander("Memory context", expanded=False):
            if memory_summary:
                st.markdown(memory_summary)
            else:
                st.caption("No learned memory yet. As you use UDIP more, Agent Lab will become more context-aware.")

    _save_states(state_mgr)

    run_col, ai_col, pin_col = st.columns([1, 1, 1])
    run_clicked = run_col.button("Run Blueprint", type="primary", width="stretch")
    if ai_col.button("Open AI Assistant", width="stretch"):
        st.session_state.active_page = "🤖 AI Assistant"
        st.rerun()
    if pin_col.button("Pin Brief", width="stretch"):
        task_brief = st.session_state.get("agent_lab_task_brief", "").strip()
        if task_brief:
            pin_to_report("insight", f"Agent Brief - {active_blueprint}", task_brief)
        else:
            st.warning("Add a task brief first so there is something meaningful to pin.")

    if run_clicked:
        groq_key = load_config().get("groq_key", "")
        task_brief = st.session_state.get("agent_lab_task_brief", "").strip()
        artifact_text = st.session_state.get("agent_lab_artifact", "").strip()
        if not groq_key:
            st.error("Groq API key not configured. Open Settings and add it before running Agent Lab.")
        elif not task_brief and not artifact_text:
            st.warning("Add a task brief or paste an artifact before running the blueprint.")
        else:
            prompt = _build_prompt(active_blueprint)
            task_mode = st.session_state.get("ai_task_mode", active_cfg["mode"]).lower()
            response_style = st.session_state.get("ai_response_style", active_cfg["style"]).lower()
            focus_notes = st.session_state.get("ai_focus_notes", active_cfg["focus"])
            selected_ds = st.session_state.get("ai_selected_dataset")
            selected_agent = st.session_state.get("selected_agent", active_cfg["agent"])

            extra_context = ""
            needs_web_context = task_mode == "research" or should_search_web(prompt)
            with st.status(f"Running {active_blueprint}...", expanded=True) as status:
                try:
                    if needs_web_context:
                        search_seed = task_brief or artifact_text or active_blueprint
                        query = extract_search_query(search_seed)
                        if query:
                            status.write(f"Searching for supporting context: {query}")
                            web_ctx = search_web(query)
                            if web_ctx:
                                extra_context = f"WEB RESEARCH:\n{web_ctx}"

                    status.write(f"Consulting {agent_options[selected_agent].name} in {task_mode} mode.")
                    agent_response = collaborative_response(
                        prompt,
                        primary_agent=selected_agent,
                        selected_dataset_id=selected_ds,
                        deep_review=st.session_state.get("ai_quality_review", True),
                        task_mode=task_mode,
                        response_style=response_style,
                        focus_notes=focus_notes,
                        extra_context=extra_context,
                        allow_web_search=not bool(extra_context),
                    )

                    response_text = agent_response["response"]
                    code_match = re.search(r"```python\n([\s\S]*?)```", response_text)
                    code = code_match.group(1).strip() if code_match else None
                    dataset_name = ""
                    if selected_ds and selected_ds in st.session_state.get("datasets", {}):
                        dataset_name = st.session_state.datasets[selected_ds]["name"]

                    run_result = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "blueprint": active_blueprint,
                        "task_brief": task_brief,
                        "constraints": st.session_state.get("agent_lab_constraints", "").strip(),
                        "deliverable": st.session_state.get("agent_lab_deliverable", "").strip(),
                        "artifact_text": artifact_text,
                        "response": response_text,
                        "review": agent_response.get("review", ""),
                        "reviewer": agent_response.get("reviewer"),
                        "review_error": agent_response.get("review_error"),
                        "agent": agent_response["agent"],
                        "role": agent_response["role"],
                        "task_mode": task_mode,
                        "response_style": response_style,
                        "dataset_id": selected_ds,
                        "dataset_name": dataset_name,
                        "code": code,
                        "prompt": prompt,
                    }
                    st.session_state.agent_lab_last_run = run_result
                    st.session_state.pop("agent_lab_sql_preview", None)
                    st.session_state.pop("agent_lab_forge_preview", None)

                    log_ai_interaction(
                        prompt,
                        response_text,
                        {
                            "surface": "agent_lab",
                            "blueprint": active_blueprint,
                            "agent": agent_response["agent"],
                            "dataset": dataset_name,
                            "task_mode": task_mode,
                            "response_style": response_style,
                            "reviewed": bool(agent_response.get("review") or agent_response.get("reviewer")),
                        },
                    )
                    add_history("Agent Lab", f"{active_blueprint} - {task_brief[:60] or 'artifact review'}")
                    status.update(label=f"Completed with {agent_response['agent']}", state="complete")
                except Exception as exc:
                    status.update(label="Agent Lab failed", state="error")
                    st.error(f"Agent Lab could not complete the request: {exc}")

    run = st.session_state.get("agent_lab_last_run")
    if run:
        operator_actions = build_operator_actions(run)
        st.divider()
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Blueprint", run["blueprint"])
        stat2.metric("Specialist", run["agent"])
        stat3.metric("Dataset", run["dataset_name"] or "None")

        response_tab, operator_tab, review_tab, code_tab, prompt_tab = st.tabs(["Response", "Operator", "Review", "Code", "Prompt"])

        with response_tab:
            st.markdown(run["response"])
            action1, action2 = st.columns(2)
            if action1.button("Pin Result", key="agent_lab_pin_result", width="stretch"):
                pin_to_report("insight", f"Agent Lab - {run['blueprint']}", run["response"], ds_name=run["dataset_name"])
            action2.download_button(
                "Download Markdown",
                run["response"],
                file_name="agent_lab_result.md",
                mime="text/markdown",
                width="stretch",
            )

        with operator_tab:
            st.markdown("### Operator Console")
            st.caption("Move this run into another UDIP surface or execute safe preview operations directly from Agent Lab.")
            action_cols = st.columns(2)
            for idx, action in enumerate(operator_actions):
                col = action_cols[idx % 2]
                col.markdown(f"**{action['label']}**")
                col.caption(action["description"])
                if col.button(action["label"], key=f"agent_operator_{action['id']}", width="stretch"):
                    if action["id"] == "run_sql_preview":
                        ok, message = run_sql_preview_from_run(run)
                        if ok:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)
                    elif action["id"] == "preview_forge_transform":
                        groq_key = load_config().get("groq_key", "")
                        with st.spinner("Generating Data Forge preview..."):
                            ok, message = preview_forge_transform_from_run(run, groq_key)
                        if ok:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)
                    else:
                        ok, message = execute_operator_action(action["id"], run)
                        if ok:
                            st.rerun()
                        st.warning(message)

            sql_preview = st.session_state.get("agent_lab_sql_preview")
            if sql_preview is not None:
                st.divider()
                st.markdown("#### SQL Preview")
                st.code(sql_preview["query"], language="sql")
                s1, s2 = st.columns(2)
                s1.metric("Rows returned", f"{sql_preview['rows']:,}")
                s2.metric("Columns", sql_preview["columns"])
                st.dataframe(sql_preview["result"].head(50), width="stretch")
                sp1, sp2 = st.columns(2)
                if sp1.button("Save SQL Preview as Dataset", key="agent_lab_save_sql_preview", width="stretch"):
                    ok, message = save_sql_preview_as_dataset()
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)
                if sp2.button("Pin SQL Preview", key="agent_lab_pin_sql_preview", width="stretch"):
                    pin_to_report("table", f"Agent Lab SQL Preview - {run['blueprint']}", sql_preview["query"], ds_name=run["dataset_name"])

            forge_preview = st.session_state.get("agent_lab_forge_preview")
            if forge_preview is not None:
                st.divider()
                st.markdown("#### Data Forge Preview")
                fp1, fp2, fp3 = st.columns(3)
                fp1.metric("Source rows", f"{forge_preview['source_rows']:,}")
                fp2.metric("Preview rows", f"{forge_preview['result_rows']:,}")
                fp3.metric("Validation", "Pass" if forge_preview["passed"] else "Review")
                st.info(forge_preview["explanation"])
                if forge_preview.get("changes"):
                    st.markdown("**Detected changes**")
                    for change in forge_preview["changes"]:
                        st.markdown(f"- {change}")
                if forge_preview.get("issues"):
                    st.markdown("**Validation notes**")
                    for issue in forge_preview["issues"]:
                        if issue.startswith("❌"):
                            st.error(issue)
                        else:
                            st.warning(issue)
                before_df = st.session_state.datasets[forge_preview["dataset_id"]]["df"]
                before_col, after_col = st.columns(2)
                before_col.markdown("**Before**")
                before_col.dataframe(before_df.head(10), width="stretch")
                after_col.markdown("**Preview result**")
                after_col.dataframe(forge_preview["result_df"].head(10), width="stretch")
                ff1, ff2 = st.columns(2)
                if ff1.button("Save Forge Preview as Dataset", key="agent_lab_save_forge_preview", width="stretch"):
                    ok, message = save_forge_preview_as_dataset()
                    if ok:
                        st.success(message)
                    else:
                        st.warning(message)
                if ff2.button("Pin Forge Summary", key="agent_lab_pin_forge_preview", width="stretch"):
                    summary = "\n".join(forge_preview.get("changes") or [forge_preview["explanation"]])
                    pin_to_report("analysis", f"Agent Lab Forge Preview - {run['blueprint']}", summary, ds_name=forge_preview["dataset_name"])

        with review_tab:
            if run.get("review"):
                st.markdown(run["review"])
            elif run.get("review_error"):
                st.caption(f"Review skipped: {run['review_error']}")
            else:
                st.caption("No separate review was generated for this run.")

        with code_tab:
            if run.get("code"):
                st.code(run["code"], language="python")
                code1, code2 = st.columns(2)
                if code1.button("Load into Code Editor", key="agent_lab_to_code", width="stretch"):
                    st.session_state.editor_code = run["code"]
                    save_code(run["code"])
                    st.session_state.active_page = "🐍 Code Editor"
                    st.rerun()
                if code2.button("Pin Code", key="agent_lab_pin_code", width="stretch"):
                    pin_to_report("code", f"Agent Lab Code - {run['blueprint']}", run["code"], ds_name=run["dataset_name"])
            else:
                st.caption("No Python block was returned for this run.")

        with prompt_tab:
            st.code(run["prompt"], language="markdown")
