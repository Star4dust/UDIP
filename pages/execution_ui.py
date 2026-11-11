import streamlit as st
import pandas as pd
from modules.core import handle_page_errors

@handle_page_errors
def render():
    """
    Renders the Pipeline Execution Flow UI.
    Provides a detailed look at the orchestrator's plan and the execution layer's steps.
    """
    st.markdown('<div class="main-header">⚡ Pipeline Execution Flow</div>', unsafe_allow_html=True)
    
    # ── Section 1: User Input ───────────────────────────────────────────────────
    with st.container():
        st.markdown("### 💬 User Request")
        query = st.session_state.get("last_query")
        if query:
            st.info(query)
        else:
            st.caption("No recent query found in session.")

    st.divider()

    # ── Section 2: Orchestration Plan ──────────────────────────────────────────
    with st.container():
        st.markdown("### 📋 Orchestration Plan")
        plan = st.session_state.get("execution_plan", {})
        if plan:
            with st.expander("View Raw Execution Plan", expanded=True):
                st.json(plan)
        else:
            st.caption("No execution plan currently active.")

    st.divider()

    # ── Section 3: Involved Agents ─────────────────────────────────────────────
    with st.container():
        st.markdown("### 🤖 Agents Involved")
        agents = st.session_state.get("agents_used", [])
        if agents:
            cols = st.columns(len(agents) if len(agents) > 0 else 1)
            for idx, agent in enumerate(agents):
                cols[idx % len(cols)].markdown(f'<div class="udip-pill udip-pill--accent">{agent}</div>', unsafe_allow_html=True)
        else:
            st.caption("No agents participated in the last run.")

    st.divider()

    # ── Section 4: Execution Workflow ──────────────────────────────────────────
    with st.container():
        st.markdown("### 🗺️ Workflow Timeline")
        logs = st.session_state.get("execution_logs", [])
        if logs:
            workflow_str = " → ".join([f"`{step.get('action')}`" for step in logs])
            st.markdown(f'<div style="background: var(--surface-soft); padding: 1rem; border-radius: 8px; border: 1px solid var(--line); margin-bottom: 1rem;">{workflow_str}</div>', unsafe_allow_html=True)
        else:
            st.caption("No workflow data available.")

    st.divider()

    # ── Section 5: Execution Steps ─────────────────────────────────────────────
    with st.container():
        st.markdown("### ⚙️ Step Details")
        if not logs:
            st.caption("No execution logs found.")
        else:
            for idx, step in enumerate(logs):
                action = step.get("action", "unknown_action")
                status = step.get("success", False)
                output = step.get("output", "")
                
                with st.expander(f"Step {idx+1}: {action}", expanded=not status):
                    if status:
                        st.success(f"Action: {action} succeeded.")
                        if output:
                            st.write(output)
                    else:
                        st.error(f"Action: {action} failed.")
                        if output:
                            st.write(output)
                        error_msg = step.get("error")
                        if error_msg:
                            st.info(f"Detail: {error_msg}")

    st.divider()

    # ── Section 6: Final Output ───────────────────────────────────────────────
    with st.container():
        st.markdown("### 🏁 Final Output")
        result = st.session_state.get("final_output")
        
        if result is None:
            st.info("No final output available for this execution.")
        else:
            # Handle DataFrame
            if isinstance(result, pd.DataFrame):
                st.markdown("#### Preview")
                st.dataframe(result.head(100), use_container_width=True)
                if st.button("Save as New Dataset"):
                    # This would ideally call a core function, but we keep it minimal per rules
                    st.info("Ready to export to local workspace.")
            
            # Handle Chart/Figure
            elif hasattr(result, "savefig") or "Figure" in str(type(result)):
                st.markdown("#### Visualization")
                st.pyplot(result)
            
            # Handle Text/Report
            elif isinstance(result, str):
                st.markdown("#### Report")
                st.write(result)
            
            # Handle Dictionary/JSON
            elif isinstance(result, dict):
                st.markdown("#### Data Structure")
                st.json(result)
            
            else:
                st.write(result)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.caption("Pipeline data is cleared when a new session starts or a new request is planned.")
