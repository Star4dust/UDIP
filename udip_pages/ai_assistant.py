"""
UDIP v3 — udip_pages/ai_assistant.py
AI Assistant page — broad, conversational, web-connected data science companion.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, re, time, io
from modules.core import (
    add_history, save_chat, save_code, load_config, save_config,
    sanitize_varname, log
)
from modules.ai_engine import (
    should_search_web, search_web, extract_search_query, update_ai_memory, ask_ai,
    build_memory_context,
    extract_data_from_pdf, extract_data_from_image, ask_groq
)
from modules.multi_model_ai import render_ai_configuration
from modules.ai_agents import collaborative_response, get_all_agents
from modules.operator_actions import render_operator_handoff
from modules.state_manager import get_state_manager

def export_to_notebook(code: str):
    """Export code to Jupyter notebook and download."""
    import nbformat as nbf
    from nbformat.v4 import new_notebook, new_code_cell
    import base64

    nb = new_notebook()
    nb.cells = [new_code_cell(code)]
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}

    # Write to string
    from io import StringIO
    sio = StringIO()
    nbf.write(nb, sio)
    content = sio.getvalue()

    # Encode for download
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="udip_code.ipynb">Download notebook</a>'
    st.markdown(href, unsafe_allow_html=True)

def render():
    st.markdown('<div class="main-header">🤖 AI Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your data, get code help, and explore insights with the assistant.")
    render_operator_handoff("🤖 AI Assistant")

    if "selected_agent" not in st.session_state:
        st.session_state.selected_agent = "udip_orchestrator"
    if "ai_quality_review" not in st.session_state:
        st.session_state.ai_quality_review = True

    state_mgr = get_state_manager()
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

    st.markdown(
        """
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Assistant Controls</div>
            <div class="udip-panel__title">Answer shape and review</div>
            <div class="udip-panel__text">
                Choose how detailed the answer should be and optionally add a short focus note when you want the
                assistant to optimize for speed, safety, code quality, or explanation depth.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_mode, c_style, c_focus = st.columns([1, 1, 1.25])
    task_mode_options = ["Auto", "Plan", "Build", "Debug", "Research", "Rewrite", "Explain", "Review"]
    style_options = ["Balanced", "Concise", "Deep"]
    with c_mode:
        st.selectbox("Task mode", task_mode_options, key="ai_task_mode")
    with c_style:
        st.selectbox("Answer style", style_options, key="ai_response_style")
    with c_focus:
        st.text_input(
            "Focus notes",
            key="ai_focus_notes",
            placeholder="Speed, safety, code quality, citations, clarity...",
        )

    preset_cols = st.columns(6)
    presets = [
        ("Plan", "Plan", "Balanced", "udip_orchestrator", "Outline the work before executing it."),
        ("Build", "Build", "Balanced", "udip_orchestrator", "Return a runnable implementation or concrete artifact."),
        ("Debug", "Debug", "Balanced", "code_architect", "Find the root cause, propose the fix, and explain how to verify it."),
        ("Research", "Research", "Deep", "udip_orchestrator", "Synthesize the best available information clearly."),
        ("Review", "Review", "Concise", "qa_sentinel", "Audit the answer for risks, bugs, and hallucinations."),
        ("Rewrite", "Rewrite", "Concise", "code_architect", "Improve clarity, structure, and polish."),
    ]
    for col, (label, mode, style, agent, note) in zip(preset_cols, presets):
        if col.button(label, key=f"ai_preset_{mode.lower()}", width="stretch"):
            st.session_state.ai_task_mode = mode
            st.session_state.ai_response_style = style
            st.session_state.ai_focus_notes = note
            st.session_state.selected_agent = agent
            st.rerun()

    ai_state.update({
        "selected_agent": st.session_state.get("selected_agent", "udip_orchestrator"),
        "task_mode": st.session_state.get("ai_task_mode", "Auto").lower(),
        "response_style": st.session_state.get("ai_response_style", "Balanced").lower(),
        "focus_notes": st.session_state.get("ai_focus_notes", ""),
    })
    state_mgr.save_page_state("ai", ai_state)

    # ── Layout, review toggle and API key (used by the chat and code panels below) ──
    ui_mode = st.radio(
        "Layout",
        ["💬 Chat only", "⚡ Chat + Code (split screen)", "💻 Code only"],
        horizontal=True,
        key="ai_ui_mode",
    )
    review_enabled = st.toggle("Quality review (second-pass check of answers)", key="ai_quality_review")
    selected_agent = st.session_state.selected_agent
    groq_key = (
        st.session_state.get("groq_key")
        or os.getenv("GROQ_API_KEY", "")
        or load_config().get("groq_key", "")
    )

    # ── AI Model Configuration ─────────────────────────────────
    with st.expander("⚙️ AI Configuration", expanded=False):
        render_ai_configuration()
    
    with st.expander("🧠 Memory Vault", expanded=False):
        mem = st.session_state.get("ai_memory", {}) or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("Sessions", mem.get("session_count", 0))
        m2.metric("Tracked ops", len(mem.get("frequent_operations", {})))
        m3.metric("Tracked columns", len(mem.get("frequent_columns", {})))
        mem_summary = build_memory_context().strip()
        if mem_summary:
            st.markdown(mem_summary)
        else:
            st.caption("No learned memory yet. Ask a few questions and UDIP will start adapting to you.")

    def get_ds_context():
        selected_ds = st.session_state.get("ai_selected_dataset")
        datasets = st.session_state.get("datasets", {})
        if selected_ds and selected_ds in datasets:
            ds = datasets[selected_ds]
            cols = ", ".join(str(c) for c in ds["df"].columns[:8])
            return (
                f"ACTIVE DATASET:\n"
                f"- Name: {ds['name']}\n"
                f"- ID: {selected_ds}\n"
                f"- Shape: {ds['df'].shape[0]:,} rows x {ds['df'].shape[1]} columns\n"
                f"- Columns: {cols}{'...' if len(ds['df'].columns) > 8 else ''}\n"
            )
        return "No active dataset selected. The AI assistant will use available datasets from session context."
    
    # ── Render code panel ─────────────────────────────────────────────────────
    def render_code_panel(container):
        """Render the AI code editor and execution interface."""
        with container:
            st.markdown("### ⚡ AI Code Studio")
            editor_val = st.session_state.get("editor_code", "")
            try:
                from streamlit_ace import st_ace
                edited_code = st_ace(
                    value=editor_val, language="python", theme="vs",
                    height=340, key="ai_live_editor_ace",
                    wrap=False,
                    auto_update=False,
                )
                if not edited_code:
                    edited_code = editor_val
            except ImportError:
                edited_code = st.text_area("Code", value=editor_val, height=340,
                                            key="ai_live_editor", label_visibility="collapsed")
            st.session_state.editor_code = edited_code
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
            run_live = c1.button("▶️ Run", type="primary", width="stretch", key="live_run_btn", help="Execute the code in a sandbox environment")
            preview_btn = c2.button("👁️ Preview", width="stretch", key="live_preview_btn", help="Show what the code will do without executing")
            validate_btn = c3.button("✅ Validate", width="stretch", key="live_validate_btn", help="Check code syntax and security")
            lint_btn = c4.button("🔧 Lint", width="stretch", key="live_lint_btn", help="Check code style and quality")
            automl_btn = c5.button("🤖 AutoML", width="stretch", key="live_automl_btn", help="Run automated machine learning on dataset")
            review_code = c6.button("🔍 Review", width="stretch", key="live_review_btn", help="Get AI feedback on code quality and best practices")
            if c7.button("💾 Save", width="stretch", key="live_save_btn", help="Save code to the Code Editor page"):
                save_code(edited_code); st.toast("✅ Saved to Code Editor")
            if c8.button("📓 Export .ipynb", width="stretch", key="live_export_btn", help="Download code as Jupyter notebook"):
                export_to_notebook(edited_code)
            if c9.button("🗑️ Clear", width="stretch", key="live_clear_btn", help="Clear the editor and reset"):
                st.session_state.editor_code = ""; st.session_state._last_ai_code = None; st.rerun()
            if preview_btn and edited_code.strip():
                from modules.ai_engine import preview_code_execution
                preview = preview_code_execution(edited_code)
                st.info("👁️ Execution Preview:")
                st.code(preview, language="text")
            if automl_btn:
                # Show AutoML interface
                st.subheader("🤖 Automated Machine Learning")
                
                datasets = st.session_state.get("datasets", {})
                if not datasets:
                    st.warning("Upload datasets first for AutoML.")
                else:
                    # Dataset selection
                    dataset_options = {ds_id: f"{ds['name']} ({len(ds['df'])} rows)" for ds_id, ds in datasets.items()}
                    selected_ds = st.selectbox("Select Dataset", options=list(dataset_options.keys()), 
                                             format_func=lambda x: dataset_options[x])
                    
                    if selected_ds:
                        df = datasets[selected_ds]["df"]
                        columns = list(df.columns)
                        
                        # Target column selection
                        target_col = st.selectbox("Target Column (what to predict)", columns)
                        
                        # Problem type
                        problem_type = st.selectbox("Problem Type", 
                                                  ["auto", "binary", "multiclass", "regression"],
                                                  help="Auto-detects if not specified")
                        
                        # Time limit
                        time_limit = st.slider("Training Time (seconds)", 60, 1800, 300)
                        
                        if st.button("🚀 Run AutoML", type="primary"):
                            with st.spinner("Training models... This may take several minutes"):
                                from modules.ai_engine import run_automl
                                result = run_automl(df, target_col, problem_type, time_limit)
                            
                            if result['success']:
                                st.success("✅ AutoML completed!")
                                
                                # Store model ID in session
                                st.session_state.last_automl_model = result['model_id']
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("🏆 Model Leaderboard")
                                    leaderboard_df = pd.DataFrame(result['leaderboard'])
                                    st.dataframe(leaderboard_df.head(10), use_container_width=True)
                                
                                with col2:
                                    st.subheader("🎯 Best Model")
                                    st.info(f"**{result['best_model']}**")
                                    st.caption(f"Model ID: {result['model_id']}")
                                    
                                    if result.get('feature_importance'):
                                        st.subheader("📊 Feature Importance")
                                        fi_df = pd.DataFrame(result['feature_importance'])
                                        st.dataframe(fi_df.head(10), use_container_width=True)
                                
                                st.subheader("🔮 Sample Predictions")
                                pred_df = pd.DataFrame(result['predictions_sample'])
                                st.dataframe(pred_df, use_container_width=True)
                                
                                # Model actions
                                col1, col2, col3 = st.columns(3)
                                if col1.button("💾 Save Model", key="save_automl"):
                                    st.success(f"Model saved with ID: {result['model_id']}")
                                if col2.button("🔮 Make Predictions", key="predict_automl"):
                                    st.info("Use the model ID in code: predictor = load_automl_model('" + result['model_id'] + "')")
                                if col3.button("📊 Evaluate", key="eval_automl"):
                                    st.info("Use the leaderboard and feature importance above to evaluate model quality.")
                                
                            else:
                                st.error(f"❌ AutoML failed: {result['error']}")
            if validate_btn and edited_code.strip():
                from modules.ai_engine import validate_code
                valid, msg = validate_code(edited_code)
                if valid:
                    st.success("✅ Code is valid and secure")
                else:
                    st.error(f"❌ Validation failed: {msg}")
            if lint_btn and edited_code.strip():
                from modules.ai_engine import lint_code
                lint_result = lint_code(edited_code)
                if "No linting issues" in lint_result:
                    st.success("✅ No linting issues")
                else:
                    st.warning(f"🔧 Linting issues:\n{lint_result}")
            if review_code and edited_code.strip():
                prompt = f"Please review this Python code for data analysis. Provide feedback on best practices, potential improvements, bugs, and efficiency:\n\n```python\n{edited_code}\n```"
                st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": time.time()})
                with st.spinner("Reviewing code..."):
                    reply, code, confidence = ask_groq(
                        prompt,
                        groq_key,
                        context=get_ds_context(),
                        selected_dataset_id=st.session_state.get("ai_selected_dataset"),
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": reply, "timestamp": time.time()})
                st.info(f"🤖 AI Confidence: {confidence:.1%}")
                st.rerun()
            if run_live and edited_code.strip():
                st.session_state._last_run_error = None
                term = st.empty()
                term.code("▶ Running...", language="text")
                t0 = time.time()
                from modules.ai_engine import run_code_with_healing as _rc
                stdout, error, success, final_code, figures = _rc(edited_code, groq_key, live_placeholder=term)
                if final_code != edited_code:
                    st.session_state.editor_code = final_code
                    st.session_state._last_ai_code = final_code
                dur = round(time.time()-t0, 2)
                if success:
                    term.empty()
                    st.success(f"✅ Completed in {dur}s")
                    if stdout.strip():
                        # Try to detect and render dataframes in output
                        _lines = stdout.strip().split("\n")
                        _has_table = any("|" in l or ("  " in l and len(l) > 30) for l in _lines[:5])
                        if _has_table or len(_lines) > 5:
                            with st.expander("📋 Output", expanded=True):
                                if len(stdout) > 2000:
                                    st.code(stdout[:2000] + "\n... (truncated)", language="text")
                                    if st.button("Show full output"):
                                        st.code(stdout, language="text")
                                else:
                                    st.code(stdout, language="text")
                                # Export options
                                col1, col2, col3 = st.columns(3)
                                if col1.button("📄 Export as TXT", key="export_txt"):
                                    st.download_button("Download TXT", stdout, "output.txt", mime="text/plain")
                                if col2.button("📊 Export as CSV", key="export_csv"):
                                    # Try to parse as dataframe
                                    try:
                                        df_out = pd.read_csv(io.StringIO(stdout), sep=r'\s{2,}', engine='python')
                                        csv_data = df_out.to_csv(index=False)
                                        st.download_button("Download CSV", csv_data, "output.csv", mime="text/csv")
                                    except:
                                        st.error("Could not parse output as CSV")
                                if col3.button("🖼️ Export as Image", key="export_img"):
                                    # If matplotlib figures were created, export last one
                                    if 'captured_figs' in locals() and captured_figs:
                                        buf = io.BytesIO()
                                        captured_figs[-1].savefig(buf, format='png')
                                        buf.seek(0)
                                        st.download_button("Download PNG", buf, "plot.png", mime="image/png")
                                    else:
                                        st.error("No plot to export")
                        else:
                            st.code(stdout, language="text")
                    # Show updated datasets preview
                    if st.session_state.datasets:
                        _ds_list = list(st.session_state.datasets.values())
                        if _ds_list:
                            with st.expander("📊 Dataset preview after run", expanded=False):
                                _preview_df = _ds_list[0]["df"]
                                st.caption(f"{len(_preview_df):,} rows × {len(_preview_df.columns)} cols")
                                st.dataframe(_preview_df.head(8), use_container_width=True)
                    add_history("Code run", edited_code[:60])
                else:
                    term.empty()
                    st.session_state._last_run_error = error

            # Variable Inspector
            if st.session_state.datasets:
                with st.expander("🔬 Variable Inspector", expanded=False):
                    ds_id = st.selectbox("Dataset", list(st.session_state.datasets.keys()),
                                          format_func=lambda k: st.session_state.datasets[k]["name"],
                                          key="var_ins_ds")
                    df_vi = st.session_state.datasets[ds_id]["df"]
                    col_vi = st.selectbox("Column", df_vi.columns.tolist(), key="var_ins_col")
                    s = df_vi[col_vi]
                    is_num = pd.api.types.is_numeric_dtype(s)
                    v1,v2,v3 = st.columns(3)
                    v1.metric("Min",    f"{s.min():.4g}" if is_num else str(s.min())[:12])
                    v2.metric("Max",    f"{s.max():.4g}" if is_num else str(s.max())[:12])
                    v3.metric("Nulls",  int(s.isnull().sum()))
                    v1.metric("Mean",   f"{s.mean():.4g}" if is_num else "—")
                    v2.metric("Std",    f"{s.std():.4g}"  if is_num else "—")
                    v3.metric("Unique", int(s.nunique()))
                    st.dataframe(s.head(8).to_frame(), width="stretch")

    def render_chat_panel(container):
        _groq_key = groq_key  # capture from outer scope
        with container:
            # Chat history
            for msg_idx, msg in enumerate(st.session_state.chat_history):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        # Split message by code blocks to interleave buttons
                        parts = re.split(r"(```python\n[\s\S]*?```)", msg["content"])
                        code_idx = 0
                        for part in parts:
                            if part.startswith("```python"):
                                st.markdown(part)
                                # Extract code for buttons
                                code_content = re.search(r"```python\n([\s\S]*?)```", part).group(1).strip()
                                
                                # Buttons for THIS specific block
                                ba, bb = st.columns([1,1])
                                with ba:
                                    if st.button(f"▶️ Run Block {code_idx+1}", key=f"irun_{msg_idx}_{code_idx}"):
                                        st.session_state._last_ai_code = code_content
                                        st.session_state.editor_code   = code_content
                                        
                                        # Execute and capture figures
                                        from modules.ai_engine import run_code_with_healing as _rc
                                        term = st.empty()
                                        stdout, error, success, final_code, figures = _rc(code_content, groq_key, live_placeholder=term)
                                        
                                        # Store in session state for persistence
                                        st.session_state.last_ai_run_results = {
                                            "stdout": stdout, "error": error, "success": success, 
                                            "final_code": final_code, "figures": figures, "block_id": f"{msg_idx}_{code_idx}"
                                        }
                                        st.rerun()
                                with bb:
                                    if st.button(f"📋 To editor", key=f"icopy_{msg_idx}_{code_idx}"):
                                        st.session_state.editor_code = code_content
                                        save_code(code_content); st.toast(f"✅ Block {code_idx+1} loaded")
                                code_idx += 1
                            else:
                                if part.strip():
                                    st.markdown(part)
                        # Action row
                        r1,r2,r3,r4,r5,_ = st.columns([1.5,1,1,1,1,3])
                        with r1:
                            if st.button("↩️ Reply", key=f"reply_{msg_idx}", help="Reply to this specific message"):
                                st.session_state._reply_quote = f"> {msg['content'][:60]}...\n\n"
                                st.rerun()
                        with r2:
                            if st.button("👍", key=f"like_{msg_idx}", help="Helpful"):
                                from modules.ai_logger import log_ai_feedback
                                log_ai_feedback(msg_idx, "like")
                                st.toast("Thanks for the feedback! 💚")
                        with r3:
                            if st.button("👎", key=f"dislike_{msg_idx}", help="Not helpful"):
                                from modules.ai_logger import log_ai_feedback
                                log_ai_feedback(msg_idx, "dislike")
                                st.toast("Feedback recorded. I'll do better! 🤖")
                        with r4:
                            if st.button("🔧", key=f"fix_{msg_idx}", help="Fix error in last code"):
                                err = st.session_state.get("_last_run_error","")
                                st.session_state._ai_trigger = (
                                    f"The code failed:\n```\n{err[:400]}\n```\nFix it."
                                    if err else "Review and improve the last code.")
                                st.rerun()
                        with r5:
                            if st.button("📌 Pin to Report", key=f"pin_{msg_idx}", help="Pin this insight to the Report Studio"):
                                from modules.core import pin_to_report
                                pin_to_report("insight", "AI Assistant Insight", msg["content"])
                                st.toast("Pinned to Report Studio! 📖")

            # --- PERSISTENT RESULTS FOR PINNING ---
            if "last_ai_run_results" in st.session_state:
                res = st.session_state.last_ai_run_results
                with st.expander("⚡ Run AI-generated code", expanded=True):
                    if res.get("figures"):
                        for fi, img_bytes in enumerate(res["figures"]):
                            st.image(img_bytes)
                            if st.button(f"📌 Pin Chart {fi+1} to Report Studio", key=f"pin_ai_fig_{fi}"):
                                from modules.core import pin_to_report
                                pin_to_report("chart", f"AI Result Chart {fi+1}", {"image_bytes": img_bytes})
                                st.toast(f"✅ Chart {fi+1} pinned!")
                    
                    if res.get("stdout"):
                        st.markdown("**Console Output:**")
                        st.code(res["stdout"], language="text")
                    
                    if res.get("error"):
                        st.error(f"❌ Execution Error:\n{res['error']}")
                    elif res.get("success"):
                        st.success("✅ Code executed successfully!")

            # Pinned responses (now managed in Report Studio)
            st.info("📌 Pinned insights are now saved directly to the **Report Studio** page.")

            # Reply box
            if st.session_state.pop("_reply_quote", None):
                st.session_state._pending_reply = st.session_state.get("_reply_quote","")
            pending = st.session_state.get("_pending_reply","")
            if pending:
                st.info(f"**Replying to:** {pending[:80]}...")
                rt = st.text_area("Reply", key="reply_box", height=70)
                c1,c2 = st.columns([1,4])
                if c1.button("Send ↗", type="primary"):
                    st.session_state._pending_reply=""
                    st.session_state._ai_trigger=pending+rt; st.rerun()
                if c2.button("Cancel"): st.session_state._pending_reply=""; st.rerun()

            # Run last AI code (chat-only mode)
            last_code = st.session_state.get("_last_ai_code")
            if last_code and ui_mode == "💬 Chat only":
                st.divider()
                st.markdown("### ⚡ Run AI-generated code")
                if st.toggle("👁️ Show code", value=True, key="show_code_tog"):
                    try:
                        from streamlit_ace import st_ace
                        st_ace(value=last_code, language="python", theme="vscode", readonly=True, height=250, key="ai_code_show", show_gutter=False)
                    except:
                        st.code(last_code, language="python")
                cr,ce,cd = st.columns(3)
                run_main = cr.button("▶️ Run on my data", type="primary", width="stretch")
                if ce.button("✏️ Open in editor", width="stretch"):
                    st.session_state.editor_code=last_code; save_code(last_code); st.toast("✅ Loaded!")
                if cd.button("✖ Dismiss", width="stretch"):
                    st.session_state._last_ai_code=None; st.session_state._last_run_error=None; st.rerun()
                if run_main:
                    st.session_state._last_run_error=None
                    term=st.empty(); term.code("▶ Starting...", language="text")
                    t0=time.time()
                    from modules.ai_engine import run_code_with_healing as _rc2
                    stdout, error, success, final_code, figures = _rc2(last_code, groq_key, live_placeholder=term)
                    if success and figures:
                        if len(figures) > 5:
                            st.warning(f"⚠️ Showing first 5 charts out of {len(figures)} to preserve performance.")
                        for fi, fig in enumerate(figures[:5]):
                            st.pyplot(fig)
                            if st.button(f"📌 Pin Chart {fi+1} to Report", key=f"main_pin_fig_{fi}"):
                                from modules.core import pin_to_report
                                import plotly.tools as tls
                                try:
                                    import plotly.io as pio
                                    plotly_fig = tls.mpl_to_plotly(fig)
                                    pin_to_report("chart", f"AI Generated Chart", {"plotly_json": pio.to_json(plotly_fig)})
                                except:
                                    pin_to_report("insight", "AI Generated Chart (Ref)", f"Chart generated from AI analysis.")
                    if final_code != last_code:
                        st.session_state.editor_code = final_code
                        st.session_state._last_ai_code = final_code
                        last_code = final_code
                    dur=round(time.time()-t0,2)
                    if success:
                        st.success(f"✅ {dur}s")
                        if stdout.strip(): term.code(stdout, language="text")
                        add_history("AI code applied", last_code[:80])
                    else:
                        term.empty(); st.session_state._last_run_error=error

            # Error panel
            if st.session_state.get("_last_run_error"):
                err = st.session_state._last_run_error
                st.error("❌ Code failed:"); st.code(err, language="text")
                if "ModuleNotFoundError" in err: st.warning("💡 Missing library — click Run again to auto-install.")
                elif "KeyError" in err:          st.warning("💡 Column not found — check Variable Inspector.")
                elif "SyntaxError" in err:       st.warning("💡 Syntax error — ask AI to fix it.")
                else:                            st.warning("💡 Ask AI to diagnose and fix below.")
                if st.button("🤖 Ask AI to fix this error", type="primary", key="fix_err_btn"):
                    fp = f"The code failed:\n\n```\n{err[:800]}\n```\n\nDiagnose the root cause and provide a corrected version."
                    if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content")!=fp:
                        st.session_state.chat_history.append({"role":"user","content":fp}); save_chat(st.session_state.chat_history)
                    st.session_state._ai_trigger=fp; st.session_state._last_run_error=None; st.rerun()

            # Dataset peek
            if st.session_state.datasets:
                with st.expander("📋 View loaded datasets", expanded=False):
                    ds_pk = st.selectbox("Dataset", list(st.session_state.datasets.keys()),
                                          format_func=lambda k: st.session_state.datasets[k]["name"], key="ai_ds_pk")
                    df_pk = st.session_state.datasets[ds_pk]["df"]
                    ta,tb,tc = st.tabs(["Sample","Columns","Stats"])
                    with ta: st.dataframe(df_pk.head(20), width="stretch")
                    with tb: st.dataframe(pd.DataFrame({"Column":df_pk.columns,"Type":df_pk.dtypes.astype(str).values,"Nulls":df_pk.isnull().sum().values,"Unique":[df_pk[c].nunique() for c in df_pk.columns]}), width="stretch")
                    with tc: st.dataframe(df_pk.describe(include="all").T, width="stretch")

            # ── AI Recipes / Templates ──────────────────────────────────────────
            with st.expander("📝 AI Recipes", expanded=False):
                st.markdown("Choose a quick task to perform on your active dataset:")
                r1, r2 = st.columns(2)
                if r1.button("📊 Describe Dataset", use_container_width=True):
                    st.session_state._ai_trigger = "Provide a comprehensive statistical description of the active dataset. Include shape, missing values summary, and key statistics for numeric columns."
                    st.rerun()
                if r2.button("🧹 Clean Data", use_container_width=True):
                    st.session_state._ai_trigger = "Write code to clean the active dataset: handle missing values (impute or drop), remove duplicates, and fix common data type issues."
                    st.rerun()
                if r1.button("📈 Build Viz", use_container_width=True):
                    st.session_state._ai_trigger = "Suggest and create the most insightful visualization for this dataset based on its columns and distribution."
                    st.rerun()
                if r2.button("🗄️ SQL Summary", use_container_width=True):
                    st.session_state._ai_trigger = "Write a SQL query using DuckDB to provide a multi-dimensional summary of the active dataset."
                    st.rerun()
                if st.button("📄 Analyze Document (PDF/Image)", use_container_width=True):
                    st.session_state._ai_trigger = "I want to analyze a document. Please tell me to upload it first using the 'Document Analysis' tool below."
                    st.rerun()
            
            # ── Document Analysis ──────────────────────────────────────────────
            with st.expander("👁️ Document Analysis (PDF/OCR)", expanded=False):
                doc_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
                if doc_file:
                    if st.button("🚀 Extract & Analyze"):
                        with st.spinner("Extracting content..."):
                            file_bytes = doc_file.read()
                            if doc_file.name.endswith(".pdf"):
                                content = extract_data_from_pdf(file_bytes)
                            else:
                                content = extract_data_from_image(file_bytes)
                            
                            st.session_state._ai_trigger = f"I have extracted the following content from a document ({doc_file.name}):\n\n{content[:4000]}\n\nPlease summarize this data and suggest how it can be converted into a UDIP dataset."
                            st.rerun()

            # Chat input
            user_input = st.chat_input("Ask UDIP AI anything about your data...")
            trigger = user_input or st.session_state.pop("_ai_trigger", None)

            if trigger:
                if not st.session_state.chat_history or st.session_state.chat_history[-1].get("content")!=trigger:
                    st.session_state.chat_history.append({"role":"user","content":trigger})
                    save_chat(st.session_state.chat_history)

                with st.chat_message("assistant", avatar="🤖"):
                    # Show more context in thinking label
                    ds_ctx = get_ds_context()
                    active_ds_name = "Dataset"
                    selected_ds = st.session_state.get("ai_selected_dataset")
                    if selected_ds and selected_ds in st.session_state.datasets:
                        active_ds_name = st.session_state.datasets[selected_ds]["name"]
                    
                    lbl = f"🌐 Searching & Thinking (using {active_ds_name})..." if should_search_web(trigger) else f"🧠 Thinking (using {active_ds_name})..."
                    with st.status(lbl, expanded=True) as status:
                        try:
                            web_ctx=""
                            if should_search_web(trigger):
                                q = extract_search_query(trigger)
                                if q:
                                    status.write(f"🌐 Searching: *{q}*...")
                                    web_ctx = search_web(q)

                            # Use collaborative AI agent system
                            task_mode = st.session_state.get("ai_task_mode", "Auto")
                            response_style = st.session_state.get("ai_response_style", "Balanced")
                            focus_notes = st.session_state.get("ai_focus_notes", "").strip()
                            status.write(f"🤖 Using the assistant in {task_mode.lower()} mode...")
                            extra_context = ""
                            if web_ctx:
                                extra_context = f"WEB RESEARCH:\n{web_ctx}"

                            agent_response = collaborative_response(
                                trigger,
                                selected_agent,
                                selected_dataset_id=st.session_state.get("ai_selected_dataset"),
                                deep_review=review_enabled,
                                task_mode=task_mode.lower(),
                                response_style=response_style.lower(),
                                focus_notes=focus_notes,
                                extra_context=extra_context,
                                allow_web_search=not bool(web_ctx),
                            )
                            reply_text = agent_response["response"]
                            agent_name = agent_response["agent"]
                            agent_role = agent_response["role"]
                            reviewer_name = agent_response.get("reviewer")

                            # Update status with agent info and dataset context
                            selected_ds = st.session_state.get("ai_selected_dataset")
                            ds_badge = ""
                            if selected_ds and selected_ds in st.session_state.datasets:
                                ds_name = st.session_state.datasets[selected_ds]["name"]
                                ds_badge = f" | 📊 {ds_name}"
                            review_badge = f" | reviewed by {reviewer_name}" if reviewer_name else ""
                            status.update(label=f"✅ {agent_name} ({agent_role}) | {task_mode}{review_badge}{ds_badge}", state="complete")

                            # Extract code and calculate confidence
                            code = None
                            cm = re.search(r"```python\n([\s\S]*?)```", reply_text)
                            if cm:
                                code = cm.group(1).strip()

                            # Calculate confidence score
                            from modules.ai_engine import calculate_confidence
                            confidence = calculate_confidence(trigger, reply_text, ds_ctx)

                            st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
                            save_chat(st.session_state.chat_history)

                            st.markdown(reply_text)
                            if agent_response.get("review"):
                                with st.expander("🛡️ Agent review", expanded=False):
                                    st.markdown(agent_response["review"])
                            elif agent_response.get("review_error"):
                                st.caption(f"Review skipped: {agent_response['review_error']}")
                            
                            # Show confidence score and dataset context
                            col_conf, col_ds = st.columns([1, 2])
                            with col_conf:
                                conf_color = "🟢" if confidence >= 0.7 else ("🟡" if confidence >= 0.5 else "🔴")
                                st.caption(f"{conf_color} Confidence: {confidence:.1%}")
                            with col_ds:
                                if selected_ds and selected_ds in st.session_state.datasets:
                                    ds = st.session_state.datasets[selected_ds]
                                    st.caption(f"📊 Using: **{ds['name']}** ({ds['df'].shape[0]:,} rows × {ds['df'].shape[1]} cols)")
                            
                            # Detailed Context Snapshot
                            if selected_ds and selected_ds in st.session_state.datasets:
                                with st.expander("🔍 Context Snapshot", expanded=False):
                                    ds = st.session_state.datasets[selected_ds]
                                    st.markdown(f"**Dataset ID:** `{selected_ds}`")
                                    st.markdown(f"**Shape:** {ds['df'].shape[0]:,} rows × {ds['df'].shape[1]} columns")
                                    st.markdown(f"**Columns:** {', '.join(ds['df'].columns[:15])}{'...' if len(ds['df'].columns) > 15 else ''}")
                                    st.caption("This info was provided to the AI for this specific query.")

                            if code:
                                st.session_state.editor_code = code
                                st.session_state._last_ai_code = code
                            
                            # Validation check
                            from modules.ai_engine import validate_ai_response
                            valid, val_msg = validate_ai_response(trigger, reply_text, ds_ctx)
                            if not valid:
                                st.warning(f"⚠️ {val_msg}")
                            
                            # Log interaction
                            from modules.ai_logger import log_ai_interaction
                            log_ai_interaction(trigger, reply_text, {
                                "dataset": active_ds_name,
                                "agent": agent_name,
                                "confidence": confidence,
                                "has_code": code is not None,
                                "validated": valid,
                                "reviewed": review_enabled,
                                "reviewer": reviewer_name,
                                "task_mode": task_mode.lower(),
                                "response_style": response_style.lower(),
                            })
                        except Exception as e:
                            err=str(e).lower(); status.update(label="❌ Error", state="error")
                            if "rate_limit" in err: st.warning("⏳ Rate limit — wait 30s and try again.")
                            elif "invalid_api_key" in err: st.error("❌ Invalid API key.")
                            else: st.error(f"Error: {e}")
                add_history("AI assistant", f"{trigger[:70]}")
                st.rerun()

            # Clear chat
            if st.session_state.chat_history:
                if st.button("🗑️ Clear conversation"):
                    st.session_state.chat_history=[]; st.session_state._last_ai_code=None
                    st.session_state._pinned_msgs=[]; save_chat([]); st.rerun()

    # ── Render based on mode ──────────────────────────────────────────────────
    if ui_mode == "💬 Chat only":
        render_chat_panel(st.container())
    elif ui_mode == "⚡ Chat + Code (split screen)":
        cc, cp = st.columns([3,2])
        with cc: render_chat_panel(st.container())
        with cp: render_code_panel(st.container())
    else:
        render_code_panel(st.container())
