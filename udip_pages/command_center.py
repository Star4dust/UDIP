import streamlit as st
import pandas as pd
import time
import json
import uuid
from modules.orchestrator import plan_actions
from modules.pipeline_runner import run_pipeline
from modules import execution_layer
from modules.system_observer import SystemObserver
from modules.dataset_memory import DatasetMemory, DatasetSnapshotManager, compare_snapshots

observer = SystemObserver()
memory = DatasetMemory()
snap_manager = DatasetSnapshotManager(memory)

def render():
    st.markdown('<div class="main-header">Workflow Runs</div>', unsafe_allow_html=True)
    st.markdown("Orchestrated processing runs, execution traces, and dataset lineage.")

    # 1. Main Layout: Chat + Live Monitoring
    left_col, right_col = st.columns([1.5, 1])

    with left_col:
        st.markdown("### Orchestrator")
        user_prompt = st.chat_input("Instruct the system (e.g., 'load sales.csv, clean it and detect anomalies')")
        
        if "command_center_runs" not in st.session_state:
            st.session_state.command_center_runs = []

        if user_prompt:
            with st.status("🛸 Orchestrating Plan...", expanded=True) as status:
                st.write("Decomposing intent...")
                plan = plan_actions(user_prompt)
                st.write("Generating dependency graph...")
                time.sleep(0.5)
                
                st.write("Executing Pipeline...")
                result = run_pipeline(plan)
                
                st.session_state.command_center_runs.append({
                    "prompt": user_prompt,
                    "plan": plan,
                    "result": result,
                    "timestamp": time.time()
                })
                status.update(label="✅ Execution Complete", state="complete")

        # Display history
        for run in reversed(st.session_state.command_center_runs[-3:]):
            with st.chat_message("user"):
                st.markdown(run["prompt"])
            
            with st.chat_message("assistant"):
                res = run["result"]
                if res["status"] == "success":
                    st.success("Success")
                else:
                    st.error(f"Failed at {res.get('errors', [{}])[0].get('action', 'unknown')}")
                
                # Plan & Logs Tabs
                t1, t2, t3 = st.tabs(["📋 Plan", "📜 Execution Trace", "📦 Output"])
                with t1:
                    st.json(run["plan"])
                with t2:
                    st.dataframe(pd.DataFrame(res.get("steps_executed", [])), width="stretch")
                with t3:
                    final_res = res.get("final_result")
                    if isinstance(final_res, pd.DataFrame):
                        st.dataframe(final_res.head(50), width="stretch")
                    else:
                        st.json(final_res)

    with right_col:
        st.markdown("### 🔍 System Observer")
        metrics = observer.get_summary_metrics()
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Success Rate", f"{metrics['overview']['success_rate']:.1f}%")
        m_col2.metric("Avg Duration", f"{metrics['overview']['avg_run_time_ms']:.0f}ms")
        
        st.markdown("#### 🤖 Agent Activity")
        logs = observer.get_detailed_logs(limit=5)
        for log in logs:
            st.caption(f"[{log['timestamp']}] {log['run_id'][:8]} - {'✅' if log['success'] else '❌'}")

    st.divider()

    # 2. Dataset & Snapshot Section
    st.markdown("### 📦 Data & Lineage")
    tab_ds, tab_snap = st.tabs(["🗄️ Active Datasets", "📸 Snapshot Comparison"])

    with tab_ds:
        datasets = st.session_state.get("datasets", {})
        if not datasets:
            st.info("No active datasets in memory.")
        else:
            for ds_id, ds in datasets.items():
                with st.expander(f"📄 {ds['name']}", expanded=False):
                    st.dataframe(ds["df"].head(10), width="stretch")
                    st.caption(f"ID: {ds_id} | Memory: {ds['df'].memory_usage().sum() / 1024:.1f} KB")

    with tab_snap:
        st.markdown("Compare system snapshots to detect drift or schema changes.")
        snapshots = snap_manager.list_snapshots()
        if len(snapshots) < 2:
            st.info("Need at least 2 snapshots to compare.")
        else:
            c1, c2 = st.columns(2)
            s1 = c1.selectbox("Source Snapshot", snapshots)
            s2 = c2.selectbox("Target Snapshot", snapshots, index=1 if len(snapshots) > 1 else 0)
            
            if st.button("⚖️ Compare Snapshots"):
                res1 = execution_layer.load_snapshot(s1)
                res2 = execution_layer.load_snapshot(s2)
                
                if res1["success"] and res2["success"]:
                    diff = compare_snapshots(res1["output"], res2["output"], memory=memory)
                    st.json(diff)
                    if diff.get("drift_detected"):
                        st.warning("⚠️ Distribution drift detected between snapshots!")
                    else:
                        st.success("✅ No significant drift detected.")
                else:
                    st.error(f"Failed to load snapshots: {res1.get('error') or res2.get('error')}")
