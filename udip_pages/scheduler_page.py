import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
UDIP v3 — pages/scheduler_page.py
Schedule automated analysis tasks.
"""

import streamlit as st
import os
from pathlib import Path
from modules.core import log, add_history, load_code, fmt_bytes
from modules.scheduler import (
    create_schedule, run_schedule_now, delete_schedule,
    register_with_windows, get_schedule_outputs, OUTPUTS_DIR
)

CODE_TEMPLATES = {
    "Daily data summary": """# Auto-generated daily summary
import pandas as pd
from datetime import datetime

print(f"=== Daily Summary — {datetime.now().strftime('%Y-%m-%d')} ===")
for ds_id, ds in datasets.items():
    df = ds['df']
    print(f"\\n{ds['name']}: {len(df):,} rows x {len(df.columns)} cols")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        print(df[num_cols].describe().round(2).to_string())
    print(f"Nulls: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
""",
    "Clean and export": """# Auto clean and export
import pandas as pd

for ds_id, ds in datasets.items():
    df = ds['df'].copy()
    rows_before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    rows_after = len(df)
    print(f"{ds['name']}: {rows_before:,} → {rows_after:,} rows ({rows_before-rows_after} removed)")
    result = df  # This gets auto-exported
    print(f"Result ready: {len(result):,} rows")
""",
    "Outlier detection report": """# Outlier detection report
import pandas as pd
import numpy as np

print("=== Outlier Detection Report ===")
for ds_id, ds in datasets.items():
    df = ds['df']
    num_cols = df.select_dtypes(include='number').columns.tolist()
    print(f"\\n{ds['name']}:")
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)]
        pct = len(outliers)/len(df)*100
        print(f"  {col}: {len(outliers):,} outliers ({pct:.1f}%)")
""",
    "Merge all datasets": """# Merge all datasets on common columns
import pandas as pd

dfs = [ds['df'] for ds in datasets.values()]
names = [ds['name'] for ds in datasets.values()]

if len(dfs) < 2:
    print("Need at least 2 datasets")
else:
    # Find common columns
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    
    if common:
        key = list(common)[0]
        print(f"Merging on: {key}")
        result = dfs[0]
        for i, df in enumerate(dfs[1:]):
            result = pd.merge(result, df, on=key, how='inner', suffixes=('','_'+str(i+2)))
        result = result.drop_duplicates()
        print(f"Merged result: {result.shape}")
    else:
        result = pd.concat(dfs, ignore_index=True)
        print(f"Concatenated: {result.shape}")
""",
}

def render():
    st.markdown('<div class="main-header">⏰ Task Scheduler</div>', unsafe_allow_html=True)
    st.markdown("Automate your data processing. Tasks run on a schedule and save outputs automatically.")

    schedules = st.session_state.get("schedules", [])

    # ── Create new schedule ────────────────────────────────────────────────────
    with st.expander("➕ Create new scheduled task", expanded=len(schedules)==0):
        col1, col2 = st.columns(2)
        with col1:
            task_name  = st.text_input("Task name", placeholder="Daily sales summary", key="sched_name")
            frequency  = st.selectbox("Frequency", ["Daily","Weekly","Hourly","On startup"], key="sched_freq")
            run_time   = st.time_input("Run at", key="sched_time")
        with col2:
            export_fmt = st.selectbox("Auto-export format", ["csv","excel","parquet"], key="sched_fmt")
            notify     = st.checkbox("Show notification when done", value=True, key="sched_notify")
            template   = st.selectbox("Start from template", ["— blank —"] + list(CODE_TEMPLATES.keys()), key="sched_tmpl")

        code_default = CODE_TEMPLATES.get(template, "# Write your automation script here\nimport pandas as pd\n\n# Access datasets:\n# for ds_id, ds in datasets.items():\n#     df = ds['df']\n\n# Name your output 'result' for auto-export\n")
        task_code = st.text_area("Script (runs with all datasets pre-loaded)", value=code_default,
                                 height=200, key="sched_code")

        if st.button("✅ Create schedule", type="primary", disabled=not task_name.strip()):
            sched = create_schedule(
                name=task_name.strip(),
                code=task_code,
                frequency=frequency,
                run_time=str(run_time)[:5],
                export_format=export_fmt,
                notify=notify,
            )
            st.session_state.schedules = st.session_state.schedules + [sched]

            # Offer to register with Windows
            if os.name == "nt":
                ok, msg = register_with_windows(sched)
                if ok:
                    st.success(f"✅ Task created and registered with Windows Task Scheduler!\n{msg}")
                else:
                    st.success(f"✅ Task created! Note: Windows registration failed ({msg}) — you can still run manually.")
            else:
                st.success(f"✅ Task '{task_name}' created! Run manually with the button below.")
            add_history("Schedule created", f"{task_name} — {frequency} at {str(run_time)[:5]}")
            st.rerun()

    # ── List existing schedules ────────────────────────────────────────────────
    if not schedules:
        st.info("No scheduled tasks yet. Create one above.")
        return

    st.markdown(f"### Active schedules ({len(schedules)})")

    for sched in schedules:
        with st.container():
            st.markdown(f"""
<div style="background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);
border-radius:10px;padding:1rem;margin-bottom:0.75rem">
<strong>{sched['name']}</strong> &nbsp;
<span style="font-size:12px;color:var(--color-text-secondary)">
{sched['frequency']} at {sched['run_time']} · Export: {sched['export_format']} · 
Last run: {sched.get('last_run','Never')} · Status: {sched.get('last_status','—')}
</span>
</div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns([2,2,2,2])
            with c1:
                if st.button("▶️ Run now", key=f"run_{sched['id']}", width="stretch"):
                    with st.spinner(f"Running '{sched['name']}'..."):
                        output, success = run_schedule_now(sched["id"])
                    if success:
                        st.success(f"✅ Task completed successfully!")
                    else:
                        st.error(f"❌ Task failed")
                    if output:
                        st.code(output[:3000], language="text")
                    st.rerun()

            with c2:
                outputs = get_schedule_outputs(sched["id"])
                if outputs:
                    latest = outputs[0]
                    data   = latest.read_bytes()
                    st.download_button(
                        f"⬇️ Latest output ({latest.suffix})",
                        data=data, file_name=latest.name,
                        key=f"dl_{sched['id']}", width="stretch"
                    )
                else:
                    st.button("No outputs yet", disabled=True, key=f"no_out_{sched['id']}", width="stretch")

            with c3:
                if st.button("📝 Edit script", key=f"edit_{sched['id']}", width="stretch"):
                    script_text = Path(sched["script_path"]).read_text() if Path(sched["script_path"]).exists() else ""
                    st.session_state.editor_code = script_text
                    st.session_state.active_page = "🐍 Code Editor"
                    st.rerun()

            with c4:
                if st.button("🗑️ Delete", key=f"del_{sched['id']}", width="stretch"):
                    if delete_schedule(sched["id"]):
                        st.success(f"Deleted '{sched['name']}'")
                        st.rerun()

            # Show output files list
            outputs = get_schedule_outputs(sched["id"])
            if outputs:
                with st.expander(f"📁 Output files ({len(outputs)})", expanded=False):
                    for f in outputs[:10]:
                        col_f, col_dl = st.columns([3,1])
                        size = fmt_bytes(f.stat().st_size)
                        col_f.markdown(f"`{f.name}` — {size}")
                        col_dl.download_button("⬇️", data=f.read_bytes(),
                                               file_name=f.name, key=f"dl_f_{f.name}")

    # ── Windows Task Scheduler info ────────────────────────────────────────────
    if os.name == "nt":
        with st.expander("ℹ️ About Windows Task Scheduler integration"):
            st.markdown("""
Tasks registered with Windows Task Scheduler run automatically even when UDIP is closed.
To manage them: open **Task Scheduler** (search in Start menu) → look for tasks starting with **UDIP_**.

If a task shows as failed in Task Scheduler, check that Python is in your PATH and the UDIP folder hasn't moved.
            """)
