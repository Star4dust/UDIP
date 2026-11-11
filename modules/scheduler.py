"""
UDIP v3 — modules/scheduler.py
Schedule automated data processing tasks using Windows Task Scheduler
or a simple time-based in-app runner.
"""

import streamlit as st
import subprocess
import sys
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from modules.core import (
    log, add_history, save_schedules, load_schedules,
    DATA_DIR
)

SCRIPTS_DIR = DATA_DIR / "scheduled_scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = DATA_DIR / "scheduled_outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

def create_schedule(name: str, code: str, frequency: str,
                    run_time: str, export_format: str = "csv",
                    notify: bool = False) -> dict:
    """Create a new scheduled task."""
    sched_id = str(uuid.uuid4())[:8]
    script_path = SCRIPTS_DIR / f"{sched_id}.py"

    # Build the full script with dataset loading preamble
    full_script = f'''"""
UDIP Scheduled Task: {name}
Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Frequency: {frequency} at {run_time}
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"{DATA_DIR.resolve()}")
DS_DIR   = DATA_DIR / "datasets"
OUTPUT   = DATA_DIR / "scheduled_outputs" / "{sched_id}"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load all saved datasets
datasets = {{}}
for meta_path in DS_DIR.glob("*.meta.json"):
    ds_id = meta_path.stem.replace(".meta","")
    df_path = DS_DIR / f"{{ds_id}}.parquet"
    if df_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["df"] = pd.read_parquet(df_path)
            datasets[ds_id] = meta
            print(f"Loaded: {{meta.get('name')}} ({{len(meta['df']):,}} rows)")
        except Exception as e:
            print(f"Could not load {{ds_id}}: {{e}}")

# ── User code ──────────────────────────────────────────────────────────────────
{code}

# ── Auto-export any DataFrame named "result" or "output" ──────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
for var_name in ["result","output","merged","cleaned","df_out"]:
    if var_name in dir():
        df_out = eval(var_name)
        if isinstance(df_out, pd.DataFrame):
            out_path = OUTPUT / f"{{var_name}}_{{timestamp}}.{export_format}"
            if "{export_format}" == "csv":
                df_out.to_csv(out_path, index=False)
            elif "{export_format}" == "parquet":
                df_out.to_parquet(out_path, index=False)
            elif "{export_format}" == "excel":
                df_out.to_excel(out_path, index=False)
            print(f"Exported: {{out_path}}")

print(f"\\nTask completed at {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
'''

    script_path.write_text(full_script, encoding="utf-8")

    schedule = {
        "id":           sched_id,
        "name":         name,
        "frequency":    frequency,
        "run_time":     run_time,
        "export_format":export_format,
        "script_path":  str(script_path),
        "output_dir":   str(OUTPUTS_DIR / sched_id),
        "created":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_run":     None,
        "last_status":  "Never run",
        "enabled":      True,
        "notify":       notify,
    }

    schedules = load_schedules()
    schedules.append(schedule)
    save_schedules(schedules)
    log.info(f"Schedule created: {name} ({frequency} at {run_time})")
    return schedule

def run_schedule_now(sched_id: str) -> tuple[str, bool]:
    """Run a scheduled task immediately and return (output, success)."""
    schedules = load_schedules()
    sched = next((s for s in schedules if s["id"] == sched_id), None)
    if not sched:
        return "Schedule not found", False

    script = sched["script_path"]
    if not Path(script).exists():
        return "Script file not found", False

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=300
        )
        output = result.stdout + (result.stderr if result.stderr else "")
        success = result.returncode == 0
        status  = "Success" if success else f"Failed (code {result.returncode})"

        # Update last run info
        for s in schedules:
            if s["id"] == sched_id:
                s["last_run"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
                s["last_status"] = status
        save_schedules(schedules)
        st.session_state.schedules = schedules
        add_history("Scheduled task", f"{sched['name']} — {status}")
        log.info(f"Schedule run: {sched['name']} → {status}")
        return output, success

    except subprocess.TimeoutExpired:
        return "Task timed out after 5 minutes", False
    except Exception as e:
        return str(e), False

def register_with_windows(sched: dict) -> tuple[bool, str]:
    """Register the task with Windows Task Scheduler via schtasks.exe."""
    if os.name != "nt":
        return False, "Windows Task Scheduler is only available on Windows."

    freq_map = {
        "Daily":   "DAILY",
        "Weekly":  "WEEKLY",
        "Hourly":  "HOURLY",
        "On startup": "ONSTART",
    }
    freq = freq_map.get(sched["frequency"], "DAILY")
    task_name = f"UDIP_{sched['id']}"
    python_exe = sys.executable
    script = sched["script_path"]

    cmd = [
        "schtasks", "/create", "/tn", task_name,
        "/tr", f'"{python_exe}" "{script}"',
        "/sc", freq,
        "/st", sched["run_time"],
        "/f",   # force overwrite
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"Windows task registered: {task_name}")
            return True, f"Task '{task_name}' registered with Windows Task Scheduler"
        else:
            return False, f"schtasks error: {result.stderr}"
    except FileNotFoundError:
        return False, "schtasks.exe not found — Windows Task Scheduler unavailable"

def delete_schedule(sched_id: str) -> bool:
    """Delete a schedule and its script file."""
    schedules = load_schedules()
    sched = next((s for s in schedules if s["id"] == sched_id), None)
    if not sched: return False

    # Delete script
    script = Path(sched["script_path"])
    if script.exists(): script.unlink()

    # Remove from Windows Task Scheduler if registered
    if os.name == "nt":
        task_name = f"UDIP_{sched_id}"
        subprocess.run(["schtasks","/delete","/tn",task_name,"/f"],
                       capture_output=True)

    schedules = [s for s in schedules if s["id"] != sched_id]
    save_schedules(schedules)
    st.session_state.schedules = schedules
    log.info(f"Schedule deleted: {sched.get('name')}")
    return True

def get_schedule_outputs(sched_id: str) -> list[Path]:
    """Get list of output files from a scheduled task."""
    out_dir = OUTPUTS_DIR / sched_id
    if not out_dir.exists(): return []
    return sorted(out_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
