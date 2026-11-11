"""
UDIP v3 — udip_pages/data_forge.py
Data Forge — Structure, clean and validate unstructured data.

Layer 1: Schema Detective — AI proposes column schema with confidence scores
Layer 2: Prompt-driven restructuring — describe your data in plain English
Layer 3: Deterministic cleaning rules — outliers, nulls, duplicates (100% accurate)
Layer 4: Validation gate — refuse to save if result fails quality checks
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from modules.core import (
    log, add_history, save_dataset, sanitize_varname,
    fmt_bytes, load_config
)
from modules.operator_actions import render_operator_handoff
from modules.shared_utilities import get_shared_logger
from modules.state_manager import get_page_memory
import uuid
from datetime import datetime

# ── HIRAX / Radio Telescope domain templates ──────────────────────────────────
DOMAIN_TEMPLATES = {
    "HIRAX Visibility Matrix": {
        "description": "Radio telescope visibility data — frequency × baseline × time",
        "rules": [
            {"type": "rename_by_position", "mapping": {0: "time_unix", 1: "frequency_hz", 2: "baseline_id", 3: "visibility_real", 4: "visibility_imag"}},
            {"type": "convert_unit", "col": "frequency_hz", "factor": 1e-6, "new_name": "frequency_mhz"},
            {"type": "unix_to_datetime", "col": "time_unix", "new_name": "timestamp"},
            {"type": "flag_rfi", "col": "visibility_real", "threshold_sigma": 5},
        ]
    },
    "HIRAX HDF5 Heights/Signal": {
        "description": "Signal confidence and height data from HIRAX HDF5 files",
        "rules": [
            {"type": "drop_constant_cols"},
            {"type": "remove_outliers", "method": "iqr", "factor": 3.0},
            {"type": "fill_nulls", "method": "interpolate"},
        ]
    },
    "Generic Numeric Array": {
        "description": "Raw numeric array with no labels — auto-label columns",
        "rules": [
            {"type": "auto_label_cols"},
            {"type": "remove_outliers", "method": "zscore", "threshold": 4.0},
            {"type": "fill_nulls", "method": "median"},
        ]
    },
    "Time Series": {
        "description": "Time series data — first column is time",
        "rules": [
            {"type": "rename_first_col", "name": "timestamp"},
            {"type": "sort_by_first_col"},
            {"type": "fill_nulls", "method": "forward_fill"},
            {"type": "remove_outliers", "method": "zscore", "threshold": 3.0},
        ]
    },
    "Log File": {
        "description": "Text log file — extract structured fields",
        "rules": [
            {"type": "parse_log_timestamp"},
            {"type": "extract_log_level"},
            {"type": "clean_whitespace"},
        ]
    },
}


# ── Layer 3: Deterministic cleaning functions (100% accurate) ─────────────────

def clean_remove_outliers(df: pd.DataFrame, col: str = None,
                           method: str = "iqr", threshold: float = 3.0,
                           factor: float = 1.5) -> tuple[pd.DataFrame, str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if col and col in num_cols:
        num_cols = [col]
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    for c in num_cols:
        if method == "iqr":
            Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            IQR = Q3 - Q1
            mask &= (df[c] >= Q1 - factor * IQR) & (df[c] <= Q3 + factor * IQR)
        elif method == "zscore":
            z = (df[c] - df[c].mean()) / (df[c].std() + 1e-10)
            mask &= z.abs() <= threshold
    df = df[mask].reset_index(drop=True)
    removed = before - len(df)
    return df, f"Removed {removed:,} outlier rows ({removed/max(before,1)*100:.1f}%)"


def clean_fill_nulls(df: pd.DataFrame, method: str = "median",
                     col: str = None) -> tuple[pd.DataFrame, str]:
    cols = [col] if col and col in df.columns else df.columns.tolist()
    total_filled = 0
    for c in cols:
        nulls = df[c].isnull().sum()
        if nulls == 0:
            continue
        if method == "median" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        elif method == "mean" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].mean())
        elif method == "forward_fill":
            df[c] = df[c].ffill()
        elif method == "backward_fill":
            df[c] = df[c].bfill()
        elif method == "interpolate" and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
        elif method == "zero":
            df[c] = df[c].fillna(0)
        elif method == "drop":
            df = df.dropna(subset=[c])
        total_filled += nulls
    return df, f"Filled {total_filled:,} null values using {method}"


def clean_remove_duplicates(df: pd.DataFrame,
                             subset: list = None) -> tuple[pd.DataFrame, str]:
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(df)
    return df, f"Removed {removed:,} duplicate rows"


def clean_drop_constant_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    return df, f"Dropped {len(const_cols)} constant columns: {const_cols}"


def clean_auto_label_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Label numeric columns with col_0, col_1 etc if they have no meaningful names."""
    generic = [c for c in df.columns
               if str(c).strip().replace(".", "").replace("-", "").isdigit()
               or str(c) in [str(i) for i in range(len(df.columns))]]
    if generic:
        rename_map = {c: f"col_{i}" for i, c in enumerate(df.columns)}
        df = df.rename(columns=rename_map)
        return df, f"Auto-labelled {len(rename_map)} columns as col_0, col_1, ..."
    return df, "Columns already have meaningful names"


def clean_convert_unit(df: pd.DataFrame, col: str, factor: float,
                        new_name: str = None) -> tuple[pd.DataFrame, str]:
    if col not in df.columns:
        return df, f"Column '{col}' not found"
    df[new_name or col] = df[col] * factor
    if new_name and new_name != col:
        df = df.drop(columns=[col])
    return df, f"Converted '{col}' × {factor} → '{new_name or col}'"


def clean_unix_to_datetime(df: pd.DataFrame, col: str,
                            new_name: str = None) -> tuple[pd.DataFrame, str]:
    if col not in df.columns:
        return df, f"Column '{col}' not found"
    target = new_name or col
    df[target] = pd.to_datetime(df[col], unit="s", errors="coerce")
    if new_name and new_name != col:
        df = df.drop(columns=[col])
    return df, f"Converted Unix timestamps in '{col}' to datetime '{target}'"


def clean_flag_rfi(df: pd.DataFrame, col: str,
                    threshold_sigma: float = 5) -> tuple[pd.DataFrame, str]:
    if col not in df.columns:
        return df, f"Column '{col}' not found"
    mean, std = df[col].mean(), df[col].std()
    flag_col = f"{col}_rfi_flag"
    df[flag_col] = ((df[col] - mean).abs() > threshold_sigma * std).astype(int)
    n_flagged = df[flag_col].sum()
    return df, f"Flagged {n_flagged:,} RFI spikes in '{col}' (>{threshold_sigma}σ) → '{flag_col}'"


def clean_rename_cols(df: pd.DataFrame,
                       rename_map: dict) -> tuple[pd.DataFrame, str]:
    # Map by position if keys are integers
    if all(isinstance(k, int) for k in rename_map.keys()):
        cols = list(df.columns)
        pos_map = {cols[k]: v for k, v in rename_map.items() if k < len(cols)}
        df = df.rename(columns=pos_map)
        return df, f"Renamed {len(pos_map)} columns: {pos_map}"
    df = df.rename(columns=rename_map)
    return df, f"Renamed columns: {rename_map}"


def clean_sort(df: pd.DataFrame, col: str = None) -> tuple[pd.DataFrame, str]:
    if col and col in df.columns:
        df = df.sort_values(col).reset_index(drop=True)
        return df, f"Sorted by '{col}'"
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    return df, f"Sorted by first column '{df.columns[0]}'"


def clean_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    return df, f"Stripped whitespace from {len(str_cols)} text columns"


def detect_schema(df: pd.DataFrame) -> list[dict]:
    """
    Proxy to execution_layer.detect_schema.
    """
    from modules.execution_layer import detect_schema as ds
    res = ds(df)
    return res["output"] if res["success"] else []


# ── Layer 2: AI prompt-driven restructuring ───────────────────────────────────

def apply_prompt_to_df(df: pd.DataFrame, prompt: str,
                        groq_key: str) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Send prompt + data description to Groq.
    AI returns a Python transformation, we execute it safely.
    Returns (transformed_df, explanation, change_log).
    """
    from modules.ai_engine import run_code

    # Build a compact data description
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        nulls  = df[col].isnull().sum()
        col_info.append(f"  '{col}' ({dtype}): sample={sample}, nulls={nulls}")
    data_desc = f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\nColumns:\n" + "\n".join(col_info)

    system = """You are a data structuring expert. You receive a description of an unstructured dataset and a user instruction.
You must return ONLY a Python code block that transforms the DataFrame.

RULES:
- The input DataFrame is available as `df`  
- Assign the final result back to `df`
- Print a brief summary of what you changed at the end
- Do NOT use plt.show() or any plotting
- Do NOT write files
- Be conservative — only do what the user explicitly asks
- If renaming columns, preserve ALL existing columns unless told to drop
- Handle errors gracefully with try/except

Return format: ```python\n<code>\n```
Then one sentence explaining what the code does."""

    user_msg = f"""Dataset description:
{data_desc}

User instruction:
{prompt}

Write Python code to transform `df` according to the instruction."""

    from groq import Groq
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg}
        ],
        max_tokens=2048,
        temperature=0.1,  # low temp = more deterministic = more accurate
    )
    reply = response.choices[0].message.content

    # Extract code block
    cm = re.search(r"```python\n([\s\S]*?)```", reply)
    if not cm:
        return df, "AI did not return valid code.", []

    code = cm.group(1).strip()

    # Execute safely
    import sys, io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    error = None
    try:
        exec_globals = {
            "df": df.copy(),
            "pd": pd, "np": np,
            "__builtins__": {k: v for k, v in __builtins__.items()
                             if k not in ("open","exec","eval","__import__")}
            if isinstance(__builtins__, dict)
            else {k: getattr(__builtins__, k) for k in dir(__builtins__)
                  if k not in ("open","exec","eval","__import__") and not k.startswith("__")},
        }
        exec(code, exec_globals)
        result_df = exec_globals.get("df", df)
        if not isinstance(result_df, pd.DataFrame):
            result_df = df
    except Exception as e:
        error = str(e)
        result_df = df
    finally:
        sys.stdout = old_stdout

    output   = buf.getvalue()
    changes  = output.strip().split("\n") if output.strip() else []

    # Extract explanation from reply (after code block)
    explanation_match = re.search(r"```\s*\n(.*)", reply, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else reply

    if error:
        return df, f"Code error: {error}", []

    # Zero-Loss Resilience / Stable Forge: Fix PyArrow Duplicate Columns Crash
    # If the AI generated duplicate columns, we fix them silently before rendering
    has_dupes = len(result_df.columns) != len(set(result_df.columns))
    if has_dupes:
        cols = pd.Series(result_df.columns)
        for dup in cols[cols.duplicated()].unique():
            mask = (cols == dup)
            cols[mask] = [f"{dup}_{i}" if i > 0 else dup for i in range(mask.sum())]
        result_df.columns = cols
        changes.append("⚠️ Automatically deduplicated column names to prevent rendering errors.")

    return result_df, explanation, changes


# ── Layer 4: Validation gate ──────────────────────────────────────────────────

def validate_result(df: pd.DataFrame,
                    original_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check the result meets minimum quality before allowing save."""
    issues = []

    if len(df) == 0:
        issues.append("❌ Result has 0 rows — transformation removed all data")

    if len(df.columns) == 0:
        issues.append("❌ Result has 0 columns")
        
    if len(df.columns) != len(set(df.columns)):
        issues.append("❌ Result contains duplicate column names which will crash the visualization engine.")

    if len(df) < 0.1 * len(original_df):
        pct = len(df) / len(original_df) * 100
        issues.append(f"⚠️ Result has only {pct:.0f}% of original rows — transformation may be too aggressive")

    null_rate = df.isnull().mean().mean()
    if null_rate > 0.8:
        issues.append(f"⚠️ {null_rate*100:.0f}% of values are null — check transformation")

    # Check for all-identical columns
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"⚠️ Column '{col}' is constant after transformation")

    return len([i for i in issues if i.startswith("❌")]) == 0, issues


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown('<div class="main-header">⚒️ Data Forge</div>', unsafe_allow_html=True)
    st.markdown("Structure, clean and validate unstructured data before analysis.")
    render_operator_handoff("⚒️ Data Forge")

    datasets = st.session_state.get("datasets", {})
    if not datasets:
        st.warning("Upload a dataset first, then come here to structure it.")
        return

    # Restore previously selected dataset
    _mem = get_page_memory("forge")
    available_ds = list(datasets.keys())
    saved_ds = _mem.get("ds_id")
    default_forge_idx = available_ds.index(saved_ds) if saved_ds in available_ds else 0

    # Dataset selector
    ds_id = st.selectbox("Select dataset to structure",
                          available_ds,
                          format_func=lambda k: datasets[k]["name"],
                          index=default_forge_idx,
                          key="forge_ds")
    _mem.set("ds_id", ds_id)
    df_original = datasets[ds_id]["df"].copy()
    df_name     = datasets[ds_id]["name"]
    df_working  = st.session_state.get("_forge_working", df_original)
    completion_pct = (1 - (df_original.isnull().sum().sum() / max(df_original.size, 1))) * 100

    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Transformation Lab</div>
            <div class="udip-panel__title">Forge a clean, validated dataset from {df_name}</div>
            <div class="udip-panel__text">
                Move from raw structure detection to prompt-guided reshaping, deterministic cleaning, and a final
                validation gate before anything gets saved back into UDIP.
            </div>
            <div class="udip-stat-strip">
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(df_original):,}</div>
                    <div class="udip-stat-tile__label">source rows</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(df_original.columns)}</div>
                    <div class="udip-stat-tile__label">source columns</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{completion_pct:.1f}%</div>
                    <div class="udip-stat-tile__label">value completion</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(df_working):,}</div>
                    <div class="udip-stat-tile__label">working rows</div>
                </div>
            </div>
            <div class="udip-pill-row">
                <div class="udip-pill udip-pill--accent">1. Schema Detective</div>
                <div class="udip-pill">2. Prompt Restructure</div>
                <div class="udip-pill">3. Clean Rules</div>
                <div class="udip-pill udip-pill--warm">4. Validation Gate</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show quick summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{len(df_original):,}")
    c2.metric("Columns", len(df_original.columns))
    c3.metric("Nulls",   f"{df_original.isnull().sum().sum():,}")
    c4.metric("Numeric cols", len(df_original.select_dtypes(include="number").columns))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Layer 1 — Schema Detective",
        "💬 Layer 2 — Prompt Restructure",
        "🧹 Layer 3 — Clean Rules",
        "🏭 Layer 4 — Full Pipeline",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1: SCHEMA DETECTIVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🔍 Schema Detective")
        st.markdown("AI analyses your columns and proposes names, types and units. "
                    "Review confidence scores and confirm before applying.")

        if st.button("🔍 Analyse schema", type="primary", key="l1_analyse"):
            with st.spinner("Analysing column patterns..."):
                proposals = detect_schema(df_original)
            st.session_state["_forge_proposals"] = proposals

        proposals = st.session_state.get("_forge_proposals")
        if proposals:
            st.markdown("**Review proposals — edit any name, then apply:**")

            edited_names = {}
            all_confident = True

            for i, p in enumerate(proposals):
                conf   = p["confidence"]
                color  = "🟢" if conf >= 0.8 else ("🟡" if conf >= 0.55 else "🔴")
                conf_pct = f"{conf*100:.0f}%"

                with st.expander(
                    f"{color} `{p['original_name']}` → proposed: `{p['proposed_name']}` "
                    f"({conf_pct} confidence)",
                    expanded=(conf < 0.7)
                ):
                    col_a, col_b = st.columns([2, 1])
                    new_name = col_a.text_input(
                        "Column name",
                        value=p["proposed_name"],
                        key=f"l1_name_{i}"
                    )
                    col_b.markdown(f"**Type:** {p['dtype']}")
                    if p["suggested_unit"]:
                        col_b.markdown(f"**Unit:** {p['suggested_unit']}")
                    for note in p["notes"]:
                        st.caption(f"💡 {note}")

                    edited_names[p["original_name"]] = new_name
                    if conf < 0.6:
                        all_confident = False

            if not all_confident:
                st.warning("⚠️ Some columns have low confidence. "
                           "Please verify the proposed names above before applying.")

            if st.button("✅ Apply schema", type="primary", key="l1_apply"):
                df_new = df_original.rename(columns=edited_names)
                
                # Zero-Loss Resilience / Stable Forge: Fix Duplicate Column Error
                # Streamlit/PyArrow crashes if multiple columns share the exact same name.
                cols = pd.Series(df_new.columns)
                for dup in cols[cols.duplicated()].unique():
                    mask = (cols == dup)
                    cols[mask] = [f"{dup}_{i}" if i > 0 else dup for i in range(mask.sum())]
                df_new.columns = cols
                
                st.session_state["_forge_working"] = df_new
                st.success(f"✅ Schema applied — {len(edited_names)} columns renamed.")
                st.dataframe(df_new.head(10), width="stretch")
                add_history("Data Forge", f"Schema applied to {df_name}")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2: PROMPT-DRIVEN RESTRUCTURE
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 💬 Prompt-Driven Restructure")
        st.markdown("Describe what your data means and what you want done. "
                    "Be specific — the more context you give, the more accurate the result.")

        cfg      = load_config()
        groq_key = cfg.get("groq_key", "")
        if not groq_key:
            st.warning("Set your Groq API key in Settings first.")
        else:
            # Working df (may already have schema applied)
            df_work = st.session_state.get("_forge_working", df_original)

            st.dataframe(df_work.head(5), width="stretch")

            # Example prompts
            with st.expander("💡 Example prompts"):
                examples = [
                    "Column 0 is time in Unix seconds, column 1 is frequency in Hz, columns 2-4 are signal amplitudes. Rename them appropriately and convert time to datetime.",
                    "This is HIRAX visibility data. The first column is baseline ID, second is timestamp. Flag any amplitude values more than 5 standard deviations from the mean as RFI.",
                    "Convert the frequency column from Hz to MHz by dividing by 1,000,000. Keep the original column too.",
                    "This data has no headers. The columns are: antenna_id, ra_degrees, dec_degrees, flux_jansky, snr. Please rename them.",
                    "Remove any rows where signal_amplitude is negative — those are invalid readings.",
                    "The timestamp column contains Unix seconds since 2000-01-01. Convert it to standard UTC datetime.",
                ]
                for ex in examples:
                    if st.button(f"Use: {ex[:60]}...", key=f"ex_{ex[:20]}"):
                        st.session_state["_forge_prompt"] = ex

            prompt = st.text_area(
                "Describe the transformation",
                value=st.session_state.get("_forge_prompt", ""),
                height=120,
                placeholder="e.g. Column 0 is frequency in MHz, column 1 is signal amplitude. Rename them and flag values above 5σ as RFI.",
                key="l2_prompt"
            )
            st.session_state["_forge_prompt"] = prompt

            col_run, col_info = st.columns([1, 3])
            run_l2 = col_run.button("▶️ Apply prompt", type="primary", key="l2_run",
                                     disabled=not prompt.strip())
            col_info.caption("AI will generate Python code from your description and run it. "
                              "You must confirm before saving.")

            if run_l2 and prompt.strip():
                with st.spinner("AI is writing transformation code..."):
                    try:
                        df_result, explanation, changes = apply_prompt_to_df(
                            df_work, prompt, groq_key
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        df_result = None

                if df_result is not None:
                    st.session_state["_forge_l2_result"]  = df_result
                    st.session_state["_forge_l2_explain"] = explanation
                    st.session_state["_forge_l2_changes"] = changes

            result   = st.session_state.get("_forge_l2_result")
            explain  = st.session_state.get("_forge_l2_explain", "")
            changes  = st.session_state.get("_forge_l2_changes", [])

            if result is not None:
                st.markdown("**AI explanation:**")
                st.info(explain)

                if changes:
                    st.markdown("**Changes made:**")
                    for c in changes:
                        st.markdown(f"- {c}")

                # Diff view
                col_before, col_after = st.columns(2)
                col_before.markdown("**Before**")
                col_before.dataframe(df_work.head(8), width="stretch")
                col_after.markdown("**After**")
                col_after.dataframe(result.head(8), width="stretch")

                # Validation gate
                passed, issues = validate_result(result, df_original)
                if issues:
                    for issue in issues:
                        if issue.startswith("❌"):
                            st.error(issue)
                        else:
                            st.warning(issue)

                if not passed:
                    st.error("❌ Validation failed — result not saved. Adjust your prompt and try again.")
                else:
                    if st.button("✅ Accept and use this result", type="primary", key="l2_accept"):
                        st.session_state["_forge_working"] = result
                        st.success("✅ Result accepted. Continue in Layer 3 or save in Layer 4.")
                        add_history("Data Forge", f"Prompt restructure applied to {df_name}")
                    if st.button("❌ Reject — try different prompt", key="l2_reject"):
                        for k in ["_forge_l2_result","_forge_l2_explain","_forge_l2_changes"]:
                            st.session_state.pop(k, None)
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3: DETERMINISTIC CLEANING RULES
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🧹 Deterministic Cleaning Rules")
        st.markdown("These rules use pure math — no AI, 100% predictable results. "
                    "Each change is shown before saving.")

        df_work = st.session_state.get("_forge_working", df_original)
        num_cols = df_work.select_dtypes(include="number").columns.tolist()
        all_cols = df_work.columns.tolist()

        # Domain template quick-apply
        st.markdown("#### Quick templates")
        tmpl_col, tmpl_btn = st.columns([3, 1])
        template = tmpl_col.selectbox("Apply domain template",
                                       ["— choose —"] + list(DOMAIN_TEMPLATES.keys()),
                                       key="l3_tmpl")
        if tmpl_btn.button("Apply template", key="l3_tmpl_apply") and template != "— choose —":
            t = DOMAIN_TEMPLATES[template]
            st.info(f"Template: {t['description']}")
            df_t = df_work.copy()
            log_msgs = []
            for rule in t["rules"]:
                rtype = rule["type"]
                try:
                    if rtype == "drop_constant_cols":
                        df_t, msg = clean_drop_constant_cols(df_t)
                    elif rtype == "auto_label_cols":
                        df_t, msg = clean_auto_label_cols(df_t)
                    elif rtype == "remove_outliers":
                        df_t, msg = clean_remove_outliers(df_t,
                            method=rule.get("method","iqr"),
                            factor=rule.get("factor",1.5),
                            threshold=rule.get("threshold",3.0))
                    elif rtype == "fill_nulls":
                        df_t, msg = clean_fill_nulls(df_t, method=rule.get("method","median"))
                    elif rtype == "rename_by_position":
                        df_t, msg = clean_rename_cols(df_t, rule.get("mapping",{}))
                    elif rtype == "convert_unit":
                        df_t, msg = clean_convert_unit(df_t, rule["col"],
                                                        rule["factor"], rule.get("new_name"))
                    elif rtype == "unix_to_datetime":
                        df_t, msg = clean_unix_to_datetime(df_t, rule["col"],
                                                            rule.get("new_name"))
                    elif rtype == "flag_rfi":
                        df_t, msg = clean_flag_rfi(df_t, rule["col"],
                                                    rule.get("threshold_sigma", 5))
                    elif rtype == "sort_by_first_col":
                        df_t, msg = clean_sort(df_t)
                    elif rtype == "clean_whitespace":
                        df_t, msg = clean_whitespace(df_t)
                    else:
                        msg = f"Unknown rule: {rtype}"
                    log_msgs.append(f"✅ {msg}")
                except Exception as e:
                    log_msgs.append(f"⚠️ {rtype} failed: {e}")

            for m in log_msgs:
                st.markdown(m)
            st.session_state["_forge_working"] = df_t
            st.success(f"Template '{template}' applied.")
            st.dataframe(df_t.head(10), width="stretch")
            add_history("Data Forge", f"Template '{template}' applied to {df_name}")

        st.divider()
        st.markdown("#### Individual rules")

        # Outlier removal
        with st.expander("🎯 Remove outliers"):
            r_col  = st.selectbox("Column (or all numeric)", ["— all numeric —"] + num_cols, key="l3_out_col")
            r_meth = st.radio("Method", ["IQR (recommended)", "Z-score"],
                               horizontal=True, key="l3_out_meth")
            if "IQR" in r_meth:
                r_fac = st.slider("IQR factor (1.5 = standard, 3.0 = lenient)",
                                   1.0, 5.0, 1.5, 0.5, key="l3_out_fac")
                if st.button("Apply outlier removal", key="l3_out_apply"):
                    col_arg = None if r_col == "— all numeric —" else r_col
                    df_t, msg = clean_remove_outliers(df_work, col=col_arg,
                                                       method="iqr", factor=r_fac)
                    st.success(msg)
                    st.session_state["_forge_working"] = df_t
                    add_history("Data Forge", msg)
            else:
                r_thr = st.slider("Z-score threshold (3 = standard, 4 = lenient)",
                                   2.0, 6.0, 3.0, 0.5, key="l3_out_thr")
                if st.button("Apply outlier removal", key="l3_out_apply_z"):
                    col_arg = None if r_col == "— all numeric —" else r_col
                    df_t, msg = clean_remove_outliers(df_work, col=col_arg,
                                                       method="zscore", threshold=r_thr)
                    st.success(msg)
                    st.session_state["_forge_working"] = df_t
                    add_history("Data Forge", msg)

        # Null filling
        with st.expander("🕳️ Fill missing values"):
            n_col  = st.selectbox("Column (or all)", ["— all columns —"] + all_cols, key="l3_null_col")
            n_meth = st.selectbox("Method",
                                   ["median","mean","forward_fill","backward_fill",
                                    "interpolate","zero","drop"],
                                   key="l3_null_meth")
            if st.button("Apply null filling", key="l3_null_apply"):
                col_arg = None if n_col == "— all columns —" else n_col
                df_t, msg = clean_fill_nulls(df_work, method=n_meth, col=col_arg)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # Duplicates
        with st.expander("👯 Remove duplicate rows"):
            if st.button("Remove duplicates", key="l3_dup_apply"):
                df_t, msg = clean_remove_duplicates(df_work)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # Column renaming
        with st.expander("✏️ Rename columns"):
            rename_pairs = {}
            for col in all_cols[:15]:  # limit to 15 columns for UI
                new = st.text_input(f"`{col}`", value=col, key=f"l3_ren_{col}")
                if new != col:
                    rename_pairs[col] = new
            if rename_pairs and st.button("Apply renames", key="l3_ren_apply"):
                df_t, msg = clean_rename_cols(df_work, rename_pairs)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # Unit conversion
        with st.expander("🔄 Convert units"):
            uc_col  = st.selectbox("Column", num_cols, key="l3_uc_col") if num_cols else None
            uc_fac  = st.number_input("Multiply by factor", value=1.0, key="l3_uc_fac")
            uc_name = st.text_input("New column name (leave blank to overwrite)", key="l3_uc_name")
            if uc_col and st.button("Apply conversion", key="l3_uc_apply"):
                df_t, msg = clean_convert_unit(df_work, uc_col, uc_fac,
                                                uc_name if uc_name else None)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # RFI flagging
        with st.expander("📡 Flag RFI spikes (radio astronomy)"):
            rfi_col = st.selectbox("Column to check", num_cols, key="l3_rfi_col") if num_cols else None
            rfi_sig = st.slider("Sigma threshold", 2.0, 10.0, 5.0, 0.5, key="l3_rfi_sig")
            if rfi_col and st.button("Flag RFI", key="l3_rfi_apply"):
                df_t, msg = clean_flag_rfi(df_work, rfi_col, rfi_sig)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # Unix to datetime
        with st.expander("🕐 Convert Unix timestamps"):
            ts_col  = st.selectbox("Timestamp column", num_cols, key="l3_ts_col") if num_cols else None
            ts_name = st.text_input("New column name", value="timestamp", key="l3_ts_name")
            if ts_col and st.button("Convert to datetime", key="l3_ts_apply"):
                df_t, msg = clean_unix_to_datetime(df_work, ts_col, ts_name)
                st.success(msg)
                st.session_state["_forge_working"] = df_t
                add_history("Data Forge", msg)

        # Current state
        df_work = st.session_state.get("_forge_working", df_original)
        st.divider()
        st.markdown(f"**Current working dataset:** {len(df_work):,} rows × {len(df_work.columns)} cols")
        st.dataframe(df_work.head(10), width="stretch")

        if st.button("↩️ Reset to original", key="l3_reset"):
            for k in ["_forge_working","_forge_proposals","_forge_l2_result",
                       "_forge_l2_explain","_forge_l2_changes"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 4: VALIDATION GATE + SAVE
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🏭 Full Pipeline — Validate & Save")
        st.markdown("Run final checks and save the structured dataset.")

        df_final = st.session_state.get("_forge_working", df_original)

        # Stats comparison
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original**")
            st.metric("Rows",    f"{len(df_original):,}")
            st.metric("Columns", len(df_original.columns))
            st.metric("Nulls",   f"{df_original.isnull().sum().sum():,}")
        with col_b:
            st.markdown("**After Data Forge**")
            row_delta = len(df_final) - len(df_original)
            st.metric("Rows",    f"{len(df_final):,}",    delta=f"{row_delta:+,}")
            st.metric("Columns", len(df_final.columns),   delta=len(df_final.columns)-len(df_original.columns))
            null_delta = df_final.isnull().sum().sum() - df_original.isnull().sum().sum()
            st.metric("Nulls",   f"{df_final.isnull().sum().sum():,}", delta=f"{null_delta:+,}")

        st.divider()
        st.markdown("**Final preview:**")
        st.dataframe(df_final.head(20), width="stretch")

        # Run validation gate
        if st.button("▶️ Run validation gate", type="primary", key="l4_validate"):
            passed, issues = validate_result(df_final, df_original)
            st.session_state["_forge_val_passed"] = passed
            st.session_state["_forge_val_issues"] = issues

        val_passed = st.session_state.get("_forge_val_passed")
        val_issues = st.session_state.get("_forge_val_issues", [])

        if val_issues is not None and len(val_issues) == 0 and val_passed is not None:
            st.success("✅ All validation checks passed — ready to save.")
        for issue in val_issues:
            if issue.startswith("❌"):
                st.error(issue)
            else:
                st.warning(issue)

        if val_passed == True:
            st.divider()
            out_name = st.text_input("Save as dataset name",
                                      value=f"{df_name.split('.')[0]}_structured",
                                      key="l4_name")
            replace  = st.checkbox("Replace original dataset", value=False, key="l4_replace")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save structured dataset", type="primary", width="stretch", key="l4_save"):
                    new_id  = ds_id if replace else str(uuid.uuid4())[:8]
                    entry   = {
                        "name":    f"{out_name}.parquet",
                        "df":      df_final,
                        "format":  "parquet",
                        "source":  "Data Forge",
                        "varname": sanitize_varname(out_name),
                    }
                    st.session_state.datasets[new_id] = entry
                    ok, err = save_dataset(new_id, entry)
                    if ok:
                        st.success(f"✅ Saved as **{out_name}** — "
                                   f"{len(df_final):,} rows × {len(df_final.columns)} cols")
                        add_history("Data Forge", f"Saved structured dataset: {out_name}")
                        get_shared_logger().log(f"Data Forge saved: {out_name} ({df_final.shape})", "info")
                        # Clear working state
                        for k in ["_forge_working","_forge_proposals","_forge_l2_result",
                                   "_forge_l2_explain","_forge_l2_changes",
                                   "_forge_val_passed","_forge_val_issues"]:
                            st.session_state.pop(k, None)
                        st.balloons()
                    else:
                        st.error(f"Save failed: {err}")
                        
            with c2:
                if st.button("📌 Pin Table", width="stretch", key="l4_pin"):
                    try:
                        import tabulate
                        table_html = tabulate.tabulate(df_final.head(20), headers='keys', tablefmt='html')
                        table_data = {
                            "dataset": df_name,
                            "shape": df_final.shape,
                            "columns": list(df_final.columns),
                            "table_html": table_html,
                            "sample_data": df_final.head(5).to_dict('records')
                        }
                        state_mgr.pin_item("table", table_data, f"Data Forge - {df_name}")
                        st.success("📌 Table pinned! Access from sidebar.")
                    except Exception as e:
                        st.error(f"Could not pin table: {e}")

        elif val_passed == False:
            st.error("❌ Validation failed — fix the issues above before saving.")
