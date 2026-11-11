"""
UDIP v3 — udip_pages/merge_wizard.py
Visual multi-dataset merge wizard — chain 5+ joins in sequence.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
from modules.core import log, add_history, save_dataset, sanitize_varname, fmt_bytes
import uuid

def render():
    st.markdown('<div class="main-header">🔗 Merge Wizard</div>', unsafe_allow_html=True)
    st.markdown("Chain multiple datasets together in sequence — like joining 5 tables at once.")

    datasets = st.session_state.get("datasets", {})
    if len(datasets) < 2:
        st.warning("Upload at least 2 datasets to use the Merge Wizard.")
        return

    ds_names = {k: v["name"] for k, v in datasets.items()}

    # ── Step 1: Build the merge chain ─────────────────────────────────────────
    st.markdown("### Step 1 — Build your merge chain")
    st.markdown("Start with a base dataset, then keep adding datasets to join onto it.")

    # Init chain in session state
    if "wizard_chain" not in st.session_state:
        st.session_state.wizard_chain = []

    # Base dataset selector
    if not st.session_state.wizard_chain:
        st.markdown("**Pick your starting (base) dataset:**")
        base_id = st.selectbox("Base dataset", list(ds_names.keys()),
                               format_func=lambda k: ds_names[k], key="wiz_base")
        if st.button("✅ Set as base", type="primary"):
            base_df = datasets[base_id]["df"]
            st.session_state.wizard_chain = [{
                "step":       0,
                "type":       "base",
                "ds_id":      base_id,
                "ds_name":    ds_names[base_id],
                "rows":       len(base_df),
                "cols":       list(base_df.columns),
            }]
            st.rerun()

    # Show chain so far
    if st.session_state.wizard_chain:
        st.markdown("**Current chain:**")
        for step in st.session_state.wizard_chain:
            if step["type"] == "base":
                st.markdown(f"🟢 **Base:** {step['ds_name']} — {step['rows']:,} rows, {len(step['cols'])} cols")
            else:
                jt = step.get("join_type","inner").upper()
                st.markdown(
                    f"↓ **{jt} JOIN** `{step['left_col']}` = `{step['right_col']}`  "
                    f"→ **{step['ds_name']}** ({step['rows']:,} rows)"
                )

        st.divider()

        # Add next join
        used_ids = {s["ds_id"] for s in st.session_state.wizard_chain}
        remaining = {k: v for k, v in ds_names.items() if k not in used_ids}

        if remaining:
            st.markdown("**Add next dataset to join:**")
            col1, col2, col3 = st.columns(3)
            next_id = col1.selectbox("Dataset to join", list(remaining.keys()),
                                     format_func=lambda k: remaining[k], key="wiz_next")
            join_type = col2.selectbox("Join type",
                                       ["inner","left","right","outer"],
                                       format_func=lambda x: x.capitalize()+" join",
                                       key="wiz_jtype")

            # Get current combined columns vs new dataset columns
            current_cols = st.session_state.wizard_chain[-1]["cols"]
            next_cols    = list(datasets[next_id]["df"].columns)
            common_cols  = [c for c in current_cols if c in next_cols]

            col3.markdown("**Join keys:**")
            left_key  = col1.selectbox("Left key",  current_cols,
                                        index=current_cols.index(common_cols[0]) if common_cols else 0,
                                        key="wiz_lk")
            right_key = col2.selectbox("Right key", next_cols,
                                        index=next_cols.index(common_cols[0]) if common_cols else 0,
                                        key="wiz_rk")

            if common_cols:
                col3.success(f"💡 Common: {', '.join(common_cols[:3])}")

            if st.button("➕ Add to chain", key="wiz_add"):
                # Preview what this join would produce
                try:
                    current_df = _build_chain_preview(datasets, st.session_state.wizard_chain)
                    next_df    = datasets[next_id]["df"]
                    merged     = pd.merge(current_df, next_df,
                                         left_on=left_key, right_on=right_key,
                                         how=join_type, suffixes=("","_joined"))
                    st.session_state.wizard_chain.append({
                        "step":       len(st.session_state.wizard_chain),
                        "type":       "join",
                        "ds_id":      next_id,
                        "ds_name":    remaining[next_id],
                        "join_type":  join_type,
                        "left_col":   left_key,
                        "right_col":  right_key,
                        "rows":       len(merged),
                        "cols":       list(merged.columns),
                    })
                    st.success(f"✅ Added {remaining[next_id]} — result will have {len(merged):,} rows")
                    st.rerun()
                except Exception as e:
                    st.error(f"Preview failed: {e}")
        else:
            st.info("All datasets have been added to the chain.")

        # ── Step 2: Options ────────────────────────────────────────────────────
        if len(st.session_state.wizard_chain) >= 2:
            st.divider()
            st.markdown("### Step 2 — Options")
            col_a, col_b = st.columns(2)
            drop_dups   = col_a.checkbox("Remove duplicate rows", value=True)
            output_name = col_b.text_input("Output name", value="wizard_merged")

            # ── Step 3: Preview & Execute ──────────────────────────────────────
            st.divider()
            st.markdown("### Step 3 — Run the merge")

            n_steps = len(st.session_state.wizard_chain) - 1
            total_input_rows = sum(
                len(datasets[s["ds_id"]]["df"])
                for s in st.session_state.wizard_chain
            )
            col_info, col_run, col_reset = st.columns([2,1,1])
            col_info.markdown(
                f"**{n_steps} join(s)** across "
                f"**{len(st.session_state.wizard_chain)} datasets** — "
                f"{total_input_rows:,} total input rows"
            )

            if col_run.button("▶️ Run all joins", type="primary", width="stretch"):
                with st.spinner("Running merge chain..."):
                    try:
                        result = _build_chain_preview(datasets, st.session_state.wizard_chain)
                        if drop_dups:
                            before = len(result)
                            result = result.drop_duplicates()
                            removed = before - len(result)
                        else:
                            removed = 0

                        ds_id_new = str(uuid.uuid4())[:8]
                        entry = {
                            "name":   f"{output_name}.parquet",
                            "df":     result,
                            "format": "parquet",
                            "source": "wizard_merge",
                            "varname": sanitize_varname(output_name),
                        }
                        st.session_state.datasets[ds_id_new] = entry
                        save_dataset(ds_id_new, entry)

                        chain_summary = " → ".join(s["ds_name"] for s in st.session_state.wizard_chain)
                        add_history("Wizard merge", f"{chain_summary} → {len(result):,} rows")

                        st.success(
                            f"✅ **Merge complete!** "
                            f"{len(result):,} rows × {len(result.columns)} columns. "
                            f"{removed:,} duplicates removed."
                        )

                        # Show stats
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Output rows",    f"{len(result):,}")
                        m2.metric("Output columns", len(result.columns))
                        m3.metric("Joins performed", n_steps)
                        m4.metric("Duplicates removed", f"{removed:,}")

                        # Preview
                        st.dataframe(result.head(20), width="stretch")
                        st.session_state.wizard_chain = []

                    except Exception as e:
                        st.error(f"Merge failed: {e}")
                        log.error(f"Wizard merge failed: {e}")

            if col_reset.button("🔄 Reset chain", width="stretch"):
                st.session_state.wizard_chain = []
                st.rerun()

        # Lineage diagram
        if len(st.session_state.wizard_chain) >= 2:
            with st.expander("📊 View merge lineage", expanded=False):
                lines = []
                for i, step in enumerate(st.session_state.wizard_chain):
                    if step["type"] == "base":
                        lines.append(f"[BASE] {step['ds_name']} ({step['rows']:,} rows)")
                    else:
                        lines.append(f"  ↓ {step['join_type'].upper()} JOIN on "
                                     f"'{step['left_col']}' = '{step['right_col']}'")
                        lines.append(f"[+{i}]  {step['ds_name']} → {step['rows']:,} rows after join")
                st.code("\n".join(lines), language="text")


def _build_chain_preview(datasets: dict, chain: list) -> pd.DataFrame:
    """Execute the full merge chain and return result DataFrame."""
    if not chain:
        raise ValueError("Empty chain")

    result = datasets[chain[0]["ds_id"]]["df"].copy()

    for step in chain[1:]:
        next_df = datasets[step["ds_id"]]["df"]
        result  = pd.merge(
            result, next_df,
            left_on=step["left_col"],
            right_on=step["right_col"],
            how=step["join_type"],
            suffixes=("", f"_{step['ds_name'][:6]}")
        )

    return result
