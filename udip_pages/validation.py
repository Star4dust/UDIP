"""
UDIP v3 — udip_pages/validation.py
Data validation rules engine — define rules, run checks, get reports.
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from modules.core import log, add_history, safe_write, DATA_DIR
import json, uuid
from datetime import datetime

RULES_FILE = DATA_DIR / "validation_rules.json"

# ── Rule types ────────────────────────────────────────────────────────────────
RULE_TYPES = {
    "Not null":           "Column must have no missing values",
    "Unique":             "All values must be unique (no duplicates)",
    "Min value":          "Numeric values must be ≥ threshold",
    "Max value":          "Numeric values must be ≤ threshold",
    "Between":            "Numeric values must be within a range",
    "Positive":           "All numeric values must be > 0",
    "No negatives":       "No negative values allowed",
    "Min length":         "Text values must be at least N characters",
    "Max length":         "Text values must be at most N characters",
    "Regex match":        "Values must match a regular expression",
    "Allowed values":     "Only specific values are permitted",
    "No whitespace":      "Text values must not have leading/trailing spaces",
    "Valid email":        "Values must be valid email addresses",
    "Valid date":         "Values must be parseable as dates",
    "No duplicates (row)":"Entire rows must be unique",
    "Row count min":      "Dataset must have at least N rows",
    "Row count max":      "Dataset must have at most N rows",
    "Column exists":      "Column must exist in dataset",
    "Is numeric":         "Values must be numeric",
    "Is integer":         "Values must be whole numbers",
    "Contains substring": "Text must contain a specific substring",
    "Date before":        "Dates must be before a specific date",
    "Date after":         "Dates must be after a specific date",
}

SEVERITY = ["🔴 Critical", "🟡 Warning", "🔵 Info"]

def load_rules() -> list:
    try:
        return json.loads(RULES_FILE.read_text()) if RULES_FILE.exists() else []
    except Exception:
        return []

def save_rules(rules: list):
    try:
        safe_write(RULES_FILE, json.dumps(rules, indent=2))
    except Exception as e:
        log.error(f"Rules save failed: {e}")

def run_rule(df: pd.DataFrame, rule: dict) -> dict:
    """Execute a single rule against a DataFrame. Returns result dict."""
    col     = rule.get("column", "")
    rtype   = rule["type"]
    passed  = True
    message = ""
    failing_rows = 0

    try:
        if rtype == "Not null":
            nulls = df[col].isnull().sum() if col in df.columns else 0
            passed = nulls == 0
            failing_rows = int(nulls)
            message = f"{nulls:,} null values found" if not passed else "No nulls ✅"

        elif rtype == "Unique":
            dups = df[col].duplicated().sum() if col in df.columns else 0
            passed = dups == 0
            failing_rows = int(dups)
            message = f"{dups:,} duplicate values" if not passed else "All unique ✅"

        elif rtype == "Min value":
            thresh = float(rule.get("threshold", 0))
            bad = (df[col] < thresh).sum() if col in df.columns else 0
            passed = bad == 0
            failing_rows = int(bad)
            message = f"{bad:,} values below {thresh}" if not passed else f"All ≥ {thresh} ✅"

        elif rtype == "Max value":
            thresh = float(rule.get("threshold", 0))
            bad = (df[col] > thresh).sum() if col in df.columns else 0
            passed = bad == 0
            failing_rows = int(bad)
            message = f"{bad:,} values above {thresh}" if not passed else f"All ≤ {thresh} ✅"

        elif rtype == "Between":
            lo = float(rule.get("min_val", 0))
            hi = float(rule.get("max_val", 100))
            bad = ((df[col] < lo) | (df[col] > hi)).sum() if col in df.columns else 0
            passed = bad == 0
            failing_rows = int(bad)
            message = f"{bad:,} values outside [{lo}, {hi}]" if not passed else f"All in [{lo},{hi}] ✅"

        elif rtype == "Positive":
            bad = (df[col] <= 0).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} non-positive values" if not passed else "All positive ✅"

        elif rtype == "No negatives":
            bad = (df[col] < 0).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} negative values" if not passed else "No negatives ✅"

        elif rtype == "Min length":
            n = int(rule.get("threshold", 1))
            bad = (df[col].astype(str).str.len() < n).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values shorter than {n} chars" if not passed else f"All ≥ {n} chars ✅"

        elif rtype == "Max length":
            n = int(rule.get("threshold", 255))
            bad = (df[col].astype(str).str.len() > n).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values longer than {n} chars" if not passed else f"All ≤ {n} chars ✅"

        elif rtype == "Regex match":
            import re
            pattern = rule.get("pattern", ".*")
            bad = (~df[col].astype(str).str.match(pattern)).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values don't match pattern" if not passed else "All match ✅"

        elif rtype == "Allowed values":
            allowed = [v.strip() for v in rule.get("allowed", "").split(",")]
            bad = (~df[col].astype(str).isin(allowed)).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values not in allowed list" if not passed else "All allowed ✅"

        elif rtype == "No whitespace":
            bad = (df[col].astype(str) != df[col].astype(str).str.strip()).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values have leading/trailing spaces" if not passed else "No whitespace ✅"

        elif rtype == "Valid email":
            import re
            pat = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            bad = (~df[col].astype(str).str.match(pat)).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} invalid emails" if not passed else "All valid emails ✅"

        elif rtype == "Valid date":
            try:
                pd.to_datetime(df[col], errors='raise')
                passed = True; message = "All valid dates ✅"
            except Exception:
                bad_count = pd.to_datetime(df[col], errors='coerce').isnull().sum()
                passed = bad_count == 0; failing_rows = int(bad_count)
                message = f"{bad_count:,} invalid dates" if not passed else "All valid dates ✅"

        elif rtype == "No duplicates (row)":
            dups = df.duplicated().sum()
            passed = dups == 0; failing_rows = int(dups)
            message = f"{dups:,} duplicate rows" if not passed else "No duplicate rows ✅"

        elif rtype == "Row count min":
            n = int(rule.get("threshold", 1))
            passed = len(df) >= n
            message = f"Dataset has {len(df):,} rows, expected ≥ {n:,}" if not passed else f"{len(df):,} rows ✅"

        elif rtype == "Row count max":
            n = int(rule.get("threshold", 1000000))
            passed = len(df) <= n
            message = f"Dataset has {len(df):,} rows, expected ≤ {n:,}" if not passed else f"{len(df):,} rows ✅"

        elif rtype == "Column exists":
            passed = col in df.columns
            message = f"Column '{col}' not found" if not passed else f"Column '{col}' exists ✅"

        elif rtype == "Is numeric":
            try:
                numeric = pd.to_numeric(df[col], errors='coerce')
                bad = numeric.isnull().sum()
                passed = bad == 0; failing_rows = int(bad)
                message = f"{bad:,} non-numeric values" if not passed else "All numeric ✅"
            except Exception:
                passed = False; message = "Error checking numeric"

        elif rtype == "Is integer":
            try:
                numeric = pd.to_numeric(df[col], errors='coerce').dropna()
                bad = ((numeric % 1) != 0).sum()
                passed = bad == 0; failing_rows = int(bad)
                message = f"{bad:,} non-integer values" if not passed else "All integers ✅"
            except Exception:
                passed = False; message = "Error checking integer"

        elif rtype == "Contains substring":
            substring = rule.get("substring", "")
            bad = (~df[col].astype(str).str.contains(substring)).sum() if col in df.columns else 0
            passed = bad == 0; failing_rows = int(bad)
            message = f"{bad:,} values don't contain '{substring}'" if not passed else f"All contain '{substring}' ✅"

        elif rtype == "Date before":
            date_str = rule.get("date", "")
            try:
                threshold = pd.to_datetime(date_str)
                dates = pd.to_datetime(df[col], errors='coerce')
                bad = (dates >= threshold).sum()
                passed = bad == 0; failing_rows = int(bad)
                message = f"{bad:,} dates not before {date_str}" if not passed else f"All before {date_str} ✅"
            except Exception:
                passed = False; message = f"Invalid date format: {date_str}"

        elif rtype == "Date after":
            date_str = rule.get("date", "")
            try:
                threshold = pd.to_datetime(date_str)
                dates = pd.to_datetime(df[col], errors='coerce')
                bad = (dates <= threshold).sum()
                passed = bad == 0; failing_rows = int(bad)
                message = f"{bad:,} dates not after {date_str}" if not passed else f"All after {date_str} ✅"
            except Exception:
                passed = False; message = f"Invalid date format: {date_str}"

    except Exception as e:
        passed = False
        message = f"Rule error: {e}"

    return {
        "rule_id":     rule["id"],
        "rule_name":   rule.get("name", rtype),
        "type":        rtype,
        "column":      col,
        "severity":    rule.get("severity", "🔴 Critical"),
        "passed":      passed,
        "message":     message,
        "failing_rows":failing_rows,
    }


def render():
    st.markdown('<div class="main-header">✅ Data Validation</div>', unsafe_allow_html=True)
    st.markdown("Define rules for your data — run checks before analysis or export.")

    datasets  = st.session_state.get("datasets", {})
    all_rules = load_rules()

    tab_define, tab_run, tab_history = st.tabs(
        ["📋 Define Rules", "▶️ Run Validation", "📜 Validation History"]
    )

    # ── TAB 1: DEFINE RULES ───────────────────────────────────────────────────
    with tab_define:
        st.markdown("### Add a new rule")

        col1, col2 = st.columns(2)
        rule_name  = col1.text_input("Rule name", placeholder="Revenue must be positive", key="vr_name")
        rule_type  = col2.selectbox("Rule type", list(RULE_TYPES.keys()), key="vr_type")
        st.caption(f"*{RULE_TYPES[rule_type]}*")

        # Column selector
        col3, col4 = st.columns(2)
        all_cols = sorted({c for ds in datasets.values() for c in ds["df"].columns})

        # Some rules don't need a column
        row_level = rule_type in ["No duplicates (row)", "Row count min", "Row count max"]
        if not row_level:
            col_choice = col3.selectbox("Column", all_cols or ["(no columns loaded)"], key="vr_col")
        else:
            col_choice = ""

        severity = col4.selectbox("Severity", SEVERITY, key="vr_sev")

        # Extra params depending on rule type
        extra = {}
        if rule_type in ["Min value","Max value","Row count min","Row count max","Min length","Max length"]:
            extra["threshold"] = st.number_input("Threshold value", value=0.0, key="vr_thresh")
        elif rule_type == "Between":
            c_a, c_b = st.columns(2)
            extra["min_val"] = c_a.number_input("Min", value=0.0, key="vr_min")
            extra["max_val"] = c_b.number_input("Max", value=100.0, key="vr_max")
        elif rule_type == "Regex match":
            extra["pattern"] = st.text_input("Regex pattern", placeholder=r"^\d{4}-\d{2}-\d{2}$", key="vr_re")
        elif rule_type == "Allowed values":
            extra["allowed"] = st.text_input("Allowed values (comma separated)",
                                              placeholder="Yes, No, Maybe", key="vr_av")

        apply_to = st.multiselect("Apply to datasets",
                                   [ds["name"] for ds in datasets.values()] or ["(upload datasets first)"],
                                   key="vr_apply")

        if st.button("➕ Save rule", type="primary"):
            if not rule_name.strip():
                st.error("Give the rule a name.")
            elif not apply_to:
                st.error("Select at least one dataset to apply this rule to.")
            else:
                new_rule = {
                    "id":       str(uuid.uuid4())[:8],
                    "name":     rule_name.strip(),
                    "type":     rule_type,
                    "column":   col_choice,
                    "severity": severity,
                    "apply_to": apply_to,
                    "created":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                    **extra,
                }
                all_rules.append(new_rule)
                save_rules(all_rules)
                st.success(f"✅ Rule '{rule_name}' saved!")
                add_history("Validation rule", f"Created: {rule_name}")
                st.rerun()

        # Show existing rules
        if all_rules:
            st.divider()
            st.markdown(f"### Saved rules ({len(all_rules)})")
            for rule in all_rules:
                col_r, col_d = st.columns([5, 1])
                sev_icon = rule['severity'].split()[0]
                col_r.markdown(
                    f"{sev_icon} **{rule['name']}** — "
                    f"*{rule['type']}*"
                    + (f" on `{rule['column']}`" if rule.get('column') else "")
                    + f" | Datasets: {', '.join(rule['apply_to'])}"
                )
                if col_d.button("🗑️", key=f"del_rule_{rule['id']}"):
                    all_rules = [r for r in all_rules if r["id"] != rule["id"]]
                    save_rules(all_rules)
                    st.rerun()
        else:
            st.info("No rules defined yet. Add your first rule above.")

    # ── TAB 2: RUN VALIDATION ─────────────────────────────────────────────────
    with tab_run:
        if not all_rules:
            st.info("Define rules in the 'Define Rules' tab first.")
        elif not datasets:
            st.warning("Upload datasets first.")
        else:
            st.markdown("### Run validation checks")

            run_ds = st.multiselect(
                "Datasets to validate",
                list(datasets.keys()),
                format_func=lambda k: datasets[k]["name"],
                default=list(datasets.keys()),
                key="vr_run_ds"
            )

            if st.button("▶️ Run all validation rules", type="primary", width="stretch"):
                results = []
                progress = st.progress(0)
                total_checks = sum(
                    1 for rule in all_rules
                    for ds_id in run_ds
                    if datasets[ds_id]["name"] in rule.get("apply_to", [])
                )
                done = 0

                for ds_id in run_ds:
                    ds = datasets[ds_id]
                    df = ds["df"]
                    for rule in all_rules:
                        if ds["name"] in rule.get("apply_to", []):
                            result = run_rule(df, rule)
                            result["dataset"] = ds["name"]
                            result["dataset_rows"] = len(df)
                            results.append(result)
                            done += 1
                            progress.progress(done / max(total_checks, 1))

                progress.empty()

                if results:
                    st.session_state["_validation_results"] = results
                    passed  = sum(1 for r in results if r["passed"])
                    failed  = sum(1 for r in results if not r["passed"])
                    critical= sum(1 for r in results if not r["passed"] and "Critical" in r["severity"])

                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Total checks", len(results))
                    m2.metric("✅ Passed",   passed,  delta=f"{passed/len(results)*100:.0f}%")
                    m3.metric("❌ Failed",   failed,  delta=f"-{failed}" if failed else "0")
                    m4.metric("🔴 Critical", critical)

                    # Quality score
                    score = round(passed / len(results) * 100, 1)
                    if score >= 90:
                        st.success(f"✅ Data quality score: **{score}%** — Excellent")
                    elif score >= 70:
                        st.warning(f"⚠️ Data quality score: **{score}%** — Needs attention")
                    else:
                        st.error(f"❌ Data quality score: **{score}%** — Poor quality")

                    add_history("Validation run",
                                f"{passed}/{len(results)} checks passed | Score: {score}%")

            # Show results
            if "_validation_results" in st.session_state:
                results = st.session_state["_validation_results"]
                st.divider()
                st.markdown("### Results")

                filter_type = st.radio("Show", ["All", "Failed only", "Passed only"],
                                       horizontal=True, key="vr_filter")

                for r in results:
                    if filter_type == "Failed only" and r["passed"]: continue
                    if filter_type == "Passed only" and not r["passed"]: continue

                    icon = "✅" if r["passed"] else r["severity"].split()[0]
                    col_r, col_rows = st.columns([4, 1])
                    col_r.markdown(
                        f"{icon} **{r['rule_name']}** on `{r['dataset']}`"
                        + (f" → `{r['column']}`" if r.get('column') else "")
                        + f": {r['message']}"
                    )
                    if r["failing_rows"] > 0:
                        pct = r["failing_rows"] / r["dataset_rows"] * 100
                        col_rows.markdown(f"**{r['failing_rows']:,}** rows ({pct:.1f}%)")

                # Export report
                st.divider()
                report_df = pd.DataFrame(results)[
                    ["dataset","rule_name","type","column","severity","passed","message","failing_rows"]
                ]
                csv_data = report_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download validation report (CSV)",
                    data=csv_data,
                    file_name=f"udip_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

    # ── TAB 3: HISTORY ────────────────────────────────────────────────────────
    with tab_history:
        val_history = [h for h in st.session_state.get("history", [])
                       if "Validation" in h.get("action", "")]
        if not val_history:
            st.info("No validation runs yet.")
        else:
            for h in val_history[:20]:
                st.markdown(
                    f"🕐 **{h['date']} {h['time']}** — {h['detail']}"
                )
