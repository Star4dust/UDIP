import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
"""
UDIP v3 — pages/sql_editor.py
SQL editor — run SQL queries on loaded datasets using DuckDB.
"""

import streamlit as st
import pandas as pd
import time
from modules.core import log, add_history, fmt_bytes, save_code
from modules.operator_actions import render_operator_handoff
from modules.sql_engine import run_sql, get_schema_info, get_sql_tables, SQL_TEMPLATES
from modules.state_manager import get_page_memory
from modules.context_engine import get_context_engine

DEFAULT_SQL = "-- Write your SQL query here\n-- Your datasets are available as tables\n\nSELECT *\nFROM your_table\nLIMIT 100;"

def render():
    st.markdown('<div class="main-header">🗄️ SQL Editor</div>', unsafe_allow_html=True)
    st.markdown("Query your datasets using standard SQL — powered by DuckDB. Fast, no setup needed.")
    render_operator_handoff("🗄️ SQL Editor")

    datasets = st.session_state.get("datasets", {})
    if not datasets:
        st.warning("Upload datasets first to use the SQL editor.")
        return

    # ── Schema sidebar info ────────────────────────────────────────────────────
    tables = get_sql_tables()

    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Query Studio</div>
            <div class="udip-panel__title">Live SQL over your in-memory datasets</div>
            <div class="udip-panel__text">
                DuckDB can query everything you have loaded right now without extra setup. Start from a template,
                inspect the schema, then export or convert the result into Python once the query is right.
            </div>
            <div class="udip-stat-strip">
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(tables)}</div>
                    <div class="udip-stat-tile__label">tables available</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{sum(len(df) for df in tables.values()):,}</div>
                    <div class="udip-stat-tile__label">rows queryable</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(SQL_TEMPLATES)}</div>
                    <div class="udip-stat-tile__label">starter templates</div>
                </div>
            </div>
            <div class="udip-pill-row">
                <div class="udip-pill udip-pill--accent">DuckDB engine</div>
                <div class="udip-pill">Schema-aware prompts</div>
                <div class="udip-pill udip-pill--warm">Export-ready results</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📋 Available tables & columns", expanded=True):
        for table_name, df in tables.items():
            st.markdown(f"**`{table_name}`** — {len(df):,} rows × {len(df.columns)} cols")
            cols_str = " · ".join(f"`{c}` ({df[c].dtype})" for c in df.columns[:12])
            if len(df.columns) > 12:
                cols_str += f" · *+{len(df.columns)-12} more*"
            st.markdown(f"<small>{cols_str}</small>", unsafe_allow_html=True)

    # ── Template selector ──────────────────────────────────────────────────────
    col_tmpl, col_fmt = st.columns([2, 1])
    with col_tmpl:
        template = st.selectbox("Load template", ["— choose —"] + list(SQL_TEMPLATES.keys()), key="sql_tmpl")
    with col_fmt:
        export_fmt = st.selectbox("Export format", ["CSV","Excel","Parquet","JSON"], key="sql_export")

    if template != "— choose —":
        tbl = list(tables.keys())[0] if tables else "my_table"
        df_ex = list(tables.values())[0] if tables else pd.DataFrame()
        num_cols = df_ex.select_dtypes(include="number").columns.tolist()
        cat_cols = df_ex.select_dtypes(exclude="number").columns.tolist()

        try:
            filled = SQL_TEMPLATES[template].format(
                table=tbl,
                table1=list(tables.keys())[0] if tables else "table1",
                table2=list(tables.keys())[1] if len(tables)>1 else "table2",
                category_col=cat_cols[0] if cat_cols else "category",
                value_col=num_cols[0] if num_cols else "value",
                column=df_ex.columns[0] if len(df_ex.columns)>0 else "column",
                key=cat_cols[0] if cat_cols else "id",
                col=num_cols[0] if num_cols else "col",
                date_col=next((c for c in df_ex.columns if "date" in c.lower()), "date_col"),
                order_col=cat_cols[0] if cat_cols else "id",
            )
        except KeyError as e:
            st.error(f"Template has missing placeholder: {e}. Using basic template.")
            filled = f"SELECT *\nFROM {tbl}\nLIMIT 100;"
            
        if "sql_query" not in st.session_state:
            st.session_state.sql_query = filled
        else:
            st.session_state.sql_query = filled

    # ── SQL Editor ─────────────────────────────────────────────────────────────
    _mem = get_page_memory("sql")

    # Seed from memory if session_state doesn't already have a query
    if "sql_query" not in st.session_state:
        st.session_state.sql_query = _mem.get("query", DEFAULT_SQL)

    # SQL editor with syntax highlighting (ace or fallback)
    try:
        from streamlit_ace import st_ace
        query = st_ace(
            value=st.session_state.sql_query,
            language="sql",
            theme="vscode",
            height=220,
            font_size=14,
            tab_size=2,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=False,
            key="sql_editor_ace",
            placeholder="SELECT * FROM your_table LIMIT 100;",
        )
        if not query:
            query = st.session_state.get("sql_query", "")
    except Exception:
        query = st.text_area(
            "SQL Query",
            value=st.session_state.sql_query,
            height=220,
            key="sql_input",
            label_visibility="collapsed",
            placeholder="SELECT * FROM your_table LIMIT 100;",
        )
    if query:
        st.session_state.sql_query = query
        _mem.set("query", query)  # persist across navigation

    col_run, col_explain, col_clear = st.columns([1,1,3])

    run_clicked     = col_run.button("▶️ Run SQL",    type="primary", width="stretch")
    explain_clicked = col_explain.button("🤖 Explain",width="stretch")

    if col_clear.button("🗑️ Clear", width="content"):
        st.session_state.sql_query = DEFAULT_SQL
        st.rerun()

    # ── Run query ──────────────────────────────────────────────────────────────
    if run_clicked:
        if not query.strip() or query.strip().startswith("--"):
            st.warning("Write a SQL query first.")
        else:
            with st.spinner("Executing SQL..."):
                result, error = run_sql(query)

            if error:
                st.error(f"❌ SQL Error: {error}")
                if "no table named" in error.lower():
                    st.info(f"Available tables: {', '.join(f'`{t}`' for t in tables.keys())}")
            else:
                st.success(f"✅ {len(result):,} rows returned")
                st.session_state["_sql_result"] = result
                st.session_state["_sql_query_run"] = query
                
                # Track in context engine
                try:
                    context_engine = get_context_engine()
                    context_engine.track_sql_query(query, {
                        "rows_returned": len(result),
                        "columns": list(result.columns),
                        "execution_time": time.time()
                    })
                    context_engine.set_active_sql_result(result, query, f"SQL Query Result")
                except Exception as e:
                    # Context engine should not break SQL functionality
                    pass

    # ── Explain with AI ────────────────────────────────────────────────────────
    if explain_clicked:
        groq_key = os.getenv("GROQ_API_KEY", st.session_state.get("groq_key",""))
        if not groq_key:
            st.warning("Set your Groq API key in Settings to use AI explain.")
        else:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                schema = get_schema_info()
                prompt = f"Explain what this SQL query does in plain English:\n\n{query}\n\nAvailable tables:\n{schema}"
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=500,
                )
                st.info(f"🤖 **AI Explanation:** {resp.choices[0].message.content}")
            except Exception as e:
                st.error(f"AI error: {e}")

    # ── Show result ────────────────────────────────────────────────────────────
    if "_sql_result" in st.session_state:
        result = st.session_state["_sql_result"]

        tab_data, tab_stats = st.tabs(["📋 Results", "📊 Quick stats"])

        with tab_data:
            n_show = st.slider("Rows to display", 10, min(10000, len(result)), min(100, len(result)), key="sql_show")
            st.dataframe(result.head(n_show), width="stretch")

            # Export
            import io
            fmt = export_fmt.lower()
            if fmt == "csv":
                data = result.to_csv(index=False).encode()
                fname = "sql_result.csv"; mime = "text/csv"
            elif fmt == "excel":
                buf = io.BytesIO(); result.to_excel(buf, index=False); buf.seek(0)
                data = buf.getvalue(); fname = "sql_result.xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif fmt == "parquet":
                buf = io.BytesIO(); result.to_parquet(buf, index=False); buf.seek(0)
                data = buf.getvalue(); fname = "sql_result.parquet"
                mime = "application/octet-stream"
            else:
                data = result.to_json(orient="records", indent=2).encode()
                fname = "sql_result.json"; mime = "application/json"

            col_dl, col_py, col_save, col_pin = st.columns(4)
            with col_dl:
                st.download_button(
                    f"⬇️ Download {export_fmt}",
                    data=data, file_name=fname, mime=mime,
                    width="stretch"
                )

            with col_py:
                groq_key = os.getenv("GROQ_API_KEY", st.session_state.get("groq_key",""))
                if groq_key:
                    if st.button("🐍 Convert SQL to Python", width="stretch"):
                        with st.spinner("AI is writing Pandas code..."):
                            try:
                                from groq import Groq
                                import re
                                client = Groq(api_key=groq_key)
                                schema = get_schema_info()
                                prompt = f"Convert this SQL query into equivalent Pandas code.\nDatasets are in a dictionary like this: `df = datasets['dataset_id']['df']`.\nUse the table names as dataset_id.\n\nSCHEMA:\n{schema}\n\nSQL:\n{st.session_state.get('_sql_query_run','')}\n\nReturn EXACTLY the Python code wrapped in ```python ... ```."
                                resp = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role":"user","content":prompt}],
                                    max_tokens=1500,
                                )
                                reply = resp.choices[0].message.content
                                code_match = re.search(r"```python\n([\s\S]*?)```", reply)
                                if code_match:
                                    st.session_state.editor_code = code_match.group(1).strip()
                                    st.session_state.active_page = "🐍 Code Editor"
                                    st.rerun()
                                else:
                                    st.warning("Failed to generate Python code. Please try asking the AI Assistant on the dashboard.")
                            except Exception as e:
                                st.error(f"AI error: {e}")

            with col_save:
                # Save as new dataset
                if st.button("💾 Save result as new dataset", width="stretch"):
                    import uuid as _uuid
                    ds_id = str(_uuid.uuid4())[:8]
                    from modules.core import save_dataset
                    entry = {"name":"sql_result.parquet","df":result,
                             "format":"parquet","source":"sql"}
                    st.session_state.datasets[ds_id] = entry
                    ok, err = save_dataset(ds_id, entry)
                    if ok:
                        st.success("✅ SQL result saved as a new dataset!")
                        add_history("SQL save", f"Saved {len(result):,} rows as new dataset")
                    else:
                        st.error(f"Could not save: {err}")

            with col_pin:
                if st.button("📌 Pin to Report", width="stretch"):
                    from modules.core import pin_to_report
                    query_run = st.session_state.get('_sql_query_run','')
                    md_content = f"### SQL Query\n```sql\n{query_run}\n```\n### Result Sample ({len(result)} rows)\n\n"
                    # Render table nicely
                    md_content += result.head(10).to_markdown(index=False)
                    pin_to_report("sql", "SQL Query Analysis", md_content)

        with tab_stats:
            if len(result) > 0:
                num_cols = result.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    st.dataframe(result[num_cols].describe().T.round(2), width="stretch")
                else:
                    st.info("No numeric columns in result to summarize.")

    # ── Query history ──────────────────────────────────────────────────────────
    sql_history = [h for h in st.session_state.get("history",[]) if h["action"]=="SQL query"]
    if sql_history:
        with st.expander(f"🕐 Recent SQL queries ({len(sql_history)})", expanded=False):
            for h in sql_history[:10]:
                col_q, col_r = st.columns([4,1])
                col_q.code(h["detail"][:80], language="sql")
                if col_r.button("↩️ Rerun", key=f"sql_rerun_{h['time']}"):
                    detail = h["detail"]
                    query_part = detail.split("...")[0] if "..." in detail else detail
                    st.session_state.sql_query = query_part
                    st.rerun()

import os
