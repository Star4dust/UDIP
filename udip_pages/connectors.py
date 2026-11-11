"""
UDIP v3 — udip_pages/connectors.py
UI for live database connections.
"""

import streamlit as st
import pandas as pd
import uuid
from modules.connectors import get_connection_url, test_connection, list_tables, query_database
from modules.core import save_dataset, add_history, log

def render():
    st.markdown('<div class="main-header">🔌 Live Connectors</div>', unsafe_allow_html=True)
    st.markdown("Connect directly to external databases and pull data into the UDIP ecosystem.")

    tab1, tab2 = st.tabs(["🔗 Connection Manager", "🔍 Data Browser"])

    with tab1:
        st.subheader("New Connection")
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "Snowflake", "SQLite"])
        
        col1, col2 = st.columns(2)
        if db_type != "SQLite":
            host = col1.text_input("Host", value="localhost")
            port = col2.number_input("Port", value=5432 if db_type=="PostgreSQL" else 3306)
            database = col1.text_input("Database Name")
            user = col2.text_input("Username")
            password = col1.text_input("Password", type="password")
        else:
            database = st.text_input("Path to SQLite file (e.g., ./data.db)")
            host = port = user = password = ""

        if st.button("⚡ Test Connection", type="secondary"):
            url = get_connection_url(db_type, host, port, database, user, password)
            ok, msg = test_connection(url)
            if ok:
                st.success(msg)
                st.session_state.active_db_url = url
            else:
                st.error(f"Failed: {msg}")

    with tab2:
        if "active_db_url" not in st.session_state:
            st.info("Connect to a database in the 'Connection Manager' tab first.")
        else:
            url = st.session_state.active_db_url
            st.success(f"Connected to {db_type}")
            
            tables = list_tables(url)
            selected_table = st.selectbox("Select Table to Preview", ["-- Choose Table --"] + tables)
            
            if selected_table != "-- Choose Table --":
                query = st.text_area("SQL Query", value=f"SELECT * FROM {selected_table} LIMIT 1000")
                if st.button("▶️ Run Query & Import"):
                    try:
                        with st.spinner("Fetching data..."):
                            df = query_database(url, query)
                            st.dataframe(df.head(100), use_container_width=True)
                            
                            ds_name = st.text_input("Dataset Name", value=f"db_{selected_table}")
                            if st.button("💾 Save as Dataset"):
                                ds_id = str(uuid.uuid4())[:8]
                                ds_entry = {
                                    "name": ds_name, "df": df,
                                    "format": "db_query", "source": f"db_{db_type}",
                                }
                                st.session_state.datasets[ds_id] = ds_entry
                                save_dataset(ds_id, ds_entry)
                                add_history("DB Import", f"Imported {len(df)} rows from {db_type}")
                                st.success(f"Dataset '{ds_name}' added to workspace!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Query failed: {e}")
