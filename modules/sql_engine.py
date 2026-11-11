"""
UDIP v3 — modules/sql_engine.py
Run SQL queries directly on loaded DataFrames using DuckDB.
"""

import streamlit as st
import pandas as pd
from modules.core import log, add_history, sanitize_varname

def get_sql_tables() -> dict[str, pd.DataFrame]:
    """Return dict of {table_name: dataframe} for all loaded datasets."""
    tables = {}
    for ds in st.session_state.get("datasets", {}).values():
        name = sanitize_varname(ds["name"])
        tables[name] = ds["df"]
    return tables

def run_sql(query: str) -> tuple[pd.DataFrame | None, str]:
    """
    Execute SQL on loaded datasets using DuckDB.
    Returns (result_df, error_message).
    """
    try:
        import duckdb
    except ImportError:
        return None, "DuckDB not installed. The app will install it automatically — try again."

    tables = get_sql_tables()
    if not tables:
        return None, "No datasets loaded. Upload some data first."

    try:
        con = duckdb.connect()
        # Register each dataset as a table
        for table_name, df in tables.items():
            con.register(table_name, df)

        result = con.execute(query).df()
        con.close()
        log.info(f"SQL executed: {query[:80]}... → {len(result):,} rows")
        add_history("SQL query", f"{query[:60]}... → {len(result):,} rows")
        return result, ""
    except Exception as e:
        error = str(e)
        log.error(f"SQL error: {error}")
        return None, error

def get_schema_info() -> str:
    """Return schema info string for SQL autocomplete hints."""
    tables = get_sql_tables()
    info = []
    for name, df in tables.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:15])
        info.append(f"  {name}({cols})")
    return "\n".join(info)

SQL_TEMPLATES = {
    "Select all rows": "SELECT *\nFROM {table}\nLIMIT 100;",
    "Count rows": "SELECT COUNT(*) as total_rows\nFROM {table};",
    "Group and sum": "SELECT {category_col}, SUM({value_col}) as total\nFROM {table}\nGROUP BY {category_col}\nORDER BY total DESC\nLIMIT 20;",
    "Filter rows": "SELECT *\nFROM {table}\nWHERE {column} = 'value'\nLIMIT 100;",
    "Join two tables": "SELECT a.*, b.{col}\nFROM {table1} a\nJOIN {table2} b ON a.{key} = b.{key}\nLIMIT 100;",
    "Top N by value": "SELECT *\nFROM {table}\nORDER BY {value_col} DESC\nLIMIT 10;",
    "Distinct values": "SELECT DISTINCT {column}, COUNT(*) as count\nFROM {table}\nGROUP BY {column}\nORDER BY count DESC;",
    "Date aggregation": "SELECT DATE_TRUNC('month', {date_col}) as month,\n       SUM({value_col}) as total\nFROM {table}\nGROUP BY month\nORDER BY month;",
    "Null check": "SELECT {column},\n       COUNT(*) as total,\n       COUNT({column}) as non_null,\n       COUNT(*) - COUNT({column}) as nulls\nFROM {table}\nGROUP BY {column};",
    "Running total": "SELECT *,\n       SUM({value_col}) OVER (ORDER BY {order_col}) as running_total\nFROM {table};",
}
