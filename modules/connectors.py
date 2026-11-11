"""
UDIP v3 — modules/connectors.py
Direct database connections and data ingestion.
"""

import pandas as pd
from sqlalchemy import create_engine, inspect
from modules.core import log

def get_connection_url(db_type: str, host: str, port: int, database: str, user: str, password: str) -> str:
    """Generate SQLAlchemy connection URL."""
    if db_type == "PostgreSQL":
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "MySQL":
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "Snowflake":
        return f"snowflake://{user}:{password}@{host}/{database}"
    elif db_type == "SQLite":
        return f"sqlite:///{database}"
    return ""

def test_connection(url: str) -> tuple:
    """Test if a database connection is successful."""
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            return True, "Connection successful!"
    except Exception as e:
        log.error(f"DB Connection failed: {e}")
        return False, str(e)

def list_tables(url: str) -> list:
    """List all tables in the connected database."""
    try:
        engine = create_engine(url)
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        log.error(f"Failed to list tables: {e}")
        return []

def query_database(url: str, query: str) -> pd.DataFrame:
    """Execute a query and return a DataFrame."""
    try:
        engine = create_engine(url)
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        log.error(f"DB Query failed: {e}")
        raise e
