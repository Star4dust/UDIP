import duckdb
import json
import datetime
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("udip.system_observer")
logger.addHandler(logging.NullHandler())

OBSERVER_DIR = Path("udip_data/system_observer")
DB_PATH = OBSERVER_DIR / "metrics.db"

class SystemObserver:
    """
    Centralized observability module for tracking agent decisions, execution performance,
    and system reliability using DuckDB for persistence.
    """
    
    def __init__(self):
        OBSERVER_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize DuckDB tables."""
        with duckdb.connect(str(DB_PATH)) as con:
            # 1. Run Summaries
            con.execute("""
                CREATE TABLE IF NOT EXISTS run_summaries (
                    run_id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP,
                    intent JSON,
                    agents_used JSON,
                    total_time_ms DOUBLE,
                    success BOOLEAN,
                    step_count INTEGER
                )
            """)
            
            # 2. Execution Steps
            con.execute("""
                CREATE TABLE IF NOT EXISTS execution_steps (
                    run_id VARCHAR,
                    step_number INTEGER,
                    action VARCHAR,
                    duration_ms DOUBLE,
                    success BOOLEAN,
                    error_tier VARCHAR, -- DataError, ExecutionError, SystemError
                    error_msg TEXT
                )
            """)
            
            # 3. Full Traces (Sampled or On-Demand)
            con.execute("""
                CREATE TABLE IF NOT EXISTS full_traces (
                    run_id VARCHAR PRIMARY KEY,
                    trace_json JSON
                )
            """)

    def log_run_summary(self, run_id: str, intent: list[str], agents: list[str], 
                        total_time_ms: float, success: bool, step_count: int):
        """Log a high-level summary of a pipeline run."""
        with duckdb.connect(str(DB_PATH)) as con:
            con.execute("""
                INSERT INTO run_summaries (run_id, timestamp, intent, agents_used, total_time_ms, success, step_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, 
                datetime.datetime.now(), 
                json.dumps(intent), 
                json.dumps(agents), 
                total_time_ms, 
                success, 
                step_count
            ))

    def log_step(self, run_id: str, step_number: int, action: str, 
                 duration_ms: float, success: bool, 
                 error_tier: Optional[str] = None, error_msg: Optional[str] = None):
        """Log metrics for a single execution step."""
        with duckdb.connect(str(DB_PATH)) as con:
            con.execute("""
                INSERT INTO execution_steps (run_id, step_number, action, duration_ms, success, error_tier, error_msg)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, 
                step_number, 
                action, 
                duration_ms, 
                success, 
                error_tier, 
                error_msg
            ))

    def log_full_trace(self, run_id: str, trace: dict):
        """Log the full execution trace (sampled or on-demand)."""
        with duckdb.connect(str(DB_PATH)) as con:
            con.execute("""
                INSERT INTO full_traces (run_id, trace_json)
                VALUES (?, ?)
            """, (run_id, json.dumps(trace)))

    def get_summary_metrics(self) -> dict:
        """Aggregate system metrics for dashboard visualization."""
        with duckdb.connect(str(DB_PATH)) as con:
            # Calculate overall success rate
            stats = con.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
                    AVG(total_time_ms) as avg_run_time_ms,
                    SUM(step_count) as total_steps_executed
                FROM run_summaries
            """).fetchone()
            
            # Error distribution
            errors = con.execute("""
                SELECT error_tier, COUNT(*) as count
                FROM execution_steps
                WHERE success = FALSE
                GROUP BY error_tier
            """).fetchall()
            
            return {
                "overview": {
                    "total_runs": int(stats[0]) if stats[0] else 0,
                    "success_rate": float(stats[1]) if stats[1] else 0.0,
                    "avg_run_time_ms": float(stats[2]) if stats[2] else 0.0,
                    "total_steps": int(stats[3]) if stats[3] else 0
                },
                "error_distribution": {str(tier): int(count) for tier, count in errors if tier}
            }

    def get_detailed_logs(self, limit: int = 50) -> list[dict]:
        """Return a list of recent run summaries."""
        with duckdb.connect(str(DB_PATH)) as con:
            rows = con.execute(f"""
                SELECT run_id, timestamp, intent, agents_used, total_time_ms, success, step_count
                FROM run_summaries
                ORDER BY timestamp DESC
                LIMIT {limit}
            """).fetchall()
            
            logs = []
            for r in rows:
                logs.append({
                    "run_id": r[0],
                    "timestamp": r[1].isoformat() if r[1] else None,
                    "intent": json.loads(r[2]) if r[2] else [],
                    "agents": json.loads(r[3]) if r[3] else [],
                    "duration_ms": r[4],
                    "success": r[5],
                    "steps": r[6]
                })
            return logs
