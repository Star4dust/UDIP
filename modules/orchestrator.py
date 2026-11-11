"""
UDIP Orchestrator Agent.

The orchestrator converts a user request into a structured execution plan.
It never executes actions and never manipulates data directly.
"""

from __future__ import annotations

import re
import uuid
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.system_observer import SystemObserver

_observer = SystemObserver()


EXECUTION_ACTIONS = {
    "run_sql",
    "load_dataset",
    "save_dataset",
    "clean_data",
    "detect_anomalies",
    "create_visualization",
    "generate_report",
    "engineer_features",
    "merge_datasets",
    "clarify_request",
}

CLEANING_METHOD_ALIASES = {
    "standard": {
        "standard",
        "default",
        "auto clean",
        "autoclean",
        "clean",
        "cleanup",
        "clean up",
        "fix data",
        "prepare data",
        "preprocess",
    },
    "drop_nulls": {
        "drop null",
        "drop nulls",
        "drop missing",
        "remove null",
        "remove nulls",
        "remove missing",
    },
    "fill_nulls": {
        "fill null",
        "fill nulls",
        "fill missing",
        "impute",
        "impute missing",
    },
    "drop_duplicates": {
        "drop duplicate",
        "drop duplicates",
        "deduplicate",
        "remove duplicate",
        "remove duplicates",
    },
    "trim_strings": {
        "trim",
        "trim strings",
        "strip whitespace",
        "remove whitespace",
        "clean text",
    },
    "normalize_columns": {
        "normalize columns",
        "normalise columns",
        "rename columns",
        "standardize columns",
        "standardise columns",
    },
    "infer_types": {
        "infer types",
        "detect types",
        "convert types",
        "parse dates",
    },
    "remove_constant_columns": {
        "remove constant",
        "remove constant columns",
        "drop constant",
        "drop constant columns",
    },
    "advanced": {
        "advanced",
        "advanced clean",
    },
    "hirax": {
        "hirax",
    },
}

LOAD_KEYWORDS = {
    "load",
    "open",
    "read",
    "upload",
    "import",
    "ingest",
}

SAVE_KEYWORDS = {
    "save",
    "export",
    "store",
    "persist",
    "write",
}

ANOMALY_KEYWORDS = {
    "anomaly",
    "anomalies",
    "outlier",
    "outliers",
    "unusual",
    "abnormal",
    "detect issues",
    "check issues",
    "quality check",
    "drift",
    "monitor",
}

SQL_KEYWORDS = {
    "sql",
    "query",
    "select",
    "group by",
    "order by",
    "where",
}

VISUALIZE_KEYWORDS = {
    "plot",
    "visualize",
    "chart",
}

REPORT_KEYWORDS = {
    "report",
    "summary",
}

ML_KEYWORDS = {
    "features",
    "engineer features",
    "ml",
    "machine learning",
    "feature engineering",
}

SUPPORTED_FILE_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".parquet",
    ".xlsx",
    ".xls",
    ".feather",
    ".h5",
    ".hdf5",
}


@dataclass(frozen=True)
class Intent:
    load: bool = False
    clean: bool = False
    detect_anomalies: bool = False
    run_sql: bool = False
    save: bool = False
    visualize: bool = False
    report: bool = False
    feature_engineering: bool = False
    intents: tuple[str, ...] = ()
    requires_clarification: bool = False
    cleaning_method: str = "standard"
    chart_type: str = "line"
    file_path: str | None = None
    sql_query: str | None = None
    save_name: str | None = None


class MLResearcherAgent:
    """Handles feature engineering and advanced ML tasks."""
    def refine_plan(self, intent: Intent) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        if intent.feature_engineering:
            steps.append({
                "id": "ml_engineer_features",
                "action": "engineer_features",
                "depends_on": ["ds_detect_anomalies", "de_clean_data", "de_load_dataset"]
            })
        return {
            "agent_name": "ml_researcher",
            "proposed_steps": steps,
            "status": "success"
        }


class DataEngineerAgent:
    """Handles loading and cleaning operations."""
    def refine_plan(self, intent: Intent) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []

        if intent.load:
            step: dict[str, Any] = {"id": "de_load_dataset", "action": "load_dataset", "depends_on": []}
            if intent.file_path:
                step["file_path"] = intent.file_path
            steps.append(step)

        if intent.run_sql:
            step = {"id": "de_run_sql", "action": "run_sql", "depends_on": ["de_load_dataset"]}
            if intent.sql_query:
                step["query"] = intent.sql_query
            steps.append(step)

        if intent.clean:
            steps.append({
                "id": "de_clean_data",
                "action": "clean_data",
                "method": intent.cleaning_method,
                "depends_on": ["de_run_sql", "de_load_dataset"]
            })

        return {
            "agent_name": "data_engineer",
            "proposed_steps": steps,
            "status": "success"
        }


class DataScientistAgent:
    """Handles analysis, anomalies, visualization, and reporting."""
    def refine_plan(self, intent: Intent) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []

        if intent.detect_anomalies:
            steps.append({"id": "ds_detect_anomalies", "action": "detect_anomalies", "depends_on": ["de_clean_data", "de_load_dataset"]})

        if intent.visualize:
            steps.append({
                "id": "ds_visualize",
                "action": "create_visualization",
                "chart_type": intent.chart_type,
                "depends_on": ["ds_detect_anomalies", "de_clean_data", "de_load_dataset"]
            })

        if intent.report:
            steps.append({"id": "ds_report", "action": "generate_report", "depends_on": ["ds_visualize", "ds_detect_anomalies", "de_load_dataset"]})

        return {
            "agent_name": "data_scientist",
            "proposed_steps": steps,
            "status": "success"
        }


class OrchestratorCoordinator:
    """Rule-based coordinator for UDIP execution-layer actions."""

    def __init__(self):
        self.data_engineer = DataEngineerAgent()
        self.data_scientist = DataScientistAgent()
        self.ml_researcher = MLResearcherAgent()

    def plan(self, user_request: str) -> dict[str, Any]:
        request = self._validate_request(user_request)
        intent = self.parse_intent(request)
        
        # Consult all agents to ensure dependencies are handled
        agents_used = ["data_engineer", "data_scientist", "ml_researcher"]
        all_proposals = [
            self.data_engineer.refine_plan(intent),
            self.data_scientist.refine_plan(intent),
            self.ml_researcher.refine_plan(intent)
        ]

        merged_steps = self._merge_proposals(all_proposals)
        sorted_steps = self._topological_sort(merged_steps)

        if intent.save:
            step = {"id": "coord_save", "action": "save_dataset", "depends_on": [s["id"] for s in sorted_steps]}
            if intent.save_name:
                step["name"] = intent.save_name
            sorted_steps.append(step)

        if not sorted_steps:
            sorted_steps.append({"id": "coord_clarify", "action": "clarify_request", "depends_on": []})

        return {
            "intent": list(intent.intents),
            "agents_used": agents_used,
            "execution_plan": sorted_steps,
            "requires_clarification": intent.requires_clarification,
            "ui_resolution": [],
            "requires_input": self._missing_inputs(sorted_steps),
        }

    def _merge_proposals(self, proposals: list[dict]) -> list[dict]:
        precedence = {"data_engineer": 1, "data_scientist": 2, "ml_researcher": 3}
        merged = {}
        
        for prop in proposals:
            agent_name = prop.get("agent_name", "unknown")
            agent_priority = precedence.get(agent_name, 99)
            
            for step in prop.get("proposed_steps", []):
                action = step["action"]
                
                if action not in merged:
                    merged[action] = {"step": step, "priority": agent_priority}
                else:
                    existing_step = merged[action]["step"]
                    existing_priority = merged[action]["priority"]
                    
                    new_step = dict(existing_step)
                    conflict = False
                    
                    for k, v in step.items():
                        if k in ["id", "depends_on"]:
                            if k == "depends_on":
                                new_step["depends_on"] = list(set(new_step.get("depends_on", []) + v))
                            continue
                            
                        if k in existing_step and existing_step[k] != v:
                            conflict = True
                        new_step[k] = v
                        
                    if not conflict:
                        merged[action]["step"] = new_step
                    else:
                        if agent_priority < existing_priority:
                            step["depends_on"] = list(set(step.get("depends_on", []) + existing_step.get("depends_on", [])))
                            merged[action] = {"step": step, "priority": agent_priority}
                        else:
                            existing_step["depends_on"] = list(set(existing_step.get("depends_on", []) + step.get("depends_on", [])))
                            merged[action]["step"] = existing_step
                            
        return [m["step"] for m in merged.values()]

    def _topological_sort(self, steps: list[dict]) -> list[dict]:
        valid_ids = {s["id"] for s in steps}
        graph = {s["id"]: set(d for d in s.get("depends_on", []) if d in valid_ids) for s in steps}
        
        sorted_ids = []
        while graph:
            ready = {node for node, deps in graph.items() if not deps}
            if not ready:
                ready = {next(iter(graph))}
                
            for node in ready:
                sorted_ids.append(node)
                del graph[node]
                
            for deps in graph.values():
                deps.difference_update(ready)
                
        step_map = {s["id"]: s for s in steps}
        return [step_map[sid] for sid in sorted_ids]

    def parse_intent(self, user_request: str) -> Intent:
        request = self._validate_request(user_request)
        lowered = request.lower()

        file_path = self._extract_file_path(request)
        sql_query = self._extract_sql_query(request)
        save_name = self._extract_save_name(request)
        cleaning_method = self._detect_cleaning_method(lowered)

        load = file_path is not None or self._contains_any(lowered, LOAD_KEYWORDS)
        clean = self._contains_cleaning_intent(lowered)
        detect_anomalies = self._contains_any(lowered, ANOMALY_KEYWORDS)
        run_sql = sql_query is not None or self._contains_sql_intent(lowered)
        save = save_name is not None or self._contains_any(lowered, SAVE_KEYWORDS)
        visualize = self._contains_any(lowered, VISUALIZE_KEYWORDS)
        report = self._contains_any(lowered, REPORT_KEYWORDS)
        feature_engineering = self._contains_any(lowered, ML_KEYWORDS)

        intents_list = []
        if load or run_sql:
            intents_list.append("data_engineering")
        if clean:
            intents_list.append("cleaning")
        if detect_anomalies:
            intents_list.append("analysis")
        if visualize:
            intents_list.append("visualization")
        if report:
            intents_list.append("reporting")
        if feature_engineering:
            intents_list.append("feature_engineering")
            
        requires_clarification = False
        if not intents_list:
            intents_list.append("uncertain")
            requires_clarification = True

        return Intent(
            load=load,
            clean=clean,
            detect_anomalies=detect_anomalies,
            run_sql=run_sql,
            save=save,
            visualize=visualize,
            report=report,
            feature_engineering=feature_engineering,
            intents=tuple(intents_list),
            requires_clarification=requires_clarification,
            cleaning_method=cleaning_method,
            file_path=file_path,
            sql_query=sql_query,
            save_name=save_name,
        )

    def _missing_inputs(self, steps: list[dict[str, Any]]) -> list[dict[str, str]]:
        missing: list[dict[str, str]] = []
        for step in steps:
            action = step["action"]
            if action == "load_dataset" and "file_path" not in step:
                missing.append({
                    "action": action,
                    "field": "file_path",
                    "reason": "A dataset path is required before execution.",
                })
            elif action == "run_sql" and "query" not in step:
                missing.append({
                    "action": action,
                    "field": "query",
                    "reason": "A SQL query is required before execution.",
                })
            elif action == "save_dataset" and "name" not in step:
                missing.append({
                    "action": action,
                    "field": "name",
                    "reason": "A dataset name is required before execution.",
                })
            elif action == "clarify_request":
                missing.append({
                    "action": action,
                    "field": "request",
                    "reason": "No supported execution-layer intent was detected.",
                })
        return missing

    def _validate_request(self, user_request: str) -> str:
        if not isinstance(user_request, str):
            return ""
        request = user_request.strip()
        if len(request) > 20_000:
            return request[:20_000] # Truncate rather than completely reject
        return request

    def _contains_any(self, text: str, terms: set[str]) -> bool:
        return any(term in text for term in terms)

    def _contains_cleaning_intent(self, text: str) -> bool:
        if self._contains_any(text, {"clean", "cleanup", "clean up", "preprocess", "prepare data"}):
            return True
        return any(alias in text for aliases in CLEANING_METHOD_ALIASES.values() for alias in aliases)

    def _contains_sql_intent(self, text: str) -> bool:
        if self._contains_any(text, SQL_KEYWORDS):
            return True
        return bool(re.search(r"\bselect\s+.+\s+from\s+", text, flags=re.IGNORECASE | re.DOTALL))

    def _detect_cleaning_method(self, text: str) -> str:
        for method, aliases in CLEANING_METHOD_ALIASES.items():
            if any(alias in text for alias in aliases):
                return method
        return "standard"

    def _extract_file_path(self, user_request: str) -> str | None:
        quoted_path = re.search(r"['\"]([^'\"]+)['\"]", user_request)
        if quoted_path:
            candidate = quoted_path.group(1).strip()
            if Path(candidate).suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                return candidate

        path_pattern = re.compile(
            r"(?P<path>(?:[A-Za-z]:\\|\.{1,2}[\\/]|[A-Za-z0-9_.-]+[\\/])"
            r"[^\n\r\t]*?"
            r"(?:\.csv|\.tsv|\.txt|\.jsonl?|\.parquet|\.xlsx?|\.feather|\.hdf5?))",
            flags=re.IGNORECASE,
        )
        path_match = path_pattern.search(user_request)
        if path_match:
            return path_match.group("path").strip(" .'\"")

        filename_pattern = re.compile(
            r"\b[\w .()-]+(?:\.csv|\.tsv|\.txt|\.jsonl?|\.parquet|\.xlsx?|\.feather|\.hdf5?)\b",
            flags=re.IGNORECASE,
        )
        filename_match = filename_pattern.search(user_request)
        if filename_match:
            return filename_match.group(0).strip(" .'\"")

        return None

    def _extract_sql_query(self, user_request: str) -> str | None:
        fenced = re.search(r"```sql\s*([\s\S]*?)```", user_request, flags=re.IGNORECASE)
        if fenced:
            query = fenced.group(1).strip()
            return query or None

        inline = re.search(
            r"\b(select|with|show|describe|explain|summarize)\b[\s\S]*",
            user_request,
            flags=re.IGNORECASE,
        )
        if inline and re.search(r"\b(select|with|show|describe|explain|summarize)\b", inline.group(0), flags=re.IGNORECASE):
            query = inline.group(0).strip()
            return query if self._looks_like_sql(query) else None

        return None

    def _looks_like_sql(self, text: str) -> bool:
        lowered = text.lower()
        return (
            lowered.startswith(("select", "with", "show", "describe", "explain", "summarize"))
            and (" from " in lowered or lowered.startswith(("show", "describe", "explain", "summarize")))
        )

    def _extract_save_name(self, user_request: str) -> str | None:
        patterns = (
            r"\bsave\s+(?:as|to|dataset\s+as)\s+['\"]?([A-Za-z0-9_. -]+)['\"]?",
            r"\bexport\s+(?:as|to)\s+['\"]?([A-Za-z0-9_. -]+)['\"]?",
            r"\bstore\s+(?:as|to)\s+['\"]?([A-Za-z0-9_. -]+)['\"]?",
        )
        for pattern in patterns:
            match = re.search(pattern, user_request, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip(" .'\"")
                if name:
                    return name
        return None


def plan_actions(user_request: str) -> dict[str, Any]:
    plan = OrchestratorCoordinator().plan(user_request)
    run_id = str(uuid.uuid4())
    plan["run_id"] = run_id
    
    # Sample full traces (e.g. 10% of runs) for observability
    if random.random() < 0.1:
        _observer.log_full_trace(run_id, plan)
        
    return plan
