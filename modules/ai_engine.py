"""
UDIP v3 — modules/ai_engine.py
AI assistant: Groq integration, session memory, code execution sandbox,
live web search via DuckDuckGo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, io, traceback, subprocess, re, time, json, ast
import os
from modules.shared_utilities import get_shared_logger
from modules.core import (
    DATA_DIR, MEMORY_FILE, create_database_from_df, sanitize_varname,
    add_history, save_chat, save_code, load_config
)
from modules.multi_model_ai import get_multi_model_ai, call_ai_with_fallback
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

from PIL import Image
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# ── Safe package allowlist ────────────────────────────────────────────────────
SAFE_PIP = {
    "seaborn":"seaborn","matplotlib":"matplotlib","sklearn":"scikit-learn",
    "scipy":"scipy","plotly":"plotly","statsmodels":"statsmodels",
    "xgboost":"xgboost","lightgbm":"lightgbm","nltk":"nltk",
    "wordcloud":"wordcloud","PIL":"Pillow","openpyxl":"openpyxl",
    "pyarrow":"pyarrow","tabulate":"tabulate","missingno":"missingno",
    "duckdb":"duckdb","sqlalchemy":"sqlalchemy","polars":"polars",
    # Extended visualization packages
    "joypy":"joypy","bokeh":"bokeh","altair":"altair",
    "plotnine":"plotnine","cufflinks":"cufflinks",
    "phik":"phik","yellowbrick":"yellowbrick",
    "shap":"shap","eli5":"eli5",
    "networkx":"networkx","pyvis":"pyvis",
    "geopandas":"geopandas","folium":"folium",
    "dtale":"dtale","sweetviz":"sweetviz",
    "kaleido":"kaleido",
}

# ── Web Search ────────────────────────────────────────────────────────────────
WEB_TRIGGER_KEYWORDS = [
    "latest", "newest", "current", "2024", "2025", "2026",
    "how to", "what is", "explain", "tutorial", "documentation",
    "best practice", "example of", "difference between",
    "library", "package", "install", "update", "version",
    "error", "bug", "fix", "solution", "workaround",
    "python", "pandas", "numpy", "plotly", "sklearn", "tensorflow",
    "pytorch", "keras", "spark", "dask", "polars",
    "api", "rest", "database", "sql", "nosql",
    "machine learning", "deep learning", "neural network",
    "regression", "classification", "clustering", "nlp",
    "power bi", "tableau", "excel", "google sheets",
    "statistics", "probability", "hypothesis", "p-value",
    "compare", "vs", "alternative", "recommend",
    "deploy", "production", "optimize", "performance",
]

def should_search_web(user_message: str) -> bool:
    """Detect whether a web search would improve the AI response."""
    msg_lower = user_message.lower()
    # Always search if user explicitly asks
    if msg_lower.startswith("/search"):
        return True
    # Don't search for pure code execution requests
    code_only = ["run", "execute", "merge my", "clean my", "show my", "plot my"]
    if any(msg_lower.startswith(p) for p in code_only):
        return False
    # Check for knowledge-seeking keywords
    match_count = sum(1 for kw in WEB_TRIGGER_KEYWORDS if kw in msg_lower)
    return match_count >= 2

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        context = "\n\nWEB SEARCH RESULTS (live from internet):\n"
        context += f"Query: {query}\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            context += f"{i}. **{title}**\n   {body[:300]}\n   Source: {href}\n\n"
        return context
    except ImportError:
        get_shared_logger().log("duckduckgo-search not installed — web search disabled", "info")
        return ""
    except Exception as e:
        get_shared_logger().log(f"Web search failed: {e}", "warning")
        return ""

def extract_search_query(user_message: str) -> str:
    """Extract a clean search query from the user message."""
    msg = user_message.strip()
    # If explicit /search command, extract the query
    if msg.lower().startswith("/search"):
        return msg[7:].strip()
    # Otherwise, clean up the message for search
    # Remove common filler words for a cleaner query
    cleaned = msg.replace("can you", "").replace("please", "").replace("help me", "")
    cleaned = cleaned.replace("I want to", "").replace("I need to", "")
    # Limit length
    words = cleaned.split()[:12]
    return " ".join(words)


# ── AI Memory ─────────────────────────────────────────────────────────────────
MEMORY_PATH = DATA_DIR / "ai_memory.json"

def load_ai_memory() -> dict:
    """Load AI memory from disk."""
    try:
        if MEMORY_PATH.exists():
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_ai_memory(mem: dict):
    """Save AI memory to disk."""
    try:
        MEMORY_PATH.write_text(json.dumps(mem, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        get_shared_logger().log(f"Could not save AI memory: {e}", "warning")

def update_ai_memory(user_msg: str, ai_reply: str):
    """Extract and persist patterns from the conversation."""
    mem = st.session_state.get("ai_memory", {})
    if not isinstance(mem, dict): mem = {}

    # Track column names mentioned
    cols_used = re.findall(r"\b([a-z_][a-z0-9_]{2,})\b", user_msg.lower())
    mem.setdefault("frequent_columns", {})
    for c in cols_used:
        mem["frequent_columns"][c] = mem["frequent_columns"].get(c, 0) + 1

    # Track operation types requested
    op_keywords = {
        "merge":"merge/join","clean":"data cleaning","pivot":"pivot tables",
        "outlier":"outlier detection","export":"export/Power BI","visuali":"visualization",
        "statistic":"statistics","predict":"machine learning","sql":"sql queries",
        "group":"groupby aggregation","date":"date/time processing",
        "correlation":"correlation analysis","regression":"regression modeling",
        "classify":"classification","cluster":"clustering","forecast":"forecasting",
        "normalize":"data normalization","transform":"data transformation",
    }
    mem.setdefault("frequent_operations", {})
    for kw, label in op_keywords.items():
        if kw in user_msg.lower():
            mem["frequent_operations"][label] = mem["frequent_operations"].get(label, 0) + 1

    # Track dataset names
    for ds in st.session_state.get("datasets", {}).values():
        name = ds.get("name","")
        mem.setdefault("known_datasets", [])
        if name and name not in mem["known_datasets"]:
            mem["known_datasets"].append(name)

    # Track user skill indicators
    advanced_keywords = ["lambda", "apply", "vectorize", "multiindex", "groupby",
                         "pivot_table", "merge", "concat", "melt", "stack",
                         "sklearn", "tensorflow", "statsmodels", "scipy"]
    basic_keywords = ["what is", "how do i", "help me", "explain", "simple"]
    
    mem.setdefault("skill_signals", {"advanced": 0, "basic": 0})
    for kw in advanced_keywords:
        if kw in user_msg.lower():
            mem["skill_signals"]["advanced"] += 1
    for kw in basic_keywords:
        if kw in user_msg.lower():
            mem["skill_signals"]["basic"] += 1

    # Track preferred libraries
    lib_mentions = {
        "plotly": "plotly", "matplotlib": "matplotlib", "seaborn": "seaborn",
        "sklearn": "scikit-learn", "scipy": "scipy", "statsmodels": "statsmodels",
    }
    mem.setdefault("preferred_libraries", {})
    for kw, lib in lib_mentions.items():
        if kw in user_msg.lower() or kw in ai_reply.lower():
            mem["preferred_libraries"][lib] = mem["preferred_libraries"].get(lib, 0) + 1

    mem["session_count"] = mem.get("session_count", 0) + 1
    mem["last_active"] = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")

    st.session_state.ai_memory = mem
    save_ai_memory(mem)

def build_memory_context() -> str:
    """Build a memory summary to prepend to AI system prompt."""
    mem = st.session_state.get("ai_memory", {})
    if not mem: return ""

    ctx = "\n\nUSER MEMORY (learned from previous sessions):\n"

    top_ops = sorted(mem.get("frequent_operations",{}).items(), key=lambda x: x[1], reverse=True)[:5]
    if top_ops:
        ctx += f"- Most used operations: {', '.join(op for op,_ in top_ops)}\n"

    top_cols = sorted(mem.get("frequent_columns",{}).items(), key=lambda x: x[1], reverse=True)[:8]
    if top_cols:
        ctx += f"- Frequently referenced columns: {', '.join(c for c,_ in top_cols)}\n"

    known_ds = mem.get("known_datasets", [])
    if known_ds:
        ctx += f"- Previously worked with: {', '.join(known_ds[-5:])}\n"

    # Skill level
    skills = mem.get("skill_signals", {})
    adv = skills.get("advanced", 0)
    bas = skills.get("basic", 0)
    if adv > bas * 2:
        ctx += "- User skill level: Advanced — use technical language, skip basic explanations\n"
    elif bas > adv * 2:
        ctx += "- User skill level: Beginner — explain concepts clearly, add more comments to code\n"
    else:
        ctx += "- User skill level: Intermediate — balance clarity with depth\n"

    # Preferred libraries
    pref_libs = sorted(mem.get("preferred_libraries",{}).items(), key=lambda x: x[1], reverse=True)[:3]
    if pref_libs:
        ctx += f"- Preferred libraries: {', '.join(lib for lib,_ in pref_libs)}\n"

    ctx += f"- Total sessions: {mem.get('session_count',1)}, last active: {mem.get('last_active','unknown')}\n"
    return ctx

# ── Dataset context ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_dataset_summary_cached(ds_id: str, df_hash: str) -> dict:
    """Build and cache a summary of the dataset for AI context."""
    datasets = st.session_state.get("datasets", {})
    if ds_id not in datasets: return {}
    
    df = datasets[ds_id]["df"]
    
    # Columns with detailed schema information
    cols_info = []
    for c in df.columns[:100]: # Increased to 100 for better context
        nulls = int(df[c].isnull().sum())
        null_str = f", {nulls} missing values" if nulls > 0 else ""
        dtype_str = str(df[c].dtype)
        
        # Add sample unique values for categorical data
        sample_vals = ""
        if df[c].nunique() < 15 and not pd.api.types.is_numeric_dtype(df[c]):
            try:
                uniques = df[c].dropna().unique()[:5]
                sample_vals = f", samples: {list(uniques)}"
            except: pass
            
        cols_info.append(f"- '{c}' (dtype: {dtype_str}{null_str}{sample_vals})")
    
    # Summary of all columns to prevent hallucination
    all_cols = list(df.columns)
    
    summary = {
        "id": ds_id,
        "name": datasets[ds_id]["name"],
        "shape": df.shape,
        "columns": cols_info,
        "all_column_names": all_cols,
        "total_cols": len(df.columns)
    }
    return summary

def build_dataset_context(query: str = "", selected_dataset_id: str | None = None) -> str:
    """
    Intelligent Context Builder (Dynamic RAG): 
    Only sends detailed metadata for datasets mentioned in the query.
    Always sends a summary list of all available datasets.
    """
    datasets = st.session_state.get("datasets", {})
    if not datasets:
        return "No datasets loaded. WARNING: Ask the user to upload data first before answering data-related questions."

    query_lower = query.lower()
    
    # Check if user is asking about data but no data is loaded/selected
    if not selected_dataset_id and any(kw in query_lower for kw in ["data", "table", "dataset", "column", "plot", "graph"]):
        # But we check if datasets exist at all
        if datasets:
            pass # We'll pick one below
        else:
            return "ERROR: No datasets are currently loaded in UDIP. I cannot answer questions about data until you upload a file."

    ctx = "AVAILABLE DATASETS (Overview):\n"
    for ds_id, ds in datasets.items():
        varname = ds.get("varname", sanitize_varname(ds["name"]))
        ctx += f"- {ds['name']} (Variable: {varname}, ID: {ds_id})\n"
    
    ctx += "\nDETAILED CONTEXT (Selective):\n"

    # Identify which datasets are actually relevant to the query
    relevant_ids = []
    for ds_id, ds in datasets.items():
        name_match = ds["name"].lower() in query_lower
        var_match = ds.get("varname", "").lower() in query_lower
        # Check columns only for short query to avoid false positives on long prompts
        col_match = any(c.lower() in query_lower for c in ds["df"].columns) if len(query) < 200 else False
        if name_match or var_match or col_match:
            relevant_ids.append(ds_id)

    # If a selected dataset exists, prioritize it
    if selected_dataset_id and selected_dataset_id in datasets:
        if selected_dataset_id not in relevant_ids:
            relevant_ids.insert(0, selected_dataset_id)

    # If nothing specific matched, include the first/most recent dataset as default
    if not relevant_ids and datasets:
        relevant_ids = [selected_dataset_id] if selected_dataset_id and selected_dataset_id in datasets else [list(datasets.keys())[-1]]

    total_chars = 0
    MAX_CHARS = 7_000 # Slightly increased
    
    for ds_id in relevant_ids:
        ds = datasets[ds_id]
        df = ds["df"]
        
        # Use cached summary
        from modules.core import df_hash as _df_hash
        summary = get_dataset_summary_cached(ds_id, _df_hash(df))
        
        block = f"=== {summary['name']} (FULL DETAILS) ===\n"
        block += f"Dataset ID: {ds_id}  <-- USE THIS EXACT ID\n"
        block += f"Access code: df = datasets['{ds_id}']['df']\n"
        block += f"Shape: {summary['shape'][0]:,} rows x {summary['shape'][1]} columns\n"
        
        block += "COLUMNS (Detailed Schema):\n"
        block += "\n".join(summary['columns'])
        if summary['total_cols'] > 100: 
            block += f"\n- ... (+{summary['total_cols']-100} more columns omitted)"
        block += "\n"
        
        # Small sample for context - ONLY for datasets that aren't massive
        if df.shape[0] < 1_000_000 and df.shape[1] < 100:
            try:
                block += f"SAMPLE (first 3 rows):\n{df.head(3).to_string(index=False)}\n"
            except Exception: pass
        elif df.shape[1] >= 100:
            try:
                # For very wide datasets, only show first few columns in sample
                block += f"SAMPLE (first 3 rows, first 10 columns):\n{df.iloc[:3, :10].to_string(index=False)}\n"
            except Exception: pass
        else:
            block += "SAMPLE: [Omitted for large dataset to save context space]\n"
        
        block += "\n"
        ctx += block
        total_chars += len(block)
        if total_chars > MAX_CHARS: break

    return ctx

def validate_code(code: str) -> tuple[bool, str]:
    """Validate Python code syntax and basic checks."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    forbidden_patterns = [
        "import os", "import subprocess", "import socket", "import requests",
        "import urllib", "import shutil", "import sys",
        "os.system", "os.popen", "os.spawn", "os.exec",
        "subprocess.", "eval(", "exec(", "pickle.", "marshal.",
        "socket.", "requests.", "urllib.", "shutil.",
        "__builtins__", "__import__",
        "open(", ".write(", ".save(",  # Block file writes
    ]
    for pattern in forbidden_patterns:
        if pattern in code:
            return False, f"Security: Forbidden pattern detected: {pattern}"

    return True, "Code is syntactically valid"

def validate_ai_response(query: str, response: str, ds_context: str) -> tuple[bool, str]:
    """Check AI response for obvious hallucinations (e.g. non-existent columns)."""
    # Extract all column names from context
    col_names = re.findall(r"- '([^']+)'", ds_context)
    if not col_names: return True, "" # No data context to check against
    
    # Check if AI uses columns NOT in the context
    # Look for patterns like df['ColName'] or df["ColName"]
    used_cols = re.findall(r"df\[['\"]([^'\"]+)['\"]\]", response)
    
    hallucinations = []
    for col in used_cols:
        if col not in col_names:
            hallucinations.append(col)
            
    if hallucinations:
        return False, f"Warning: AI referenced non-existent columns: {', '.join(hallucinations)}"
    
    return True, ""

def lint_code(code: str) -> str:
    """Lint code using flake8 and return issues."""
    try:
        import subprocess
        result = subprocess.run(['flake8', '--stdin-display-name', 'code', '-'], 
                              input=code, text=True, capture_output=True, timeout=10)
        if result.returncode == 0:
            return "No linting issues found."
        else:
            return result.stdout or result.stderr
    except Exception as e:
        return f"Linting failed: {e}"

def calculate_confidence(query: str, reply: str, ds_context: str) -> float:
    """Calculate confidence score based on context relevance."""
    query_lower = query.lower()
    reply_lower = reply.lower()
    
    # Extract column names from context
    import re
    col_matches = re.findall(r"- '([^']+)'", ds_context)
    columns = [c.lower() for c in col_matches]
    
    # Count how many columns are mentioned in query and reply
    query_col_count = sum(1 for col in columns if col in query_lower)
    reply_col_count = sum(1 for col in columns if col in reply_lower)
    
    # Base confidence
    confidence = 0.5
    
    # Boost if columns are used appropriately
    if query_col_count > 0:
        confidence += 0.2 * (reply_col_count / query_col_count)
    
    # Boost if code is present
    if "```python" in reply:
        confidence += 0.2
    
    # Cap at 1.0
    return min(confidence, 1.0)

def preview_code_execution(code: str) -> str:
    """Preview what the code will do without executing."""
    try:
        ast_tree = ast.parse(code)
        preview_lines = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                targets = [getattr(t, 'id', str(t)) for t in node.targets if hasattr(t, 'id')]
                preview_lines.append(f"Assign to: {', '.join(targets)}")
            elif isinstance(node, ast.Call):
                func_name = getattr(node.func, 'id', str(node.func))
                preview_lines.append(f"Call function: {func_name}")
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                preview_lines.append(f"Import: {', '.join(modules)}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                preview_lines.append(f"From {module} import: {', '.join(names)}")
            
            # Check dataset ID access
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "datasets":
                    if isinstance(node.slice, ast.Constant):
                        ds_id = node.slice.value
                        datasets = st.session_state.get("datasets", {})
                        if ds_id not in datasets:
                            preview_lines.append(f"⚠️ Access UNKNOWN dataset ID: {ds_id}")
                        else:
                            preview_lines.append(f"Access dataset: {datasets[ds_id]['name']}")

        if preview_lines:
            return "Code Preview:\n" + "\n".join(f"- {line}" for line in preview_lines[:12])
        else:
            return "Code appears to be valid but contains no executable statements."
    except SyntaxError as e:
        return f"Syntax error: {e}"
    except Exception as e:
        return f"Preview failed: {e}"

def run_automl(df: pd.DataFrame, target_column: str, problem_type: str = 'auto', time_limit: int = 300) -> dict:
    """Run AutoML on dataset using AutoGluon."""
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile, os, uuid
        
        # Create unique model directory
        model_id = str(uuid.uuid4())[:8]
        model_dir = DATA_DIR / "models" / f"automl_{model_id}"
        model_dir.mkdir(exist_ok=True)
        
        # Initialize predictor
        predictor = TabularPredictor(
            label=target_column,
            path=str(model_dir),
            problem_type=problem_type,
            eval_metric='accuracy' if problem_type == 'binary' or problem_type == 'multiclass' else 'root_mean_squared_error'
        )
        
        # Fit model
        predictor.fit(
            df,
            time_limit=time_limit,
            presets='best_quality'
        )
        
        # Get results
        leaderboard = predictor.leaderboard()
        best_model = predictor.get_model_best()
        
        # Make predictions on training data for demo
        predictions = predictor.predict(df.head(10))
        
        return {
            'success': True,
            'model_id': model_id,
            'leaderboard': leaderboard.to_dict(),
            'best_model': best_model,
            'predictions_sample': predictions.head().to_dict(),
            'feature_importance': predictor.feature_importance(df).to_dict() if hasattr(predictor, 'feature_importance') else None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def load_automl_model(model_id: str):
    """Load a saved AutoML model."""
    try:
        from autogluon.tabular import TabularPredictor
        model_dir = DATA_DIR / "models" / f"automl_{model_id}"
        if not model_dir.exists():
            return None
        predictor = TabularPredictor.load(str(model_dir))
        return predictor
    except Exception as e:
        get_shared_logger().log(f"Failed to load AutoML model {model_id}: {e}", "error")
        return None

def suggest_features(df: pd.DataFrame, ds_context: str, groq_key: str | None = None) -> str:
    """Ask AI to suggest new features for a dataset."""
    prompt = f"Given the following dataset context, suggest 5-8 new features (columns) that would be highly valuable for analysis or machine learning. For each, provide a brief rationale and the Pandas code to create it.\n\nCONTEXT:\n{ds_context}"
    return ask_groq(prompt, groq_key=groq_key)

def extract_data_from_pdf(file_bytes) -> str:
    """Extract text and tables from a PDF file."""
    if not HAS_PDF:
        return "ERROR: pdfplumber is not installed. Please run: pip install pdfplumber"
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                text += f"--- Page {i+1} ---\n"
                text += page.extract_text() or ""
                tables = page.extract_tables()
                if tables:
                    text += "\nTABLES DETECTED:\n"
                    for table in tables:
                        df_tmp = pd.DataFrame(table)
                        text += df_tmp.to_string(index=False) + "\n"
                text += "\n"
        return text
    except Exception as e:
        get_shared_logger().log(f"PDF extraction failed: {e}", "error")
        return f"Error extracting PDF: {e}"

def extract_data_from_image(file_bytes) -> str:
    """Extract text from an image using OCR."""
    if not HAS_TESSERACT:
        return "OCR Error: 'pytesseract' is not installed or configured. Please install Tesseract-OCR and pytesseract."
    
    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        get_shared_logger().log(f"OCR failed: {e}", "error")
        return f"Error performing OCR: {e}"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are UDIP AI — a brilliant, versatile data science companion built into the Universal Data Integration Platform.

YOUR IDENTITY:
- You are confident, precise, enthusiastic, and deeply knowledgeable
- You focus on data science, data engineering, analytics, statistics, and machine learning
- You can also discuss adjacent topics: software engineering, Python programming, career advice, tool comparisons, research methodology, industry trends
- You hold engaging, natural conversations while remaining technically excellent

YOUR EXPERTISE:
- Python data analysis: pandas, numpy, scipy, statsmodels, polars
- Data cleaning, merging, reshaping, pivoting, ETL pipelines
- Statistical analysis: hypothesis testing, ANOVA, chi-square, regression, Bayesian methods
- Machine learning: scikit-learn, XGBoost, random forests, neural networks, model evaluation
- Deep learning concepts: CNNs, RNNs, transformers, NLP fundamentals
- Data visualization: plotly, seaborn, matplotlib, dashboard design principles
- SQL queries on DataFrames (DuckDB integration)
- Exporting for Power BI, Tableau, Excel, and other BI tools
- Date/time processing, time series analysis, forecasting
- Data engineering: ETL, data pipelines, data warehousing concepts
- Big data concepts: Spark, Dask, distributed computing
- Research methodology: experiment design, A/B testing, sampling techniques

CONVERSATIONAL STYLE:
- Be warm, helpful, and encouraging — like pairing with a senior data scientist friend
- For simple questions: give direct, concise answers
- For complex questions: give thorough, well-structured explanations
- When you have web search results: synthesize them naturally into your response, citing sources
- If asked about something outside your core expertise, give your best answer and note your limitations
- Use analogies and real-world examples to explain complex concepts
- Never refuse a reasonable question — always find an angle to be helpful
- DO NOT use internal model tags like <|python_tag|> or <|thought|> in your response
- Use ### for headers instead of # to keep the UI clean
- CRITICAL: When dealing with wide datasets (thousands of columns), NEVER use df.hist() or df.plot() on the whole dataframe. It will crash the platform. Select a few columns first (e.g., df.iloc[:, :10].hist()).
- POLARS RULE: In Polars, use .describe() for statistical summaries. NEVER use .summary() as it does not exist.
- ACCURACY RULE: You are working with professional enterprise data. Inaccuracy is UNACCEPTABLE. Always double-check column names and data types from the provided context before writing code. If you are unsure, ask for clarification.

AGENTIC RESPONSE PROTOCOL:
- Think like a senior planner, builder, reviewer, tutor, and researcher in one agent.
- Internally follow: understand -> plan -> answer -> self-check. Do not reveal hidden reasoning.
- If the task is broad, split it into the smallest useful steps before answering.
- If the task is ambiguous and a wrong answer would be risky, ask one precise clarifying question.
- For simple questions, answer directly and concisely.
- For complex work, lead with the recommendation, then give the rationale, then the next step.
- When writing code, return runnable code with exact imports and exact dataset column names from context.
- When reviewing or debugging, call out the likely cause, the fix, and how to verify it.
- When the user asks for the best or most flawless solution, optimize for correctness, clarity, and safe execution over brevity.
- Respect any TASK DIRECTIVES in the prompt, especially mode, style, and focus notes.

CRITICAL CODE RULES — FOLLOW EVERY ONE:

1. NEVER call plt.show() — figures display automatically.
2. NEVER use os.system(), subprocess, or open files for writing.
3. ALWAYS include ALL imports at the top.

━━━ PRIMARY DATASET PRIORITY (MOST IMPORTANT) ━━━
If the "SELECTED DATASET:" section appears in context, THAT IS THE PRIMARY DATASET.
- Always prioritize the selected dataset when answering questions
- Reference it by exact Dataset ID from context
- If user says "my data" or "the dataset", they mean the selected dataset
- If selected dataset exists but is not explicitly mentioned in the query, still consider it as primary context
- Access via: df = datasets["EXACT_ID"]["df"]
- NEVER invent or assume column names; always check SELECTED DATASET columns first
- If dataset ID is missing, refuse to write code and ask user to select a dataset
- If you cannot find any relevant data loaded, explicitly warn the user: "I don't see any data loaded for that query."

━━━ DATASET ACCESS ━━━
Datasets live in a dict called `datasets`. Use the EXACT Dataset ID from context:
    df = datasets["EXACT_ID"]["df"]
If only one dataset is loaded, you may also use `df = list(datasets.values())[0]["df"]`.
NEVER write: df = pd.read_csv(...) or df = pd.read_excel(...) or df = DatasetName
NEVER guess the Dataset ID; it MUST come from the context.

- Use exact column names from the COLUMNS section.
- If a dataset has a `varname` hint, that variable is available in code too.

━━━ COLUMN NAMES (SECOND MOST IMPORTANT) ━━━
Column names are CASE-SENSITIVE and may contain spaces.
ALWAYS use the EXACT column names shown in the COLUMNS section of the dataset context.
NEVER guess or lowercase column names.
Examples:
    CORRECT: df["Order Date"]   — if context shows "Order Date"
    WRONG:   df["order_date"]   — unless context explicitly shows that spelling
    CORRECT: df["Ship Mode"]    — if context shows "Ship Mode"
    WRONG:   df["ship_mode"]

If you need to clean column names first, do it explicitly:
    df.columns = df.columns.str.strip()  # only if needed

━━━ BEFORE WRITING ANY CODE ━━━
1. Look at the COLUMNS list in the dataset context
2. Use those exact column names
3. Check the dtype of each column you use
4. For dates shown as object dtype, convert first: pd.to_datetime(df["Col Name"])
5. Print df.head(3) and df.dtypes at the start to verify

After transforms, always print shape and sample so the user can verify results.
For large datasets (>500k rows): df = df.sample(50000) before plotting.
For wide datasets (>50 columns): NEVER plot more than 10 columns at once. Use df.iloc[:, :10].plot() or select specific columns.
Ploting thousands of columns will crash the UI.

━━━ CODE TEMPLATES (Use these patterns) ━━━
Common operations:
- Load data: df = datasets["ID"]["df"]
- Filter: df_filtered = df[df["column"] > value]
- Group by: df.groupby("column").agg({"col": "mean"})
- Plot: df.plot(kind="bar", x="x_col", y="y_col"); plt.show()
- SQL: result = duckdb.sql("SELECT * FROM df WHERE col > 10").df()
- Create database: con = duckdb.connect("database.db"); con.execute("CREATE TABLE table_name AS SELECT * FROM df"); con.close()
- AutoML: result = run_automl(df, "target_column", problem_type="binary", time_limit=300)

━━━ POLARS (QUANTUM ENGINE) ━━━
- Convert: df_pl = pl.from_pandas(df)
- Summary: df_pl.describe()  # CRITICAL: DO NOT use .summary()
- Filter: df_pl.filter(pl.col("col") > val)
- Select: df_pl.select(["col1", "col2"])
- GroupBy: df_pl.group_by("col").agg(pl.col("val").mean())

Always check df.shape and df.head() after operations.
For large data, use df.sample(1000) for plotting.

━━━ DATABASE CREATION ━━━
To create a persistent database from a dataset:
1. Choose a dataset ID from context
2. Use duckdb.connect("path/to/database.db") to create file
3. CREATE TABLE table_name AS SELECT * FROM df
4. Close connection
5. Database saved to disk, can be loaded later

━━━ AUTOMATED MACHINE LEARNING ━━━
For AutoML on datasets:
1. Identify target column from COLUMNS list
2. Determine problem type: 'binary', 'multiclass', 'regression', or 'auto'
3. Call run_automl(df, "target_col", problem_type="auto", time_limit=300)
3. Call run_automl(df, "target_column", problem_type="auto", time_limit=300)
4. Returns model leaderboard, predictions, and feature importance
5. Use for quick model prototyping and evaluation
"""

# ── Code executor ─────────────────────────────────────────────────────────────
def run_code(code: str, live_placeholder=None, datasets_ctx: dict = None) -> tuple[str, str, bool, list[bytes]]:
    """
    Executes Python code in a safe sandbox, captures stdout and matplotlib figures as bytes.
    Returns (stdout, error_message, success, captured_images_as_bytes).
    """
    MAX_LINES = 500
    
    # Use provided context or pull from session state safely
    active_datasets = datasets_ctx if datasets_ctx is not None else st.session_state.get("datasets", {})

    class StreamBuf(io.StringIO):
        def __init__(self, ph): super().__init__(); self._ph=ph; self._lines=[]
        def write(self, s):
            super().write(s)
            if s.strip():
                self._lines.append(s.rstrip())
                if len(self._lines) > MAX_LINES:
                    self._lines = self._lines[-MAX_LINES:]
                    self._lines[0] = f"[truncated — showing last {MAX_LINES} lines]"
                if self._ph:
                    self._ph.code("\n".join(self._lines), language="text")
            return len(s)

    # Sandbox builtins
    def safe_open(file, mode="r", **kw):
        if any(c in str(mode) for c in ("w","a","x","+")):
            raise PermissionError("File writing is blocked for safety.")
        return open(file, mode, **kw)

    # Use full builtins — only override 'open' to block file writing
    import builtins as _builtins
    SAFE_BUILTINS = vars(_builtins).copy()
    SAFE_BUILTINS["open"] = safe_open

    install_log = []

    for attempt in range(6):
        # Pre-compile
        try:
            compiled = compile(code, "<udip_ai>", "exec")
        except SyntaxError as e:
            return "", f"SyntaxError on line {e.lineno}: {e.msg}\n{e.text or ''}", False

        buf = StreamBuf(live_placeholder)
        old_stdout = sys.stdout
        sys.stdout  = buf
        error = None

        try:
            import matplotlib
            matplotlib.use("Agg", force=True) # Force backend to ensure thread-safety
            import matplotlib.pyplot as _plt
            _plt.close("all") # Clear any stale figures from previous runs
            captured_images = []
            orig_show = _plt.show
            def patched_show(*a, **kw):
                fig = _plt.gcf()
                if len(captured_images) < 5:
                    if fig.get_axes():
                        buf_img = io.BytesIO()
                        fig.savefig(buf_img, format="png", bbox_inches="tight")
                        captured_images.append(buf_img.getvalue())
                _plt.close(fig)
            _plt.show = patched_show

            import psutil, os
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            t_start = time.time()

            import duckdb
            import polars as pl
            # Inject datasets dict AND individual varnames so either access method works
            exec_globals = {
                "pd": pd, "np": np, "plt": _plt, "duckdb": duckdb, "pl": pl,
                "polars": pl, # Alias so pl. works even if AI forgot import
                "datasets": active_datasets,
                "create_database_from_df": create_database_from_df,
                "run_automl": run_automl,
                "load_automl_model": load_automl_model,
                "__builtins__": SAFE_BUILTINS,
            }
            # Expose each dataset under its varname AND clean filename so AI code works regardless
            import re as _re
            for _ds_id, _ds in active_datasets.items():
                _vname = _ds.get("varname", "")
                if _vname:
                    exec_globals[_vname] = _ds["df"]
                _clean = _re.sub(r"[^a-zA-Z0-9_]", "_", _ds["name"].rsplit(".", 1)[0])
                if _clean and _clean[0].isalpha():
                    exec_globals[_clean] = _ds["df"]
            # Always expose first dataset as 'df' if only one loaded
            if len(active_datasets) == 1:
                exec_globals["df"] = list(active_datasets.values())[0]["df"]
            exec(compiled, exec_globals)
            
            # Final sweep to capture any figures not shown via plt.show()
            # We do this twice to catch anything that might have been deferred
            for _ in range(2):
                for num in _plt.get_fignums():
                    fig = _plt.figure(num)
                    if not fig.get_axes(): continue
                    buf_img = io.BytesIO()
                    fig.savefig(buf_img, format="png", bbox_inches="tight")
                    img_bytes = buf_img.getvalue()
                    if img_bytes not in captured_images:
                        captured_images.append(img_bytes)
            _plt.close("all")
            
            _plt.show = orig_show

            # Performance profiling
            t_end = time.time()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            exec_time = t_end - t_start
            mem_used = mem_after - mem_before
            get_shared_logger().log(f"Code execution: {exec_time:.2f}s, Memory: {mem_used:+.1f}MB", "info")

            if "datasets" in exec_globals:
                st.session_state.datasets = exec_globals["datasets"]

        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        stdout = buf.getvalue()

        if error is None:
            prefix = "\n".join(f"📦 Auto-installed: {p}" for p in install_log)
            return (prefix + "\n\n" if prefix else "") + stdout, "", True, captured_images

        match = re.search(r"ModuleNotFoundError: No module named [\x27\x22]([\w\.]+)[\x27\x22]", error)
        if not match or attempt == 5:
            suffix = f"\nAuto-installed: {', '.join(install_log)}" if install_log else ""
            return stdout, error + suffix, False, []

        module  = match.group(1).split(".")[0]
        pip_pkg = SAFE_PIP.get(module)
        if pip_pkg is None:
            return stdout, f"Package '{module}' not in safe allowlist.\nAllowed: {', '.join(sorted(SAFE_PIP.values()))}\n\n{error}", False, []

        ph = st.empty()
        ph.info(f"📦 Auto-installing {pip_pkg}...")
        try:
            subprocess.run([sys.executable,"-m","pip","install",pip_pkg,"--quiet"], check=True, capture_output=True)
            install_log.append(pip_pkg)
            get_shared_logger().log(f"Auto-installed: {pip_pkg}", "info")
            ph.success(f"✅ Installed {pip_pkg} — retrying...")
            time.sleep(0.3); ph.empty()
        except subprocess.CalledProcessError as e:
            ph.error(f"❌ Could not install {pip_pkg}")
            get_shared_logger().log(f"pip install failed: {pip_pkg}: {e.stderr}", "error")
            return stdout, error, False, []

    return stdout, error or "", False, []

def run_code_with_healing(code: str, groq_key: str, live_placeholder=None, max_retries=2, datasets_ctx: dict = None) -> tuple[str, str, bool, str, list]:
    """
    Executes code and automatically asks the LLM to fix it if it fails.
    Returns (stdout, error, success, final_code, figures).
    """
    # Validate code first
    valid, msg = validate_code(code)
    if not valid:
        return "", msg, False, code
    
    current_code = code
    stdout, error, success, figures = run_code(current_code, live_placeholder, datasets_ctx=datasets_ctx)
    
    retries = 0
    while not success and retries < max_retries and error and groq_key:
        if live_placeholder:
            live_placeholder.warning(f"⚠️ Code failed. Self-healing (Attempt {retries+1}/{max_retries})...")
            
        # Do not try to heal missing modules that pip install failed on
        if "Package '" in error and "not in safe allowlist" in error:
            break
            
        fix_prompt = f"The following code failed during execution:\n\nCODE:\n```python\n{current_code}\n```\n\nERROR:\n```text\n{error}\n```\n\nDiagnose the issue and rewrite the code to fix it. Ensure you use EXACT column names as they appear in the dataset context, and check for missing values or incorrect data types if applicable. Return ONLY the fixed python code in a ```python ... ``` block."
        
        try:
            from modules.ai_engine import ask_groq
            reply, new_code, _ = ask_groq(
                fix_prompt,
                groq_key,
                selected_dataset_id=st.session_state.get("ai_selected_dataset"),
            )
            if new_code and new_code.strip() != current_code.strip():
                current_code = new_code
                if live_placeholder:
                    live_placeholder.info("🔧 Running fixed code...")
                stdout, error, success, figures = run_code(current_code, live_placeholder, datasets_ctx=datasets_ctx)
            else:
                break # LLM couldn't provide a different code
        except Exception as e:
            error += f"\n[Auto-fix failed: {str(e)}]"
            break
            
        retries += 1
        
    return stdout, error or "", success, current_code, figures

# ── Milestone 3: Autonomous Scientist (Agentic Loop) ─────────────────────────

def run_sql_on_data(sql: str) -> str:
    """Execute SQL on loaded datasets using DuckDB."""
    import duckdb
    try:
        # Register all datasets in duckdb
        con = duckdb.connect(database=':memory:')
        for ds_id, ds in st.session_state.datasets.items():
            varname = ds.get("varname", sanitize_varname(ds["name"]))
            con.register(varname, ds["df"])
        
        rel = con.execute(sql)
        df_res = rel.df()
        return f"SQL Result ({df_res.shape[0]} rows):\n{df_res.head(10).to_string()}\nFull result cached."
    except Exception as e:
        return f"SQL Error: {str(e)}"

def autonomous_solve(goal: str, groq_key: str, status_ph=None) -> str:
    """
    Experimental Autonomous Agent loop. 
    Cycle: Thought -> Action -> Observation (up to 5 turns).
    """
    from groq import Groq
    client = Groq(api_key=groq_key)
    
    agent_prompt = f"""You are the UDIP Autonomous Scientist. 
    Goal: {goal}
    
    You have these tools:
    1. SQL_QUERY: Run SQL on loaded datasets. Example: Action: SQL_QUERY, Action Input: SELECT * FROM df_name LIMIT 5
    2. USE_PYTHON: Run Python code. Access data via datasets['ID']['df']. Example: Action: USE_PYTHON, Action Input: print(list(datasets.values())[0]['df'].describe())
    3. WEB_SEARCH: Search internet. Example: Action: WEB_SEARCH, Action Input: current trends in data science 2026
    4. FINAL_ANSWER: When you are done. Example: Action: FINAL_ANSWER, Action Input: The analysis shows...
    
    Rules:
    - Operate in cycles: Thought -> Action -> Action Input.
    - Wait for the Observation (which will be provided to you) before next Thought.
    - Always state your Thought before an Action.
    - Max 5 cycles.
    """
    
    history = [{"role": "system", "content": agent_prompt}]
    
    for i in range(5):
        if status_ph: status_ph.info(f"🧠 Agent thinking (Turn {i+1}/5)...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history + [{"role": "user", "content": "Continue your analysis. If you have the final answer, use FINAL_ANSWER."}],
            temperature=0.2
        )
        
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        
        if status_ph: st.markdown(f"**Turn {i+1} Thought**: {reply.split('Action:')[0].replace('Thought:','').strip()}")
        
        # Parse Action
        if "Action: FINAL_ANSWER" in reply:
            return reply.split("Action Input:")[1].strip() if "Action Input:" in reply else reply
        
        # Extract Action and Input
        try:
            action_match = re.search(r"Action:\s*(\w+)", reply)
            input_match = re.search(r"Action Input:\s*([\s\S]*)", reply)
            
            if not action_match or not input_match:
                obs = "Error: Could not parse Action/Input. Please use the format: Action: TOOL_NAME, Action Input: ... "
            else:
                action = action_match.group(1).upper()
                act_input = input_match.group(1).strip()
                
                if status_ph: status_ph.warning(f"🛠️ Executing: {action}")
                
                if action == "SQL_QUERY":
                    obs = run_sql_on_data(act_input)
                elif action == "USE_PYTHON":
                    res_out, res_err, _ = run_code(act_input)
                    obs = f"Python Output: {res_out}\nErrors: {res_err}"
                elif action == "WEB_SEARCH":
                    obs = search_web(act_input)
                else:
                    obs = f"Unknown action: {action}"
            
            history.append({"role": "user", "content": f"Observation: {obs}"})
        except Exception as e:
            history.append({"role": "user", "content": f"System Error during execution: {e}"})

    return "Agent reached maximum turns (5) without a final answer. Please review the trace above."

# ── Groq chat ─────────────────────────────────────────────────────────────────
# Greetings answered locally — no API call needed, no rate limit risk
_LOCAL_GREETINGS = {
    "hi": "Hi! I'm UDIP AI. Upload a dataset and ask me anything about it — I can analyse, clean, merge, visualise, and run Python code on your data.",
    "hello": "Hello! Ready to help with your data. What would you like to do?",
    "hey": "Hey! What can I help you analyse today?",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "Happy to help! Anything else you'd like to explore?",
    "ok": "Got it. What would you like to do next?",
    "hola": "¡Hola! Ready to help with your data analysis.",
    "hi there": "Hi there! What data would you like to work with today?",
    "hello there": "Hello! I'm here to help with your data. What do you need?",
}

def _resolve_groq_key(groq_key: str | None = None) -> str | None:
    if groq_key:
        return groq_key
    session_key = st.session_state.get("groq_key")
    if session_key:
        return session_key
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        return env_key
    try:
        cfg = load_config()
        if cfg.get("groq_key"):
            return cfg["groq_key"]
    except Exception:
        pass
    return None

def ask_groq(
    user_message: str,
    groq_key: str | None = None,
    context: str = "",
    selected_dataset_id: str | None = None,
    allow_web_search: bool = True,
    include_dataset_context: bool = True,
) -> tuple[str, str | None, float]:
    """
    Send message to Groq. Returns (reply_text, extracted_code_or_None, confidence_score).
    - Greetings answered locally (no API call, no rate limit)
    - Rate limit errors handled with automatic retry after wait
    """
    import time as _time

    # Handle greetings locally — never hits the API
    msg_clean = user_message.lower().strip().strip("?!." )
    if msg_clean in _LOCAL_GREETINGS:
        return _LOCAL_GREETINGS[msg_clean], None, 1.0

    groq_key = _resolve_groq_key(groq_key)
    if not groq_key:
        return (
            "Groq API key is not configured. Open settings > api keys, save your Groq key, and try again.",
            None,
            0.0,
        )

    from groq import Groq
    client = Groq(api_key=groq_key)

    if include_dataset_context:
        if len(user_message) < 10:
            ds_context = build_dataset_context(selected_dataset_id=selected_dataset_id)
            mem_context = build_memory_context()
            web_context = ""
        else:
            ds_context = build_dataset_context(query=user_message, selected_dataset_id=selected_dataset_id)
            mem_context = build_memory_context()
            web_context = ""
            if allow_web_search and should_search_web(user_message):
                query = extract_search_query(user_message)
                if query:
                    web_context = search_web(query)
    else:
        ds_context = ""
        mem_context = ""
        web_context = ""

    full_system = ""
    if st.session_state.get("quantum_mode", False):
        full_system += "⚠️ VANGUARD ENGINE ACTIVE (QUANTUM MODE): The user is working with MASSIVE datasets. You MUST use Polars (import polars as pl) for all data processing tasks. DO NOT USE PANDAS. Polars is essential for performance here. Access datasets as: df_pl = pl.from_pandas(datasets['ID']['df']).\n\n"
    
    full_system += SYSTEM_PROMPT
    
    if context:
        full_system += f"\n\nADDITIONAL CONTEXT:\n{context}"
    full_system += "\n\nDATASET CONTEXT:\n" + ds_context
    if web_context:
        full_system += web_context

    recent = st.session_state.get("chat_history", [])[-60:]
    messages = [{"role": m["role"], "content": m["content"]}
                for m in recent if m["role"] in ("user","assistant")]

    # Try with retries on rate limit
    for attempt in range(3):
        try:
            # Ensure the current user message is included
            current_messages = [{"role": "system", "content": full_system}] + messages + [{"role": "user", "content": user_message}]
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=current_messages,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise e
            continue

def ask_ai(
    user_message: str,
    groq_key: str,
    context: str = "",
    selected_dataset_id: str | None = None,
    allow_web_search: bool = True,
    include_dataset_context: bool = True,
) -> tuple[str, str | None, float]:
    """Backward-compatible alias for ask_groq."""
    return ask_groq(
        user_message=user_message,
        groq_key=groq_key,
        context=context,
        selected_dataset_id=selected_dataset_id,
        allow_web_search=allow_web_search,
        include_dataset_context=include_dataset_context,
    )

# ── Omni-Vision Extraction (Milestone 2) ──────────────────────────────────────
def extract_data_from_multimodal(file_bytes: bytes, file_type: str, groq_key: str) -> str:
    """
    Use Llama-3.2 Vision to extract structured tabular data from images/PDFs.
    Returns a JSON string of the extracted data for conversion to DataFrame.
    """
    import base64
    from modules.multi_model_ai import get_multi_model_ai
    ai = get_multi_model_ai()
    
    b64_data = base64.b64encode(file_bytes).decode('utf-8')
    mime_type = "image/jpeg" if "jpg" in file_type or "jpeg" in file_type else ("image/png" if "png" in file_type else "application/pdf")
    
    # PDF handling: since Groq vision is primarily for images, 
    # we tell the model it's a document scan if it's a PDF.
    
    prompt = """Analyze this document/image and extract ALL tabular data into a clean JSON format.
    Return ONLY a JSON array of objects where each object is a row. 
    Example: [{"Date": "2023-01-01", "Item": "Apple", "Price": 1.50}, ...]
    If there are multiple tables, merge them into one reasonable structure if possible, or return the main one.
    Do not include explanations, only the raw JSON array.
    """
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_data}"
                        }
                    }
                ]
            }
        ]
        
        response_text, cost = ai.call_ai(
            messages=messages,
            model="llama-3.2-11b-vision-preview",
            temperature=0.1,
            max_tokens=4096
        )
        
        # Create mock response object for compatibility
        class MockResponse:
            def __init__(self, content):
                self.choices = [{"message": {"content": content}}]
        
        response = MockResponse(response_text)
        
        content = response.choices[0].message.content
        # Clean up Markdown JSON blocks if present
        clean_json = re.sub(r"```json\s*|\s*```", "", content).strip()
        return clean_json
    except Exception as e:
        get_shared_logger().log(f"Vision extraction failed: {e}", "error")
        return f"Error: {str(e)}"
