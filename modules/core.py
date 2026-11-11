"""
UDIP v3 — modules/core.py
Persistence, encryption, helpers, session state.
Shared by all pages and modules.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import socket
import sys
import time
import traceback
import uuid
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Version ────────────────────────────────────────────────────────────────────
APP_VERSION = "3.0.0"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("udip")

# ── Polars optional ────────────────────────────────────────────────────────────
try:
    import polars as _polars
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# ── Memory limits ──────────────────────────────────────────────────────────────
MEMORY_WARN_MB  = 800
MEMORY_LIMIT_MB = 2_000

# ── File export helpers ────────────────────────────────────────────────────────
EXT  = {"CSV": ".csv", "JSON": ".json", "Parquet": ".parquet", "Excel": ".xlsx", "TSV": ".tsv"}
MIME = {
    "CSV":     "text/csv",
    "JSON":    "application/json",
    "Parquet": "application/octet-stream",
    "Excel":   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "TSV":     "text/tab-separated-values",
}

# ── Paths ──────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent.parent.resolve()

def _resolve_data_dir() -> Path:
    """Return the active data directory, respecting user config."""
    cfg_path = _APP_DIR / "udip_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("data_dir"):
                p = Path(cfg["data_dir"])
                if p.is_absolute():
                    return p
        except Exception:
            pass
    return _APP_DIR / "udip_data"

BASE_DATA_DIR = _resolve_data_dir()
DATA_DIR      = BASE_DATA_DIR
DS_DIR        = DATA_DIR / "datasets"
MEMORY_FILE   = DATA_DIR / "ai_memory.json"

# Ensure directories exist
for _d in [DATA_DIR, DS_DIR, DATA_DIR / "history", DATA_DIR / "chats",
           DATA_DIR / "code", DATA_DIR / "snapshots", DATA_DIR / "raw_hdf5"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Encryption ─────────────────────────────────────────────────────────────────
def _get_machine_key() -> str:
    """Return a deterministic 32-char hex key derived from this machine."""
    seed = f"{socket.gethostname()}-{os.environ.get('USERNAME', os.environ.get('USER', 'udip'))}"
    return hashlib.sha256(seed.encode()).hexdigest()[:32]

def encrypt(text: str) -> str:
    """Encrypt text with a machine-local key (Fernet when available, XOR fallback)."""
    if not text:
        return ""
    try:
        from cryptography.fernet import Fernet
        key = base64.urlsafe_b64encode(_get_machine_key().encode()[:32].ljust(32, b"="))
        return Fernet(key).encrypt(text.encode()).decode()
    except Exception:
        # XOR fallback
        k = _get_machine_key()
        xored = bytes(c ^ ord(k[i % len(k)]) for i, c in enumerate(text.encode()))
        return base64.b64encode(xored).decode()

def decrypt(ciphertext: str) -> str:
    """Decrypt text encrypted by `encrypt`. Returns '' on failure."""
    if not ciphertext:
        return ""
    try:
        from cryptography.fernet import Fernet
        key = base64.urlsafe_b64encode(_get_machine_key().encode()[:32].ljust(32, b"="))
        return Fernet(key).decrypt(ciphertext.encode()).decode()
    except Exception:
        pass
    try:
        k = _get_machine_key()
        raw = base64.b64decode(ciphertext.encode())
        return bytes(c ^ ord(k[i % len(k)]) for i, c in enumerate(raw)).decode()
    except Exception:
        return ""

# ── Safe file write ────────────────────────────────────────────────────────────
def safe_write(path: Path, content: str) -> bool:
    """Atomically write text to path, creating parent dirs."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception as exc:
        log.warning(f"safe_write failed for {path}: {exc}")
        return False

# ── Config ─────────────────────────────────────────────────────────────────────
_CFG_PATH = _APP_DIR / "udip_config.json"

def load_config() -> dict:
    """Load UDIP config from disk. Returns {} on failure."""
    try:
        if _CFG_PATH.exists():
            return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_config(cfg: dict) -> None:
    """Save UDIP config to disk."""
    try:
        _CFG_PATH.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning(f"save_config failed: {exc}")

def update_paths(data_dir: str) -> None:
    """Update the DATA_DIR at runtime and persist to config."""
    global DATA_DIR, DS_DIR, MEMORY_FILE, BASE_DATA_DIR
    p = Path(data_dir)
    BASE_DATA_DIR = p
    DATA_DIR      = p
    DS_DIR        = p / "datasets"
    MEMORY_FILE   = p / "ai_memory.json"
    for _d in [DATA_DIR, DS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["data_dir"] = str(p)
    save_config(cfg)

# ── Memory helpers ─────────────────────────────────────────────────────────────
def get_memory_mb() -> float:
    """Return process resident memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0

def fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

# Alias used by some pages
fmt = fmt_bytes

# ── Variable sanitizer ─────────────────────────────────────────────────────────
def sanitize_varname(name: str) -> str:
    """Return a Python-safe variable name derived from name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if name and name[0].isdigit():
        name = "_" + name
    return name or "_unnamed"

# ── DataFrame I/O ──────────────────────────────────────────────────────────────
def df_to_bytes(df: pd.DataFrame, fmt_key: str = "CSV") -> bytes:
    """Serialize a DataFrame to bytes in the requested format."""
    buf = io.BytesIO()
    k = fmt_key.upper()
    if k == "CSV":
        return df.to_csv(index=False).encode()
    elif k == "TSV":
        return df.to_csv(index=False, sep="\t").encode()
    elif k == "JSON":
        return df.to_json(orient="records", indent=2).encode()
    elif k == "PARQUET":
        df.to_parquet(buf, index=False)
        return buf.getvalue()
    elif k == "EXCEL":
        df.to_excel(buf, index=False)
        return buf.getvalue()
    return df.to_csv(index=False).encode()

def df_hash(df: pd.DataFrame) -> str:
    """Return a short hash fingerprint of a DataFrame."""
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]
    except Exception:
        return uuid.uuid4().hex[:12]

# ── Database helper ────────────────────────────────────────────────────────────
def create_database_from_df(df: pd.DataFrame, name: str) -> str:
    """Create a DuckDB database from a DataFrame. Returns db path."""
    try:
        import duckdb
        db_path = str(DATA_DIR / f"{sanitize_varname(name)}.duckdb")
        con = duckdb.connect(db_path)
        tbl = sanitize_varname(name)
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
        con.execute(f"CREATE TABLE {tbl} AS SELECT * FROM df")
        con.close()
        return db_path
    except Exception as exc:
        log.warning(f"create_database_from_df failed: {exc}")
        return ""

# ── Complex column preservation ────────────────────────────────────────────────
_COMPLEX_PREFIX = "__complex__"

def _prepare_dataframe_for_storage(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Convert complex columns to real+imag pairs for Parquet storage."""
    out = df.copy()
    for col in df.columns:
        if pd.api.types.is_complex_dtype(df[col]):
            out[f"{_COMPLEX_PREFIX}{col}__real"] = df[col].apply(
                lambda v: float(v.real) if pd.notna(v) else np.nan
            )
            out[f"{_COMPLEX_PREFIX}{col}__imag"] = df[col].apply(
                lambda v: float(v.imag) if pd.notna(v) else np.nan
            )
            out = out.drop(columns=[col])
        elif df[col].dtype == object:
            # Convert object cols with arrays to strings for storage
            try:
                sample = df[col].dropna().head(1)
                if len(sample) and isinstance(sample.iloc[0], (list, np.ndarray)):
                    out[col] = df[col].apply(lambda v: str(v) if v is not None else None)
            except Exception:
                pass
    return out

def _restore_complex_columns(df: pd.DataFrame, meta: dict = None) -> pd.DataFrame:
    """Restore complex columns from their real+imag storage form."""
    real_cols = [c for c in df.columns if c.startswith(_COMPLEX_PREFIX) and c.endswith("__real")]
    out = df.copy()
    for rc in real_cols:
        base = rc[len(_COMPLEX_PREFIX):-6]  # strip prefix and __real
        ic = f"{_COMPLEX_PREFIX}{base}__imag"
        if ic in out.columns:
            out[base] = out[rc].values + 1j * out[ic].values
            out = out.drop(columns=[rc, ic])
    return out

# ── Dataset persistence ────────────────────────────────────────────────────────
def normalize_dataset_path(path: str) -> str:
    """Ensure dataset path starts with /."""
    return path if path.startswith("/") else "/" + path.lstrip("/")

def save_dataset(ds_id: str, ds: dict) -> Tuple[bool, str]:
    """Persist a dataset entry to disk as Parquet + metadata JSON."""
    try:
        name    = ds.get("name", ds_id)
        source  = ds.get("source", "unknown")
        reason  = ds.get("reason", "unknown")
        df      = ds.get("df", pd.DataFrame())
        meta    = {
            "id":      ds_id,
            "name":    name,
            "source":  source,
            "reason":  reason,
            "timestamp": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows":    len(df),
            "cols":    len(df.columns),
            "columns": list(df.columns),
        }
        snap_dir = DS_DIR / ds_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        df_save = _prepare_dataframe_for_storage(df, meta)
        df_save.to_parquet(snap_dir / "data.parquet", index=False)
        (snap_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return True, ""
    except Exception as exc:
        log.warning(f"save_dataset failed for {ds_id}: {exc}")
        return False, str(exc)

def delete_dataset_from_disk(ds_id: str) -> None:
    """Remove a persisted dataset from disk."""
    import shutil
    try:
        snap_dir = DS_DIR / ds_id
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
    except Exception as exc:
        log.warning(f"delete_dataset_from_disk failed for {ds_id}: {exc}")

# ── History ────────────────────────────────────────────────────────────────────
_HIST_FILE = DATA_DIR / "history" / "history.json"

def add_history(entry: dict | str, action: str | None = None) -> None:
    """Append an entry to the in-memory and on-disk history.

    Backward compatible with legacy calls like add_history("Category", "message").
    """
    if isinstance(entry, str):
        if action is None:
            raise TypeError("add_history requires a message when called with a string action.")
        entry = {"action": entry, "message": action}
    elif not isinstance(entry, dict):
        raise TypeError("add_history expects a dict or action/message strings.")
    else:
        if action is not None and isinstance(action, str):
            entry.setdefault("action", action)

    if "history" not in st.session_state:
        st.session_state["history"] = []
    entry.setdefault("timestamp", _dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.session_state["history"].append(entry)
    try:
        hist = load_history()
        hist.append(entry)
        _HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HIST_FILE.write_text(json.dumps(hist[-500:], indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

def load_history() -> list:
    """Load history from disk."""
    try:
        if _HIST_FILE.exists():
            return json.loads(_HIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

# ── Code / Chat persistence ────────────────────────────────────────────────────
def save_code(name: str, code: str) -> None:
    code_dir = DATA_DIR / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    safe_write(code_dir / f"{sanitize_varname(name)}.py", code)

def load_code(name: str) -> str:
    p = DATA_DIR / "code" / f"{sanitize_varname(name)}.py"
    try:
        return p.read_text(encoding="utf-8") if p.exists() else ""
    except Exception:
        return ""

def save_chat(messages: list) -> None:
    chat_dir = DATA_DIR / "chats"
    chat_dir.mkdir(parents=True, exist_ok=True)
    fname = chat_dir / f"chat_{_dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        fname.write_text(json.dumps(messages, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning(f"save_chat failed: {exc}")

# ── Dashboard persistence ──────────────────────────────────────────────────────
def save_dashboard(pins: list) -> None:
    p = DATA_DIR / "dashboard_pins.json"
    try:
        p.write_text(json.dumps(pins, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning(f"save_dashboard failed: {exc}")

def pin_to_report(item: dict) -> None:
    """Pin a chart/result to the dashboard report."""
    pins = st.session_state.get("dashboard_pins", [])
    item.setdefault("ts", _dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    pins.append(item)
    st.session_state["dashboard_pins"] = pins
    save_dashboard(pins)

# ── Schedule persistence ───────────────────────────────────────────────────────
def load_schedules() -> list:
    """Load scheduled tasks from disk (empty list if none saved yet)."""
    p = DATA_DIR / "schedules.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        return data if isinstance(data, list) else []
    except Exception as exc:
        log.warning(f"load_schedules failed: {exc}")
        return []

def save_schedules(schedules: list) -> None:
    """Persist scheduled tasks to disk."""
    p = DATA_DIR / "schedules.json"
    try:
        p.write_text(json.dumps(schedules, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning(f"save_schedules failed: {exc}")

# ── Snapshots ──────────────────────────────────────────────────────────────────
def save_snapshot() -> str:
    """Save a snapshot of all current datasets. Returns snapshot name."""
    import shutil
    name = f"snapshot_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    snap_root = DATA_DIR / "snapshots" / name
    snap_root.mkdir(parents=True, exist_ok=True)
    datasets = st.session_state.get("datasets", {})
    for ds_id, ds in datasets.items():
        save_dataset(ds_id, ds)
        src = DS_DIR / ds_id
        if src.exists():
            shutil.copytree(src, snap_root / ds_id, dirs_exist_ok=True)
    meta = {"name": name, "timestamp": _dt.now().isoformat(), "dataset_ids": list(datasets)}
    (snap_root / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return name

def list_snapshots(ds_id: str | None = None) -> list:
    """Return list of snapshot metadata dicts.

    When *ds_id* is None (the default), every snapshot in the store is
    returned — preserving the original behaviour used by
    ``command_center.py`` and other whole-system consumers.

    When *ds_id* is provided, the flat list is filtered so that only
    snapshots whose manifest ``dataset_ids`` list contains *ds_id* are
    returned.  Snapshots with an unparseable manifest (which lack a
    ``dataset_ids`` field) are excluded from the filtered result, since
    their dataset membership cannot be confirmed.

    This signature is purely additive: existing call sites that pass no
    argument (``list_snapshots()``) are unaffected.
    """
    snaps = []
    snap_dir = DATA_DIR / "snapshots"
    if not snap_dir.exists():
        return []
    for d in sorted(snap_dir.iterdir()):
        m = d / "manifest.json"
        if m.exists():
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                meta = {"name": d.name}
            snaps.append(meta)
    if ds_id is not None:
        snaps = [
            s for s in snaps 
            if isinstance(s, dict) and isinstance(s.get("dataset_ids"), list) and ds_id in s["dataset_ids"]
        ]
    return snaps
def restore_snapshot(name: str) -> bool:
    """Restore datasets from a named snapshot into session state."""
    import shutil
    try:
        snap_root = DATA_DIR / "snapshots" / name
        if not snap_root.exists():
            return False
        for ds_dir in snap_root.iterdir():
            if not ds_dir.is_dir() or ds_dir.name == "manifest.json":
                continue
            dest = DS_DIR / ds_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ds_dir, dest)
        return True
    except Exception as exc:
        log.warning(f"restore_snapshot failed: {exc}")
        return False

# ── Error handler decorator ────────────────────────────────────────────────────
def handle_page_errors(func):
    """Decorator that catches exceptions in a page render and shows them in the UI."""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            st.error(f"🚨 Page Error: Something went wrong while rendering this section.\n\n**Technical details**\n\n{exc}")
            log.error(f"Page error in {func.__name__}: {traceback.format_exc()}")
    return wrapper

# ── Session init ───────────────────────────────────────────────────────────────
_SESSION_DEFAULTS: dict = {
    "datasets":          {},
    "history":           [],
    "chat_history":      [],
    "editor_code":       "",
    "dashboard_pins":    [],
    "ai_memory":         {},
    "schedules":         [],
    "active_page":       "📌 Dashboard",
    "groq_key":          "",
    "_app_loaded":       False,
    "quantum_mode":      False,
    "uploader_cache":    set(),
    "dark_mode":         False,
    "authenticated":     False,
    "_saved_widget_states": {},
}

def init_session() -> None:
    """Initialize all required Streamlit session-state keys."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    # Load persisted history on first run
    if not st.session_state.get("_app_loaded"):
        st.session_state["history"] = load_history()
        st.session_state["_app_loaded"] = True
    # Restore persisted datasets
    _reload_datasets_from_disk()

def _reload_datasets_from_disk() -> None:
    """Reload datasets that were saved in a previous session."""
    if st.session_state.get("datasets"):
        return
    if not DS_DIR.exists():
        return
    for ds_dir in DS_DIR.iterdir():
        if not ds_dir.is_dir():
            continue
        parquet = ds_dir / "data.parquet"
        meta_f  = ds_dir / "meta.json"
        if not parquet.exists():
            continue
        try:
            df   = pd.read_parquet(parquet)
            df   = _restore_complex_columns(df)
            meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
            ds_id = ds_dir.name
            st.session_state["datasets"][ds_id] = {
                "df":     df,
                "name":   meta.get("name", ds_id),
                "source": meta.get("source", "disk"),
                "reason": meta.get("reason", "restored"),
            }
        except Exception as exc:
            log.warning(f"Failed to reload dataset {ds_dir.name}: {exc}")

# ── HDF5 metadata helpers ──────────────────────────────────────────────────────
def normalize_hdf5_attrs(attrs) -> dict:
    """Normalize HDF5 attributes dict to plain Python types."""
    result = {}
    for k, v in dict(attrs).items():
        try:
            if isinstance(v, bytes):
                result[k] = v.decode("utf-8", errors="replace")
            elif isinstance(v, (np.bytes_,)):
                result[k] = v.tobytes().decode("utf-8", errors="replace")
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.integer,)):
                result[k] = int(v)
            elif isinstance(v, (np.floating,)):
                result[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                result[k] = bool(v)
            else:
                result[k] = v
        except Exception:
            result[k] = str(v)
    return result

def infer_scientific_axis_metadata(path: str, shape: tuple, attrs: dict, overrides: dict = None) -> dict:
    """Infer axis roles and labels. Delegates to scientific_hdf5 implementation."""
    from modules.scientific_hdf5 import infer_scientific_axis_metadata as _real
    return _real(path, tuple(shape) if shape else (), attrs or {}, overrides)

# ── Universal file reader ──────────────────────────────────────────────────────
def read_uploaded_file(uploaded_file) -> Any:
    """
    Read any uploaded file into a DataFrame (or dict for HDF5).
    Returns pd.DataFrame or dict{str->pd.DataFrame}.
    """
    import os as _os
    name = getattr(uploaded_file, "name", "") or ""
    name_lower = name.lower()

    try:
        size = uploaded_file.seek(0, 2)
        uploaded_file.seek(0)
    except Exception:
        size = 0

    # ── HDF5 ──────────────────────────────────────────────────────────────────
    if name_lower.endswith((".h5", ".hdf5", ".hdf")):
        return _h5_to_df(uploaded_file)

    # ── CSV / TSV ──────────────────────────────────────────────────────────────
    if name_lower.endswith((".csv", ".txt")):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep="\t")

    if name_lower.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")

    # ── Excel ──────────────────────────────────────────────────────────────────
    if name_lower.endswith((".xlsx", ".xls", ".xlsm")):
        xl = pd.ExcelFile(uploaded_file)
        if len(xl.sheet_names) == 1:
            return xl.parse(xl.sheet_names[0])
        return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

    # ── JSON ───────────────────────────────────────────────────────────────────
    if name_lower.endswith(".json"):
        data = json.load(uploaded_file)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)
            return pd.DataFrame([data])
        return pd.DataFrame([{"value": data}])

    # ── Parquet ────────────────────────────────────────────────────────────────
    if name_lower.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    # ── Feather ────────────────────────────────────────────────────────────────
    if name_lower.endswith(".feather"):
        return pd.read_feather(uploaded_file)

    # ── XML ────────────────────────────────────────────────────────────────────
    if name_lower.endswith(".xml"):
        return pd.read_xml(uploaded_file)

    # ── SQLite / DuckDB ────────────────────────────────────────────────────────
    if name_lower.endswith((".sqlite", ".sqlite3", ".db")):
        import tempfile, sqlite3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            con = sqlite3.connect(tmp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)
            if tables.empty:
                return pd.DataFrame()
            frames = {}
            for tbl in tables["name"]:
                frames[f"{name}_{tbl}"] = pd.read_sql(f"SELECT * FROM [{tbl}]", con)
            con.close()
            return frames if frames else pd.DataFrame()
        finally:
            try: _os.unlink(tmp_path)
            except: pass

    # ── DuckDB ────────────────────────────────────────────────────────────────
    if name_lower.endswith(".duckdb"):
        import tempfile, duckdb
        with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            con = duckdb.connect(tmp_path, read_only=True)
            tables = con.execute("SHOW TABLES").fetchdf()
            frames = {}
            for table_name in tables.get("name", []):
                frames[f"{name}_{table_name}"] = con.execute(f"SELECT * FROM {table_name}").fetchdf()
            con.close()
            return frames if frames else pd.DataFrame()
        finally:
            try: _os.unlink(tmp_path)
            except: pass

    # ── Fallback: try CSV ──────────────────────────────────────────────────────
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()


def _h5_to_df(uploaded_file) -> Any:
    """
    Read an HDF5 file and return a dict of DataFrames keyed by dataset path,
    or a single metadata DataFrame for the file structure.

    For HIRAX-style files the primary numeric datasets (vis, gain, flags, etc.)
    are each converted to a DataFrame with proper numeric columns:
      - 3-D complex (time, freq, baseline): collapse to per-baseline amplitude columns
      - 2-D (time, baseline): collapse to per-baseline columns
      - 1-D structured (field1, field2, …): one column per field
      - 1-D flat: single numeric column
    """
    import h5py, tempfile, os as _os

    # Write buffer to a temp file so h5py can open it
    data = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    frames: dict[str, pd.DataFrame] = {}
    metadata_rows: list[dict] = []

    try:
        with h5py.File(tmp_path, "r") as f:

            # Collect all datasets in the file
            all_datasets: list[tuple[str, Any]] = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.append((name, obj))
            f.visititems(_collect)

            # Sort by size descending, prefer the largest arrays first
            def _ds_size(item):
                try: return item[1].size
                except: return 0
            sorted_datasets = sorted(all_datasets, key=_ds_size, reverse=True)

            for ds_path, ds_obj in sorted_datasets:
                try:
                    raw   = ds_obj[()]
                    shape = ds_obj.shape
                    dtype = ds_obj.dtype
                    attrs = normalize_hdf5_attrs(ds_obj.attrs)
                    name  = ds_path.split("/")[-1]
                    size  = int(np.prod(shape)) if shape else 0
                    kind  = _infer_scientific_kind_fast(ds_path, attrs, shape)

                    # Build metadata row
                    metadata_rows.append({
                        "category":            _path_to_category(ds_path),
                        "name":                name,
                        "path":                ds_path,
                        "shape":               str(shape),
                        "dtype":               str(dtype),
                        "size":                size,
                        "scientific_kind":     kind,
                        "axis_labels":         str(attrs.get("axis", "")),
                        "component_labels":    str(attrs.get("input_map", attrs.get("prod", ""))),
                        "axis_override_applied": False,
                    })

                    # Skip non-numeric or tiny metadata datasets
                    if dtype.names:
                        # Structured dtype — expand each field
                        df_dict = {}
                        for field in dtype.names:
                            col = raw[field]
                            if np.issubdtype(col.dtype, np.number):
                                if col.ndim == 1:
                                    df_dict[f"{name}_{field}"] = col.astype(float)
                                else:
                                    flat = col.reshape(col.shape[0], -1)
                                    for i in range(min(flat.shape[1], 8)):
                                        df_dict[f"{name}_{field}_{i}"] = flat[:, i].astype(float)
                        if df_dict:
                            n = max(len(v) for v in df_dict.values())
                            df_dict = {k: v[:n] for k, v in df_dict.items()}
                            frames[ds_path] = pd.DataFrame(df_dict)

                    elif np.iscomplexobj(raw):
                        # Complex array: produce amplitude + real + imag columns
                        arr = np.asarray(raw)
                        if arr.ndim == 3:
                            # (time, freq, baseline) → per-baseline mean amplitude over freq
                            n_time, n_freq, n_bl = arr.shape
                            df_dict = {}
                            for bl in range(min(n_bl, 20)):
                                amp = np.abs(arr[:, :, bl]).mean(axis=1)
                                df_dict[f"{name}_b{bl}"] = amp
                            frames[ds_path] = pd.DataFrame(df_dict)
                        elif arr.ndim == 4:
                            # (time, freq, baseline, pol)
                            n_time, n_freq, n_bl, n_pol = arr.shape
                            df_dict = {}
                            for bl in range(min(n_bl, 10)):
                                for pol in range(min(n_pol, 2)):
                                    amp = np.abs(arr[:, :, bl, pol]).mean(axis=1)
                                    df_dict[f"{name}_b{bl}_p{pol}"] = amp
                            frames[ds_path] = pd.DataFrame(df_dict)
                        elif arr.ndim == 2:
                            # (time, baseline)
                            n_time, n_bl = arr.shape
                            df_dict = {}
                            for bl in range(min(n_bl, 20)):
                                df_dict[f"{name}_amp_{bl}"] = np.abs(arr[:, bl])
                            frames[ds_path] = pd.DataFrame(df_dict)
                        elif arr.ndim == 1:
                            frames[ds_path] = pd.DataFrame({
                                f"{name}_real": arr.real,
                                f"{name}_imag": arr.imag,
                                f"{name}_amp":  np.abs(arr),
                            })

                    elif np.issubdtype(dtype, np.number):
                        arr = np.asarray(raw)
                        if arr.ndim == 1 and arr.size > 0:
                            frames[ds_path] = pd.DataFrame({name: arr.astype(float)})
                        elif arr.ndim == 2:
                            n_row, n_col = arr.shape
                            if n_col <= 32:
                                df_dict = {f"{name}_{i}": arr[:, i].astype(float) for i in range(n_col)}
                            else:
                                df_dict = {f"{name}_mean": arr.mean(axis=1),
                                           f"{name}_std":  arr.std(axis=1),
                                           f"{name}_max":  arr.max(axis=1),
                                           f"{name}_min":  arr.min(axis=1)}
                            frames[ds_path] = pd.DataFrame(df_dict)
                        elif arr.ndim == 3:
                            # Collapse to (time, n_bl) mean
                            n0, n1, n2 = arr.shape
                            df_dict = {}
                            for i in range(min(n2, 20)):
                                df_dict[f"{name}_b{i}"] = arr[:, :, i].mean(axis=1)
                            frames[ds_path] = pd.DataFrame(df_dict)

                except Exception as exc:
                    log.debug(f"HDF5 dataset {ds_path} skipped: {exc}")

    finally:
        try: _os.unlink(tmp_path)
        except: pass

    # Always include the metadata overview table
    if metadata_rows:
        frames["__metadata__"] = pd.DataFrame(metadata_rows)

    return frames if frames else pd.DataFrame()


def _path_to_category(path: str) -> str:
    """Derive a human-readable category from the HDF5 dataset path."""
    p = path.lower()
    if "index_map" in p: return "index_map"
    if any(t in p for t in ("vis", "flag", "weight")): return "datasets"
    if any(t in p for t in ("gain", "calib")): return "telescope"
    if "freq" in p or "chan" in p: return "frequencies"
    if "time" in p: return "time"
    return "datasets"

def _infer_scientific_kind_fast(path: str, attrs: dict, shape: tuple) -> str:
    """Fast heuristic for scientific kind without importing scientific_hdf5."""
    p = path.lower()
    a = " ".join(str(v).lower() for v in attrs.values())
    blob = p + " " + a
    if any(k in blob for k in ("vis", "visibility", "visibilities", "correlat")):
        return "visibility_tensor"
    if any(k in blob for k in ("gain", "calib", "bandpass")):
        return "calibration_solution"
    if any(k in blob for k in ("freq", "frequency", "chan")):
        return "frequency_axis"
    if any(k in blob for k in ("time", "timestamp", "fpga")):
        return "time_axis"
    if any(k in blob for k in ("baseline", "prod", "uv")):
        return "baseline_coordinates"
    return "generic_array"

# ── Scientific HDF5 reader ─────────────────────────────────────────────────────
def read_scientific_hdf5(uploaded_file, preserve_structure: bool = True) -> dict:
    """
    Read an HDF5 file and return a structured scientific_data dict with keys:
      datasets, telescope, interferometry, frequencies, time, axis_overrides
    Used by the HIRAX Inspector and observation session engine.
    """
    import h5py, tempfile, os as _os
    from modules.scientific_hdf5 import (
        infer_scientific_kind, apply_axis_overrides,
        infer_scientific_axis_metadata as infer_axis_metadata
    )

    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        name = getattr(uploaded_file, "name", "unknown.h5")
    else:
        with open(uploaded_file, "rb") as fh:
            data = fh.read()
        name = str(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    scientific_data: dict = {
        "datasets":       {},
        "telescope":      {},
        "interferometry": {},
        "frequencies":    {},
        "time":           {},
        "axis_overrides": {},
        "file_name":      name,
    }
    axis_index: dict = {}

    def _load_axis_overrides(fobj) -> dict:
        overrides: dict = {}
        for key in ("axis_overrides", "udip_axis_overrides", "axis_meta"):
            if key in fobj.attrs:
                try:
                    raw = fobj.attrs[key]
                    if isinstance(raw, (bytes, np.bytes_)):
                        raw = raw.decode("utf-8", errors="replace")
                    overrides.update(json.loads(raw) if isinstance(raw, str) else {})
                except Exception:
                    pass
            if key in fobj:
                try:
                    grp = fobj[key]
                    if isinstance(grp, h5py.Dataset):
                        raw = grp[()]
                        if hasattr(raw, "decode"):
                            raw = raw.decode("utf-8", errors="replace")
                        overrides.update(json.loads(raw) if isinstance(raw, str) else {})
                except Exception:
                    pass
        return overrides

    try:
        with h5py.File(tmp_path, "r") as f:
            axis_overrides = _load_axis_overrides(f)
            scientific_data["axis_overrides"] = axis_overrides

            def _walk(group, path_prefix: str = "") -> None:
                for key in group:
                    item      = group[key]
                    item_path = f"{path_prefix}/{key}" if path_prefix else key
                    item_path = item_path.lstrip("/")
                    if isinstance(item, h5py.Dataset):
                        try:
                            raw       = item[()]
                            shape     = item.shape
                            dtype     = item.dtype
                            attrs     = normalize_hdf5_attrs(item.attrs)
                            kind      = infer_scientific_kind(item_path, shape, attrs)
                            axis_meta = infer_axis_metadata(item_path, shape, attrs)
                            override  = axis_overrides.get(item_path) or axis_overrides.get(key)
                            if override:
                                axis_meta = apply_axis_overrides(axis_meta, override)
                            # Convert raw to DataFrame
                            if dtype.names:
                                rows = [dict(zip(dtype.names, row)) for row in raw.tolist()] \
                                       if raw.ndim == 1 else []
                                df = pd.DataFrame(rows) if rows else pd.DataFrame({"value": [str(raw)]})
                            else:
                                arr = np.asarray(raw)
                                if arr.size == 0:
                                    df = pd.DataFrame()
                                elif arr.ndim == 1:
                                    df = pd.DataFrame({"value": arr.tolist()})
                                elif arr.ndim == 2:
                                    cols = [f"col_{i}" for i in range(arr.shape[1])]
                                    df = pd.DataFrame(arr, columns=cols)
                                else:
                                    flat = arr.reshape(-1, arr.shape[-1])
                                    df = pd.DataFrame(flat, columns=[f"col_{i}" for i in range(flat.shape[1])])
                            ds_info = {
                                "df":              df,
                                "data":            raw,
                                "shape":           shape,
                                "dtype":           str(dtype),
                                "size":            int(np.prod(shape)) if shape else 0,
                                "scientific_kind": kind,
                                "axis_meta":       axis_meta,
                                "attributes":      attrs,
                                "path":            item_path,
                                "name":            key,
                            }
                            scientific_data["datasets"][item_path] = ds_info
                            axis_index[item_path] = axis_meta
                            key_lower = item_path.lower()
                            if any(t in key_lower for t in ("vis", "visibility")):
                                scientific_data["interferometry"]["visibilities"] = ds_info
                            elif any(t in key_lower for t in ("gain", "calib")):
                                scientific_data["telescope"]["calibration"] = ds_info
                            elif any(t in key_lower for t in ("freq", "chan")) and "index_map" in key_lower:
                                scientific_data["frequencies"]["channels"] = ds_info
                            elif "time" in key_lower and "index_map" in key_lower:
                                scientific_data["time"]["timestamps"] = ds_info
                            elif any(t in key_lower for t in ("prod", "baseline")):
                                scientific_data["interferometry"]["baselines"] = ds_info
                        except Exception as exc:
                            log.debug(f"HDF5 walk skipped {item_path}: {exc}")
                    elif isinstance(item, h5py.Group):
                        _walk(item, item_path)

            _walk(f)
            scientific_data["metadata"] = {
                "attributes": normalize_hdf5_attrs(f.attrs),
                "file_name": name,
            }

    finally:
        try: _os.unlink(tmp_path)
        except: pass

    return scientific_data
