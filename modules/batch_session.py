"""Batch session engine for the 50-file HIRAX research corpus."""
from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from modules.hirax_research_pipeline import run_research_pipeline_for_file
from modules.hirax_research_corpus import build_corpus_manifest

BATCH_EXPORT_VERSION = "1.0"


def _stable_hash(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _aggregate(per_file: list[dict]) -> dict:
    valid = [r for r in per_file if r.get("reproducible_ok")]
    n = len(per_file)
    n_valid = len(valid)
    pct_valid = (n_valid / n * 100.0) if n else 0.0

    # Aggregate floats across files (NaN-safe).
    def _col(key):
        return np.array([r.get(key, np.nan) for r in valid], dtype=float)

    def _stats(cols):
        if cols.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
        finite = cols[np.isfinite(cols)]
        if finite.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    summary = {
        "n_files_total": n,
        "n_files_reproducible": n_valid,
        "pct_reproducible": pct_valid,
        "research_confidence": _stats(_col("research_confidence")),
        "trxr_k": _stats(_col("trxr_k")),
        "tsys_k_mean": _stats(_col("tsys_k_mean")),
        "t_brightness_k_mean": _stats(_col("t_brightness_k_mean")),
        "delta_t_sky_rms_k": _stats(_col("delta_t_sky_rms_k")),
        "beam_squint_arcmin": _stats(_col("beam_squint_arcmin")),
        "wedge_window_ratio": _stats(_col("wedge_window_ratio")),
        "sk_flagged_pct": _stats(_col("sk_flagged_pct")),
        "drift_rate_hz_per_min": _stats(_col("drift_rate_hz_per_min")),
    }
    return summary


def _categorise(per_file: list[dict]) -> dict:
    """Group files by category and report per-category stats."""
    by_cat: dict[str, list[dict]] = {}
    for row in per_file:
        by_cat.setdefault(row.get("category", "uncategorized"), []).append(row)
    out = {}
    for cat, rows in by_cat.items():
        conf = np.array([r.get("research_confidence", np.nan) for r in rows], dtype=float)
        conf_finite = conf[np.isfinite(conf)]
        out[cat] = {
            "n_files": len(rows),
            "research_confidence_mean": float(np.mean(conf_finite)) if conf_finite.size else float("nan"),
            "n_reproducible": int(sum(1 for r in rows if r.get("reproducible_ok"))),
        }
    return out


def _make_conclusion(per_file: list[dict], aggregate: dict, by_category: dict) -> dict:
    n = aggregate["n_files_total"]
    n_valid = aggregate["n_files_reproducible"]
    pct = aggregate["pct_reproducible"]
    trxr_med = aggregate["trxr_k"].get("median", float("nan"))
    tsys_med = aggregate["tsys_k_mean"].get("median", float("nan"))
    squint_med = aggregate["beam_squint_arcmin"].get("median", float("nan"))
    ratio_med = aggregate["wedge_window_ratio"].get("median", float("nan"))
    conf_med = aggregate["research_confidence"].get("median", float("nan"))
    dr_med = aggregate["drift_rate_hz_per_min"].get("median", float("nan"))
    sk_med = aggregate["sk_flagged_pct"].get("median", float("nan"))

    if pct >= 90.0 and conf_med >= 0.7 and ratio_med > 1.0:
        grade = "RESEARCH-GRADE"
    elif pct >= 70.0 and conf_med >= 0.5:
        grade = "RESEARCH SUPPORT"
    elif pct >= 40.0:
        grade = "EDUCATIONAL"
    else:
        grade = "INSUFFICIENT"

    text = (
        f"Session conclusion: {n} files processed, {n_valid} ({pct:.1f}%) "
        f"reproducible. Median Trxr = {trxr_med:.1f} K, median Tsys = {tsys_med:.1f} K, "
        f"median beam-squint = {squint_med:.2f} arcmin. Foreground wedge / window power "
        f"ratio = {ratio_med:.2f} (>1 means foreground wedge dominates as expected). "
        f"Median spectral-kurtosis flagged fraction = {sk_med:.2f} %, "
        f"median drifting-RFI rate = {dr_med:.3g} Hz/min. "
        f"Overall research confidence median = {conf_med:.3f}. "
        f"Grade: {grade}."
    )
    return {"text": text, "grade": grade, "generated_utc": datetime.now(timezone.utc).isoformat()}


def run_batch_session(file_paths: Sequence[str | Path], *, n_workers: int = 1) -> dict:
    """Run the HIRAX research pipeline against a list of files and aggregate."""
    file_paths = [Path(p) for p in file_paths]
    t0 = time.perf_counter()
    per_file: list[dict] = []
    for path in file_paths:
        try:
            res = run_research_pipeline_for_file(path)
        except Exception as exc:
            res = {
                "file_name": path.name,
                "category": "uncategorized",
                "errors": [str(exc)],
                "warnings": [],
                "research_confidence": float("nan"),
                "trxr_k": float("nan"),
                "tsys_k_mean": float("nan"),
                "t_brightness_k_mean": float("nan"),
                "delta_t_sky_rms_k": float("nan"),
                "beam_squint_arcmin": float("nan"),
                "wedge_window_ratio": float("nan"),
                "sk_flagged_pct": float("nan"),
                "drift_rate_hz_per_min": float("nan"),
                "reproducible_ok": False,
            }
        per_file.append(res)
    elapsed = time.perf_counter() - t0
    aggregate = _aggregate(per_file)
    by_category = _categorise(per_file)
    conclusion = _make_conclusion(per_file, aggregate, by_category)
    reproducibility_signature = _stable_hash([r.get("reproducible_signature") for r in per_file])
    return {
        "version": BATCH_EXPORT_VERSION,
        "n_workers": int(n_workers),
        "elapsed_seconds": float(elapsed),
        "per_file": per_file,
        "per_file_table": pd.DataFrame(per_file),
        "aggregate": aggregate,
        "by_category": by_category,
        "conclusion": conclusion,
        "reproducibility_signature": reproducibility_signature,
    }


def run_batch_validation_corpus(corpus_dir: str | Path) -> dict:
    corpus_dir = Path(corpus_dir)
    files = sorted(corpus_dir.glob("*.h5"))
    manifest = build_corpus_manifest(corpus_dir)
    return run_batch_session(files)


def export_batch_session(batch: dict, output_dir: str | Path, stem: str = "udip_research_session") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    table = batch.get("per_file_table")
    if isinstance(table, pd.DataFrame):
        table.to_csv(csv_path, index=False)
    payload = {
        "version": batch.get("version"),
        "aggregate": batch.get("aggregate"),
        "by_category": batch.get("by_category"),
        "conclusion": batch.get("conclusion"),
        "reproducibility_signature": batch.get("reproducibility_signature"),
        "elapsed_seconds": batch.get("elapsed_seconds"),
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return {"csv": str(csv_path), "json": str(json_path)}


__all__ = [
    "run_batch_session",
    "run_batch_validation_corpus",
    "export_batch_session",
    "BATCH_EXPORT_VERSION",
]
