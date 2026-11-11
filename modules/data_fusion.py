import pandas as pd
import numpy as np
import difflib
import logging
from typing import Any

logger = logging.getLogger("udip.data_fusion")
logger.addHandler(logging.NullHandler())

class DataFusionEngine:
    """
    Engine for aligning schemas and merging datasets using fuzzy semantic matching
    and statistical similarity.
    """
    
    def __init__(self, mode: str = "SAFE"):
        self.mode = mode.upper() # SAFE or AUTO
        
    def _get_column_profile(self, df: pd.DataFrame) -> dict:
        profile = {}
        for col in df.columns:
            series = df[col].dropna()
            col_type = str(df[col].dtype)
            
            stats = {
                "name": str(col),
                "type": col_type,
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "count": len(series)
            }
            
            if stats["is_numeric"] and not series.empty:
                stats.update({
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max())
                })
            profile[col] = stats
        return profile

    def _calculate_similarity(self, p1: dict, p2: dict) -> float:
        """
        Calculate a similarity score (0-1) between two columns based on name, type, and stats.
        """
        # 1. Name Similarity (0.0 to 1.0)
        name_sim = difflib.SequenceMatcher(None, p1["name"].lower(), p2["name"].lower()).ratio()
        
        # 2. Type Similarity
        type_sim = 1.0 if p1["type"] == p2["type"] else 0.5 if (p1["is_numeric"] == p2["is_numeric"]) else 0.0
        
        # 3. Statistical Similarity (for numeric)
        stat_sim = 0.0
        if p1["is_numeric"] and p2["is_numeric"]:
            # Check overlap of range
            range1 = p1.get("max", 0) - p1.get("min", 0)
            range2 = p2.get("max", 0) - p2.get("min", 0)
            
            if range1 > 0 and range2 > 0:
                # Overlap calculation
                overlap_min = max(p1["min"], p2["min"])
                overlap_max = min(p1["max"], p2["max"])
                if overlap_max > overlap_min:
                    overlap_ratio = (overlap_max - overlap_min) / max(range1, range2)
                    stat_sim = overlap_ratio
                
                # Mean proximity
                mean_diff = abs(p1.get("mean", 0) - p2.get("mean", 0))
                max_val = max(abs(p1.get("mean", 0)), abs(p2.get("mean", 0)), 1e-9)
                mean_sim = 1.0 - min(mean_diff / max_val, 1.0)
                stat_sim = (stat_sim + mean_sim) / 2.0

        # Weighted score
        weights = {"name": 0.4, "type": 0.3, "stats": 0.3}
        score = (name_sim * weights["name"]) + (type_sim * weights["type"]) + (stat_sim * weights["stats"])
        return score

    def propose_alignment(self, df_list: list[pd.DataFrame]) -> dict:
        """
        Generate a proposal for aligning columns across multiple DataFrames.
        """
        if not df_list or len(df_list) < 2:
            return {"mappings": [], "confidence": {}}
            
        profiles = [self._get_column_profile(df) for df in df_list]
        proposal = {"mappings": [], "confidence": {}}
        
        # Pivot around the first DataFrame as the target schema
        target_profile = profiles[0]
        
        for i, source_profile in enumerate(profiles[1:], start=1):
            mapping = {}
            for s_col, s_p in source_profile.items():
                best_match = None
                best_score = 0.0
                
                for t_col, t_p in target_profile.items():
                    score = self._calculate_similarity(s_p, t_p)
                    if score > best_score:
                        best_score = score
                        best_match = t_col
                
                if best_match and best_score > 0.6: # Threshold for proposal
                    mapping[s_col] = best_match
                    proposal["confidence"][f"df_{i}.{s_col}"] = float(best_score)
            
            proposal["mappings"].append(mapping)
            
        return proposal

    def align_schemas(self, df_list: list[pd.DataFrame], proposal: dict = None, mode: str = None) -> list[pd.DataFrame]:
        """
        Align schemas based on proposal. In SAFE mode (default), returns original DataFrames with metadata.
        In AUTO mode, renames columns.
        """
        exec_mode = (mode or self.mode).upper()
        
        if not proposal:
            proposal = self.propose_alignment(df_list)
            
        aligned_dfs = []
        # First DF is the target, so it stays as is
        aligned_dfs.append(df_list[0].copy() if exec_mode == "AUTO" else df_list[0])
        
        for i, df in enumerate(df_list[1:], start=0):
            mapping = proposal["mappings"][i] if i < len(proposal["mappings"]) else {}
            if exec_mode == "AUTO":
                new_df = df.rename(columns=mapping)
                aligned_dfs.append(new_df)
            else:
                # SAFE mode: add metadata but don't mutate data
                df_meta = df.copy()
                df_meta.attrs["alignment_proposal"] = mapping
                aligned_dfs.append(df_meta)
                
        return aligned_dfs

    def merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, key: str, mode: str = None) -> dict:
        """
        Merge two datasets by key, handling schema alignment and conflict detection.
        """
        exec_mode = (mode or self.mode).upper()
        proposal = self.propose_alignment([df1, df2])
        mapping = proposal["mappings"][0] if proposal["mappings"] else {}
        
        # Check if key needs alignment
        target_key = key
        source_key = key
        if key not in df2.columns:
            # Look for key in mapping
            for s_col, t_col in mapping.items():
                if t_col == key:
                    source_key = s_col
                    break
        
        report = {
            "mode": exec_mode,
            "key_used": {"df1": target_key, "df2": source_key},
            "alignment_proposal": mapping,
            "conflicts": {}
        }
        
        if exec_mode == "AUTO":
            df2_aligned = df2.rename(columns=mapping)
            merged = pd.merge(df1, df2_aligned, on=target_key, how="inner", suffixes=("_1", "_2"))
            
            # Conflict Detection
            common_cols = set(df1.columns).intersection(set(df2_aligned.columns)) - {target_key}
            for col in common_cols:
                c1 = f"{col}_1"
                c2 = f"{col}_2"
                if c1 in merged.columns and c2 in merged.columns:
                    mismatches = merged[merged[c1] != merged[c2]]
                    if not mismatches.empty:
                        report["conflicts"][col] = int(len(mismatches))
        else:
            # SAFE mode: try standard merge on provided keys without renaming
            try:
                merged = pd.merge(df1, df2, left_on=target_key, right_on=source_key, how="inner")
            except Exception as e:
                report["error"] = str(e)
                merged = None
                
        return {
            "merged_df": merged,
            "fusion_report": report
        }
