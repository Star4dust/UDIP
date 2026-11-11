import pandas as pd
import numpy as np
import logging
from typing import Any

logger = logging.getLogger("udip.feature_alchemy")
logger.addHandler(logging.NullHandler())

class FeatureAlchemy:
    """
    Automated feature engineering and selection module.
    """
    
    def __init__(self, mode: str = "BALANCED", budget: int = 20):
        self.mode = mode.upper()
        self.budget = budget
        
    def _classify_columns(self, df: pd.DataFrame) -> dict:
        """
        Classify columns into continuous, categorical, ID-like, or low-variance.
        """
        classification = {
            "continuous": [],
            "categorical": [],
            "id_like": [],
            "low_variance": [],
            "datetime": []
        }
        
        num_rows = len(df)
        if num_rows == 0:
            return classification
            
        for col in df.columns:
            series = df[col]
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series):
                classification["datetime"].append(col)
                continue
            
            # ID-like detection
            unique_count = series.nunique()
            if unique_count == num_rows and num_rows > 1 and not pd.api.types.is_float_dtype(series):
                classification["id_like"].append(col)
                continue
                
            # Numeric
            if pd.api.types.is_numeric_dtype(series):
                # Low variance check
                if len(series) > 1:
                    if series.std() < 1e-6:
                        classification["low_variance"].append(col)
                        continue
                else:
                    classification["low_variance"].append(col)
                    continue

                # Check for ID-like integers
                if pd.api.types.is_integer_dtype(series) and unique_count > 0.9 * num_rows:
                    classification["id_like"].append(col)
                else:
                    classification["continuous"].append(col)
            else:
                classification["categorical"].append(col)
                
        return classification

    def generate_features(self, df: pd.DataFrame, mode: str = None, budget: int = None) -> dict:
        """
        Generate new features based on time and numeric transformations.
        """
        exec_mode = (mode or self.mode).upper()
        exec_budget = budget or self.budget
        
        transformed_df = df.copy()
        report = {"added_features": [], "transformations": []}
        
        classification = self._classify_columns(df)
        
        # 1. Time-based Features
        for col in classification["datetime"]:
            if len(report["added_features"]) >= exec_budget: break
            series = df[col]
            
            new_feats = {
                f"{col}_year": series.dt.year,
                f"{col}_month": series.dt.month,
                f"{col}_day": series.dt.day,
                f"{col}_dayofweek": series.dt.dayofweek
            }
            
            for name, data in new_feats.items():
                if len(report["added_features"]) < exec_budget:
                    transformed_df[name] = data
                    report["added_features"].append(name)
            
            report["transformations"].append(f"Extracted time features from {col}")

        # 2. Numeric Transformations
        continuous_cols = classification["continuous"]
        
        # BASIC, BALANCED, AGGRESSIVE
        for col in continuous_cols:
            if len(report["added_features"]) >= exec_budget: break
            
            series = df[col]
            
            # Scaling (Always apply if BASIC+)
            if exec_mode in ["BASIC", "BALANCED", "AGGRESSIVE"]:
                # Simple Min-Max scaling
                c_min, c_max = series.min(), series.max()
                if c_max > c_min:
                    transformed_df[f"{col}_scaled"] = (series - c_min) / (c_max - c_min)
                    report["added_features"].append(f"{col}_scaled")
                
            # BALANCED+ Transformations
            if exec_mode in ["BALANCED", "AGGRESSIVE"] and len(report["added_features"]) < exec_budget:
                # Log transform
                if (series > 0).all():
                    transformed_df[f"{col}_log"] = np.log1p(series)
                    report["added_features"].append(f"{col}_log")
                
                # Square transform
                if len(report["added_features"]) < exec_budget:
                    transformed_df[f"{col}_sq"] = series ** 2
                    report["added_features"].append(f"{col}_sq")

        # 3. Interactions (AGGRESSIVE only)
        if exec_mode == "AGGRESSIVE":
            for i, col1 in enumerate(continuous_cols):
                if len(report["added_features"]) >= exec_budget: break
                for col2 in continuous_cols[i+1:]:
                    if len(report["added_features"]) >= exec_budget: break
                    
                    name = f"{col1}_x_{col2}"
                    transformed_df[name] = df[col1] * df[col2]
                    report["added_features"].append(name)

        return {
            "transformed_df": transformed_df,
            "report": report
        }

    def select_features(self, df: pd.DataFrame, target: str, variance_threshold: float = 0.01) -> dict:
        """
        Select high-value features using variance thresholding and correlation.
        """
        report = {"dropped_features": [], "selection_criteria": []}
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")
            
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if target not in numeric_df.columns:
             # Try to convert target if possible, or fail
             try:
                 numeric_df[target] = pd.to_numeric(df[target])
             except Exception:
                 raise ValueError(f"Target column '{target}' must be numeric for selection.")

        # 1. Variance Thresholding
        variances = numeric_df.var()
        low_var_cols = variances[variances < variance_threshold].index.tolist()
        if target in low_var_cols: low_var_cols.remove(target)
        
        report["dropped_features"].extend(low_var_cols)
        report["selection_criteria"].append(f"Variance Threshold: {variance_threshold}")
        
        # 2. Correlation Filtering
        correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
        # Drop if correlation < 0.1
        weak_cols = correlations[correlations < 0.1].index.tolist()
        
        report["dropped_features"].extend([c for c in weak_cols if c not in report["dropped_features"]])
        report["selection_criteria"].append("Correlation Threshold: 0.1")
        
        # Identify final features
        final_cols = [c for c in numeric_df.columns if c not in report["dropped_features"]]
        selected_df = df[final_cols]
        
        report["correlation_scores"] = {str(k): float(v) for k, v in correlations.to_dict().items()}
        
        return {
            "selected_df": selected_df,
            "report": report
        }
