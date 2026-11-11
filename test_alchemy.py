import pandas as pd
import numpy as np
import json
from modules.feature_alchemy import FeatureAlchemy

# 1. Prepare sample data
df = pd.DataFrame({
    "id": range(100),
    "date": pd.date_range("2023-01-01", periods=100),
    "target": np.random.randn(100),
    "feat_high_var": np.random.randn(100) * 10,
    "feat_low_var": [0.001] * 100, # Should be filtered by variance
    "feat_uncorrelated": np.random.randn(100) # Should be filtered by correlation
})

# Add a correlated feature
df["feat_correlated"] = df["target"] * 2 + np.random.randn(100) * 0.1

alchemy = FeatureAlchemy(mode="BALANCED", budget=10)

# 2. Test Feature Generation
print("=== FEATURE GENERATION (BALANCED) ===")
gen_res = alchemy.generate_features(df)
print(f"Original columns: {len(df.columns)}")
print(f"Transformed columns: {len(gen_res['transformed_df'].columns)}")
print("Report:", json.dumps(gen_res["report"], indent=2))

# 3. Test Feature Selection
print("\n=== FEATURE SELECTION ===")
# We use the transformed DF so it has many features
sel_res = alchemy.select_features(gen_res["transformed_df"], target="target")
print(f"Final columns: {len(sel_res['selected_df'].columns)}")
print("Selected columns:", sel_res["selected_df"].columns.tolist())
print("Report Summary:", json.dumps({k: v for k, v in sel_res["report"].items() if k != "correlation_scores"}, indent=2))
