import pandas as pd
import numpy as np
import json
from modules.data_fusion import DataFusionEngine

# 1. Prepare sample data
df1 = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "revenue": [100.0, 200.0, 300.0],
    "region": ["North", "South", "East"]
})

df2 = pd.DataFrame({
    "cust_id": [1, 2, 4],
    "sales": [105.0, 210.0, 400.0], # Similar values to revenue
    "loc": ["N", "S", "W"]
})

engine = DataFusionEngine(mode="SAFE")

# 2. Test Proposal
print("=== ALIGNMENT PROPOSAL ===")
proposal = engine.propose_alignment([df1, df2])
print(json.dumps(proposal, indent=2))

# 3. Test Safe Merge
print("\n=== SAFE MERGE (No renaming) ===")
res_safe = engine.merge_datasets(df1, df2, key="customer_id", mode="SAFE")
print(f"Success: {res_safe['merged_df'] is not None}")
print("Report:", json.dumps(res_safe["fusion_report"], indent=2))

# 4. Test Auto Merge
print("\n=== AUTO MERGE (Renaming enabled) ===")
res_auto = engine.merge_datasets(df1, df2, key="customer_id", mode="AUTO")
if res_auto["merged_df"] is not None:
    print("Merged columns:", res_auto["merged_df"].columns.tolist())
    print("Merged rows:", len(res_auto["merged_df"]))
print("Report:", json.dumps(res_auto["fusion_report"], indent=2))
