import json
import pathlib
import sys
import time

# Ensure modules can be imported
sys.path.append(str(pathlib.Path(__file__).parent))

from modules.orchestrator import plan_actions
from modules.pipeline_runner import run_pipeline
from modules.system_observer import SystemObserver

observer = SystemObserver()

# 1. Run a successful pipeline
request = "load the file 'udip_data/execution_layer/sample.csv', clean it and save as 'observed_test'"
print("--- RUNNING SUCCESSFUL PIPELINE ---")
plan = plan_actions(request)
res_success = run_pipeline(plan)
print(f"Status: {res_success['status']}")

# 2. Run a failing pipeline (malicious SQL)
request_fail = "run sql 'DROP TABLE users'" # This will fail validation or execution
print("\n--- RUNNING FAILING PIPELINE ---")
plan_fail = plan_actions(request_fail)
res_fail = run_pipeline(plan_fail)
print(f"Status: {res_fail['status']}")

# 3. Check Metrics
print("\n--- SYSTEM METRICS ---")
metrics = observer.get_summary_metrics()
print(json.dumps(metrics, indent=2))

print("\n--- RECENT LOGS ---")
logs = observer.get_detailed_logs(limit=5)
for log in logs:
    print(f"[{log['timestamp']}] Run: {log['run_id']} | Success: {log['success']} | Duration: {log['duration_ms']:.2f}ms")
