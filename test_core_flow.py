import json, pathlib, sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from modules.orchestrator import plan_actions
from modules.pipeline_runner import run_pipeline

# Prepare a tiny CSV if not exists (sample.csv already created)
request = "load the file 'udip_data/execution_layer/sample.csv', clean it with drop_nulls and save as 'cleaned_test'"
plan = plan_actions(request)
print('=== PLAN ===')
print(json.dumps(plan, indent=2))
result = run_pipeline(plan)
print('\n=== PIPELINE RESULT ===')
print(json.dumps(result, indent=2))
