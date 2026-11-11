import json
import os
import datetime
from pathlib import Path
from typing import Dict, Any, List
from modules.core import DATA_DIR, log

EXPERIMENT_DIR = DATA_DIR / "hirax_experiments"
EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)

def log_experiment(dataset_id: str, pipeline: List[Dict], parameters: Dict, results: Dict) -> str:
    """
    Log a HIRAX experiment run to disk as JSON.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{dataset_id}_{timestamp}"
    
    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "pipeline": pipeline,
        "parameters": parameters,
        "results": {k: v for k, v in results.items() if isinstance(v, (str, int, float, list, dict))}
    }
    
    file_path = EXPERIMENT_DIR / f"{experiment_id}.json"
    try:
        with open(file_path, "w") as f:
            json.dump(entry, f, indent=2)
        log.info(f"Experiment logged: {experiment_id}")
        return experiment_id
    except Exception as e:
        log.error(f"Failed to log experiment: {e}")
        return ""

def list_experiments() -> List[Dict]:
    """
    List all logged experiments.
    """
    experiments = []
    for file in EXPERIMENT_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                experiments.append(json.load(f))
        except:
            continue
    return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

def get_experiment(experiment_id: str) -> Dict:
    """
    Get a specific experiment by ID.
    """
    file_path = EXPERIMENT_DIR / f"{experiment_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    return {}
