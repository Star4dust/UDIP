import shutil
from pathlib import Path

base_dir = Path("udip_data")
default_ws = base_dir / "default"
default_ws.mkdir(exist_ok=True)

to_move = [
    "config.json", "ai_memory.json", "chat_history.json", "dashboard.json",
    "history.json", "report_artifacts.json", "saved_code.py", "udip.log",
    "databases", "datasets", "models", "snapshots"
]

for item_name in to_move:
    src = base_dir / item_name
    dst = default_ws / item_name
    if src.exists():
        print(f"Moving {src} to {dst}")
        if src.is_dir():
            # If dst dir exists, we might need to merge or delete
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))
    else:
        print(f"Skipping {item_name} (not found)")
