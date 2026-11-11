import os
import shutil
import zipfile
import tempfile
from pathlib import Path
import streamlit as st
import datetime

def create_portable_bundle():
    """
    Creates a .zip bundle containing the app code and current data.
    This allows 'deployment' by simply moving the folder to another PC.
    """
    workspace = Path(os.getcwd())
    temp_dir = Path(tempfile.mkdtemp())
    bundle_name = "udip_vanguard_portable"
    bundle_path = temp_dir / bundle_name
    bundle_path.mkdir()

    # 1. Copy essential modules and pages
    shutil.copytree(workspace / "modules", bundle_path / "modules", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(workspace / "udip_pages", bundle_path / "udip_pages", ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
    # 2. Copy main app components
    shutil.copy(workspace / "app.py", bundle_path / "app.py")
    shutil.copy(workspace / "requirements.txt", bundle_path / "requirements.txt")
    if (workspace / "LICENSE").exists():
        shutil.copy(workspace / "LICENSE", bundle_path / "LICENSE")
    
    # 3. Copy current datasets
    data_dest = bundle_path / "udip_data"
    data_dest.mkdir()
    if (workspace / "udip_data").exists():
        for f in (workspace / "udip_data").glob("*"):
            if f.is_file():
                shutil.copy(f, data_dest / f.name)

    # 4. Create README
    readme_content = f"""# UDIP VANGUARD PORTABLE BUNDLE
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## How to run:
1. Ensure you have Python 3.9+ installed.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Launch UDIP: `streamlit run app.py`

This bundle contains your code, configurations, and all currently saved datasets.
"""
    (bundle_path / "README.md").write_text(readme_content)

    # 5. Zip it up
    zip_output = workspace / f"{bundle_name}.zip"
    with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(bundle_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), bundle_path)
                zipf.write(os.path.join(root, file), arcname=os.path.join(bundle_name, rel_path))

    # Cleanup temp
    shutil.rmtree(temp_dir)
    return str(zip_output)
