@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat
start "" http://localhost:8501
streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
