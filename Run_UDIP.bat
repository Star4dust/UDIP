@echo off
setlocal
title UDIP v3 - Universal Data Integration Platform
cd /d "%~dp0"

:: ── Check venv ────────────────────────────────────────────────────────────────
if not exist "venv\Scripts\python.exe" (
    echo First time setup - running installer...
    call "%~dp0installer\Install_UDIP.bat"
    exit /b
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call "%~dp0venv\Scripts\activate.bat"

:: ── Install missing packages silently ─────────────────────────────────────────
python -c "import streamlit, pystray, psutil" >nul 2>&1
if errorlevel 1 (
    echo Installing packages...
    pip install -r "%~dp0requirements.txt" --quiet
)

:: ── Clear cache ────────────────────────────────────────────────────────────────
if exist "%~dp0__pycache__" rmdir /s /q "%~dp0__pycache__" >nul 2>&1
if exist "%~dp0modules\__pycache__" rmdir /s /q "%~dp0modules\__pycache__" >nul 2>&1
if exist "%~dp0udip_pages\__pycache__" rmdir /s /q "%~dp0udip_pages\__pycache__" >nul 2>&1

:: ── Kill any existing Streamlit on port 8501 ──────────────────────────────────
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: ── Start Streamlit ───────────────────────────────────────────────────────────
echo.
echo  ==========================================
echo   UDIP v3 - Universal Data Integration
echo  ==========================================
echo.
echo  Starting... please wait.
echo.

start /b streamlit run "%~dp0app.py" ^
    --server.port 8501 ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --logger.level error

:: ── Wait until server responds ────────────────────────────────────────────────
echo  Waiting for server...
:WAIT_LOOP
timeout /t 1 /nobreak >nul
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501', timeout=1)" >nul 2>&1
if errorlevel 1 goto WAIT_LOOP

:: ── Open browser ─────────────────────────────────────────────────────────────
echo  Server ready - opening browser...
start "" "http://localhost:8501"
echo.
echo  ==========================================
echo   UDIP is running at http://localhost:8501
echo.
echo   Keep this window open while using app.
echo   To stop: close this window.
echo  ==========================================

:: ── Keep window open (keep Streamlit alive) ───────────────────────────────────
:KEEP_ALIVE
timeout /t 5 /nobreak >nul
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501', timeout=2)" >nul 2>&1
if not errorlevel 1 goto KEEP_ALIVE

echo.
echo  UDIP has stopped. Press any key to exit.
pause >nul
