@echo off
setlocal EnableDelayedExpansion
title UDIP v3 — Installer
color 0A

cd /d "%~dp0"
cd ..
set APP_DIR=%CD%
echo  App folder: %APP_DIR%

echo.
echo  ================================================
echo    UDIP v3 — Universal Data Integration Platform
echo    Installer
echo  ================================================
echo.
echo  This installer will:
echo   1. Check for Python (install if missing)
echo   2. Install all required packages
echo   3. Create a Desktop shortcut
echo   4. Create a Start Menu entry
echo   5. Register UDIP with Windows (Add/Remove Programs)
echo.
echo  Press any key to continue, or close this window to cancel.
pause > nul

:: ── Check Python ─────────────────────────────────────────────────────────────
echo.
echo  [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python not found. Opening download page...
    echo  IMPORTANT: When installing Python, tick "Add python.exe to PATH"
    echo  After installing Python, run this installer again.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo  [OK] Python %PYVER%

:: ── Setup virtual environment ─────────────────────────────────────────────────
echo.
echo  [2/6] Setting up app environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo  [ERROR] Could not create environment.
        pause & exit /b 1
    )
)
call venv\Scripts\activate.bat

:: ── Install packages ──────────────────────────────────────────────────────────
echo.
echo  [3/6] Installing packages (may take 3-5 minutes first time)...
echo  This may take a few minutes - please wait...
pip install -r "%APP_DIR%\requirements.txt" --disable-pip-version-check --no-warn-script-location 2>nul
pip install pystray pillow psutil --quiet --no-warn-script-location 2>nul
echo  [OK] All packages installed

echo  [OK] All packages installed

:: ── Create launcher batch file ─────────────────────────────────────────────────
echo.
echo  [4/6] Creating launcher...
set INSTALL_DIR=%~dp0
set LAUNCHER=%INSTALL_DIR%Launch_UDIP.bat

(
echo @echo off
echo cd /d "%INSTALL_DIR%"
echo call venv\Scripts\activate.bat
echo start "" http://localhost:8501
echo streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
) > "%LAUNCHER%"

:: ── Desktop shortcut ──────────────────────────────────────────────────────────
echo.
echo  [5/6] Creating Desktop shortcut...
set SHORTCUT_PATH=%USERPROFILE%\Desktop\UDIP v3.lnk
set VBS_TEMP=%TEMP%\create_shortcut.vbs

(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%SHORTCUT_PATH%"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%APP_DIR%\Run_UDIP.bat"
echo oLink.WorkingDirectory = "%APP_DIR%"
echo oLink.Description = "UDIP v3 — Universal Data Integration Platform"
echo oLink.IconLocation = "%APP_DIR%\udip.ico"
echo oLink.WindowStyle = 1
echo oLink.Save
) > "%VBS_TEMP%"

cscript //nologo "%VBS_TEMP%"
del "%VBS_TEMP%" >nul 2>&1
echo  [OK] Desktop shortcut created

:: ── Start Menu entry ──────────────────────────────────────────────────────────
set START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\UDIP
if not exist "%START_MENU%" mkdir "%START_MENU%"
set START_SHORTCUT=%START_MENU%\UDIP v3.lnk

(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = "%START_SHORTCUT%"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "%APP_DIR%\Run_UDIP.bat"
echo oLink.WorkingDirectory = "%APP_DIR%"
echo oLink.Description = "UDIP v3 — Universal Data Integration Platform"
echo oLink.IconLocation = "%APP_DIR%\udip.ico"
echo oLink.Save
) > "%VBS_TEMP%"

cscript //nologo "%VBS_TEMP%"
del "%VBS_TEMP%" >nul 2>&1
echo  [OK] Start Menu entry created

:: ── Register with Windows (optional, skipped if it hangs) ─────────────────────
echo.
echo  [6/6] Registering with Windows...
set REG_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UDIP_v3
reg add "%REG_KEY%" /v "DisplayName" /t REG_SZ /d "UDIP v3" /f >nul 2>&1
reg add "%REG_KEY%" /v "DisplayVersion" /t REG_SZ /d "3.0.0" /f >nul 2>&1
reg add "%REG_KEY%" /v "InstallLocation" /t REG_SZ /d "%APP_DIR%" /f >nul 2>&1
echo  [OK] Registration complete

:: ── Create uninstaller ────────────────────────────────────────────────────────
(
echo @echo off
echo echo Uninstalling UDIP v3...
echo del /q "%USERPROFILE%\Desktop\UDIP v3.lnk" 2^>nul
echo rmdir /s /q "%START_MENU%" 2^>nul
echo reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UDIP_v3" /f 2^>nul
echo echo.
echo echo UDIP has been uninstalled. Your data in the udip_data folder has been kept.
echo echo To remove your data too, delete the folder: %INSTALL_DIR%
echo pause
) > "%APP_DIR%\Uninstall_UDIP.bat"

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Installation complete!
echo  ================================================
echo.
echo   UDIP v3 has been installed. You can now:
echo.
echo   Option 1: Double-click "UDIP v3" on your Desktop
echo   Option 2: Search "UDIP" in the Start Menu
echo   Option 3: Run Launch_UDIP.bat in this folder
echo.
echo   To uninstall: Run Uninstall_UDIP.bat or use
echo   Control Panel > Add/Remove Programs > UDIP v3
echo.

set /p LAUNCH="Launch UDIP now? (Y/N): "
if /i "%LAUNCH%"=="Y" (
    echo.
    echo  Starting UDIP...
    call "%APP_DIR%\Run_UDIP.bat"
)

echo.
echo  Installation finished. You can close this window.
pause
