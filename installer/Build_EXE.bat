@echo off
title UDIP v3 - Build EXE
cd /d "%~dp0"
cd ..
set APP_DIR=%CD%

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Run Install_UDIP.bat first.
    pause & exit /b 1
)

call "venv\Scripts\activate.bat"

:: Write the app path so launcher.py can find it
echo %APP_DIR%> "_udip_path.txt"

pip install pyinstaller --quiet --no-warn-script-location

echo Building UDIP.exe...
pyinstaller --onefile --windowed --icon=udip.ico --name=UDIP --add-data="_udip_path.txt;." --noconfirm launcher.py

if exist "dist\UDIP.exe" (
    copy "dist\UDIP.exe" "UDIP.exe" >nul
    echo.
    echo SUCCESS! UDIP.exe is ready. Double-click it to launch.
    echo Updating Desktop shortcut...
    set VBS=%TEMP%\sc.vbs
    (
    echo Set o = WScript.CreateObject^("WScript.Shell"^)
    echo Set s = o.CreateShortcut^("%USERPROFILE%\Desktop\UDIP v3.lnk"^)
    echo s.TargetPath = "%APP_DIR%\UDIP.exe"
    echo s.WorkingDirectory = "%APP_DIR%"
    echo s.IconLocation = "%APP_DIR%\udip.ico"
    echo s.Save
    ) > "%VBS%"
    cscript //nologo "%VBS%"
    del "%VBS%" >nul 2>&1
    echo Done!
) else (
    echo BUILD FAILED - scroll up to see error.
)
pause
