@echo off
echo Uninstalling UDIP v3...
del /q "%USERPROFILE%\Desktop\UDIP v3.lnk" 2>nul
rmdir /s /q "%USERPROFILE%\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\UDIP" 2>nul
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\UDIP_v3" /f 2>nul
echo.
echo UDIP has been uninstalled. Your data in the udip_data folder has been kept.
echo To remove your data too, delete the folder: %~dp0udip_data\
pause
