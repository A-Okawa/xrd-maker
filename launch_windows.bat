@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 初回セットアップが必要です。setup_windows.bat を先に実行してください。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" desktop.py %*
set "XRD_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%XRD_LAUNCH_EXIT%"=="0" pause
exit /b %XRD_LAUNCH_EXIT%
