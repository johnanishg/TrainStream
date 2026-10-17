@echo off
REM Starts the TrainStream live dashboard in its own console window and opens it.
REM Close that window (or press Ctrl+C in it) to stop the server.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

REM Already running? Just open the browser.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8765/api/system | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
  echo TrainStream dashboard is already running.
  start "" http://127.0.0.1:8765
  exit /b 0
)

start "TrainStream dashboard" "%PY%" -u dashboard\server.py %*

echo Waiting for the dashboard to start...
for /l %%i in (1,1,60) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8765/api/system | Out-Null; exit 0 } catch { exit 1 }"
  if not errorlevel 1 goto ready
  timeout /t 1 /nobreak >nul
)
echo The dashboard did not start within 60 seconds; check the "TrainStream dashboard" window for errors.
exit /b 1

:ready
start "" http://127.0.0.1:8765
echo Dashboard running at http://127.0.0.1:8765
