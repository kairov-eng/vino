@echo off
REM Double-click to start Vino Svoe local frontend + backend (two windows).
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo Freeing ports 8091 / 8092 ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\dev-free-ports.ps1"
if errorlevel 1 (
  echo Warning: free-ports script returned an error — continuing anyway.
)

echo Starting backend on 8092 ...
start "vino-backend :8092" powershell -NoProfile -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location -LiteralPath '%ROOT%\backend'; Write-Host 'Backend http://127.0.0.1:8092' -ForegroundColor Cyan; python run_dev.py"

timeout /t 2 /nobreak >nul

echo Starting frontend on 8091 ...
start "vino-frontend :8091" powershell -NoProfile -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location -LiteralPath '%ROOT%\frontend'; Write-Host 'Frontend http://127.0.0.1:8091' -ForegroundColor Cyan; npm run dev"

echo.
echo Opened two windows. UI: http://127.0.0.1:8091
echo Close this window anytime; servers keep running in the other two.
timeout /t 4 >nul
endlocal
