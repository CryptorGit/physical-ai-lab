@echo off
setlocal

set "ISAAC_SIM_HOME=C:\isaacsim"

if not exist "%ISAAC_SIM_HOME%\isaac-sim.bat" (
    echo [ERROR] Isaac Sim was not found:
    echo %ISAAC_SIM_HOME%\isaac-sim.bat
    pause
    exit /b 1
)

cd /d "%ISAAC_SIM_HOME%"

echo Starting Isaac Sim...
call isaac-sim.bat

if errorlevel 1 (
    echo.
    echo Isaac Sim exited with an error.
    pause
)

endlocal
