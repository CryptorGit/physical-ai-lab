@echo off
setlocal

set "MUJOCO_DIR=%USERPROFILE%\workspace\physical-ai-lab\experiments\mujoco"

if not exist "%MUJOCO_DIR%\.venv\Scripts\Activate.ps1" (
    echo [ERROR] MuJoCo virtual environment was not found.
    echo %MUJOCO_DIR%\.venv\Scripts\Activate.ps1
    pause
    exit /b 1
)

powershell.exe -NoExit -ExecutionPolicy Bypass -Command ^
  "Set-Location '%MUJOCO_DIR%'; .\.venv\Scripts\Activate.ps1"

endlocal
