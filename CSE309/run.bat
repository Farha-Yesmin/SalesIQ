@echo off
title SalesIQ
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo.
    echo  No virtual environment found in this folder.
    echo  Run these commands once from CSE309:
    echo    python -m venv venv
    echo    venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting SalesIQ...
echo Open: http://127.0.0.1:5000/login
echo Press Ctrl+C to stop.
echo.
venv\Scripts\python.exe app.py
if errorlevel 1 pause
