@echo off
rem Starts the phishing-detection API with the project's own Python (.venv).
rem Double-click this file, or run start_app.bat from PowerShell in this folder.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The .venv folder is missing. Create it first:
    echo   py -3.13 -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting the API (the first start can take up to a minute)...
echo When you see "Uvicorn running", open http://127.0.0.1:8000/ in your browser.
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
