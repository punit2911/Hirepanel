@echo off
rem Windows one-click start: double-click this file, or run scripts\run.bat
cd /d "%~dp0\.."
where python >nul 2>nul || (echo Install Python 3.10+ first: https://www.python.org/downloads/ & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
python scripts\setup_env.py
echo.
echo Starting HirePanel at http://localhost:8000   (Ctrl+C to stop)
uvicorn app.asgi:app --host 127.0.0.1 --port 8000
pause
