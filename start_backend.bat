@echo off
echo Starting ATC Backend Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Run from project root so 'backend' is a package
echo Backend server starting on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload