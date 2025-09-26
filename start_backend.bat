@echo off
echo Starting ATC Backend Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Change to backend directory
cd backend

REM Start the FastAPI server
echo Backend server starting on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python main.py