@echo off
echo Starting ATC Frontend...
echo.

REM Activate virtual environment
call venv\Scripts\activate

REM Change to frontend directory
cd frontend

REM Start Streamlit
echo Frontend starting on http://localhost:8501
echo Press Ctrl+C to stop the frontend
echo.
streamlit run app.py