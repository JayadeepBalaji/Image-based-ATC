@echo off
echo ========================================
echo  Animal Type Classification Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Setting up virtual environment...

REM Create virtual environment
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start the system:
echo 1. Backend:  cd backend ^&^& python main.py
echo 2. Frontend: cd frontend ^&^& streamlit run app.py
echo.
echo Or use the provided start scripts:
echo - start_backend.bat
echo - start_frontend.bat
echo.
pause