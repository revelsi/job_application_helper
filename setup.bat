@echo off
setlocal enabledelayedexpansion

REM Job Application Helper - Simple Windows Setup Script
REM Ensures proper isolation with minimal complexity

echo 🚀 JOB APPLICATION HELPER - SETUP
echo ============================================
echo Setting up isolated development environment
echo ============================================

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Please run this script from the project root directory
    exit /b 1
)
if not exist "backend" (
    echo ❌ Backend directory not found
    exit /b 1
)
if not exist "frontend" (
    echo ❌ Frontend directory not found
    exit /b 1
)

REM Find Python 3.9+
echo ℹ️  Searching for Python 3.9+...
set PYTHON_CMD=
for %%p in (python3.13 python3.12 python3.11 python3.10 python3.9 python3 python) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%v in ('%%p --version 2^>^&1') do (
            for /f "tokens=1,2 delims=." %%a in ("%%v") do (
                set major=%%a
                set minor=%%b
                if !major! equ 3 if !minor! geq 9 (
                    echo ✅ Found Python %%v: %%p
                    set PYTHON_CMD=%%p
                    goto :found_python
                )
            )
        )
    )
)

echo ❌ Python 3.9+ required but not found!
echo Install from python.org and add to PATH
exit /b 1

:found_python

REM Create root data directories
echo ℹ️  Creating data directories...
mkdir data\documents 2>nul
mkdir data\cache 2>nul
echo ✅ Data directories created

REM Setup backend with strict isolation
echo ℹ️  Setting up backend with strict isolation...
cd backend

REM Remove existing venv if it exists
if exist "venv" (
    echo ℹ️  Removing existing virtual environment...
    rmdir /s /q venv
)

REM Create isolated virtual environment
echo ℹ️  Creating isolated virtual environment...
%PYTHON_CMD% -m venv venv
if !errorlevel! neq 0 (
    echo ❌ Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Verify isolation
for /f "tokens=*" %%i in ('where python') do (
    echo %%i | find "venv" >nul
    if !errorlevel! neq 0 (
        echo ❌ Virtual environment not properly isolated!
        exit /b 1
    )
)

echo ✅ Virtual environment isolated

REM Upgrade pip and install dependencies
echo ℹ️  Installing backend dependencies...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if exist "requirements-dev.txt" (
    echo ℹ️  Installing development dependencies...
    pip install -r requirements-dev.txt --quiet
)

REM Setup .env file
if not exist ".env" (
    if exist "env.example" (
        copy env.example .env >nul
        echo ✅ .env file created from template
    )
)

REM Create backend data directories
mkdir data\documents 2>nul
mkdir data\cache 2>nul

echo ✅ Backend setup complete
call deactivate
cd ..

REM Setup frontend with proper isolation
echo ℹ️  Setting up frontend with proper isolation...

REM Check Node.js
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Node.js not found! Install Node.js 18+ from nodejs.org
    exit /b 1
)

for /f "tokens=1 delims=v" %%v in ('node --version') do (
    for /f "tokens=1 delims=." %%a in ("%%v") do (
        set node_major=%%a
        if !node_major! lss 18 (
            echo ❌ Node.js 18+ required. Found: %%v
            exit /b 1
        )
        echo ✅ Using Node.js %%v
    )
)

cd frontend

REM Install dependencies locally (not globally)
echo ℹ️  Installing frontend dependencies...
npm install --no-audit --no-fund --quiet
if !errorlevel! neq 0 (
    echo ❌ Frontend dependencies installation failed
    exit /b 1
)

echo ✅ Frontend setup complete
cd ..

REM Verify setup
echo ℹ️  Verifying setup...
cd backend
call venv\Scripts\activate.bat

REM Quick import test
python -c "import sys; print(f'Python: {sys.version.split()[0]}'); print(f'Location: {sys.executable}'); from src.api.main import app; print('✅ Backend imports working')"
if !errorlevel! neq 0 (
    echo ❌ Backend import test failed
    exit /b 1
)

call deactivate
cd ..

REM Check frontend
cd frontend
if not exist "node_modules" (
    echo ❌ Frontend dependencies not installed!
    exit /b 1
)
cd ..

echo ✅ Setup verification complete

REM Show completion message
echo.
echo 🚀 JOB APPLICATION HELPER - SETUP
echo ============================================
echo ✅ Setup completed successfully!
echo.
echo ℹ️  Next steps:
echo   1. Start backend: cd backend ^&^& venv\Scripts\activate.bat ^&^& python start_api.py
echo   2. Start frontend: cd frontend ^&^& npm run dev
echo   3. Or use: launch_app.bat (starts both)
echo.
echo ⚠️  All environments are properly isolated!

pause 