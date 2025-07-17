@echo off
REM Job Application Helper - Unified Launch Script for Windows
REM This script provides easy launching of backend and frontend services

setlocal enabledelayedexpansion

REM Configuration
set BACKEND_PORT=8000
set FRONTEND_PORT=8080

echo 🚀 JOB APPLICATION HELPER - UNIFIED LAUNCHER
echo ============================================================
echo Modern AI-powered job application assistance
echo Backend: FastAPI + Document Processing ^| Frontend: React + TypeScript
echo ============================================================

REM Function to check if a port is in use
:check_port
set port=%1
netstat -an | findstr ":%port% " | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Function to check prerequisites
:check_prerequisites
echo ℹ️  Checking prerequisites...

REM Check virtual environment
if not exist "backend\venv" (
    echo ❌ Virtual environment not found. Please run setup first.
    echo 💡 Run: setup.bat
    exit /b 1
)

REM Check Node.js for frontend
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Node.js not found. Frontend will not be available.
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo ⚠️  Frontend dependencies not installed. Run: cd frontend ^&^& npm install
    exit /b 1
)

echo ✅ Prerequisites check passed
exit /b 0

REM Function to start backend
:start_backend
echo ℹ️  Starting backend server...

REM Check if backend is already running
call :check_port %BACKEND_PORT%
if %errorlevel% equ 0 (
    echo ⚠️  Backend already running on port %BACKEND_PORT%
    exit /b 0
)

REM Activate virtual environment and start backend
call backend\venv\Scripts\activate.bat
cd backend

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating from template...
    if exist "env.example" (
        copy env.example .env >nul
        echo ℹ️  .env file created from template
        echo ℹ️  API keys are optional for initial setup
    ) else (
        echo ⚠️  env.example not found. API keys may not be loaded.
    )
)

REM Start the API server
echo ✅ Backend starting on http://localhost:%BACKEND_PORT%
start /B python start_api.py
timeout /t 3 /nobreak >nul

REM Check if backend started successfully
call :check_port %BACKEND_PORT%
if %errorlevel% equ 0 (
    echo ✅ Backend started successfully
) else (
    echo ❌ Backend failed to start
    cd ..
    exit /b 1
)

cd ..
exit /b 0

REM Function to start frontend
:start_frontend
echo ℹ️  Starting frontend server...

REM Check if frontend is already running
call :check_port %FRONTEND_PORT%
if %errorlevel% equ 0 (
    echo ⚠️  Frontend already running on port %FRONTEND_PORT%
    exit /b 0
)

cd frontend

REM Start frontend development server
echo ✅ Frontend starting on http://localhost:%FRONTEND_PORT%
start /B npm run dev
timeout /t 5 /nobreak >nul

REM Check if frontend started successfully
call :check_port %FRONTEND_PORT%
if %errorlevel% equ 0 (
    echo ✅ Frontend started successfully
) else (
    echo ❌ Frontend failed to start
    cd ..
    exit /b 1
)

cd ..
exit /b 0

REM Function to stop services
:stop_services
echo ℹ️  Stopping services...

REM Kill processes on backend port
for /f "tokens=5" %%i in ('netstat -ano ^| findstr ":%BACKEND_PORT%"') do taskkill /PID %%i /F >nul 2>&1

REM Kill processes on frontend port
for /f "tokens=5" %%i in ('netstat -ano ^| findstr ":%FRONTEND_PORT%"') do taskkill /PID %%i /F >nul 2>&1

echo ✅ Services stopped
exit /b 0

REM Function to show status
:show_status
echo ℹ️  Service Status:

call :check_port %BACKEND_PORT%
if %errorlevel% equ 0 (
    echo ✅ Backend: Running on http://localhost:%BACKEND_PORT%
    echo    - API Docs: http://localhost:%BACKEND_PORT%/docs
    echo    - Health: http://localhost:%BACKEND_PORT%/health
) else (
    echo ⚠️  Backend: Not running
)

call :check_port %FRONTEND_PORT%
if %errorlevel% equ 0 (
    echo ✅ Frontend: Running on http://localhost:%FRONTEND_PORT%
) else (
    echo ⚠️  Frontend: Not running
)
exit /b 0

REM Function to show help
:show_help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   start     Start both backend and frontend (default)
echo   stop      Stop all services
echo   restart   Restart all services
echo   status    Show service status
echo   help      Show this help message
echo.
echo Examples:
echo   %0         # Start all services
echo   %0 start   # Start all services
echo   %0 stop    # Stop all services
echo   %0 status  # Show status
exit /b 0

REM Main execution
set command=%1
if "%command%"=="" set command=start

if "%command%"=="start" (
    call :check_prerequisites
    if %errorlevel% neq 0 exit /b 1
    
    call :start_backend
    if %errorlevel% neq 0 exit /b 1
    
    call :start_frontend
    if %errorlevel% neq 0 exit /b 1
    
    echo.
    echo ✅ Application started successfully!
    echo.
    echo ℹ️  Access the application:
    echo    Frontend: http://localhost:%FRONTEND_PORT%
    echo    Backend API: http://localhost:%BACKEND_PORT%
    echo    API Documentation: http://localhost:%BACKEND_PORT%/docs
    echo.
    echo ℹ️  API keys are optional for initial setup
    echo ℹ️  Add them through the web interface when ready
    echo.
    echo ℹ️  Opening browser...
    start http://localhost:%FRONTEND_PORT%
    echo.
    echo Press Ctrl+C to stop all services
    pause >nul
) else if "%command%"=="stop" (
    call :stop_services
    echo ✅ All services stopped
) else if "%command%"=="restart" (
    call :stop_services
    timeout /t 2 /nobreak >nul
    call %0 start
) else if "%command%"=="status" (
    call :show_status
) else if "%command%"=="help" (
    call :show_help
) else (
    echo ❌ Unknown command: %command%
    call :show_help
    exit /b 1
) 