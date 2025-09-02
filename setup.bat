@echo off
setlocal enabledelayedexpansion

REM Job Application Helper - UV Windows Setup Script
REM Modern setup using UV package manager for faster, more reliable builds

echo 🚀 JOB APPLICATION HELPER - UV SETUP
echo ============================================
echo Setting up with UV (ultra-fast Python package manager)
echo Features: 10-100x faster installs, better dependency resolution
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

REM Install UV if not present
echo ℹ️  Checking for UV package manager...
uv --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ℹ️  Installing UV package manager...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if !errorlevel! neq 0 (
        echo ❌ Failed to install UV. Please install manually:
        echo Visit: https://docs.astral.sh/uv/getting-started/installation/
        exit /b 1
    )
)

echo ✅ UV package manager ready

REM Create root data directories
echo ℹ️  Creating data directories...
mkdir data\documents 2>nul
mkdir data\cache 2>nul
echo ✅ Data directories created

REM Create Docker secrets directory for production deployments (optional)
echo ℹ️  Creating Docker secrets directory...
mkdir backend\.secrets 2>nul
echo ✅ Docker secrets directory created
echo ℹ️  To use custom encryption keys in Docker, add them to backend\.secrets\encryption_key

REM Setup backend with UV
echo ℹ️  Setting up backend with UV...
cd backend

REM UV automatically manages Python versions and virtual environments
echo ℹ️  Installing dependencies with UV (this may take a moment)...
uv sync --no-dev
echo ✅ Production dependencies installed

REM Ask about development dependencies
echo.
echo ℹ️  Development dependencies include testing, linting, and documentation tools.
echo ℹ️  They are NOT required to run the application.
echo.
set /p install_dev="Install development dependencies? (y/N): "
if /i "!install_dev!"=="y" (
    uv sync --extra dev
    echo ✅ Development dependencies installed
) else (
    echo ℹ️  Skipping development dependencies (recommended for users)
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

REM Quick import test using UV
uv run python -c "import sys; print(f'Python: {sys.version.split()[0]}'); print(f'Location: {sys.executable}'); from src.api.main import app; print('✅ Backend imports working with UV')"
if !errorlevel! neq 0 (
    echo ❌ Backend import test failed
    exit /b 1
)

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
echo 🎉 SETUP COMPLETED SUCCESSFULLY WITH UV!
echo.
echo ℹ️  Environment Details:
echo ℹ️    Backend:  UV-managed virtual environment in backend/.venv/
echo ℹ️    Frontend: Node.js packages in frontend/node_modules/
echo ℹ️    Data:     Local storage in data/ directory
echo.
echo ℹ️  Next steps:
echo ℹ️    1. Start backend: cd backend ^&^& uv run python start_api.py
echo ℹ️    2. Start frontend: cd frontend ^&^& npm run dev
echo ℹ️    3. Or use: launch_app.bat (starts both with UV)
echo.
echo ⚠️  UV Benefits:
echo ⚠️    ✅ 10-100x faster package installs
echo ⚠️    ✅ Better dependency resolution
echo ⚠️    ✅ Built-in Python version management
echo ⚠️    ✅ Lockfile for reproducible builds

pause 