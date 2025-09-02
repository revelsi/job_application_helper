@echo off
setlocal enabledelayedexpansion

REM Job Application Helper - Docker Environment Setup (Windows)
REM This script helps set up the Docker environment with proper encryption keys

echo 🔐 Job Application Helper - Docker Environment Setup
echo ====================================================

REM Create data directory if it doesn't exist
if not exist "data" (
    echo 📁 Creating data directory...
    mkdir data\documents 2>nul
    mkdir data\cache 2>nul
    echo    ✅ Data directory created
)

REM Create Docker secrets directory for production deployments (optional)
if not exist "backend\.secrets" (
    echo 🔐 Creating Docker secrets directory...
    mkdir backend\.secrets 2>nul
    echo    ✅ Docker secrets directory created
    echo    💡 To use custom encryption keys in Docker, add them to backend\.secrets\encryption_key
)

REM Check if .env file exists
if not exist ".env" (
    if exist "backend\env.example" (
        echo 📝 Creating .env file from template...
        copy backend\env.example .env >nul
        echo    ✅ .env file created
    )
)

REM Generate encryption key if not set
findstr /C:"ENCRYPTION_KEY=" .env >nul 2>&1
if !errorlevel! neq 0 (
    echo 🔑 Generating encryption key...
    
    REM Generate a new Fernet key using Python
    for /f "delims=" %%i in ('python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2^>nul') do set ENCRYPTION_KEY=%%i
    
    if "!ENCRYPTION_KEY!"=="" (
        echo    ❌ Failed to generate encryption key. Please install cryptography:
        echo    pip install cryptography
        echo    Then run this script again.
        pause
        exit /b 1
    )
    
    REM Update .env file
    findstr /C:"ENCRYPTION_KEY=" .env >nul 2>&1
    if !errorlevel! equ 0 (
        powershell -Command "(Get-Content .env) -replace 'ENCRYPTION_KEY=.*', 'ENCRYPTION_KEY=!ENCRYPTION_KEY!' | Set-Content .env"
    ) else (
        echo ENCRYPTION_KEY=!ENCRYPTION_KEY!>> .env
    )
    
    echo    ✅ Encryption key generated and saved to .env
) else (
    echo 🔑 Encryption key already configured
)

echo.
echo ✅ Docker environment setup complete!
echo.
echo Next steps:
echo 1. Add your OpenAI API key to .env file:
echo    OPENAI_API_KEY=your_actual_api_key_here
echo.
echo 2. Start the application:
echo    docker compose up -d
echo.
echo 3. Access the application at http://localhost:8080
echo.
echo 💡 Your encryption key is safely stored in .env file
echo    Back up this file to preserve access to your data!

pause
