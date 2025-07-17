@echo off
setlocal enabledelayedexpansion

REM Job Application Helper - Container Management Script (Windows)
REM Convenient wrapper for Docker Compose operations

echo 🚀 JOB APPLICATION HELPER - CONTAINER LAUNCHER
echo ==============================================
echo Docker-based deployment with guaranteed consistency
echo ==============================================

REM Check if Docker is installed and running
:check_docker
echo ℹ️  Checking Docker installation...

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker first.
    echo 💡 Visit: https://docs.docker.com/get-docker/
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker first.
    exit /b 1
)

REM Check for Docker Compose (both old and new syntax)
where docker-compose >nul 2>&1
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if !errorlevel! neq 0 (
        echo ❌ Docker Compose is not installed. Please install Docker Compose first.
        echo 💡 Visit: https://docs.docker.com/compose/install/
        exit /b 1
    )
)

echo ✅ Docker and Docker Compose are available
goto :check_files

REM Check if required files exist
:check_files
echo ℹ️  Checking required files...

if "%COMPOSE_FILE%"=="" set COMPOSE_FILE=docker-compose.yml

if not exist "%COMPOSE_FILE%" (
    echo ❌ %COMPOSE_FILE% not found
    exit /b 1
)

if not exist "backend\Dockerfile" (
    echo ❌ backend\Dockerfile not found
    exit /b 1
)

if not exist "frontend\Dockerfile" (
    echo ❌ frontend\Dockerfile not found
    exit /b 1
)

echo ✅ All required files found
goto :setup_env

REM Setup environment file
:setup_env
echo ℹ️  Setting up environment configuration...

if not exist "backend\.env" (
    if exist "backend\env.example" (
        echo ℹ️  Creating .env file from template...
        copy "backend\env.example" "backend\.env" >nul
        echo ⚠️  Please configure your API keys in backend\.env
    ) else (
        echo ❌ backend\env.example not found
        exit /b 1
    )
)

echo ✅ Environment configuration ready
goto :setup_data

REM Create data directories
:setup_data
echo ℹ️  Creating data directories...
mkdir data\documents 2>nul
mkdir data\cache 2>nul

echo ✅ Data directories created
goto :start_containers

REM Start containers
:start_containers
echo ℹ️  Starting containers...

REM Build and start services
docker-compose -f "%COMPOSE_FILE%" up --build -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start containers
    exit /b 1
)

echo ✅ Containers started successfully

REM Wait for services to be healthy
echo ℹ️  Waiting for services to be ready...

REM Wait for backend (simplified check)
timeout /t 30 /nobreak >nul
docker-compose -f "%COMPOSE_FILE%" ps backend | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo ✅ Backend is healthy
) else (
    echo ⚠️  Backend may still be starting up
)

REM Wait for frontend (simplified check)
timeout /t 10 /nobreak >nul
docker-compose -f "%COMPOSE_FILE%" ps frontend | findstr "healthy" >nul
if %errorlevel% equ 0 (
    echo ✅ Frontend is healthy
) else (
    echo ⚠️  Frontend may still be starting up
)

goto :show_status

REM Show status
:show_status
echo ℹ️  Container Status:
docker-compose -f "%COMPOSE_FILE%" ps

echo.
echo ℹ️  Service URLs:
echo    Frontend: http://localhost:8080
echo    Backend API: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
goto :open_browser

REM Open browser
:open_browser
echo ℹ️  Opening browser...
start http://localhost:8080
goto :end

REM Stop containers
:stop_containers
echo ℹ️  Stopping containers...
docker-compose -f "%COMPOSE_FILE%" down
echo ✅ Containers stopped
goto :end

REM Clean up containers and images
:cleanup
echo ℹ️  Cleaning up containers and images...
docker-compose -f "%COMPOSE_FILE%" down --rmi all --volumes --remove-orphans
echo ✅ Cleanup completed
goto :end

REM Show logs
:show_logs
set service=%2
if "%service%"=="" (
    docker-compose -f "%COMPOSE_FILE%" logs -f
) else (
    docker-compose -f "%COMPOSE_FILE%" logs -f %service%
)
goto :end

REM Show help
:show_help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   start     Start all containers (default)
echo   stop      Stop all containers
echo   restart   Restart all containers
echo   status    Show container status
echo   logs      Show logs (optionally for specific service)
echo   cleanup   Stop containers and remove images
echo   help      Show this help message
echo.
echo Environment Variables:
echo   COMPOSE_FILE  Docker Compose file to use (default: docker-compose.yml)
echo.
echo Examples:
echo   %0                                                  # Start all containers
echo   %0 start                                            # Start all containers
echo   set COMPOSE_FILE=docker-compose.prod.yml && %0     # Start with production config
echo   %0 logs backend                                     # Show backend logs
echo   %0 stop                                             # Stop all containers
echo   %0 cleanup                                          # Clean up everything
goto :end

REM Main execution
set command=%1
if "%command%"=="" set command=start

if "%command%"=="start" (
    goto :check_docker
) else if "%command%"=="stop" (
    goto :stop_containers
) else if "%command%"=="restart" (
    call :stop_containers
    timeout /t 2 /nobreak >nul
    goto :check_docker
) else if "%command%"=="status" (
    goto :show_status
) else if "%command%"=="logs" (
    goto :show_logs
) else if "%command%"=="cleanup" (
    goto :cleanup
) else if "%command%"=="help" (
    goto :show_help
) else (
    echo ❌ Unknown command: %command%
    goto :show_help
)

:end
echo.
echo ✅ Application started successfully!
echo ℹ️  Use 'docker-compose logs' to view logs
echo ℹ️  Use '%0 stop' to stop all containers 