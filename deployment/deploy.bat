@echo off
REM MedExtract Deployment Script for Windows

echo MedExtract Deployment Script
echo ================================

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed.
    echo Please install Docker Desktop and ensure it's running.
    echo Visit: https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)

REM Check for GPU support (NVIDIA)
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo GPU detected - using GPU-enabled configuration
    set COMPOSE_FILE=docker-compose.yml
) else (
    echo No GPU detected - using CPU configuration
    set COMPOSE_FILE=docker-compose.cpu.yml
)

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

REM Show menu
echo.
echo Select deployment mode:
echo 1) Development mode (with logs)
echo 2) Production mode (background)
echo 3) Quick start
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Starting in development mode...
    docker-compose -f %COMPOSE_FILE% up --build
) else if "%choice%"=="2" (
    echo Starting in production mode...
    docker-compose -f %COMPOSE_FILE% up -d --build
    echo.
    echo MedExtract is running!
    echo Frontend: http://localhost:3000
    echo Backend API: http://localhost:8000
    echo.
    echo To view logs: docker-compose logs -f
    echo To stop: docker-compose down
) else if "%choice%"=="3" (
    echo Starting quick deployment...
    docker-compose -f %COMPOSE_FILE% up -d
    echo.
    echo MedExtract is running!
    echo Opening browser...
    timeout /t 5 /nobreak >nul
    start http://localhost:3000
) else (
    echo Invalid choice.
    pause
    exit /b 1
)

echo.
echo Deployment complete!
pause