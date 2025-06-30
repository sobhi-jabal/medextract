#!/bin/bash

# MedExtract Deployment Script
# This script helps deploy MedExtract with proper configuration

set -e

echo "MedExtract Deployment Script"
echo "============================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    echo "        Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for docker compose (new style) or docker-compose (old style)
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
    echo "[OK] Using Docker Compose (plugin)"
elif command_exists docker-compose; then
    DOCKER_COMPOSE="docker-compose"
    echo "[OK] Using docker-compose (standalone)"
else
    echo "[ERROR] Docker Compose is not installed. Please install Docker Desktop or Docker Compose."
    echo "        Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "[OK] Prerequisites check passed"

# Detect GPU availability
echo ""
echo "Checking for GPU support..."
if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    echo "[OK] NVIDIA GPU detected"
    GPU_AVAILABLE=true
    COMPOSE_FILE="deployment/docker-compose.yml"
else
    echo "[INFO] No GPU detected, will use CPU mode"
    GPU_AVAILABLE=false
    COMPOSE_FILE="deployment/docker-compose.cpu.yml"
fi

# Create necessary directories in parent directory
echo ""
echo "Creating necessary directories..."
cd "$(dirname "$0")/.." || exit 1
mkdir -p data output checkpoints logs

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
    elif [ -f deployment/.env.example ]; then
        cp deployment/.env.example .env
    fi
    
    # Update default model based on GPU availability
    if [ "$GPU_AVAILABLE" = false ] && [ -f .env ]; then
        sed -i.bak 's/DEFAULT_MODEL=.*/DEFAULT_MODEL=phi3:mini/' .env
    fi
fi

# Ask user for deployment mode
echo ""
echo "Select how to start Docker:"
echo "1) Development - Shows logs in terminal, rebuilds on code changes"
echo "2) Production - Runs in background, optimized for daily use"
echo "3) Quick start - Just start everything quickly"
echo ""
echo "Note: All pipeline features (RAG, models, etc.) are configured in the web UI"
read -p "Enter your choice (1-3): " DEPLOY_MODE

case $DEPLOY_MODE in
    1)
        echo "Starting in development mode..."
        $DOCKER_COMPOSE -f $COMPOSE_FILE up --build
        ;;
    2)
        echo "Starting in production mode..."
        $DOCKER_COMPOSE -f $COMPOSE_FILE up -d --build
        echo ""
        echo "[OK] MedExtract is running in production mode!"
        echo "     Frontend: http://localhost:3000"
        echo "     Backend API: http://localhost:8000"
        echo "     Ollama API: http://localhost:11434"
        echo ""
        echo "To view logs: $DOCKER_COMPOSE logs -f"
        echo "To stop: $DOCKER_COMPOSE down"
        ;;
    3)
        echo "Starting quick deployment..."
        # Pull pre-built images if available, otherwise build
        $DOCKER_COMPOSE -f $COMPOSE_FILE up -d
        echo ""
        echo "[OK] MedExtract is running!"
        echo "     Frontend: http://localhost:3000"
        echo ""
        echo "Waiting for services to be ready..."
        sleep 10
        
        # Open browser
        if command_exists xdg-open; then
            xdg-open http://localhost:3000
        elif command_exists open; then
            open http://localhost:3000
        else
            echo "Please open http://localhost:3000 in your browser"
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Show helpful commands
echo ""
echo "Useful commands:"
echo "   View logs:         $DOCKER_COMPOSE logs -f [service]"
echo "   Stop services:     $DOCKER_COMPOSE down"
echo "   Restart service:   $DOCKER_COMPOSE restart [service]"
echo "   Pull new models:   docker exec medextract-ollama ollama pull [model]"
echo "   List models:       docker exec medextract-ollama ollama list"
echo ""
echo "Troubleshooting:"
echo "   If Ollama fails:   $DOCKER_COMPOSE restart ollama"
echo "   If backend fails:  $DOCKER_COMPOSE logs backend"
echo "   Clear all data:    $DOCKER_COMPOSE down -v"