#!/bin/bash

# MedExtract Quick Start Script
# Just starts the application without asking questions

echo "Starting MedExtract..."
echo ""

# Detect docker compose command
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Detect GPU
if nvidia-smi >/dev/null 2>&1; then
    echo "[OK] GPU detected - using optimized configuration"
    COMPOSE_FILE="deployment/docker-compose.yml"
else
    echo "[INFO] No GPU - using CPU configuration"
    COMPOSE_FILE="deployment/docker-compose.cpu.yml"
fi

# Change to parent directory
cd "$(dirname "$0")/.." || exit 1

# Create directories
mkdir -p data output checkpoints logs 2>/dev/null

# Create .env if doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env 2>/dev/null || true
    elif [ -f deployment/.env.example ]; then
        cp deployment/.env.example .env 2>/dev/null || true
    fi
fi

# Start services
echo ""
echo "Starting services..."
$DOCKER_COMPOSE -f $COMPOSE_FILE up -d --build

# Wait for services
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check health
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo ""
    echo "[OK] MedExtract is ready!"
    echo ""
    echo "Open in your browser: http://localhost:3000"
    echo ""
    echo "All extraction features (RAG, models, etc.) are configured in the web interface"
    echo ""
    echo "To stop: $DOCKER_COMPOSE down"
    echo "To view logs: $DOCKER_COMPOSE logs -f"
    
    # Try to open browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:3000 2>/dev/null || true
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:3000 2>/dev/null || true
    fi
else
    echo ""
    echo "[WARNING] Services are still starting. Please wait a moment and refresh http://localhost:3000"
    echo ""
    echo "To check status: $DOCKER_COMPOSE logs"
fi