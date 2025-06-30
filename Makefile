# MedExtract Makefile
# Simplifies common Docker operations

# Detect docker compose command
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: help build up down logs clean reset dev prod cpu gpu pull-models test health

# Default target
help:
	@echo "MedExtract Docker Commands:"
	@echo "  make build       - Build all Docker images"
	@echo "  make up          - Start all services (auto-detect GPU)"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make clean       - Stop services and remove volumes"
	@echo "  make reset       - Complete reset (removes all data)"
	@echo "  make dev         - Start in development mode"
	@echo "  make prod        - Start in production mode"
	@echo "  make cpu         - Force CPU-only mode"
	@echo "  make gpu         - Force GPU mode"
	@echo "  make pull-models - Download recommended models"
	@echo "  make test        - Run health checks"
	@echo "  make health      - Check service health"

# Build Docker images
build:
	$(DOCKER_COMPOSE) build

# Start services (auto-detect GPU)
up:
	@if nvidia-smi > /dev/null 2>&1; then \
		echo "GPU detected, using GPU configuration"; \
		$(DOCKER_COMPOSE) up -d; \
	else \
		echo "No GPU detected, using CPU configuration"; \
		$(DOCKER_COMPOSE) -f docker-compose.cpu.yml up -d; \
	fi
	@echo "MedExtract is starting..."
	@echo "Frontend: http://localhost:3000"
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

# Stop services
down:
	$(DOCKER_COMPOSE) down

# View logs
logs:
	$(DOCKER_COMPOSE) logs -f

# Clean up (including volumes)
clean:
	$(DOCKER_COMPOSE) down -v
	rm -rf output/* checkpoints/*

# Complete reset
reset: clean
	rm -rf data/* logs/*
	rm -f .env
	cp .env.example .env
	@echo "Reset complete. Run 'make up' to start fresh."

# Development mode with live logs
dev:
	@if nvidia-smi > /dev/null 2>&1; then \
		$(DOCKER_COMPOSE) up --build; \
	else \
		$(DOCKER_COMPOSE) -f docker-compose.cpu.yml up --build; \
	fi

# Production mode (detached)
prod: up
	@echo "Running in production mode"

# Force CPU mode
cpu:
	$(DOCKER_COMPOSE) -f docker-compose.cpu.yml up -d
	@echo "Running in CPU-only mode"

# Force GPU mode
gpu:
	$(DOCKER_COMPOSE) up -d
	@echo "Running in GPU mode"

# Pull recommended models
pull-models:
	@echo "Pulling recommended models..."
	@if nvidia-smi > /dev/null 2>&1; then \
		echo "GPU detected, pulling GPU-optimized models"; \
		docker exec medextract-ollama ollama pull phi4:latest || true; \
		docker exec medextract-ollama ollama pull llama3.2:3b || true; \
	else \
		echo "No GPU detected, pulling CPU-optimized models"; \
		docker exec medextract-ollama ollama pull phi3:mini || true; \
		docker exec medextract-ollama ollama pull tinyllama:latest || true; \
	fi
	@docker exec medextract-ollama ollama list

# Run tests
test: health
	@echo "Running basic extraction test..."
	@curl -X POST http://localhost:8000/extract \
		-H "Content-Type: multipart/form-data" \
		-F "file=@test_data.csv" \
		-F 'config={"text_column":"text","datapoint_configs":[{"name":"test","instruction":"Extract test data"}]}' \
		|| echo "Test failed - ensure services are running and test_data.csv exists"

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "Backend not responding"
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool || echo "Ollama not responding"
	@echo "Health check complete"

# Service-specific logs
logs-backend:
	$(DOCKER_COMPOSE) logs -f backend

logs-frontend:
	$(DOCKER_COMPOSE) logs -f frontend

logs-ollama:
	$(DOCKER_COMPOSE) logs -f ollama

# Restart specific services
restart-backend:
	$(DOCKER_COMPOSE) restart backend

restart-frontend:
	$(DOCKER_COMPOSE) restart frontend

restart-ollama:
	$(DOCKER_COMPOSE) restart ollama

# Shell access
shell-backend:
	docker exec -it medextract-backend /bin/bash

shell-ollama:
	docker exec -it medextract-ollama /bin/sh

# Model management
list-models:
	docker exec medextract-ollama ollama list

pull-model:
	@read -p "Enter model name (e.g., phi3:mini): " model; \
	docker exec medextract-ollama ollama pull $$model

delete-model:
	@read -p "Enter model name to delete: " model; \
	docker exec medextract-ollama ollama rm $$model