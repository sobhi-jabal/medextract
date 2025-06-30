#!/bin/bash

# MedExtract Setup Checker
# This script checks if your system is ready for MedExtract deployment

echo "🔍 MedExtract Setup Checker"
echo "=========================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Initialize status
all_good=true

# Check Docker
echo -n "✓ Checking Docker... "
if command_exists docker && docker version >/dev/null 2>&1; then
    echo "OK ($(docker version --format 'v{{.Server.Version}}')"
else
    echo "NOT FOUND"
    echo "  → Please install Docker Desktop: https://docs.docker.com/get-docker/"
    all_good=false
fi

# Check Docker Compose
echo -n "✓ Checking Docker Compose... "
if docker compose version >/dev/null 2>&1; then
    echo "OK (plugin)"
elif command_exists docker-compose; then
    echo "OK (standalone)"
else
    echo "NOT FOUND"
    echo "  → Docker Compose should be included with Docker Desktop"
    all_good=false
fi

# Check GPU
echo -n "✓ Checking GPU support... "
if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "NVIDIA GPU found ($gpu_name)"
    echo "  → Will use GPU-optimized configuration"
else
    echo "No GPU detected"
    echo "  → Will use CPU-optimized configuration"
fi

# Check available memory
echo -n "✓ Checking available memory... "
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    total_mem=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    total_mem=$(free -g | awk '/^Mem:/ {print $2}')
else
    total_mem="Unknown"
fi
echo "${total_mem}GB"
if [[ "$total_mem" =~ ^[0-9]+$ ]] && [ "$total_mem" -lt 8 ]; then
    echo "  → Warning: Less than 8GB RAM. Consider using smaller models."
fi

# Check disk space
echo -n "✓ Checking disk space... "
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    free_space=$(df -g . | awk 'NR==2 {print $4}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    free_space=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
else
    free_space="Unknown"
fi
echo "${free_space}GB free"
if [[ "$free_space" =~ ^[0-9]+$ ]] && [ "$free_space" -lt 10 ]; then
    echo "  → Warning: Less than 10GB free space. Models require ~5-10GB."
fi

# Check ports
echo -n "✓ Checking required ports... "
ports_in_use=""
for port in 3000 8000 11434; do
    if lsof -i :$port >/dev/null 2>&1 || netstat -an | grep -q ":$port "; then
        ports_in_use="$ports_in_use $port"
    fi
done
if [ -z "$ports_in_use" ]; then
    echo "All clear"
else
    echo "CONFLICT"
    echo "  → Ports in use:$ports_in_use"
    echo "  → Stop conflicting services or modify docker-compose.yml"
    all_good=false
fi

echo ""
if [ "$all_good" = true ]; then
    echo "✅ Your system is ready for MedExtract!"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./deploy.sh"
    echo "2. Select option 3 (Quick start)"
    echo "3. Wait for services to start"
    echo "4. Open http://localhost:3000"
else
    echo "❌ Please fix the issues above before deploying."
fi