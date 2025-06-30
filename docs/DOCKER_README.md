# MedExtract Docker Deployment Guide

MedExtract is a medical text extraction tool that uses LLMs to extract structured data from unstructured medical reports. This guide will help you deploy MedExtract locally using Docker.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- At least 8GB of RAM available
- 10GB of free disk space

### One-Command Deployment

**On macOS/Linux:**
```bash
./deploy.sh
```

**On Windows:**
```cmd
deploy.bat
```

Select option 3 (Quick start) and the application will automatically:
1. Set up all required services
2. Download a small LLM model
3. Open the web interface at http://localhost:3000

## 📋 System Requirements

### Minimum Requirements (CPU Mode)
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

### Recommended Requirements (GPU Mode)
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GPU with 6GB+ VRAM
- Storage: 20GB
- NVIDIA Docker runtime installed

## 🏗️ Architecture

MedExtract consists of three main services:

1. **Frontend**: React/Next.js web interface (port 3000)
2. **Backend**: FastAPI Python server (port 8000)
3. **Ollama**: Local LLM runtime (port 11434)

## 🔧 Manual Setup

### 1. Clone or Download the Repository
```bash
git clone <repository-url>
cd medextract
```

### 2. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to customize:
- `DEFAULT_MODEL`: LLM model to use
- `BATCH_SIZE`: Number of documents to process in parallel
- `USE_RAG_DEFAULT`: Enable/disable RAG by default

### 3. Start Services

**For GPU-enabled systems:**
```bash
docker-compose up -d
```

**For CPU-only systems:**
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

### 4. Access the Application
- Web Interface: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 📊 Using MedExtract

### 1. Upload Data
- Click "Upload Data" or drag-drop a CSV/Excel file
- Select the column containing medical text

### 2. Configure Extraction
Choose a preset or configure custom datapoints:
- **Indications & Impressions**: Extract clinical indications and impressions
- **NIH Grant Analysis**: Extract diseases, organs, and AI techniques
- **BT-RADS Assessment**: Extract medication status and calculate scores

### 3. Start Extraction
- Click "Start Extraction"
- Monitor real-time progress
- Download results as CSV or Excel

## 🛠️ Common Operations

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Stop Services
```bash
docker-compose down
```

### Update Models
```bash
# List available models
docker exec medextract-ollama ollama list

# Pull a new model
docker exec medextract-ollama ollama pull llama3.2:3b
```

### Reset Everything
```bash
# Stop and remove all data
docker-compose down -v
rm -rf data output checkpoints
```

## 🔍 Troubleshooting

### Ollama Connection Issues
```bash
# Restart Ollama
docker-compose restart ollama

# Check Ollama status
curl http://localhost:11434/api/tags
```

### Out of Memory Errors
1. Reduce `BATCH_SIZE` in `.env`
2. Use a smaller model (e.g., `tinyllama:latest`)
3. Increase Docker memory allocation

### Slow Performance
1. Enable GPU support if available
2. Reduce `CHUNK_SIZE` for faster processing
3. Use `single_call` extraction strategy

### Port Conflicts
If ports are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "3001:3000"  # Change frontend port
  - "8001:8000"  # Change backend port
```

## 📦 Available Models

### CPU-Optimized Models
- `phi3:mini` (2.5GB) - Fastest, good accuracy
- `tinyllama:latest` (680MB) - Very fast, basic tasks
- `orca-mini:3b` (1.9GB) - Balanced performance

### GPU-Optimized Models
- `phi4:latest` (3.8GB) - Best accuracy for medical text
- `llama3.2:3b` (2.0GB) - Fast and accurate
- `mixtral:8x7b` (26GB) - Highest accuracy, requires 32GB+ RAM

## 🔐 Security Considerations

- All processing happens locally - no data leaves your machine
- Services are only accessible from localhost by default
- To expose services externally, update firewall rules carefully
- Consider using HTTPS proxy for production deployments

## 📞 Support

For issues or questions:
1. Check the logs: `docker-compose logs`
2. Verify services are healthy: `http://localhost:8000/health`
3. Ensure models are downloaded: `docker exec medextract-ollama ollama list`
4. Create an issue with error logs and system details

## 🎯 Tips for Best Results

1. **Text Quality**: Ensure medical reports are complete and well-formatted
2. **Column Selection**: Choose the column with the most complete text
3. **RAG Settings**: Enable RAG for long documents (>2000 words)
4. **Few-Shot Examples**: Add examples for better extraction accuracy
5. **Batch Size**: Start with 5, increase if system handles it well

## 📈 Performance Tuning

### For Faster Processing
- Use CPU-optimized models
- Disable RAG if not needed
- Increase batch size
- Use `single_call` strategy

### For Better Accuracy
- Use larger models (GPU required)
- Enable RAG
- Add few-shot examples
- Use `multi_call` strategy
- Lower temperature (0.0-0.2)

---

Happy extracting! 🏥✨