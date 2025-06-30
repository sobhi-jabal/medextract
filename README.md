# MedExtract - Medical Report Information Extraction

A comprehensive toolkit for extracting structured information from unstructured medical reports using LLMs.

## Directory Structure

```
medextract/
├── medextract-ui/              # NEW: Web-based UI application (actively developed)
│   ├── frontend/              # React/Next.js frontend
│   ├── backend/               # FastAPI backend with unified extraction pipeline
│   ├── docker-compose.yml     # Docker configuration for the UI app
│   └── run_test.sh           # Quick test script (no LLM required)
│
├── medextract-ui_backup_2/     # BACKUP: Original job-based architecture (untouched)
│
├── original_examples/          # Example extraction scripts
│   ├── ds_btrads_c1_6_NO_rag.py        # BT-RADS extraction with optional RAG
│   ├── llm_updated_allbatches_*.py      # Indications/impressions extraction
│   └── nih_ai_rad_all.py               # NIH grant analysis extraction
│
├── deployment/                 # Docker deployment configurations
│   ├── docker-compose.yml     # Main Docker configuration
│   ├── docker-compose.cpu.yml # CPU-only configuration
│   ├── deploy.sh              # Interactive deployment script
│   ├── quick_start.sh         # Quick start script
│   └── check_setup.sh         # System requirements checker
│
├── docs/                       # Documentation
│   ├── DOCKER_README.md       # Docker deployment guide
│   └── QUICK_START.md         # Quick start guide
│
├── medextract.py              # Original CLI tool
├── requirements.txt           # Python dependencies for CLI
├── setup.py                   # Package setup for CLI
└── config/                    # Configuration files
    ├── config.yaml
    └── default_config.yaml
```

## Quick Start

### Option 1: Test the Web UI (Recommended)
```bash
cd medextract-ui
./run_test.sh
```
Then open http://localhost:3000 in your browser.

### Option 2: Full Deployment with LLM
```bash
cd deployment
./deploy.sh
```

### Option 3: Use Original CLI
```bash
pip install -r requirements.txt
python medextract.py --help
```

## Components

### medextract-ui/ (New Web Application)
- **Purpose**: Modern web interface for medical text extraction
- **Features**:
  - Drag-and-drop file upload
  - Configure extraction parameters
  - Real-time progress tracking
  - Export results as CSV/Excel
  - Support for RAG, few-shot examples, and multiple LLM models
- **Status**: Actively developed, replaces job-based architecture with direct extraction

### original_examples/
- **Purpose**: Reference implementations for different extraction tasks
- **Contents**:
  - BT-RADS assessment extraction
  - Medical report indications/impressions extraction
  - NIH grant analysis extraction
- **Usage**: These scripts demonstrate various extraction patterns and can be run standalone

### deployment/
- **Purpose**: Docker-based deployment configurations
- **Features**:
  - Automatic GPU/CPU detection
  - Ollama integration for local LLM inference
  - One-command deployment
- **Usage**: Run `./deploy.sh` from the deployment directory

## Development Status

- **Active Development**: `medextract-ui/` - The new web-based extraction tool
- **Stable**: Original CLI tool (`medextract.py`) and example scripts
- **Archived**: `medextract-ui_backup_2/` - Original job-based architecture

## Getting Help

1. Check the documentation in `docs/`
2. For UI issues, see `medextract-ui/README.md`
3. For deployment issues, see `deployment/check_setup.sh`

## License

See LICENSE file for details.