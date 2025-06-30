# MedExtract Quick Start Guide

## 🚀 Start in 30 Seconds

### 1. Install Docker Desktop
- **Mac/Windows**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Docker is likely already installed

### 2. Run MedExtract
```bash
./quick_start.sh
```

That's it! The browser will open automatically at http://localhost:3000

## 📊 How to Use

1. **Upload your data** - Drag & drop a CSV or Excel file with medical reports
2. **Select text column** - Choose which column contains the medical text
3. **Configure extraction** - Use a preset or create custom datapoints:
   - **Preset: Indications & Impressions** - Extract clinical findings
   - **Preset: NIH Grants** - Extract diseases, organs, AI techniques
   - **Preset: BT-RADS** - Extract medication status
4. **Click Start Extraction** - Watch real-time progress
5. **Download results** - Get CSV or Excel file with extracted data

## 🎛️ All Features Are in the UI

No need to configure anything in Docker! Everything is adjustable in the web interface:
- ✅ RAG (Retrieval-Augmented Generation) toggle
- ✅ Model selection (phi3:mini, tinyllama, etc.)
- ✅ Temperature and other LLM settings
- ✅ Extraction strategy (single vs multi-call)
- ✅ Few-shot examples
- ✅ Custom datapoint configuration

## 🛑 Stop MedExtract

```bash
docker compose down
```

## ❓ Need Help?

- **Check if running**: http://localhost:8000/health
- **View logs**: `docker compose logs`
- **Full documentation**: See DOCKER_README.md

---

**Note**: First run will take ~5 minutes to download the AI model. Subsequent runs start in seconds.