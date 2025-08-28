# Job Application Helper

AI-powered job application assistant with document processing and chat interface. Documents stored locally, AI processing via external APIs.

## Quick Start

**Note:** Ensure Docker Desktop is running before executing Docker commands.

### Option 1: Docker with Cloud AI Providers (Recommended)
```bash
git clone <repository-url>
cd job_application_helper
docker compose up -d
```
**Access:** http://localhost:8080 → Configure API keys in the web interface

### Option 2: Docker with Local GPU Ollama
```bash
git clone <repository-url>
cd job_application_helper

# Setup local Ollama with GPU support
./setup-local-ollama.sh

# Start application (connects to local Ollama)
docker compose up -d
```
**Access:** http://localhost:8080

### Option 3: Local Setup (No Docker)
```bash
git clone <repository-url>
cd job_application_helper
./setup.sh       # Linux/macOS
setup.bat        # Windows
./launch_app.sh  # Linux/macOS
launch_app.bat   # Windows
```

**Access:** http://localhost:8080

## Requirements

- **Docker** (recommended) OR **Python 3.9+** + **Node.js 18+**
- **API Keys** (for cloud providers) - configure via web interface
- **Ollama** (for local LLMs) - automatically installed by setup script

## Setup Options

**Note:** Ensure Docker Desktop is running before executing Docker commands.

### Docker with Cloud AI Providers (Recommended)
```bash
docker compose up -d
```

### Docker with Local GPU Ollama
```bash
# Setup local Ollama with GPU acceleration
./setup-local-ollama.sh

# Start application with Docker
docker compose up -d
```

### Container Scripts
```bash
./start-containers.sh         # Start with health checks
./start-containers.sh stop    # Stop containers
./start-containers.sh logs    # View logs
```

### Local Setup (No Docker)
```bash
./setup.sh && ./launch_app.sh  # Linux/macOS
setup.bat && launch_app.bat    # Windows
```

### Manual Setup
```bash
# Backend
cd backend
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp env.example .env
python start_api.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Troubleshooting

- **Docker not running:** Start Docker Desktop before running Docker commands
- **Port conflicts:** Check ports 8000/8080 are free
- **Python/Node missing:** Install required versions
- **API keys:** Optional for testing, add via web interface
- **Ollama not found:** Run `./setup-local-ollama.sh` to install and configure
- **Slow Ollama responses:** Ensure GPU support is enabled (check `ollama list` output)
- **First query delay:** Initial requests may take ~25 seconds for model warmup

**Logs:** `docker compose logs` or `./start-containers.sh logs`

## AI Provider Notes

**Cloud AI Providers (Recommended):**
- **OpenAI**: GPT-4 models with reasoning capabilities
- **Anthropic**: Claude models with reasoning capabilities  
- **Mistral AI**: High-performance models with reasoning capabilities
- **Novita**: GPT-OSS-20B model with reasoning capabilities

**Local LLMs (Ollama):**
- **GPU Support**: Local Ollama with GPU acceleration (Apple Silicon/NVIDIA)
- **Models**: Gemma 3 (1B), Llama 3.2 (1B) - lightweight and fast

## Features
- Upload PDF, DOCX, TXT files
- AI chat with document context using multiple providers:
  - **Cloud AI**: OpenAI (GPT-4), Anthropic (Claude), Mistral AI, Novita (GPT-OSS-20B)
  - **Local LLMs**: Ollama (Gemma 3, Llama 3.2) with GPU acceleration
- Local document storage and processing
- Encrypted API key storage
- Vector-based document search and retrieval

## Privacy & Data Handling

**Local Storage:**
- All uploaded documents are stored locally on your machine
- No documents are uploaded to external services
- Database and files remain on your system

**AI API Usage:**
- AI chat functionality requires sending your messages and relevant document excerpts to external LLM providers (OpenAI, etc.)
- This is essential for the app's core functionality but means some content is processed by external services
- OpenAI states they don't use API data for training, but review their privacy policy for current terms
- The app cannot function without these AI API calls

**Recommendation:** Avoid including highly sensitive information (SSNs, passwords, etc.) in documents or chat, as this content may be sent to AI providers.

## License
Apache License 2.0 — see [LICENSE](LICENSE) 