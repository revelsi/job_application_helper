# Job Application Helper

AI-powered job application assistant with document processing and chat interface. Documents stored locally, AI processing via external APIs.

## Quick Start

**Note:** Ensure Docker Desktop is running before executing Docker commands.

### Option 1: Docker with Cloud AI Providers (Recommended)
```bash
git clone <repository-url>
cd job_application_helper

# Setup Docker environment (creates necessary directories and configs)
./setup-docker-env.sh        # Linux/macOS
setup-docker-env.bat         # Windows

# Start application
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

### Option 3: Local Setup with UV (No Docker)
```bash
git clone <repository-url>
cd job_application_helper
./setup-uv.sh       # Linux/macOS
setup.bat           # Windows
./launch_app-uv.sh  # Linux/macOS
launch_app.bat      # Windows
```

**Access:** http://localhost:5173

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

### Backend Scripts
```bash
cd backend
./run_scripts.sh test         # Run all tests
./run_scripts.sh test -v      # Run tests with verbose output
./run_scripts.sh check        # Run system check
./run_scripts.sh status       # Check system status
```

### Local Setup with UV (No Docker)
```bash
./setup-uv.sh && ./launch_app-uv.sh  # Linux/macOS
setup.bat && launch_app.bat          # Windows
```

### Manual Setup with UV
```bash
# Backend
cd backend
uv sync
cp env.example .env
uv run python start_api.py

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
- **OpenAI**: GPT-5-mini with reasoning capabilities
- **Mistral AI**: Small/Medium models with function calling
- **Novita**: Open-source models (GPT-OSS-20B, Qwen3-32B, GLM-4.5)

**Local LLMs (Ollama):**
- **GPU Support**: Local Ollama with GPU acceleration (Apple Silicon/NVIDIA)
- **Models**: Gemma 3 (1B), Llama 3.2 (1B) - lightweight and fast

## Features
- Upload PDF, DOCX, TXT files
- AI chat with document context using multiple providers:
  - **Cloud AI**: OpenAI (GPT-5-mini), Mistral AI, Novita (open-source models)
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