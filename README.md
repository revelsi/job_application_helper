# Job Application Helper

AI-powered job application assistant with document processing and chat interface. Documents stored locally, AI processing via external APIs.

## Quick Start

**Docker (Recommended):**
```bash
git clone <repository-url>
cd job_application_helper
docker compose up --build
```

**Local Setup:**
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

- **Docker** (easiest) OR **Python 3.9+** + **Node.js 18+**

## Setup Options

### Docker (Recommended)
```bash
docker compose up --build
```

### Container Scripts
```bash
./start-containers.sh         # Start with health checks
./start-containers.sh stop    # Stop containers
./start-containers.sh logs    # View logs
```

### Local Development
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

- **Docker not running:** Start Docker Desktop
- **Port conflicts:** Check ports 8000/8080 are free
- **Python/Node missing:** Install required versions
- **API keys:** Optional for testing, add via web interface

**Logs:** `docker compose logs` or `./start-containers.sh logs`

## Features
- Upload PDF, DOCX, TXT files
- AI chat with document context
- Local document storage and processing
- Encrypted API key storage

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