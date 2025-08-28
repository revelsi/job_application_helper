#!/bin/bash

# Setup Local Ollama with GPU Support for Job Application Helper
# This script installs and configures Ollama locally for optimal GPU performance

set -e

echo "🚀 Setting up local Ollama with GPU support for Job Application Helper..."

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
    OLLAMA_VERSION=$(ollama --version)
    echo "   Version: $OLLAMA_VERSION"
else
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Add Ollama to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ Ollama installed successfully"
fi

# Check if Ollama service is running
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama service is already running"
else
    echo "🚀 Starting Ollama service..."
    # Start Ollama in background
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    
    # Verify Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama service started successfully"
    else
        echo "❌ Failed to start Ollama service"
        exit 1
    fi
fi

# Check GPU support
echo "🔍 Checking GPU support..."
if ollama list 2>&1 | grep -q "Metal"; then
    echo "✅ GPU support detected (Apple Silicon Metal)"
    GPU_SUPPORT=true
elif ollama list 2>&1 | grep -q "CUDA"; then
    echo "✅ GPU support detected (NVIDIA CUDA)"
    GPU_SUPPORT=true
else
    echo "⚠️  No GPU support detected - will run on CPU"
    GPU_SUPPORT=false
fi

# Download required models
echo "📥 Downloading required models..."

echo "   Downloading gemma3:1b (815MB)..."
ollama pull gemma3:1b

echo "   Downloading llama3.2:1b (1.3GB)..."
ollama pull llama3.2:1b

# Verify models are downloaded
echo "🔍 Verifying models..."
ollama list

echo ""
echo "✅ Local Ollama setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Run 'docker compose up -d' to start the application"
echo "   2. Access the app at http://localhost:8080"
echo "   3. Ollama will use ${GPU_SUPPORT:+GPU}${GPU_SUPPORT:-CPU} for optimal performance"
echo ""
echo "💡 Tips:"
echo "   - Ollama service will auto-start on system boot"
echo "   - Models are cached locally for faster loading"
echo "   - For Docker CPU-only mode, use: docker compose -f docker-compose.yml -f docker-compose.cpu-ollama.yml up -d"
echo ""
echo "🔧 Management commands:"
echo "   - Stop Ollama: pkill -f 'ollama serve'"
echo "   - Start Ollama: ollama serve"
echo "   - List models: ollama list"
echo "   - Remove model: ollama rm <model_name>"
