#!/bin/bash

# Job Application Helper - Docker Environment Setup
# This script helps set up the Docker environment with proper encryption keys

echo "🔐 Job Application Helper - Docker Environment Setup"
echo "=================================================="

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p data/documents data/cache
    chmod 755 data/
    echo "   ✅ Data directory created"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp backend/env.example .env
    echo "   ✅ .env file created"
fi

# Generate encryption key if not set
if ! grep -q "ENCRYPTION_KEY=" .env || grep -q "ENCRYPTION_KEY=your_encryption_key_here" .env; then
    echo "🔑 Generating encryption key..."
    
    # Generate a new Fernet key using Python
    ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "")
    
    if [ -z "$ENCRYPTION_KEY" ]; then
        echo "   ❌ Failed to generate encryption key. Please install cryptography:"
        echo "   pip install cryptography"
        echo "   Then run this script again."
        exit 1
    fi
    
    # Update .env file
    if grep -q "ENCRYPTION_KEY=" .env; then
        sed -i.bak "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
    else
        echo "ENCRYPTION_KEY=$ENCRYPTION_KEY" >> .env
    fi
    
    echo "   ✅ Encryption key generated and saved to .env"
else
    echo "🔑 Encryption key already configured"
fi

echo ""
echo "✅ Docker environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your OpenAI API key to .env file:"
echo "   OPENAI_API_KEY=your_actual_api_key_here"
echo ""
echo "2. Start the application:"
echo "   docker compose up -d"
echo ""
echo "3. Access the application at http://localhost:8080"
echo ""
echo "💡 Your encryption key is safely stored in .env file"
echo "   Back up this file to preserve access to your data!" 