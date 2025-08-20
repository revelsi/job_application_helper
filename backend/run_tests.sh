#!/bin/bash
# Run tests with UV (modern Python package manager)

# Ensure we're in the backend directory
cd "$(dirname "$0")"

# Add UV to PATH if needed
export PATH="$HOME/.local/bin:$PATH"

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or use legacy script: python -m pytest tests"
    exit 1
fi

# Run tests with UV (automatically handles virtual environment and Python path)
echo "🧪 Running tests with UV..."
uv run python -m pytest tests "$@" 