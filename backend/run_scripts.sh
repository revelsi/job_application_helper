#!/bin/bash
# Run backend scripts with UV (modern Python package manager)

# Ensure we're in the backend directory
cd "$(dirname "$0")"

# Add UV to PATH if needed
export PATH="$HOME/.local/bin:$PATH"

# Check if UV is available
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or use legacy approach: PYTHONPATH=. python script.py"
    exit 1
fi

# Run the specified script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name> [arguments...]"
    echo "Example: $0 scripts/check_system.py"
    exit 1
fi

echo "🔧 Running script with UV..."
uv run python "$@" 