#!/bin/bash
# Run backend scripts with correct Python environment and paths

# Ensure we're in the backend directory
cd "$(dirname "$0")"

# Set Python path to current directory so 'src' imports work
export PYTHONPATH=.

# Run the specified script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_name> [arguments...]"
    echo "Example: $0 scripts/check_system.py"
    exit 1
fi

python "$@" 