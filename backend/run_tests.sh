#!/bin/bash
# Run tests with correct Python environment and paths

# Ensure we're in the backend directory
cd "$(dirname "$0")"

# Set Python path to current directory so 'src' imports work
export PYTHONPATH=.

# Use the Python from the virtual environment
python -m pytest tests "$@" 