#!/bin/bash

# Job Application Helper - Simple Setup Script
# Sets up the simplified job application helper without RAG complexity

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[OK] $1${NC}"; }
print_warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
print_error() { echo -e "${RED}[ERROR] $1${NC}"; }
print_info() { echo -e "${BLUE}[INFO] $1${NC}"; }

print_banner() {
    echo -e "${GREEN}JOB APPLICATION HELPER - SIMPLIFIED SETUP${NC}"
    echo "============================================"
    echo "Setting up simplified job application helper"
    echo "Features: Document storage, AI chat, local only"
    echo "============================================"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Find Python 3.9+
find_python() {
    print_info "Searching for Python 3.9+..."
    
    for cmd in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v $cmd &> /dev/null; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [ -n "$version" ]; then
                local major=$(echo $version | cut -d. -f1)
                local minor=$(echo $version | cut -d. -f2)
                if [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; then
                    print_status "Found Python $version: $cmd"
                    PYTHON_CMD=$cmd
                    return 0
                fi
            fi
        fi
    done
    
    print_error "Python 3.9+ is required but not found!"
    print_error ""
    print_error "Please install Python 3.9+ first:"
    print_error "  macOS:    brew install python@3.9 (or newer)"
    print_error "  Ubuntu:   sudo apt install python3.9 python3.9-venv (or newer)"
    print_error "  Windows:  Download from python.org"
    print_error ""
    print_error "Note: This script creates a virtual environment but does not install Python itself."
    print_error "Python must be installed at the system level first."
    exit 1
}

# Setup backend with strict isolation
setup_backend() {
    print_info "Setting up backend with strict isolation..."
    
    # Find Python
    find_python
    
    cd backend
    
    # Create isolated virtual environment
    if [ -d "venv" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    print_info "Creating isolated virtual environment..."
    print_info "Using: $PYTHON_CMD -m venv venv"
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Verify isolation
    if [[ "$(which python)" != *"venv"* ]]; then
        print_error "Virtual environment not properly isolated!"
        exit 1
    fi
    
    print_status "Virtual environment isolated: $(which python)"
    
    # Upgrade pip in isolation
    python -m pip install --upgrade pip --quiet
    
    # Install runtime dependencies in isolation
    print_info "Installing runtime dependencies..."
    pip install -r requirements.txt --quiet
    
    # Ask about development dependencies
    echo ""
    print_info "Development dependencies include testing, linting, and documentation tools."
    print_info "They are NOT required to run the application."
    echo ""
    read -p "Install development dependencies? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-dev.txt" ]; then
            print_info "Installing development dependencies..."
            pip install -r requirements-dev.txt --quiet
            print_status "Development dependencies installed"
        fi
    else
        print_info "Skipping development dependencies (recommended for users)"
    fi
    
    # Setup .env file
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_status ".env file created from template"
        fi
    fi
    
    # Create data directories
    mkdir -p data/{documents,cache}
    
    print_status "Backend setup complete"
    cd ..
}

# Setup frontend with proper isolation
setup_frontend() {
    print_info "Setting up frontend with proper isolation..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js 18+ is required but not found!"
        print_error ""
        print_error "Please install Node.js 18+ first:"
        print_error "  macOS:    brew install node@18"
        print_error "  Ubuntu:   sudo apt install nodejs npm"
        print_error "  Windows:  Download from nodejs.org"
        print_error ""
        print_error "Recommended: Use Node Version Manager (nvm) for better version control:"
        print_error "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
        print_error "  nvm install 18"
        print_error ""
        print_error "Note: This script installs packages locally but does not install Node.js itself."
        exit 1
    fi
    
    local node_version=$(node --version | cut -d'v' -f2)
    local major=$(echo $node_version | cut -d'.' -f1)
    
    if [ "$major" -lt 18 ]; then
        print_error "Node.js 18+ is required. Found: $node_version"
        print_error ""
        print_error "Please upgrade Node.js:"
        print_error "  Using nvm: nvm install 18 && nvm use 18"
        print_error "  Or download latest from nodejs.org"
        exit 1
    fi
    
    print_status "Using Node.js $node_version"
    
    cd frontend
    
    # Install dependencies locally (not globally) - this is Node.js's "virtual environment"
    print_info "Installing frontend dependencies locally in node_modules/..."
    npm install --no-audit --no-fund --quiet
    
    print_status "Frontend setup complete"
    print_info "Packages installed in: $(pwd)/node_modules"
    cd ..
}

# Create root data directories
setup_data() {
    print_info "Creating data directories..."
    mkdir -p data/{documents,cache}
    print_status "Data directories created"
}

# Verify complete setup
verify_setup() {
    print_info "Verifying setup..."
    
    # Check backend
    cd backend
    source venv/bin/activate
    
    # Quick import test (without exposing API keys)
    python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
print(f'Location: {sys.executable}')
try:
    from src.api.main import app
    print('[OK] Backend imports working')
except Exception as e:
    print(f'[ERROR] Backend import error: {e}')
    sys.exit(1)
" || exit 1
    
    deactivate
    cd ..
    
    # Check frontend
    cd frontend
    if [ ! -d "node_modules" ]; then
        print_error "Frontend dependencies not installed!"
        exit 1
    fi
    cd ..
    
    print_status "Setup verification complete"
}

# Show completion message
show_completion() {
    print_banner
    print_status "Setup completed successfully!"
    echo ""
    print_info "Environment Details:"
    print_info "  Backend:  Python virtual environment in backend/venv/"
    print_info "  Frontend: Node.js packages in frontend/node_modules/"
    print_info "  Data:     Local storage in data/ directory"
    echo ""
    print_info "Next steps:"
    print_info "  1. Start backend: cd backend && source venv/bin/activate && python start_api.py"
    print_info "  2. Start frontend: cd frontend && npm run dev"
    print_info "  3. Or use: ./launch_app.sh (starts both)"
    echo ""
    print_warning "All environments are properly isolated!"
    print_warning "No system-wide packages were installed!"
}

# Main execution
main() {
    print_banner
    check_directory
    setup_data
    setup_backend
    setup_frontend
    verify_setup
    show_completion
}

# Run main function
main "$@" 