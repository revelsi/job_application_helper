#!/bin/bash

# Job Application Helper - UV Setup Script
# Modern setup using UV package manager for faster, more reliable builds

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
    echo -e "${GREEN}JOB APPLICATION HELPER - UV SETUP${NC}"
    echo "========================================="
    echo "Setting up with UV (ultra-fast Python package manager)"
    echo "Features: 10-100x faster installs, better dependency resolution"
    echo "========================================="
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Install UV if not present
install_uv() {
    if command -v uv &> /dev/null; then
        local uv_version=$(uv --version | cut -d' ' -f2)
        print_status "UV already installed: $uv_version"
        return 0
    fi
    
    print_info "Installing UV package manager..."
    
    # Try different installation methods
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &> /dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        print_error "Neither curl nor wget found. Please install UV manually:"
        print_error "  macOS: brew install uv"
        print_error "  Or visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        local uv_version=$(uv --version | cut -d' ' -f2)
        print_status "UV installed successfully: $uv_version"
    else
        print_error "UV installation failed. Please install manually."
        exit 1
    fi
}

# Setup backend with UV
setup_backend() {
    print_info "Setting up backend with UV..."
    
    cd backend
    
    # UV automatically manages Python versions and virtual environments
    print_info "Installing dependencies with UV (this may take a moment)..."
    
    # Install production dependencies
    uv sync --no-dev
    print_status "Production dependencies installed"
    
    # Ask about development dependencies
    echo ""
    print_info "Development dependencies include testing, linting, and documentation tools."
    print_info "They are NOT required to run the application."
    echo ""
    read -p "Install development dependencies? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv sync --extra dev
        print_status "Development dependencies installed"
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
    
    # Install dependencies locally
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
    
    # Quick import test using UV
    uv run python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
print(f'Location: {sys.executable}')
try:
    from src.api.main import app
    print('[OK] Backend imports working with UV')
except Exception as e:
    print(f'[ERROR] Backend import error: {e}')
    sys.exit(1)
" || exit 1
    
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
    print_status "Setup completed successfully with UV!"
    echo ""
    print_info "Environment Details:"
    print_info "  Backend:  UV-managed virtual environment in backend/.venv/"
    print_info "  Frontend: Node.js packages in frontend/node_modules/"
    print_info "  Data:     Local storage in data/ directory"
    echo ""
    print_info "Next steps:"
    print_info "  1. Start backend: cd backend && uv run python start_api.py"
    print_info "  2. Start frontend: cd frontend && npm run dev"
    print_info "  3. Or use: ./launch_app-uv.sh (starts both with UV)"
    echo ""
    print_warning "UV Benefits:"
    print_warning "  ✅ 10-100x faster package installs"
    print_warning "  ✅ Better dependency resolution"
    print_warning "  ✅ Built-in Python version management"
    print_warning "  ✅ Lockfile for reproducible builds"
}

# Main execution
main() {
    print_banner
    check_directory
    install_uv
    setup_data
    setup_backend
    setup_frontend
    verify_setup
    show_completion
}

# Run main function
main "$@"
