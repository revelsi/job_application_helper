#!/bin/bash

# Job Application Helper - UV Launch Script
# Launches both backend and frontend using UV for the backend

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
    echo -e "${GREEN}JOB APPLICATION HELPER - UV LAUNCHER${NC}"
    echo "========================================="
    echo "Starting application with UV backend"
    echo "========================================="
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "README.md" ] || [ ! -d "backend" ] || [ ! -d "frontend" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Check UV installation
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "UV not found! Please run setup-uv.sh first or install UV manually:"
        print_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    
    # Add to PATH if needed
    export PATH="$HOME/.local/bin:$PATH"
    
    local uv_version=$(uv --version | cut -d' ' -f2)
    print_status "Using UV $uv_version"
}

# Check if backend is set up
check_backend() {
    if [ ! -f "backend/pyproject.toml" ]; then
        print_error "Backend not set up with UV! Please run setup-uv.sh first"
        exit 1
    fi
    
    if [ ! -d "backend/.venv" ]; then
        print_error "Backend virtual environment not found! Please run setup-uv.sh first"
        exit 1
    fi
}

# Check if frontend is set up
check_frontend() {
    if [ ! -d "frontend/node_modules" ]; then
        print_error "Frontend not set up! Please run setup-uv.sh first"
        exit 1
    fi
}

# Kill any existing processes
cleanup() {
    print_info "Cleaning up any existing processes..."
    
    # Kill any existing backend processes
    pkill -f "start_api.py" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    
    # Kill any existing frontend processes  
    pkill -f "vite" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    
    sleep 1
}

# Start backend with UV
start_backend() {
    print_info "Starting backend with UV..."
    
    cd backend
    
    # Start backend in background with UV
    uv run python start_api.py &
    BACKEND_PID=$!
    
    cd ..
    
    # Wait a moment for backend to start
    sleep 3
    
    # Check if backend is running
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_status "Backend started successfully (PID: $BACKEND_PID)"
    else
        print_error "Backend failed to start"
        exit 1
    fi
}

# Start frontend
start_frontend() {
    print_info "Starting frontend..."
    
    cd frontend
    
    # Start frontend in background
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
    
    # Wait a moment for frontend to start
    sleep 3
    
    # Check if frontend is running
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_status "Frontend started successfully (PID: $FRONTEND_PID)"
    else
        print_error "Frontend failed to start"
        exit 1
    fi
}

# Show running info
show_info() {
    echo ""
    print_status "🚀 Application is running!"
    echo ""
    print_info "URLs:"
    print_info "  Frontend: http://localhost:5173"
    print_info "  Backend:  http://localhost:8000"
    print_info "  API Docs: http://localhost:8000/docs"
    echo ""
    print_info "Processes:"
    print_info "  Backend PID:  $BACKEND_PID"
    print_info "  Frontend PID: $FRONTEND_PID"
    echo ""
    print_warning "Press Ctrl+C to stop both services"
}

# Handle shutdown
shutdown() {
    echo ""
    print_info "Shutting down services..."
    
    # Kill backend
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        print_status "Backend stopped"
    fi
    
    # Kill frontend
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        print_status "Frontend stopped"
    fi
    
    # Additional cleanup
    pkill -f "start_api.py" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    
    print_status "Application stopped"
    exit 0
}

# Set up signal handling
trap shutdown SIGINT SIGTERM

# Main execution
main() {
    print_banner
    check_directory
    check_uv
    check_backend
    check_frontend
    cleanup
    start_backend
    start_frontend
    show_info
    
    # Wait for user to stop
    while true; do
        sleep 1
    done
}

# Run main function
main "$@"
