#!/bin/bash

# Job Application Helper - Simple Launch Script
# Starts backend and frontend services

set -e

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
    echo -e "${GREEN}JOB APPLICATION HELPER - LAUNCHER${NC}"
    echo "============================================"
    echo "Starting backend and frontend services"
    echo "============================================"
}

# Check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if [ ! -d "backend/venv" ]; then
        print_error "Backend virtual environment not found. Run: ./setup.sh"
        exit 1
    fi
    
    if [ ! -d "frontend/node_modules" ]; then
        print_error "Frontend dependencies not found. Run: ./setup.sh"
        exit 1
    fi
    
    print_status "Prerequisites check passed"
}

# Start backend
start_backend() {
    print_info "Starting backend server..."
    
    if check_port 8000; then
        print_warning "Backend already running on port 8000"
        return 0
    fi
    
    cd backend
    source venv/bin/activate
    
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp env.example .env
    fi
    
    print_status "Backend starting on http://localhost:8000"
    python start_api.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    
    cd ..
    
    # Wait longer for backend to start and add retry mechanism
    print_info "Waiting for backend to start..."
    for i in {1..10}; do
        sleep 2
        if check_port 8000; then
            print_status "Backend started successfully"
            return 0
        fi
        if [ $i -lt 10 ]; then
            print_info "Backend not ready yet, waiting... (attempt $i/10)"
        fi
    done
    
    print_error "Backend failed to start after 20 seconds"
    exit 1
}

# Start frontend
start_frontend() {
    print_info "Starting frontend server..."
    
    if check_port 8080; then
        print_warning "Frontend already running on port 8080"
        return 0
    fi
    
    cd frontend
    
    print_status "Frontend starting on http://localhost:8080"
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    
    cd ..
    sleep 3
    
    if check_port 8080; then
        print_status "Frontend started successfully"
    else
        print_error "Frontend failed to start"
        exit 1
    fi
}

# Stop services
stop_services() {
    print_info "Stopping services..."
    
    # Stop backend
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            print_status "Backend stopped"
        fi
        rm -f backend.pid
    fi
    
    # Stop frontend
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_status "Frontend stopped"
        fi
        rm -f frontend.pid
    fi
    
    # Kill any remaining processes
    if check_port 8000; then
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    fi
    
    if check_port 8080; then
        lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    fi
}

# Show status
show_status() {
    print_info "Service Status:"
    
    if check_port 8000; then
        print_status "Backend: Running on http://localhost:8000"
    else
        print_error "Backend: Not running"
    fi
    
    if check_port 8080; then
        print_status "Frontend: Running on http://localhost:8080"
    else
        print_error "Frontend: Not running"
    fi
}

# Handle script termination
cleanup() {
    echo ""
    print_info "Shutting down services..."
    stop_services
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
case "${1:-start}" in
    start)
        print_banner
        check_prerequisites
        start_backend
        start_frontend
        echo ""
        print_status "All services started successfully!"
        print_info "Backend: http://localhost:8000"
        print_info "Frontend: http://localhost:8080"
        print_info "API Docs: http://localhost:8000/docs"
        echo ""
        print_info "Press Ctrl+C to stop all services"
        
        # Wait for user to stop services
        while true; do
            sleep 1
        done
        ;;
    stop)
        print_info "Stopping all services..."
        stop_services
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        echo "  start   - Start both backend and frontend (default)"
        echo "  stop    - Stop all services"
        echo "  status  - Show service status"
        exit 1
        ;;
esac 