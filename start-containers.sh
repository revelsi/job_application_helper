#!/bin/bash

# Job Application Helper - Container Management Script
# Convenient wrapper for Docker Compose operations

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
    echo -e "${GREEN}JOB APPLICATION HELPER - CONTAINER LAUNCHER${NC}"
    echo "=============================================="
    echo "Docker-based deployment with guaranteed consistency"
    echo "=============================================="
}

# Helper function to run Docker Compose commands with proper syntax
docker_compose_cmd() {
    local compose_file=${COMPOSE_FILE:-docker-compose.yml}
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$compose_file" "$@"
    else
        docker compose -f "$compose_file" "$@"
    fi
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check for Docker Compose (both old and new syntax)
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_info "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_status "Docker and Docker Compose are available"
}

# Check if required files exist
check_files() {
    local compose_file=${COMPOSE_FILE:-docker-compose.yml}
    
    if [ ! -f "$compose_file" ]; then
        print_error "$compose_file not found"
        exit 1
    fi
    
    if [ ! -f "backend/Dockerfile" ]; then
        print_error "backend/Dockerfile not found"
        exit 1
    fi
    
    if [ ! -f "frontend/Dockerfile" ]; then
        print_error "frontend/Dockerfile not found"
        exit 1
    fi
    
    print_status "All required files found"
}

# Setup environment file
setup_env() {
    if [ ! -f "backend/.env" ]; then
        if [ -f "backend/env.example" ]; then
            print_info "Creating .env file from template..."
            cp backend/env.example backend/.env
            print_warning "Please configure your API keys in backend/.env"
        else
            print_error "backend/env.example not found"
            exit 1
        fi
    fi
    
    print_status "Environment configuration ready"
}

# Create data directories
setup_data() {
    print_info "Creating data directories..."
    mkdir -p data/{documents,cache}
    print_status "Data directories created"
}

# Start containers
start_containers() {
    print_info "Starting containers..."
    
    # Build and start services
    docker_compose_cmd up --build -d
    
    print_status "Containers started successfully"
    
    # Wait for services to be healthy
    print_info "Waiting for services to be ready..."
    
    # Wait for backend
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker_compose_cmd ps backend | grep -q "healthy"; then
            print_status "Backend is healthy"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend failed to start properly"
        docker_compose_cmd logs backend
        exit 1
    fi
    
    # Wait for frontend
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker_compose_cmd ps frontend | grep -q "healthy"; then
            print_status "Frontend is healthy"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Frontend failed to start properly"
        docker_compose_cmd logs frontend
        exit 1
    fi
}

# Show status
show_status() {
    print_info "Container Status:"
    docker_compose_cmd ps
    
    echo ""
    print_info "Service URLs:"
    print_info "  Frontend: http://localhost:8080"
    print_info "  Backend API: http://localhost:8000"
    print_info "  API Documentation: http://localhost:8000/docs"
}

# Stop containers
stop_containers() {
    print_info "Stopping containers..."
    docker_compose_cmd down
    print_status "Containers stopped"
}

# Clean up containers and images
cleanup() {
    print_info "Cleaning up containers and images..."
    docker_compose_cmd down --rmi all --volumes --remove-orphans
    print_status "Cleanup completed"
}

# Show logs
show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        docker_compose_cmd logs -f "$service"
    else
        docker_compose_cmd logs -f
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start all containers (default)"
    echo "  stop      Stop all containers"
    echo "  restart   Restart all containers"
    echo "  status    Show container status"
    echo "  logs      Show logs (optionally for specific service)"
    echo "  cleanup   Stop containers and remove images"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  COMPOSE_FILE  Docker Compose file to use (default: docker-compose.yml)"
    echo ""
    echo "Examples:"
    echo "  $0                                        # Start all containers"
    echo "  $0 start                                  # Start all containers"
    echo "  COMPOSE_FILE=docker-compose.prod.yml $0  # Start with production config"
    echo "  $0 logs backend                           # Show backend logs"
    echo "  $0 stop                                   # Stop all containers"
    echo "  $0 cleanup                                # Clean up everything"
}

# Main execution
case "${1:-start}" in
    start)
        print_banner
        check_docker
        check_files
        setup_env
        setup_data
        start_containers
        echo ""
        print_status "Application started successfully!"
        show_status
        echo ""
        print_info "Opening browser..."
        if command -v open &> /dev/null; then
            open http://localhost:8080
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8080
        fi
        ;;
    stop)
        stop_containers
        ;;
    restart)
        stop_containers
        sleep 2
        $0 start
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    cleanup)
        cleanup
        ;;
    help)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 