#!/bin/bash
# Run backend scripts and tests with UV (modern Python package manager)

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

# Function to show usage
show_usage() {
    echo "Usage: $0 <command> [arguments...]"
    echo ""
    echo "Commands:"
    echo "  script <script_name> [args...]  Run a Python script"
    echo "  test [pytest_args...]           Run tests"
    echo "  check                           Run system check script"
    echo "  status                          Run status script"
    echo "  list-docs                       List documents"
    echo "  evaluate                        Run system evaluation"
    echo ""
    echo "Examples:"
    echo "  $0 script scripts/check_system.py"
    echo "  $0 test tests/test_credentials.py"
    echo "  $0 test -v"
    echo "  $0 check"
}

# Main execution
case "${1:-}" in
    script)
        if [ $# -lt 2 ]; then
            echo "❌ Script name required"
            show_usage
            exit 1
        fi
        shift
        echo "🔧 Running script with UV..."
        uv run python "$@"
        ;;
    test)
        shift
        echo "🧪 Running tests with UV..."
        uv run python -m pytest tests "$@"
        ;;
    check)
        echo "🔍 Running system check..."
        uv run python scripts/check_system.py
        ;;
    status)
        echo "📊 Running status check..."
        uv run python scripts/status.py
        ;;
    list-docs)
        echo "📄 Listing documents..."
        uv run python scripts/list_documents.py
        ;;
    evaluate)
        echo "📈 Running system evaluation..."
        uv run python scripts/evaluate_system.py
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 