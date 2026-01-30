#!/bin/bash
# Run the Leverage Simulator Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
USE_LOCAL=false
for arg in "$@"; do
    case $arg in
        --local)
            USE_LOCAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local    Run with local Python venv instead of Docker"
            echo "  --help     Show this help message"
            exit 0
            ;;
    esac
done

if [ "$USE_LOCAL" = true ]; then
    # Run with local venv
    echo "Running with local Python environment..."

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi

    echo "Starting Leverage Simulator..."
    echo "Open http://localhost:8501 in your browser"
    echo ""
    streamlit run app/main.py --server.headless true
else
    # Run with Docker (default)
    echo "Starting Leverage Simulator with Docker..."
    echo "Open http://localhost:8501 in your browser"
    echo ""

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed or not in PATH"
        echo "Install Docker or use --local flag to run with Python venv"
        exit 1
    fi

    # Use docker compose
    docker compose up --build
fi
