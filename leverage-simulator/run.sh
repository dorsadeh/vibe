#!/bin/bash
# Run the Leverage Simulator Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run Streamlit
echo "Starting Leverage Simulator..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app/main.py --server.headless true
