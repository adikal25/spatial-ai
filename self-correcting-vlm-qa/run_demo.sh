#!/bin/bash
# Simple script to run the demo

echo "🔍 Starting Self-Correcting VLM QA Demo..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Running setup..."
    ./setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f "config/.env" ]; then
    echo "❌ Error: config/.env not found!"
    echo "Please copy config/.env.example to config/.env and add your Anthropic API key"
    exit 1
fi

# Start API in background
echo "Starting API server..."
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 3

# Start Streamlit demo
echo "Starting Streamlit demo..."
echo ""
echo "✅ Demo will open in your browser at http://localhost:8501"
echo ""
streamlit run demo/app.py

# Cleanup - kill API server when Streamlit exits
kill $API_PID 2>/dev/null
