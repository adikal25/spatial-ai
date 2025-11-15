#!/bin/bash
# Render startup script for self-correcting-vlm-qa

# Set the working directory to the project root
cd "$(dirname "$0")"

# Add the project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Print debug info
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "PORT: $PORT"

# Start the FastAPI server
exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
