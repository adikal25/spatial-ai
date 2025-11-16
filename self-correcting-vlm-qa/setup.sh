#!/bin/bash
# Setup script for Self-Correcting VLM QA

echo "🔍 Self-Correcting Vision-Language QA - Setup"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Set up environment file
echo "Setting up environment configuration..."
if [ ! -f "config/.env" ]; then
    cp config/.env.example config/.env
    echo "✅ Created config/.env from template"
    echo "⚠️  Please edit config/.env and add your API keys"
else
    echo "✅ config/.env already exists"
fi
echo ""

# Download MiDaS model (cache it)
echo "Downloading MiDaS model (this may take a few minutes)..."
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_Small', pretrained=True)" 2>&1 | grep -v "Downloading"
echo "✅ MiDaS model downloaded"
echo ""

echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config/.env and add your Anthropic API key"
echo "   ANTHROPIC_API_KEY=your_key_here"
echo "   HUGGINGFACEHUB_API_TOKEN=your_hf_token"
echo "   ENABLE_TRIPO_RECONSTRUCTION=true   # disable if you don't need meshes"
echo ""
echo "2. Run the demo:"
echo "   ./run_demo.sh"
echo ""
echo "   Or manually:"
echo "   - Terminal 1: make run-api"
echo "   - Terminal 2: make run-demo"
echo ""
echo "For help: make help"
echo "=============================================="
