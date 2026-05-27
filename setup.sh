#!/bin/bash
# OCR Pipeline API Setup Script

set -e

echo "🚀 OCR Pipeline API - Setup Script"
echo "===================================="

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3.11 --version 2>&1 || python3 --version 2>&1)
echo "✓ Found: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv || python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate || . .venv/Scripts/activate 2>/dev/null || true
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e .
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ No pyproject.toml or requirements.txt found!"
    exit 1
fi
echo "✓ Dependencies installed"

# Setup environment file
echo ""
echo "Setting up environment..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
    fi
else
    echo "✓ .env already exists"
fi

# Create model directories
echo ""
echo "Creating model directories..."
mkdir -p model_path/text_detection
mkdir -p model_path/text_recognition
echo "✓ Model directories created"

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Review and update .env file if needed"
echo "2. Activate venv: source .venv/bin/activate"
echo "3. Start server: python -m uvicorn app.main:app --reload"
echo "4. Visit: http://localhost:8000/api/docs"
echo ""
echo "📚 For more information, see QUICKSTART.md or README.md"
