#!/bin/bash
# Jenga-AI Testing Environment Setup Script
# ===========================================
# Sets up a complete testing environment with all dependencies

set -e  # Exit on error

echo "========================================================================"
echo "  JENGA-AI TESTING ENVIRONMENT SETUP"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo -e "${RED}Error: Please run this script from the Jenga-AI root directory${NC}"
    exit 1
fi

echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "  Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${YELLOW}  ⚠️  Warning: Python 3.9+ recommended, but we'll proceed with $PYTHON_VERSION${NC}"
    echo "  Some features may not work optimally"
else
    echo -e "${GREEN}  ✓ Python version is sufficient${NC}"
fi

echo ""
echo "Step 2: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  Virtual environment already exists"
    read -p "  Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        python3 -m venv .venv
        echo -e "${GREEN}  ✓ Created fresh virtual environment${NC}"
    fi
else
    python3 -m venv .venv
    echo -e "${GREEN}  ✓ Created virtual environment${NC}"
fi

echo ""
echo "Step 3: Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}  ✓ Activated .venv${NC}"

echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}  ✓ Pip upgraded${NC}"

echo ""
echo "Step 5: Installing PyTorch (CPU version)..."
echo "  This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
echo -e "${GREEN}  ✓ PyTorch installed${NC}"

echo ""
echo "Step 6: Installing Transformers and core dependencies..."
pip install transformers datasets accelerate > /dev/null 2>&1
echo -e "${GREEN}  ✓ Transformers installed${NC}"

echo ""
echo "Step 7: Installing additional dependencies..."
pip install \
    numpy pandas \
    scikit-learn \
    pyyaml \
    tqdm \
    psutil \
    pytest pytest-cov \
    mlflow \
    tensorboard \
    > /dev/null 2>&1
echo -e "${GREEN}  ✓ Additional dependencies installed${NC}"

echo ""
echo "Step 8: Installing PEFT and LLM dependencies..."
pip install peft bitsandbytes > /dev/null 2>&1 || echo -e "${YELLOW}  ⚠️  Warning: Some LLM dependencies failed (bitsandbytes requires GPU)${NC}"
echo -e "${GREEN}  ✓ PEFT installed${NC}"

echo ""
echo "Step 9: Installing Jenga-AI in editable mode..."
pip install -e . > /dev/null 2>&1
echo -e "${GREEN}  ✓ Jenga-AI installed${NC}"

echo ""
echo "Step 10: Creating test directories..."
mkdir -p tests/{unit,integration,performance,configs,data,utils}
mkdir -p docs/testing
echo -e "${GREEN}  ✓ Test directories created${NC}"

echo ""
echo "Step 11: Making test scripts executable..."
chmod +x tests/environment_check.py
chmod +x tests/unit/test_imports.py
chmod +x tests/utils/create_synthetic_data.py
chmod +x tests/utils/memory_monitor.py
echo -e "${GREEN}  ✓ Scripts made executable${NC}"

echo ""
echo "Step 12: Running environment check..."
python tests/environment_check.py

echo ""
echo "========================================================================"
echo "  SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Review the testing plan: less todo.md"
echo "  2. Read the quick start: less QUICK_START_TESTING.md"
echo "  3. Generate test data: python tests/utils/create_synthetic_data.py --all"
echo "  4. Run import tests: python tests/unit/test_imports.py"
echo ""
echo "Happy testing! 🧪"
echo "========================================================================"


