#!/usr/bin/env python3
"""
Google Colab Setup Script for Jenga-AI
======================================
Automatically installs all dependencies and sets up the environment in Google Colab.

Usage in Colab:
    !python setup_colab.py
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("  JENGA-AI GOOGLE COLAB SETUP")
    print("=" * 60)
    print()
    
    # Install core dependencies
    dependencies = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "transformers>=4.35.0",
        "datasets>=2.14.0", 
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("📦 Installing Dependencies...")
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep.split()[0]}"):
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # Install Jenga-AI in development mode
    print()
    run_command("pip install -e .", "Installing Jenga-AI framework")
    
    # Set up environment variables
    print()
    print("🔧 Setting up environment...")
    os.environ['PYTHONPATH'] = os.getcwd()
    
    # Create necessary directories
    directories = ['tests/temp', 'experiments', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    # Verify installation
    print()
    print("🧪 Verifying Installation...")
    
    # Test imports
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
        "import datasets; print(f'Datasets: {datasets.__version__}')",
        "import numpy as np; print(f'NumPy: {np.__version__}')",
        "import pandas as pd; print(f'Pandas: {pd.__version__}')"
    ]
    
    for test_cmd in test_imports:
        try:
            exec(test_cmd)
        except Exception as e:
            print(f"❌ Import test failed: {e}")
    
    # Test GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 GPU not available, using CPU mode")
    except:
        print("❌ Unable to check GPU status")
    
    # Test Jenga-AI imports
    print()
    print("🔍 Testing Jenga-AI Imports...")
    
    try:
        from multitask_bert.core.config import ExperimentConfig
        print("✅ Core config import successful")
    except Exception as e:
        print(f"❌ Core config import failed: {e}")
    
    try:
        from multitask_bert.data.data_processing import DataProcessor
        print("✅ Data processing import successful")
    except Exception as e:
        print(f"❌ Data processing import failed: {e}")
    
    # Run environment check
    print()
    print("🏁 Running Environment Check...")
    if os.path.exists("tests/environment_check.py"):
        run_command("python tests/environment_check.py", "Environment validation")
    else:
        print("⚠️  Environment check script not found")
    
    print()
    print("=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 Next Steps:")
    print("1. Run tests: !python tests/run_test_suite.py --unit-only")
    print("2. Check imports: !python tests/unit/test_imports.py")
    print("3. Start experimenting with the framework!")
    print()

if __name__ == "__main__":
    main()