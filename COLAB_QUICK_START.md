# Jenga-AI Google Colab Quick Start

**Run Jenga-AI testing framework in Google Colab in under 10 minutes!** 🚀

## 📦 Step 1: Upload to Google Drive

1. **Download ZIP**: Download the `jenga-ai-core.zip` file
2. **Upload to Drive**: Upload ZIP to your Google Drive
3. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/)

## 🔧 Step 2: Setup in Colab

Create a new notebook and run these cells:

### Cell 1: Mount Drive and Extract
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract the project
import zipfile
import os

# Update this path to your ZIP file location in Drive
zip_path = '/content/drive/MyDrive/jenga-ai-core.zip'
extract_path = '/content/jenga-ai'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Change to project directory
%cd /content/jenga-ai
!ls -la
```

### Cell 2: Install Dependencies
```python
# Run automated setup
!python setup_colab.py
```

This will automatically:
- Install PyTorch with CUDA support
- Install transformers, datasets, and other dependencies
- Set up the Jenga-AI framework
- Verify GPU availability
- Run environment checks

### Cell 3: Verify Setup
```python
# Quick verification
!python tests/environment_check.py
```

### Cell 4: Run Tests
```python
# Run unit tests (fast - 2-3 minutes)
!python tests/run_test_suite.py --unit-only --verbose

# Or run specific tests
!python tests/unit/test_imports.py
!python tests/unit/test_data_loading.py
```

## 🧪 Available Test Suites

### Unit Tests (Fast - 2-3 minutes)
```python
!python tests/run_test_suite.py --unit-only
```
Tests core components: data loading, model components, configuration

### Integration Tests (Medium - 5-10 minutes)  
```python
!python tests/run_test_suite.py --integration
```
Tests end-to-end training workflows

### All Tests (Complete - 10-15 minutes)
```python
!python tests/run_test_suite.py --all
```
Complete testing suite with detailed reporting

### Individual Test Files
```python
# Data processing tests
!python tests/unit/test_data_loading.py
!python tests/unit/test_data_preprocessing.py

# Model component tests  
!python tests/unit/test_attention_fusion.py
!python tests/unit/test_multitask_model.py

# Configuration tests
!python tests/unit/test_config_validation.py

# Training tests
!python tests/integration/test_single_task_training.py
```

## 📊 Expected Results

### ✅ Success Indicators
```
✅ All dependencies installed
✅ GPU detected (Tesla T4/V100/etc.)
✅ Jenga-AI imports working
✅ Tests passing (80%+ success rate)
```

### ⚠️ Common Issues

**GPU Not Detected**
- Enable GPU: Runtime → Change runtime type → GPU

**Import Errors**  
- Re-run setup: `!python setup_colab.py`
- Check dependencies: `!pip list | grep torch`

**Memory Issues**
- Tests are designed for Colab's ~15GB RAM
- Use `--fast` flag to skip memory-intensive tests

## 🔬 Advanced Usage

### Custom Test Execution
```python
# Generate detailed JSON report
!python tests/run_test_suite.py --all --json test_results.json

# View results
import json
with open('test_results.json', 'r') as f:
    results = json.load(f)
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['total_passed']}")
```

### Memory Monitoring
```python
# Monitor GPU memory during tests
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Run with memory monitoring
!python tests/run_test_suite.py --unit-only --verbose
```

### Training Examples
```python
# Quick training test (uses tiny model)
!python tests/integration/test_single_task_training.py

# View training logs
!find . -name "*.log" -exec head -20 {} \;
```

## 📈 Test Categories

| Category | Command | Duration | Purpose |
|----------|---------|----------|---------|
| Imports | `test_imports.py` | 30s | Validate all modules load |
| Data | `test_data_*.py` | 1-2 min | Test data processing |
| Models | `test_*_model.py` | 2-3 min | Test model components |
| Config | `test_config_*.py` | 1 min | Test configuration |
| Training | `test_*_training.py` | 5-10 min | Test training workflows |

## 🎯 Troubleshooting

### Setup Issues
```python
# Check Python environment
import sys
print(f"Python: {sys.version}")

# Check current directory
import os
print(f"Current dir: {os.getcwd()}")
!ls -la

# Reinstall if needed
!pip install --upgrade torch transformers datasets
```

### Test Failures
```python
# Debug specific test
!python tests/unit/test_imports.py -v

# Check detailed error logs
!python tests/run_test_suite.py --unit-only --verbose 2>&1 | tail -50
```

### Performance Issues
```python
# Check GPU utilization
!nvidia-smi

# Monitor memory during execution
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```

## 🚀 Next Steps After Testing

1. **Analyze Results**: Review test output for any failures
2. **Document Bugs**: Note any issues discovered
3. **Experiment**: Try training custom models
4. **Contribute**: Report findings to improve the framework

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Verify all setup steps were completed
3. Try restarting the Colab runtime
4. Check that GPU is enabled if tests are slow

---

**Happy Testing in Colab!** 🧪

This setup provides a complete Jenga-AI testing environment in Google Colab with GPU acceleration and all necessary dependencies.