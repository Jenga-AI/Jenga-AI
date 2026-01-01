# Jenga-AI Testing Quick Start Guide

**Get testing in under 5 minutes!** 🚀

## ⚡ Quick Setup

### 1. Install Dependencies
```bash
# Option A: Using pip
pip install torch transformers datasets numpy pandas scikit-learn pyyaml tqdm psutil

# Option B: Using uv (faster)
uv pip install torch transformers datasets numpy pandas scikit-learn pyyaml tqdm psutil

# Option C: CPU-only PyTorch (lighter)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets numpy pandas scikit-learn pyyaml tqdm psutil
```

### 2. Verify Setup
```bash
python3 tests/environment_check.py
```
Should show ✓ for all critical dependencies.

### 3. Run Tests
```bash
# Quick unit tests (2-3 minutes)
python3 tests/run_test_suite.py --unit-only

# Include integration tests (5-10 minutes)  
python3 tests/run_test_suite.py --all

# Verbose output to see details
python3 tests/run_test_suite.py --unit-only --verbose
```

## 🧪 Individual Test Examples

### Test Data Loading
```bash
python3 tests/unit/test_data_loading.py
```
**Tests:** CSV, JSON, JSONL file loading, malformed data handling

### Test Model Components  
```bash
python3 tests/unit/test_attention_fusion.py
python3 tests/unit/test_multitask_model.py
```
**Tests:** Attention fusion mechanism, multi-task model architecture

### Test Training Pipeline
```bash
python3 tests/integration/test_single_task_training.py
```
**Tests:** End-to-end training for sentiment analysis and NER

### Test Configuration
```bash
python3 tests/unit/test_config_validation.py
```
**Tests:** YAML config parsing, validation, error handling

## 🔍 What Each Test Validates

| Test File | Purpose | Key Validations |
|-----------|---------|----------------|
| `test_imports.py` | Module imports | All components load correctly |
| `test_data_loading.py` | Data formats | CSV/JSON/JSONL loading, error handling |
| `test_data_preprocessing.py` | Tokenization | Text processing, batching, edge cases |
| `test_tasks.py` | Task definitions | Classification, NER, multi-label tasks |
| `test_attention_fusion.py` | Fusion mechanism | Attention weights, gradient flow |
| `test_multitask_model.py` | Model architecture | Forward pass, device compatibility |
| `test_config_validation.py` | Configuration | YAML parsing, validation, defaults |
| `test_single_task_training.py` | Training pipeline | End-to-end training, inference |

## 🐛 Expected Results (Without Full Setup)

**If dependencies are missing**, you'll see import errors like:
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:** Install dependencies as shown above.

**If dependencies are installed**, you should see:
```
✓ All tests passed! Framework is working correctly.
```

**If some tests fail**, this is expected and valuable! Each failure represents a potential bug in the framework.

## 📊 Success Metrics

### Excellent Results (80%+ pass rate)
- Most core functionality working
- Only minor edge case failures
- Framework is production-ready

### Good Results (50-80% pass rate)  
- Core components functional
- Some integration issues to fix
- Framework is development-ready

### Poor Results (<50% pass rate)
- Major setup or dependency issues
- Need to check installation
- Framework needs significant fixes

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Check Python version (need 3.9+)
python3 --version

# Verify transformers installation
python3 -c "import transformers; print(transformers.__version__)"

# Install missing packages
pip install <missing_package>
```

**Memory Issues**
```bash
# Check available RAM
free -h

# Use smaller batch sizes in configs
# Our tests are designed for 16GB+ RAM
```

**Slow Performance**
```bash
# Use CPU-only mode (tests are CPU-optimized)
# Tests use bert-tiny (4.4M params) for speed
# Expected: 2-3 mins for unit tests, 5-10 mins for integration
```

## 🎯 Next Steps After Testing

### If Tests Pass
1. **Run Full Suite**: Execute all test categories
2. **Generate Report**: `python3 tests/run_test_suite.py --json results.json`
3. **Document Findings**: Any edge cases or performance notes
4. **Contribute**: Create GitHub issues for improvements

### If Tests Fail
1. **Document Failures**: Note which specific tests fail and why
2. **Check Dependencies**: Ensure all required packages installed
3. **Create Issues**: Submit bug reports with reproduction steps
4. **Fix & Retest**: Address issues and re-run tests

## 📈 Advanced Usage

### Performance Profiling
```bash
# Monitor memory during tests
htop  # In separate terminal while tests run

# Generate detailed reports
python3 tests/run_test_suite.py --all --json detailed_report.json
```

### Custom Test Execution
```bash
# Skip slow tests
python3 tests/run_test_suite.py --fast

# Only run specific categories
python3 tests/run_test_suite.py --unit-only
python3 tests/run_test_suite.py --integration
```

## 🤝 Contributing Test Results

After running tests, please share your results! This helps improve the framework:

1. **Test Report**: Share pass/fail statistics
2. **System Info**: OS, Python version, RAM, CPU
3. **Bug Reports**: Any failures with reproduction steps
4. **Performance Data**: Memory usage, execution time
5. **Suggestions**: Ideas for additional tests or improvements

---

**Happy Testing!** 🧪

This testing framework is designed to systematically validate Jenga-AI's multi-task learning capabilities and help identify areas for improvement. Your testing contribution helps make the framework more robust and reliable for the community.