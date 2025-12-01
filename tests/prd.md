# Product Requirements Document (PRD)
## Jenga-AI Framework Testing & Validation

---

### 📋 Document Information

**Project**: Jenga-AI Multi-Task NLP Framework Testing  
**Version**: 1.0  
**Date**: November 24, 2025  
**Author**: QA/Testing Contributor  
**Environment**: Ubuntu 22.04/24.04, 16GB RAM, UV Package Manager  
**Status**: In Progress  

---

## 1. Executive Summary

### 1.1 Purpose
This document outlines the comprehensive testing strategy for the Jenga-AI multi-task learning framework. As a testing contributor, the goal is to validate functionality, identify bugs, document issues, and ensure the framework works as advertised for African-context NLP tasks.

### 1.2 Scope
- Validate core framework functionality
- Test multi-task training capabilities
- Verify data processing pipelines
- Document bugs and create actionable GitHub issues
- Ensure 16GB RAM compatibility
- Test with synthetic and real datasets (free only)

### 1.3 Success Criteria
- All core modules import successfully
- Single-task training completes without errors
- Multi-task training (2+ tasks) works correctly
- Memory usage stays under 12GB during training
- Inference produces valid predictions
- Comprehensive bug documentation created

---

## 2. Technical Environment

### 2.1 System Specifications
```yaml
Operating System: Ubuntu 22.04/24.04 LTS
RAM: 16GB
Package Manager: UV (https://github.com/astral-sh/uv)
Python Version: 3.9+
GPU: Optional (CUDA compatible if available)
Storage: ~5GB for models and datasets
```

### 2.2 Development Setup

#### Environment Setup with UV
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Jenga-AI/Jenga-AI.git
cd Jenga-AI

# Create virtual environment with UV
uv venv .venv
source .venv/bin/activate

# Install dependencies with UV
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers datasets pandas numpy scikit-learn
uv pip install pytorch-lightning wandb
uv pip install pytest pytest-cov black flake8

# Install Jenga-AI in editable mode
uv pip install -e .
```

### 2.3 Directory Structure
```
JengaAI/
├── multitask_bert/
│   ├── tasks/
│   ├── data/
│   ├── core/
│   ├── training/
│   ├── deployment/
│   └── analysis/
├── tests/                    # Testing directory (create this)
│   ├── data/                # Test datasets
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── outputs/             # Test outputs
├── docs/                    # Documentation
│   └── testing/            # Testing documentation
└── prd.md                   # This file
```

---

## 3. Testing Strategy

### 3.1 Testing Phases

#### Phase 1: Environment Validation (Days 1-2)
**Objective**: Ensure development environment is properly configured

**Tasks**:
- Verify UV installation and package management
- Test Python environment and dependencies
- Validate GPU availability (if applicable)
- Check RAM and storage capacity
- Test basic Python/PyTorch functionality

**Deliverables**:
- `tests/environment_check.py` - System validation script
- Environment report document

#### Phase 2: Module Import Testing (Days 2-3)
**Objective**: Verify all framework modules can be imported

**Tasks**:
- Test imports for all modules in `multitask_bert/`
- Identify missing dependencies
- Document import errors
- Create dependency requirements file

**Deliverables**:
- `tests/test_imports.py` - Import validation suite
- `requirements-complete.txt` - Full dependency list
- Bug reports for import failures

#### Phase 3: Synthetic Data Testing (Days 3-5)
**Objective**: Test with lightweight synthetic datasets

**Tasks**:
- Create mini datasets (50-100 samples each):
  - Sentiment analysis (Swahili + English)
  - Named Entity Recognition
  - Agricultural disease classification
- Test data loading pipelines
- Verify data preprocessing
- Validate tokenization

**Deliverables**:
- `tests/data/sentiment_mini.csv`
- `tests/data/ner_mini.jsonl`
- `tests/data/agriculture_mini.csv`
- `tests/test_data_loading.py`
- Data pipeline bug reports

#### Phase 4: Single-Task Training (Days 5-7)
**Objective**: Validate training on individual tasks

**Tasks**:
- Test classification task training
- Test NER task training
- Monitor memory usage
- Validate model checkpointing
- Test training with tiny models (bert-tiny, 4M params)

**Deliverables**:
- `tests/test_single_task.py`
- Training logs and metrics
- Memory usage reports
- Performance benchmarks

#### Phase 5: Multi-Task Training (Days 7-10)
**Objective**: Test core multi-task learning functionality

**Tasks**:
- Test 2-task training (sentiment + NER)
- Test 3-task training
- Verify task fusion mechanisms
- Test uncertainty weighting
- Validate shared encoder architecture

**Deliverables**:
- `tests/test_multi_task.py`
- Multi-task training logs
- Fusion mechanism validation report
- Bug reports for multi-task issues

#### Phase 6: Inference & Deployment (Days 10-12)
**Objective**: Test model inference and deployment

**Tasks**:
- Test single sample inference
- Test batch inference
- Validate output formats
- Test model serialization/deserialization
- Verify deployment pipeline

**Deliverables**:
- `tests/test_inference.py`
- Inference performance benchmarks
- Deployment documentation

#### Phase 7: Edge Cases & Stress Testing (Days 12-14)
**Objective**: Test framework robustness

**Tasks**:
- Test with empty datasets
- Test with malformed data
- Test memory limits
- Test with maximum sequence lengths
- Test error handling

**Deliverables**:
- `tests/test_edge_cases.py`
- Stress test report
- Error handling documentation

#### Phase 8: Documentation & Reporting (Days 14-15)
**Objective**: Compile comprehensive testing report

**Tasks**:
- Create master bug report
- Document all test results
- Write testing guide for contributors
- Create GitHub issues for all bugs
- Prepare recommendations

**Deliverables**:
- `docs/testing/TEST_RESULTS.md`
- `docs/testing/BUG_REPORT.md`
- `docs/testing/TESTING_GUIDE.md`
- GitHub issues (with labels: bug, enhancement, documentation)

---

## 4. Detailed Test Specifications

### 4.1 Unit Tests

#### Test Suite 1: Task Definitions
```python
# tests/unit/test_tasks.py
def test_classification_task_creation()
def test_ner_task_creation()
def test_task_label_mapping()
def test_task_serialization()
```

#### Test Suite 2: Data Processing
```python
# tests/unit/test_data.py
def test_csv_data_loading()
def test_jsonl_data_loading()
def test_tokenization()
def test_data_batching()
def test_data_augmentation()
```

#### Test Suite 3: Model Components
```python
# tests/unit/test_model.py
def test_shared_encoder()
def test_task_specific_heads()
def test_fusion_mechanisms()
def test_forward_pass()
```

### 4.2 Integration Tests

#### Test Suite 4: End-to-End Training
```python
# tests/integration/test_training.py
def test_single_task_training_flow()
def test_multi_task_training_flow()
def test_checkpoint_saving()
def test_checkpoint_loading()
def test_training_resume()
```

#### Test Suite 5: Inference Pipeline
```python
# tests/integration/test_inference.py
def test_single_prediction()
def test_batch_prediction()
def test_multi_task_prediction()
def test_inference_speed()
```

### 4.3 Performance Tests

#### Memory Usage Tests
```python
# tests/performance/test_memory.py
def test_training_memory_usage()
def test_inference_memory_usage()
def test_memory_cleanup()
def test_gradient_accumulation()
```

#### Speed Benchmarks
```python
# tests/performance/test_speed.py
def test_training_speed()
def test_inference_speed()
def test_data_loading_speed()
```

---

## 5. Test Data Specifications

### 5.1 Synthetic Datasets

#### Sentiment Analysis Dataset
```yaml
Name: sentiment_mini.csv
Size: 100 samples
Format: CSV
Columns:
  - text: string (Swahili/English mixed)
  - label: int (0=Negative, 1=Positive)
Balance: 50/50 split
Languages: Swahili (40%), English (40%), Code-switched (20%)
```

#### NER Dataset
```yaml
Name: ner_mini.jsonl
Size: 50 samples
Format: JSONL
Fields:
  - text: string
  - tokens: list of strings
  - labels: list of ints
Label Types:
  - 0: O (Other)
  - 1: B-LOC (Location)
  - 2: B-PER (Person)
  - 3: B-THREAT (Security threat)
```

#### Agricultural Classification
```yaml
Name: agriculture_mini.csv
Size: 80 samples
Format: CSV
Columns:
  - text: string (disease descriptions)
  - label: int (0=Healthy, 1=Disease)
Balance: 50/50 split
```

### 5.2 Real Datasets (Free Only)

#### Option 1: Hugging Face Datasets
```python
# Small subsets only (100-200 samples)
datasets = [
    "swahili_news",     # Swahili classification
    "afrisent-sw",      # African sentiment
    "masakhaner",       # African NER
]
```

#### Option 2: Public Kaggle Datasets
```yaml
- Kenya Agricultural Data (if available)
- East African News Sentiment
- Swahili Text Classification
Note: Only download small datasets (<100MB)
```

---

## 6. Resource Constraints & Optimization

### 6.1 Memory Management

#### Training Configuration (16GB RAM)
```yaml
Model Size: tiny (<10M params) or small (<50M params)
Batch Size: 2-4 samples
Max Sequence Length: 64-128 tokens
Gradient Accumulation: 4-8 steps
Mixed Precision: FP16 (if GPU available)
Dataset Loading: Streaming mode preferred
```

#### Recommended Models for Testing
```yaml
Primary: "prajjwal1/bert-tiny" (4.4M params)
Secondary: "prajjwal1/bert-mini" (11M params)
Fallback: "prajjwal1/bert-small" (29M params)
Avoid: "bert-base-multilingual-cased" (180M params - too large)
```

### 6.2 UV Package Manager Optimizations

#### Dependency Management
```bash
# Cache dependencies for faster reinstalls
uv pip install --cache-dir ~/.cache/uv <package>

# Install specific versions to avoid conflicts
uv pip install "torch==2.1.0" "transformers==4.35.0"

# Use UV's faster resolver
uv pip install --resolution=highest <package>
```

#### Environment Management
```bash
# Quick environment recreation
uv venv .venv --python 3.10
source .venv/bin/activate

# Export dependencies
uv pip freeze > requirements-frozen.txt

# Sync environment
uv pip sync requirements-frozen.txt
```

---

## 7. Bug Reporting Standards

### 7.1 Bug Report Template

```markdown
## Bug Report: [Brief Description]

**Priority**: Critical | High | Medium | Low
**Component**: tasks | data | core | training | deployment
**Environment**: Ubuntu 22.04, 16GB RAM, UV package manager

### Description
[Clear description of the bug]

### Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Error Messages
```
[Paste full error traceback]
```

### System Information
- Python version:
- PyTorch version:
- Transformers version:
- GPU: Yes/No
- RAM available:

### Code to Reproduce
```python
# Minimal reproducible example
```

### Screenshots/Logs
[If applicable]

### Suggested Fix
[If you have ideas]

### Related Issues
[Link to related GitHub issues]
```

### 7.2 Issue Labels
```yaml
bug: Something isn't working
enhancement: New feature or request
documentation: Documentation improvements
testing: Related to testing framework
memory: Memory usage issues
performance: Speed/efficiency issues
data: Data loading/processing issues
training: Training loop issues
inference: Inference/deployment issues
good-first-issue: Good for newcomers
help-wanted: Extra attention needed
```

---

## 8. Testing Tools & Scripts

### 8.1 Core Testing Scripts

#### Environment Checker
```bash
# tests/environment_check.py
python tests/environment_check.py
# Output: System report, dependency check, GPU info
```

#### Quick Test Suite
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=multitask_bert --cov-report=html

# Run specific test
pytest tests/unit/test_tasks.py::test_classification_task_creation
```

#### Memory Monitor
```bash
# tests/monitor_memory.py
python tests/monitor_memory.py --duration 300
# Monitors RAM/GPU usage during training
```

#### Training Test
```bash
# tests/quick_train.py
python tests/quick_train.py \
  --model prajjwal1/bert-tiny \
  --task sentiment \
  --data tests/data/sentiment_mini.csv \
  --epochs 2 \
  --batch-size 4
```

### 8.2 Automation Scripts

#### Daily Test Runner
```bash
#!/bin/bash
# tests/run_daily_tests.sh

echo "Running Jenga-AI Daily Tests..."
date

# Run import tests
python tests/test_imports.py

# Run unit tests
pytest tests/unit/ -v --tb=short

# Run memory tests
python tests/test_memory_usage.py

# Generate report
python tests/generate_report.py

echo "Tests complete!"
```

---

## 9. Success Metrics

### 9.1 Quantitative Metrics
```yaml
Code Coverage: >70% for core modules
Test Pass Rate: >95% for unit tests
Memory Usage: <12GB peak during training
Training Time: <5 minutes for mini datasets
Issues Created: >20 documented bugs/enhancements
Documentation: 100% of bugs documented
```

### 9.2 Qualitative Metrics
```yaml
- Framework understanding: Deep
- Bug documentation: Comprehensive
- Testing guide: Clear and actionable
- Community contribution: Active (issues, discussions)
- Code quality: Following best practices
```

---

## 10. Deliverables Checklist

### Week 1 (Days 1-7)
- [ ] Environment setup complete
- [ ] All modules import successfully
- [ ] Synthetic datasets created
- [ ] Data loading tests pass
- [ ] Single-task training works
- [ ] Memory usage documented

### Week 2 (Days 8-14)
- [ ] Multi-task training works
- [ ] Inference pipeline tested
- [ ] Edge cases identified
- [ ] All bugs documented
- [ ] GitHub issues created
- [ ] Performance benchmarks complete

### Week 3 (Day 15)
- [ ] Final test report written
- [ ] Testing guide created
- [ ] Bug report compiled
- [ ] Recommendations documented
- [ ] Code contributions submitted

---

## 11. Risk Management

### 11.1 Potential Risks

#### Risk 1: Incomplete Implementation
**Impact**: High  
**Probability**: Medium  
**Mitigation**: Document all missing features, focus on testing what exists, create enhancement requests

#### Risk 2: Memory Constraints
**Impact**: Medium  
**Probability**: Low  
**Mitigation**: Use tiny models, small batch sizes, gradient accumulation

#### Risk 3: Dependency Conflicts
**Impact**: Medium  
**Probability**: Medium  
**Mitigation**: Use UV's dependency resolver, pin versions, test in clean environment

#### Risk 4: No Real Datasets Available
**Impact**: Low  
**Probability**: Low  
**Mitigation**: Use synthetic data primarily, Hugging Face datasets as backup

---

## 12. Communication Plan

### 12.1 Internal Documentation
- Update this PRD weekly
- Maintain testing log in `docs/testing/PROGRESS.md`
- Document daily findings in `docs/testing/DAILY_LOG.md`

### 12.2 GitHub Communication
- Create issues for all bugs (use template)
- Open discussions for questions
- Submit PRs for bug fixes (if capable)
- Respond to maintainer comments within 24h

### 12.3 Reporting Schedule
```yaml
Daily: Update progress log
Weekly: Create summary report
End of Testing: Final comprehensive report
```

---

## 13. Future Considerations

### 13.1 Extended Testing (Optional)
- Test with larger datasets (if resources allow)
- Test with larger models (bert-base)
- Integration with LLMs (Phase 1 of roadmap)
- Agent framework testing (Phase 3)

### 13.2 Continuous Integration
- Set up GitHub Actions for automated testing
- Create pre-commit hooks
- Implement code quality checks

---

## 14. References

### 14.1 Documentation Links
- Jenga-AI Repository: https://github.com/Jenga-AI/Jenga-AI
- UV Package Manager: https://github.com/astral-sh/uv
- PyTorch Documentation: https://pytorch.org/docs
- Transformers Documentation: https://huggingface.co/docs/transformers

### 14.2 Related Research
- Multi-Task Learning in NLP
- Attention Fusion Mechanisms
- African Language NLP
- Low-Resource Model Training

---

## 15. Appendices

### Appendix A: Command Reference
```bash
# UV commands
uv venv .venv                    # Create environment
uv pip install <package>         # Install package
uv pip list                      # List packages
uv pip freeze                    # Export dependencies

# Testing commands
pytest tests/                    # Run all tests
pytest -k test_name             # Run specific test
pytest --cov                    # With coverage
pytest -v --tb=short            # Verbose with short traceback

# Memory monitoring
htop                            # System monitor
nvidia-smi                      # GPU monitor (if applicable)
```

### Appendix B: Useful Python Snippets
```python
# Check memory usage
import psutil
ram = psutil.virtual_memory()
print(f"RAM: {ram.percent}% used")

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Monitor training
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your training code
prof.export_chrome_trace("trace.json")
```

---

**Document Version Control**
- v1.0 (2025-11-24): Initial PRD creation
- Future updates will be tracked here

**Sign-off**
Testing Contributor: _________________ Date: _________
Project Maintainer: __________________ Date: _________