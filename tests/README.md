# Jenga-AI Testing Suite

This directory contains all testing infrastructure for the Jenga-AI framework.

## 📁 Directory Structure

```
tests/
├── README.md                    # This file
├── environment_check.py         # System validation script
├── unit/                        # Unit tests
│   ├── test_imports.py          # Module import tests
│   ├── test_data_loading.py     # Data loading tests (TODO)
│   ├── test_tasks.py            # Task definition tests (TODO)
│   └── test_attention_fusion.py # Fusion mechanism tests (TODO)
├── integration/                 # Integration tests
│   ├── test_single_task.py      # Single-task training (TODO)
│   ├── test_multi_task.py       # Multi-task training (TODO)
│   └── test_inference.py        # Inference pipeline (TODO)
├── performance/                 # Performance tests
│   ├── test_memory_usage.py     # Memory profiling (TODO)
│   └── test_speed.py            # Speed benchmarks (TODO)
├── configs/                     # Test configurations
│   └── (YAML config files)
├── data/                        # Test datasets
│   ├── sentiment_mini.csv
│   ├── ner_mini.jsonl
│   └── (other test data)
└── utils/                       # Testing utilities
    ├── create_synthetic_data.py # Data generator
    └── memory_monitor.py        # Memory monitoring
```

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# From project root
bash setup_testing_env.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Environment Check

```bash
python tests/environment_check.py
```

This validates:
- Python version (3.9+)
- System resources (RAM, CPU, disk)
- PyTorch installation
- All dependencies
- Jenga-AI modules import correctly

### 3. Generate Test Data

```bash
python tests/utils/create_synthetic_data.py --all
```

Creates small datasets for testing:
- Sentiment analysis (100 samples)
- NER (50 samples)
- Agriculture classification (80 samples)
- QA scoring (60 samples)

### 4. Run Import Tests

```bash
python tests/unit/test_imports.py
```

Validates all modules can be imported.

### 5. Run Full Test Suite (when complete)

```bash
pytest tests/ -v
```

## 📝 Testing Guidelines

### Writing Tests

1. **Follow naming convention:** `test_*.py`
2. **Use descriptive names:** `test_sentiment_classification_with_tiny_model()`
3. **Document expected behavior:** Add docstrings
4. **Test edge cases:** Empty data, malformed input, etc.
5. **Keep tests fast:** Use tiny models and small datasets

### Test Categories

#### Unit Tests (`tests/unit/`)
- Test individual functions/classes
- Mock external dependencies
- Fast (<1 second per test)
- No GPU required

Example:
```python
def test_attention_fusion_initialization():
    """Test AttentionFusion layer initializes correctly."""
    from multitask_bert.core.fusion import AttentionFusion
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
    fusion = AttentionFusion(config, num_tasks=2)
    
    assert fusion.num_tasks == 2
    assert fusion.task_embeddings.num_embeddings == 2
```

#### Integration Tests (`tests/integration/`)
- Test end-to-end workflows
- Use real (but small) models
- Moderate speed (1-5 minutes)
- CPU-friendly configurations

Example:
```python
def test_single_task_training():
    """Test complete training pipeline for single task."""
    # Load small model, small data
    # Train for 2 epochs
    # Verify metrics are computed
    # Check checkpoint is saved
```

#### Performance Tests (`tests/performance/`)
- Measure memory usage
- Benchmark speed
- Profile bottlenecks
- Document optimal configurations

Example:
```python
def test_training_memory_usage():
    """Profile memory usage during training."""
    from tests.utils.memory_monitor import MemoryMonitor
    
    monitor = MemoryMonitor()
    monitor.start()
    
    # Train model
    
    monitor.stop()
    peak_memory = monitor.get_peak_memory()
    
    assert peak_memory < 12000  # Less than 12GB
```

## 🔧 Testing Utilities

### Memory Monitor

```python
from tests.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()
monitor.start()

# Your code here
monitor.log_checkpoint("After model load")

# More code
monitor.log_checkpoint("After training")

monitor.stop()
monitor.print_report()
```

### Synthetic Data Generator

```bash
# Generate all datasets
python tests/utils/create_synthetic_data.py --all

# Generate specific dataset
python tests/utils/create_synthetic_data.py --sentiment --num-samples 200

# Custom output directory
python tests/utils/create_synthetic_data.py --all --output-dir /tmp/test_data
```

## 🐛 Bug Reporting

When you find a bug:

1. **Document it:** Create entry in `docs/testing/BUGS.md`
2. **Create minimal repro:** Write smallest code to reproduce
3. **Log details:**
   - Python version
   - PyTorch version
   - Full error traceback
   - System info (RAM, CPU)
4. **Create GitHub issue:** Use bug template

## 📊 CPU Testing Best Practices

### Memory Management

```yaml
# Safe configuration for 16GB RAM
model: prajjwal1/bert-tiny   # 4.4M params, ~20MB
batch_size: 2                # Small batches
max_length: 64               # Short sequences
gradient_accumulation: 4     # Effective batch = 8
num_epochs: 2                # Quick experiments
```

### Speed Optimization

- Use `bert-tiny` for most tests (4.4M params)
- Keep datasets under 1000 samples for testing
- Use short sequence lengths (64-128)
- Enable gradient accumulation instead of large batches

### Expected Performance (CPU)

| Model | Dataset Size | Batch Size | Time/Epoch | Peak Memory |
|-------|-------------|------------|------------|-------------|
| bert-tiny | 100 samples | 2 | ~1-2 min | ~500 MB |
| bert-mini | 100 samples | 2 | ~3-4 min | ~1 GB |
| distilbert | 100 samples | 2 | ~5-6 min | ~2 GB |

## 🎯 Testing Phases

Follow the testing plan in `../todo.md`:

1. **Phase 1:** Environment & Setup (Days 1-2)
2. **Phase 2:** Module Import Testing (Days 2-3)
3. **Phase 3:** Synthetic Data Creation (Days 3-4)
4. **Phase 4:** Single-Task Training (Days 5-7)
5. **Phase 5:** Multi-Task Training (Days 7-9)
6. **Phase 6:** Inference & Deployment (Days 9-10)
7. **Phase 7:** LLM Fine-tuning (Days 10-11)
8. **Phase 8:** Edge Cases & Stress Testing (Days 11-12)
9. **Phase 9:** Bug Documentation (Days 12-14)
10. **Phase 10:** Final Documentation (Days 14-15)

## 📚 Additional Resources

- **Testing Plan:** `../todo.md`
- **Testing Summary:** `../TESTING_SUMMARY.md`
- **Quick Start:** `../QUICK_START_TESTING.md`
- **PRD:** `prd.md`

## 🤝 Contributing

When contributing tests:

1. Follow existing patterns
2. Add docstrings
3. Use descriptive names
4. Keep tests focused (one concept per test)
5. Document any special setup required
6. Update this README if adding new utilities

## ✅ Checklist for New Tests

- [ ] Test has descriptive name
- [ ] Test has docstring explaining what it tests
- [ ] Test uses CPU-friendly configuration
- [ ] Test cleans up resources (models, data)
- [ ] Test has clear assertion with helpful message
- [ ] Test is added to appropriate category (unit/integration/performance)
- [ ] Test runs in reasonable time (<5 minutes)

---

**Happy Testing!** 🧪

For questions or issues, refer to the main documentation or create a GitHub issue.


