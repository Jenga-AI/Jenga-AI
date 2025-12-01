# Jenga-AI Testing Plan - Executive Summary

**Date:** November 24, 2025  
**Tester:** Collins  
**Environment:** 16GB RAM, No GPU, CPU-only, Ubuntu Linux

---

## 🎯 What We're Testing

**Jenga-AI** is a multi-task NLP framework designed for African-context applications. It enables training a single model to handle multiple NLP tasks simultaneously using a shared encoder and task-specific heads.

### Core Architecture

```
┌─────────────────────────────────────────┐
│        Input Text (Swahili/English)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Shared Encoder (BERT-based)          │
│    - bert-tiny / bert-mini / distilbert │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Attention Fusion Layer (Optional)     
│    - Task-specific weighting            │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│   Task 1     │ │   Task 2     │
│ (Sentiment)  │ │    (NER)     │
└──────────────┘ └──────────────┘
```

### Three Main Components

1. **`multitask_bert/`** - Core multi-task learning framework
   - Shared encoder with task-specific heads
   - Attention fusion mechanisms
   - Round-robin task sampling during training

2. **`llm_finetuning/`** - LLM fine-tuning with PEFT (LoRA)
   - Teacher-student distillation
   - Parameter-efficient training
   - Quantization support (4-bit/8-bit)

3. **`seq2seq_models/`** - Sequence-to-sequence models
   - Translation, summarization tasks
   - Encoder-decoder architectures

---

## 🧠 Key Concepts to Test

### 1. Multi-Task Learning
- **What it does:** Trains one model on multiple tasks simultaneously
- **Why test:** Core functionality - must verify tasks don't interfere
- **Risk:** Negative transfer (one task hurts another)
- **Test approach:** Compare multi-task vs single-task performance

### 2. Attention Fusion
- **What it does:** Learns task-specific attention weights for shared representations
- **Why test:** Novel feature - needs validation
- **Risk:** May not improve performance or cause training instability
- **Test approach:** Compare with/without fusion, visualize attention patterns

### 3. Round-Robin Task Sampling
- **What it does:** Alternates between tasks during training
- **Why test:** Ensures balanced training across tasks
- **Risk:** Unequal dataset sizes may cause imbalance
- **Test approach:** Monitor loss convergence per task

### 4. Memory Efficiency
- **What it does:** Share encoder parameters across tasks
- **Why test:** Critical for our 16GB RAM constraint
- **Risk:** Memory leaks, inefficient implementations
- **Test approach:** Profile memory at each training stage

---

## 🎪 Testing Strategy - Smart & Resource-Conscious

### Our Constraints
✅ **What we have:**
- 16GB RAM (plenty for testing with tiny models)
- Multi-core CPU (slower but sufficient)
- UV package manager (fast, modern)
- Full codebase access

❌ **What we don't have:**
- GPU (no problem - use CPU-optimized configs)
- Large datasets (we'll create synthetic mini-datasets)
- Unlimited time (15 days - need smart prioritization)

### Smart Approach

#### 1. **Tiny Models Only**
```yaml
Primary: prajjwal1/bert-tiny (4.4M params)
- Memory: ~200MB model + ~500MB training = <1GB total
- Speed: ~50 samples/sec on CPU
- Perfect for testing

Secondary: prajjwal1/bert-mini (11M params)
- Memory: ~500MB model + ~1GB training = ~1.5GB total
- Speed: ~20 samples/sec on CPU
- Use for final validation

Avoid: bert-base-multilingual-cased (180M params)
- Memory: ~2GB model + ~8GB training = too slow for testing
```

#### 2. **Mini Datasets**
```yaml
Sentiment: 100 samples (50 pos, 50 neg)
NER: 50 samples (varied entities)
Agriculture: 80 samples
QA Scoring: 60 samples

Total: ~300 samples across all tasks
Training time: 3-5 minutes per experiment
```

#### 3. **CPU-Optimized Training Config**
```yaml
batch_size: 2              # Very small batches
gradient_accumulation: 4   # Effective batch size = 8
max_length: 64             # Short sequences
num_epochs: 2              # Quick experiments
learning_rate: 2e-5        # Standard
```

#### 4. **Incremental Testing**
```
Day 1-2:  Setup → Can we import modules?
Day 3-4:  Data → Can we load data?
Day 5-7:  Single Task → Can we train one task?
Day 7-9:  Multi-Task → Can we train two tasks together?
Day 9-10: Inference → Can we make predictions?
Day 11-12: Edge Cases → What breaks the system?
Day 13-15: Documentation → Report everything
```

---

## 🐛 Bug Hunting - What to Look For

### Critical Bugs (System Broken)
🔴 **Memory Leaks**
- Symptom: Memory usage grows with each epoch
- Test: Train for 10 epochs, monitor RAM
- Expected: Stable memory usage

🔴 **Training Crashes**
- Symptom: NaN losses, gradient explosions
- Test: Various learning rates, batch sizes
- Expected: Stable loss curves

🔴 **Multi-Task Interference**
- Symptom: Task 2 performance degrades when training with Task 1
- Test: Compare single-task vs multi-task metrics
- Expected: Similar or better performance

🔴 **Data Loading Failures**
- Symptom: Hangs, crashes, incorrect data
- Test: Different data formats (CSV, JSON, JSONL)
- Expected: Clean loading with validation

### Important Bugs (Functionality Issues)
🟡 **Incorrect Metrics**
- Symptom: F1 score doesn't match manual calculation
- Test: Small dataset with known expected metrics
- Expected: Accurate metric computation

🟡 **Checkpoint Issues**
- Symptom: Can't load saved model
- Test: Save → Load → Inference
- Expected: Identical predictions before/after loading

🟡 **Config Parsing Errors**
- Symptom: YAML config not parsed correctly
- Test: Various config combinations
- Expected: Clear error messages for invalid configs

### Minor Bugs (Quality Issues)
🟢 **Poor Error Messages**
- Symptom: Cryptic errors that don't help debugging
- Test: Intentionally break things
- Expected: Clear, actionable error messages

🟢 **Documentation Gaps**
- Symptom: Missing or unclear docs
- Test: Follow documentation as new user
- Expected: Complete, clear instructions

---

## 📊 Success Criteria

### Must Have (Critical)
- ✅ All modules import successfully
- ✅ Single-task training completes without errors
- ✅ Multi-task training works for 2+ tasks
- ✅ Memory usage stays under 12GB
- ✅ Inference produces valid predictions
- ✅ >20 bugs/issues documented with repro steps

### Should Have (Important)
- ✅ Attention fusion mechanism works
- ✅ Checkpoint save/load works
- ✅ All data formats supported (CSV, JSON, JSONL)
- ✅ Basic error handling for edge cases
- ✅ Comprehensive testing guide created

### Nice to Have (Optional)
- ✅ LLM fine-tuning module tested
- ✅ Performance benchmarks documented
- ✅ Code coverage >70%
- ✅ Visualization tools tested

---

## 🚀 Immediate Action Plan (Next 3 Days)

### Today (Day 1)
```bash
# 1. Environment Setup (2 hours)
□ Install UV if not present
□ Create clean virtual environment
□ Install dependencies
□ Run environment check script

# 2. Import Testing (2 hours)
□ Test all module imports
□ Document missing dependencies
□ Fix any import errors

# 3. Create Synthetic Data (2 hours)
□ Generate sentiment_mini.csv (100 samples)
□ Generate ner_mini.jsonl (50 samples)
□ Validate data formats
```

### Tomorrow (Day 2)
```bash
# 1. Data Loading Tests (2 hours)
□ Test CSV loading
□ Test JSONL loading
□ Test JSON loading
□ Fix any data processing bugs

# 2. First Training Test (3 hours)
□ Create simple config for sentiment task
□ Run training with bert-tiny
□ Monitor memory usage
□ Document training time

# 3. Memory Profiling (1 hour)
□ Profile memory at each stage
□ Document safe limits
□ Create memory monitoring script
```

### Day After (Day 3)
```bash
# 1. Single-Task Validation (3 hours)
□ Test NER task training
□ Test multi-label classification
□ Validate metrics calculation
□ Test checkpoint save/load

# 2. Bug Documentation (2 hours)
□ Document all bugs found so far
□ Create GitHub issues for critical bugs
□ Update testing progress

# 3. Multi-Task Prep (1 hour)
□ Create 2-task config
□ Prepare datasets
□ Plan multi-task tests
```

---

## 💡 Pro Tips for CPU Testing

### Memory Management
```python
# Before each test run
import gc
import torch

gc.collect()  # Clear Python garbage
torch.cuda.empty_cache()  # Clears some CPU memory too

# Monitor memory during training
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1e9:.2f} GB")
```

### Speed Optimization
```python
# Use smaller models
model_name = "prajjwal1/bert-tiny"  # ✅ Good
model_name = "bert-base-uncased"    # ❌ Too slow

# Use smaller sequence lengths
max_length = 64   # ✅ Fast, good for testing
max_length = 512  # ❌ Slow, unnecessary for most tests

# Use gradient accumulation for larger effective batch sizes
batch_size = 2              # Small batches fit in memory
gradient_accumulation = 4   # Effective batch = 8
```

### Quick Smoke Test
```bash
# Create a 5-minute smoke test that validates basics
python tests/quick_test.py

Expected output:
✓ Imports working
✓ Data loads correctly
✓ Model initializes
✓ Forward pass works
✓ Training step completes
✓ Inference works

If all pass → Environment is ready!
```

---

## 📈 Expected Timeline

```
Week 1: Foundation & Single-Task
├── Day 1-2: Environment + Imports + Data ✅
├── Day 3-4: Data Loading + Synthetic Data ✅
├── Day 5-7: Single-Task Training Tests ✅

Week 2: Multi-Task & Advanced Features
├── Day 7-9: Multi-Task Training ✅
├── Day 9-10: Inference + Deployment ✅
├── Day 11-12: Edge Cases + LLM Module ✅

Week 3: Documentation & Reporting
├── Day 13-14: Bug Documentation + GitHub Issues ✅
├── Day 15: Final Report + Testing Guide ✅
```

---

## 🎯 Key Takeaways

### What Makes This Plan Smart?

1. **Resource-Conscious**: Uses tiny models, small datasets, optimized configs
2. **Incremental**: Tests simple things first, builds complexity
3. **Practical**: Focuses on finding bugs, not perfect coverage
4. **Well-Documented**: Every bug gets a detailed report
5. **Realistic**: 15-day timeline with achievable daily goals

### What We'll Deliver

📄 **Documentation**
- Comprehensive bug report with 20+ issues
- Testing guide for future contributors
- CPU training optimization guide
- Test results and benchmarks

🧪 **Test Suite**
- Unit tests for core modules
- Integration tests for training pipelines
- Performance tests for memory/speed
- Edge case tests for robustness

🐛 **Bug Fixes** (if time permits)
- Critical bugs that break core functionality
- Simple fixes with clear solutions
- Documentation improvements

🤝 **Community Contribution**
- GitHub issues with detailed repro steps
- Active participation in discussions
- Professional, constructive feedback

---

## 🔗 Quick Links

- **PRD Document**: `tests/prd.md`
- **TODO List**: `todo.md` (this file - detailed checklist)
- **Main README**: `README.MD`
- **Example Configs**: `examples/*.yaml`
- **Existing Tests**: `multitask_bert/tests/`

---

## 📞 Getting Help

If stuck:
1. Check existing tests in `multitask_bert/tests/`
2. Review example configs in `examples/`
3. Read module docstrings
4. Search GitHub issues
5. Ask in discussions

---

**Remember:** Our goal is to find bugs, document them thoroughly, and help improve the framework. We're not aiming for perfection, but for comprehensive understanding and clear documentation.

**Let's make Jenga-AI more robust! 🏗️🧪**


