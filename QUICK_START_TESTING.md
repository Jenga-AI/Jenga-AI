# Quick Start Guide - Jenga-AI Testing

**For:** Testing Team / QA Contributors  
**Machine:** 16GB RAM, No GPU, CPU-only  
**Time to Complete:** 30 minutes for initial setup

---

## 🚀 Step 1: Environment Check (5 minutes)

First, validate your environment is ready:

```bash
cd /home/collins/Documents/Jenga-AI/Jenga-AI

# Make scripts executable
chmod +x tests/environment_check.py
chmod +x tests/unit/test_imports.py
chmod +x tests/utils/create_synthetic_data.py
chmod +x tests/utils/memory_monitor.py

# Run environment check
python tests/environment_check.py
```

**Expected Output:**
- ✓ Python 3.9+ detected
- ✓ 16GB RAM available
- ✓ PyTorch installed
- ✓ Transformers installed
- ✓ All dependencies present

**If any checks fail:**
```bash
# Install missing dependencies
pip install torch transformers datasets numpy pandas pyyaml tqdm scikit-learn psutil

# Or with UV (faster):
uv pip install torch transformers datasets numpy pandas pyyaml tqdm scikit-learn psutil
```

---

## 🧪 Step 2: Test Module Imports (5 minutes)

Check if all Jenga-AI modules can be imported:

```bash
python tests/unit/test_imports.py
```

**Expected Output:**
- All import tests should PASS
- If any fail, note them down (this is a bug!)

**Common Issues:**
- Missing dependencies → Install them
- Circular imports → Note in bug report
- Syntax errors → Critical bug, document it

---

## 📊 Step 3: Create Synthetic Data (5 minutes)

Generate small test datasets:

```bash
python tests/utils/create_synthetic_data.py --all
```

**This creates:**
- `tests/data/sentiment_mini.csv` (100 samples)
- `tests/data/sentiment_mini.jsonl` (100 samples)
- `tests/data/ner_mini.jsonl` (50 samples)
- `tests/data/agriculture_mini.csv` (80 samples)
- `tests/data/qa_scoring_mini.json` (60 samples)

**Verify files were created:**
```bash
ls -lh tests/data/
```

---

## 💾 Step 4: Check Memory (2 minutes)

Verify memory is sufficient:

```bash
python tests/utils/memory_monitor.py check
```

**You should see:**
- Total: ~16 GB
- Available: >8 GB (if less, close other apps)

**Test memory monitoring (optional):**
```bash
python tests/utils/memory_monitor.py demo
```

---

## 🎯 Step 5: First Training Test (10 minutes)

Now let's try a real training test! Create a minimal config:

```bash
cat > tests/configs/test_tiny_sentiment.yaml << 'EOF'
project_name: "Test_Sentiment_Tiny"

model:
  base_model: "prajjwal1/bert-tiny"
  dropout: 0.1

tokenizer:
  max_length: 64
  padding: "max_length"
  truncation: true

training:
  output_dir: "./test_results"
  learning_rate: 2.0e-5
  batch_size: 2
  num_epochs: 2
  weight_decay: 0.01
  warmup_steps: 10
  device: "cpu"
  logging:
    service: "tensorboard"
    experiment_name: "Test_Run"

tasks:
  - name: "SentimentTest"
    type: "single_label_classification"
    data_path: "tests/data/sentiment_mini.jsonl"
    heads:
      - name: "sentiment_head"
        num_labels: 2
        weight: 1.0
EOF
```

Now run a test training:

```bash
# Monitor memory during training
python -c "
from tests.utils.memory_monitor import MemoryMonitor
import subprocess
import sys

monitor = MemoryMonitor(name='Training Test')
monitor.start()
monitor.log_checkpoint('Before training')

# Run training (you would run your actual training script here)
print('Would run: python examples/run_experiment.py tests/configs/test_tiny_sentiment.yaml')
print('(Skipping actual training for now - just testing memory monitoring)')

monitor.log_checkpoint('After training')
monitor.stop()
monitor.print_report()
"
```

**Note:** The actual training will take 3-5 minutes. For now, we're just testing the setup.

---

## ✅ Step 6: Verify Everything Works

Check your progress:

```bash
# 1. Environment check passed?
echo "✓ Environment ready"

# 2. All modules import?
echo "✓ Modules working"

# 3. Data created?
ls tests/data/*.csv tests/data/*.jsonl tests/data/*.json
echo "✓ Test data ready"

# 4. Memory sufficient?
free -h | grep Mem
echo "✓ Memory OK"
```

---

## 📝 Next Steps

You're now ready to start serious testing! Follow this order:

### Today (Day 1)
1. ✅ Environment setup (DONE!)
2. ✅ Import testing (DONE!)
3. ✅ Data creation (DONE!)
4. 🔄 Review the codebase:
   ```bash
   # Read key files to understand architecture
   cat multitask_bert/core/model.py
   cat multitask_bert/training/trainer.py
   cat multitask_bert/tasks/base.py
   ```

### Tomorrow (Day 2)
1. Test data loading pipeline
2. Run first single-task training
3. Document any bugs found

### This Week
- Complete Phase 1-3 of testing plan (see `todo.md`)
- Document all bugs in `docs/testing/BUGS.md`
- Create GitHub issues for critical bugs

---

## 📚 Important Files

**Documentation:**
- `todo.md` - Complete testing checklist (YOUR MAIN GUIDE)
- `TESTING_SUMMARY.md` - Testing strategy overview
- `tests/prd.md` - Product requirements document

**Testing Scripts:**
- `tests/environment_check.py` - System validation
- `tests/unit/test_imports.py` - Import tests
- `tests/utils/create_synthetic_data.py` - Data generator
- `tests/utils/memory_monitor.py` - Memory monitoring

**Example Configs:**
- `examples/experiment.yaml` - Multi-task example
- `examples/hackathon_mvp.yaml` - MVP configuration
- `hackathon_mvp.yaml` - Another example

---

## 🐛 When You Find Bugs

Document each bug using this template:

```markdown
## Bug #XX: [Short Description]

**Priority:** Critical/High/Medium/Low
**Component:** core/tasks/data/training/deployment
**Found:** YYYY-MM-DD

### Description
Clear description of what's wrong.

### Steps to Reproduce
1. Step 1
2. Step 2
3. Error occurs

### Expected Behavior
What should happen.

### Actual Behavior
What actually happens.

### Error Message
```
Full error traceback here
```

### System Info
- Python: 3.x
- PyTorch: x.x
- RAM: 16GB
- GPU: None

### Suggested Fix
(If you have ideas)
```

Save in: `docs/testing/BUGS.md`

---

## 💡 Pro Tips

### Memory Management
```python
# Before each test
import gc
gc.collect()

# Monitor during training
from tests.utils.memory_monitor import MemoryMonitor
monitor = MemoryMonitor()
monitor.start()
# ... training ...
monitor.stop()
monitor.print_report()
```

### Quick Tests
```bash
# Test a single function
python -c "from multitask_bert.core.model import MultiTaskModel; print('✓ Model imports')"

# Test data loading
python -c "import pandas as pd; df = pd.read_csv('tests/data/sentiment_mini.csv'); print(f'✓ Loaded {len(df)} samples')"
```

### CPU Optimization
```yaml
# Always use these settings for testing:
model: prajjwal1/bert-tiny  # 4.4M params
batch_size: 2               # Small batches
max_length: 64              # Short sequences
num_epochs: 2               # Quick experiments
```

---

## 🆘 Troubleshooting

### "ImportError: No module named X"
```bash
pip install X
# or
uv pip install X
```

### "Out of memory" error
```yaml
# Reduce these in your config:
batch_size: 1              # Smallest
max_length: 32             # Shorter
# Or close other applications
```

### "Model download failed"
```bash
# Check internet connection
ping huggingface.co

# Try downloading manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('prajjwal1/bert-tiny')"
```

### Tests taking too long
```yaml
# Use even smaller settings:
num_epochs: 1
batch_size: 1
max_length: 32
# Sample only 20 items from data
```

---

## 📞 Getting Help

1. **Check existing tests:** `multitask_bert/tests/`
2. **Review examples:** `examples/*.yaml`
3. **Read module docs:** Look for docstrings in code
4. **Search issues:** Check GitHub issues
5. **Ask in discussions:** Create a discussion thread

---

## ✨ You're Ready!

Your environment is set up and you're ready to start testing. Remember:

1. **Start small** - Test simple things first
2. **Document everything** - Every bug, every observation
3. **Be systematic** - Follow the todo.md checklist
4. **Ask questions** - If stuck, ask for help
5. **Have fun!** - You're helping make Jenga-AI better! 🎉

**Good luck with testing!** 🧪🔬

---

**Next:** Open `todo.md` and start checking off tasks!


