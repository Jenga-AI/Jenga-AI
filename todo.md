# Jenga-AI Testing TODO List
## System Tester - Comprehensive Bug Detection & Validation Plan

**Machine Specs:** 16GB RAM, No GPU, CPU-only  
**Testing Period:** 15 days  
**Status:** In Progress  
**Last Updated:** November 24, 2025

---

## 📋 PHASE 1: ENVIRONMENT & SETUP (Days 1-2)
**Status:** 🔄 In Progress  
**Goal:** Validate environment setup and understand the codebase structure

### ✅ Completed
- [x] Read and comprehend PRD document
- [x] Understand system architecture (multi-task BERT framework)
- [x] Identify three main components: multitask_bert, llm_finetuning, seq2seq_models
- [x] Review existing requirements.txt

### 🔲 TODO - Environment Setup

#### ENV-1: System Validation
- [ ] Create `tests/environment_check.py` script
  - [ ] Check Python version (3.9+)
  - [ ] Verify available RAM (should have 16GB)
  - [ ] Confirm CPU cores available
  - [ ] Check disk space (need ~5GB minimum)
  - [ ] Test PyTorch CPU installation
  - [ ] Verify transformers library works

#### ENV-2: Dependency Testing
- [ ] Test UV package manager installation
- [ ] Create clean virtual environment with UV
- [ ] Install dependencies in clean env
- [ ] Document any missing or conflicting packages
- [ ] Create `requirements-testing.txt` with pinned versions
- [ ] Test basic imports (torch, transformers, datasets)

#### ENV-3: Memory Baseline
- [ ] Create memory monitoring script `tests/utils/memory_monitor.py`
- [ ] Record baseline memory usage (idle system)
- [ ] Test memory with small model load (bert-tiny)
- [ ] Document memory overhead for different model sizes

---

## 📋 PHASE 2: MODULE IMPORT TESTING (Days 2-3)
**Status:** 🔲 Not Started  
**Goal:** Verify all framework modules can be imported without errors

#### IMP-1: Core Modules Import Test
- [ ] Create `tests/unit/test_imports.py`
- [ ] Test import: `multitask_bert.core.model`
- [ ] Test import: `multitask_bert.core.fusion`
- [ ] Test import: `multitask_bert.core.config`
- [ ] Test import: `multitask_bert.core.registry`
- [ ] Document any import errors with full traceback

#### IMP-2: Task Modules Import Test
- [ ] Test import: `multitask_bert.tasks.base`
- [ ] Test import: `multitask_bert.tasks.classification`
- [ ] Test import: `multitask_bert.tasks.ner`
- [ ] Test import: `multitask_bert.tasks.question_answering`
- [ ] Test import: `multitask_bert.tasks.qa_qc`
- [ ] Test import: `multitask_bert.tasks.sentiment_analysis`
- [ ] Test import: `multitask_bert.tasks.regression`

#### IMP-3: Data & Training Modules
- [ ] Test import: `multitask_bert.data.data_processing`
- [ ] Test import: `multitask_bert.data.universal`
- [ ] Test import: `multitask_bert.data.custom`
- [ ] Test import: `multitask_bert.training.trainer`
- [ ] Test import: `multitask_bert.training.callbacks`
- [ ] Test import: `multitask_bert.training.data`

#### IMP-4: LLM Fine-tuning Modules
- [ ] Test import: `llm_finetuning.model.model_factory`
- [ ] Test import: `llm_finetuning.model.teacher_student`
- [ ] Test import: `llm_finetuning.training.trainer`
- [ ] Test import: `llm_finetuning.data.data_processing`

#### IMP-5: Utility & Analysis Modules
- [ ] Test import: `multitask_bert.analysis.attention`
- [ ] Test import: `multitask_bert.analysis.metrics`
- [ ] Test import: `multitask_bert.analysis.visualization`
- [ ] Test import: `multitask_bert.deployment.inference`
- [ ] Test import: `multitask_bert.deployment.export`

---

## 📋 PHASE 3: SYNTHETIC DATA CREATION (Days 3-4)
**Status:** 🔲 Not Started  
**Goal:** Create small test datasets for CPU-friendly training

#### DATA-1: Sentiment Analysis Dataset (Mini)
- [ ] Create `tests/data/sentiment_mini.csv`
  - Target: 100 samples (50 positive, 50 negative)
  - Mix: 40% Swahili, 40% English, 20% Code-switched
  - Columns: text, label (0=Negative, 1=Positive)
- [ ] Create `tests/data/sentiment_mini.jsonl` (same data, JSONL format)
- [ ] Validate data format and balance

#### DATA-2: NER Dataset (Mini)
- [ ] Create `tests/data/ner_mini.jsonl`
  - Target: 50 samples
  - Labels: O (0), B-LOC (1), B-PER (2), B-THREAT (3)
  - Include varied sequence lengths (10-30 tokens)
  - Format: {"text": "...", "tokens": [...], "labels": [...]}
- [ ] Validate label distribution

#### DATA-3: Agricultural Classification Dataset
- [ ] Create `tests/data/agriculture_mini.csv`
  - Target: 80 samples (40 healthy, 40 disease)
  - Disease types: Maize blight, Cassava mosaic, etc.
  - Columns: text, label (0=Healthy, 1=Disease)

#### DATA-4: Multi-Label QA Scoring Dataset
- [ ] Create `tests/data/qa_scoring_mini.json`
  - Target: 60 samples
  - Multiple heads: opening, listening, proactiveness, resolution, hold, closing
  - Keep structure similar to existing qa_score_data.json

#### DATA-5: Data Loading Test Suite
- [ ] Create `tests/unit/test_data_loading.py`
  - [ ] Test CSV loading
  - [ ] Test JSONL loading
  - [ ] Test JSON loading
  - [ ] Test data validation
  - [ ] Test tokenization with different max_lengths

---

## 📋 PHASE 4: SINGLE-TASK TRAINING TESTS (Days 5-7)
**Status:** 🔲 Not Started  
**Goal:** Validate training on individual tasks with memory constraints

#### TRAIN-1: Classification Task Training
- [ ] Create `tests/integration/test_single_classification.py`
- [ ] Create config: `tests/configs/single_classification_cpu.yaml`
  - Model: `prajjwal1/bert-tiny` (4.4M params)
  - Batch size: 2
  - Max length: 64
  - Epochs: 2
  - Gradient accumulation: 4
- [ ] Test sentiment classification training
- [ ] Monitor memory usage during training
- [ ] Validate checkpoint saving
- [ ] Test model loading from checkpoint
- [ ] Document training time and peak memory

#### TRAIN-2: NER Task Training
- [ ] Create `tests/integration/test_single_ner.py`
- [ ] Create config: `tests/configs/single_ner_cpu.yaml`
  - Same constraints as above
- [ ] Test NER training with mini dataset
- [ ] Validate token-level predictions
- [ ] Test padding and masking (-100 labels)
- [ ] Check metric calculation (F1, precision, recall)

#### TRAIN-3: Multi-Label Classification
- [ ] Create `tests/integration/test_single_multilabel.py`
- [ ] Create config: `tests/configs/single_multilabel_cpu.yaml`
- [ ] Test QA scoring task (multiple heads)
- [ ] Validate all heads produce outputs
- [ ] Test per-head loss calculation
- [ ] Verify weighted loss combination

#### TRAIN-4: Memory Profiling
- [ ] Create `tests/performance/test_memory_usage.py`
- [ ] Profile memory at different stages:
  - Model loading
  - Data loading
  - Forward pass
  - Backward pass
  - Optimizer step
- [ ] Test with different batch sizes (1, 2, 4)
- [ ] Document memory-batch size relationship
- [ ] Identify memory leaks (if any)

---

## 📋 PHASE 5: MULTI-TASK TRAINING TESTS (Days 7-9)
**Status:** 🔲 Not Started  
**Goal:** Test core multi-task learning functionality

#### MULTI-1: Two-Task Training (Sentiment + NER)
- [ ] Create `tests/integration/test_two_task_training.py`
- [ ] Create config: `tests/configs/two_task_cpu.yaml`
  - Combine sentiment + NER
  - Round-robin task sampling
  - Batch size: 2 per task
- [ ] Test shared encoder functionality
- [ ] Validate task-specific heads work correctly
- [ ] Test round-robin iterator
- [ ] Monitor if one task dominates training
- [ ] Validate separate eval metrics for each task

#### MULTI-2: Three-Task Training
- [ ] Create `tests/integration/test_three_task_training.py`
- [ ] Create config: `tests/configs/three_task_cpu.yaml`
  - Sentiment + NER + Agriculture
- [ ] Test task balancing with unequal dataset sizes
- [ ] Validate all tasks train without errors
- [ ] Check for task interference (negative transfer)
- [ ] Document convergence behavior

#### MULTI-3: Attention Fusion Testing
- [ ] Create `tests/unit/test_attention_fusion.py`
- [ ] Test AttentionFusion module initialization
- [ ] Test task embedding creation
- [ ] Test attention weight calculation
- [ ] Validate fused representation shape
- [ ] Test with different hidden sizes
- [ ] Check gradient flow through fusion layer
- [ ] Compare with/without fusion performance

#### MULTI-4: Task Registry & Configuration
- [ ] Test task registration mechanism
- [ ] Test task ID mapping
- [ ] Validate task config parsing from YAML
- [ ] Test dynamic task addition
- [ ] Test task removal/disabling

---

## 📋 PHASE 6: INFERENCE & DEPLOYMENT (Days 9-10)
**Status:** 🔲 Not Started  
**Goal:** Test model inference and deployment pipeline

#### INF-1: Single Sample Inference
- [ ] Create `tests/integration/test_inference.py`
- [ ] Test single text classification prediction
- [ ] Test single NER prediction
- [ ] Test multi-head output parsing
- [ ] Validate output format consistency
- [ ] Test with edge cases (empty text, very long text)

#### INF-2: Batch Inference
- [ ] Test batch prediction (size 4, 8, 16)
- [ ] Compare batch vs single inference speed
- [ ] Validate batch results match individual predictions
- [ ] Test memory usage during batch inference

#### INF-3: Model Export & Loading
- [ ] Test model serialization (save_pretrained)
- [ ] Test model deserialization (from_pretrained)
- [ ] Test checkpoint compatibility
- [ ] Test loading partial checkpoints (encoder only)
- [ ] Verify task heads saved correctly

#### INF-4: Deployment API Test
- [ ] Test `multitask_bert.deployment.inference` module
- [ ] Create simple REST API wrapper test
- [ ] Test concurrent inference requests
- [ ] Measure inference latency
- [ ] Test model warm-up behavior

---

## 📋 PHASE 7: LLM FINE-TUNING MODULE (Days 10-11)
**Status:** 🔲 Not Started  
**Goal:** Test LLM fine-tuning capabilities (CPU-compatible)

#### LLM-1: Model Factory Testing
- [ ] Create `tests/unit/test_model_factory.py`
- [ ] Test loading tiny LLM model (e.g., distilgpt2)
- [ ] Test quantization config (skip if no GPU)
- [ ] Test LoRA config application
- [ ] Test PEFT model creation
- [ ] Validate trainable parameters count

#### LLM-2: Teacher-Student Setup
- [ ] Test teacher model loading
- [ ] Test student model initialization
- [ ] Test TeacherStudentModel wrapper
- [ ] Validate forward pass works
- [ ] Test distillation loss calculation (if implemented)

#### LLM-3: LLM Data Processing
- [ ] Test `llm_finetuning.data.data_processing`
- [ ] Test data formatting for causal LM
- [ ] Test instruction format handling
- [ ] Validate tokenization for LLMs
- [ ] Test with small dataset (<50 samples)

#### LLM-4: Small-Scale LLM Training
- [ ] Create `tests/integration/test_llm_finetuning_cpu.py`
- [ ] Use smallest available model (distilgpt2, ~82M params)
- [ ] Batch size: 1, gradient accumulation: 8
- [ ] Train for 10 steps only (not full epoch)
- [ ] Monitor memory (expect 8-12GB usage)
- [ ] Document training speed (steps/sec)
- [ ] Test if LoRA reduces memory

---

## 📋 PHASE 8: EDGE CASES & STRESS TESTING (Days 11-12)
**Status:** 🔲 Not Started  
**Goal:** Test framework robustness and error handling

#### EDGE-1: Data Edge Cases
- [ ] Create `tests/unit/test_edge_cases.py`
- [ ] Test empty dataset handling
- [ ] Test single sample dataset
- [ ] Test dataset with all same label
- [ ] Test malformed JSON/CSV (missing columns)
- [ ] Test extremely long sequences (>512 tokens)
- [ ] Test special characters and emojis
- [ ] Test mixed encodings

#### EDGE-2: Model Edge Cases
- [ ] Test with max_length=1
- [ ] Test with batch_size=1
- [ ] Test with 0 warmup steps
- [ ] Test with very high learning rate (1.0)
- [ ] Test with 0 epochs (should skip training)
- [ ] Test loading model from non-existent checkpoint

#### EDGE-3: Training Edge Cases
- [ ] Test training with no eval dataset
- [ ] Test early stopping when metric doesn't improve
- [ ] Test resume training from checkpoint
- [ ] Test training interruption handling
- [ ] Test multi-task with one empty dataset
- [ ] Test gradient accumulation with batch_size=1

#### EDGE-4: Memory Limit Testing
- [ ] Test model with increasing batch sizes until OOM
- [ ] Test with longest possible sequence length
- [ ] Test loading multiple models simultaneously
- [ ] Test memory cleanup after training
- [ ] Document maximum safe configuration

---

## 📋 PHASE 9: BUG DOCUMENTATION & REPORTING (Days 12-14)
**Status:** 🔲 Not Started  
**Goal:** Document all bugs and create GitHub issues

#### BUG-1: Bug Tracking
- [ ] Create `docs/testing/BUG_REPORT.md`
- [ ] Document all found bugs with:
  - Bug ID, Priority, Component
  - Steps to reproduce
  - Expected vs actual behavior
  - Error messages/tracebacks
  - System info
  - Suggested fix (if known)

#### BUG-2: GitHub Issue Creation
- [ ] Create GitHub issues for each critical bug
- [ ] Create GitHub issues for high-priority bugs
- [ ] Add appropriate labels (bug, enhancement, memory, etc.)
- [ ] Link related issues
- [ ] Provide minimal reproducible examples

#### BUG-3: Known Issues Documentation
- [ ] Document CPU-specific limitations
- [ ] Document memory constraints workarounds
- [ ] Document compatible model sizes
- [ ] Document optimal configurations for 16GB RAM

---

## 📋 PHASE 10: FINAL DOCUMENTATION (Day 14-15)
**Status:** 🔲 Not Started  
**Goal:** Create comprehensive testing documentation

#### DOC-1: Test Results Report
- [ ] Create `docs/testing/TEST_RESULTS.md`
- [ ] Summary of all test phases
- [ ] Pass/fail rates for each test suite
- [ ] Performance benchmarks
- [ ] Memory usage analysis
- [ ] Known issues and limitations

#### DOC-2: Testing Guide
- [ ] Create `docs/testing/TESTING_GUIDE.md`
- [ ] How to set up testing environment
- [ ] How to run test suites
- [ ] How to create new tests
- [ ] Testing best practices
- [ ] CPU-specific testing guidelines

#### DOC-3: Recommendations
- [ ] Create `docs/testing/RECOMMENDATIONS.md`
- [ ] Suggested improvements for memory efficiency
- [ ] Code quality recommendations
- [ ] Documentation improvements needed
- [ ] Feature requests
- [ ] Performance optimization suggestions

#### DOC-4: CPU Training Guide
- [ ] Create `docs/CPU_TRAINING_GUIDE.md`
- [ ] Optimal configurations for 16GB RAM
- [ ] Model size recommendations
- [ ] Batch size and gradient accumulation guide
- [ ] Expected training times
- [ ] Troubleshooting common issues

---

## 🐛 CRITICAL BUGS TO LOOK FOR

### High Priority Issues
- [ ] Memory leaks during training
- [ ] Incorrect loss calculation in multi-task scenarios
- [ ] Task interference (negative transfer)
- [ ] Attention fusion not working as expected
- [ ] Checkpoint loading failures
- [ ] Data loader hanging with CPU
- [ ] Incorrect gradient accumulation
- [ ] Missing error handling for edge cases

### Medium Priority Issues
- [ ] Inefficient data loading
- [ ] Unnecessary model copies in memory
- [ ] Missing validation checks
- [ ] Unclear error messages
- [ ] Logging configuration issues
- [ ] Config parsing edge cases

### Low Priority Issues
- [ ] Documentation gaps
- [ ] Code style inconsistencies
- [ ] Missing type hints
- [ ] Unused imports
- [ ] Inefficient implementations

---

## 📊 SUCCESS METRICS

### Quantitative Goals
- [ ] >70% code coverage for core modules
- [ ] >95% test pass rate for unit tests
- [ ] Peak memory <12GB during training
- [ ] Training time <5 min for mini datasets
- [ ] >20 documented bugs/enhancements
- [ ] 100% of bugs documented with repro steps

### Qualitative Goals
- [ ] Deep understanding of framework architecture
- [ ] Comprehensive bug documentation
- [ ] Clear and actionable testing guide
- [ ] Active community contribution (issues, discussions)
- [ ] Professional code quality in tests

---

## 🛠️ TESTING TOOLS & SCRIPTS TO CREATE

### Essential Scripts
1. `tests/environment_check.py` - System validation
2. `tests/utils/memory_monitor.py` - Memory tracking
3. `tests/utils/create_synthetic_data.py` - Generate test data
4. `tests/run_all_tests.sh` - Run complete test suite
5. `tests/quick_test.py` - Fast smoke test

### Config Files Needed
1. `tests/configs/cpu_tiny_model.yaml` - Minimal config for testing
2. `tests/configs/single_task_template.yaml` - Single task base
3. `tests/configs/multi_task_template.yaml` - Multi-task base

### Monitoring Scripts
1. `tests/performance/benchmark_inference.py` - Speed tests
2. `tests/performance/memory_profiler.py` - Detailed memory analysis
3. `tests/performance/training_monitor.py` - Real-time training stats

---

## 📝 NOTES & OBSERVATIONS

### Memory Optimization Strategies
- Use `bert-tiny` (4.4M params) as primary test model
- Batch size: 2 (max 4 with gradient accumulation)
- Max sequence length: 64 (max 128 if needed)
- Gradient accumulation steps: 4-8
- Enable `torch.no_grad()` during evaluation
- Clear cache between runs: `torch.cuda.empty_cache()` (even on CPU, clears some memory)

### Time Estimates (CPU Training)
- bert-tiny, 100 samples, 2 epochs, batch_size=2: ~3-5 minutes
- bert-mini, 100 samples, 2 epochs, batch_size=2: ~8-12 minutes
- Multi-task (2 tasks), 200 samples total: ~10-15 minutes

### Expected Issues
1. **Import errors** - Missing dependencies, version conflicts
2. **Memory errors** - OOM with larger models/batches
3. **Data format errors** - Inconsistent data formats
4. **Config errors** - YAML parsing issues, missing required fields
5. **Training errors** - NaN losses, gradient issues, task balancing
6. **Inference errors** - Shape mismatches, output format issues

---

## 🚀 IMMEDIATE NEXT STEPS (Priority Order)

1. **Create environment check script** - Validate setup
2. **Test all module imports** - Find missing dependencies
3. **Create synthetic mini datasets** - Enable testing
4. **Run single-task classification test** - Validate basic training
5. **Test multi-task with 2 tasks** - Core functionality
6. **Document all bugs found** - Track issues
7. **Create memory optimization guide** - Help future users

---

**Last Updated:** November 24, 2025  
**Tester:** Collins  
**Machine:** Ubuntu, 16GB RAM, No GPU, UV Package Manager  
**Framework Version:** Jenga-AI v1.0 (In Testing)


