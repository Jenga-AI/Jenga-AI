# CPU-Friendly LLM Finetuning Setup Guide

## Overview
This guide provides a complete setup for finetuning language models on synthetic data using CPU-friendly configurations with MLflow tracking.

## What Has Been Completed

### 1. Framework Exploration ✅
- Analyzed the `llm_finetuning` module structure
- Identified integration points for MLflow
- Located synthetic test data in `tests/outputs/test_data.json`

### 2. Configuration Files Created ✅

#### `cpu_finetuning_config.yaml`
- Model: microsoft/DialoGPT-small (CPU-friendly)
- PEFT/LoRA configuration with reduced parameters (r=4)
- Batch size: 1 (optimized for CPU memory)
- Gradient accumulation: 8 steps
- Max sequence length: 256 tokens
- MLflow tracking enabled

### 3. Scripts Created ✅

#### `run_cpu_finetuning.py`
Full-featured finetuning script with:
- MLflow experiment tracking
- CPU optimization settings
- Synthetic data processing
- Complete integration with the Jenga-AI framework

#### `simple_cpu_finetune.py`
Lightweight version for testing without heavy dependencies:
- Basic data processing
- Simple MLflow directory structure
- Metrics logging

### 4. Data Processing ✅
- Processed 50 samples from synthetic test data
- Saved to `cpu_training_data_simple.json`
- Format: JSON with text, sample_id, and quality fields

### 5. MLflow Tracking ✅
- Created experiment structure in `mlruns/`
- Run tracking with timestamps and metrics
- Ready for full MLflow UI integration

## Installation Instructions

### Using uv (Recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install core dependencies
uv pip install torch transformers peft accelerate datasets mlflow pyyaml

# For CPU-only torch (smaller download)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Using pip

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers peft accelerate datasets mlflow pyyaml

# For CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running the Finetuning

### Quick Test (No Dependencies)
```bash
python3 simple_cpu_finetune.py
```

### Full Training (After Installing Dependencies)
```bash
# Initialize MLflow
python3 scripts/initialize_mlflow.py

# Run finetuning
python3 run_cpu_finetuning.py --config cpu_finetuning_config.yaml

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

## CPU Optimization Settings

The configuration is optimized for CPU training:

1. **Model Size**: Using small models (125M-345M parameters)
2. **Batch Size**: 1 sample per batch
3. **Gradient Accumulation**: 8 steps to simulate larger batches
4. **Sequence Length**: Limited to 256 tokens
5. **LoRA Rank**: Reduced to 4 for efficiency
6. **Threading**: Optimized to 4 CPU threads

## Memory Requirements

- Minimum RAM: 8GB
- Recommended RAM: 16GB+
- Disk Space: ~5GB for models and data

## Monitoring Training

### Using MLflow UI
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Then navigate to http://localhost:5000

### Check Logs
```bash
# View training metrics
cat mlruns/cpu-finetuning-experiment/*/metrics.json

# View run information
cat mlruns/cpu-finetuning-experiment/*/run_info.json
```

## Customization Options

### Adjust Training Parameters
Edit `cpu_finetuning_config.yaml`:
- `batch_size`: Increase if you have more RAM
- `num_epochs`: Increase for longer training
- `learning_rate`: Adjust based on convergence

### Use Different Models
CPU-friendly model options:
- `microsoft/DialoGPT-small` (117M params)
- `EleutherAI/gpt-neo-125M` (125M params)
- `google/flan-t5-small` (60M params)

### Modify Data Processing
Edit data processing in `run_cpu_finetuning.py`:
- Change sample limit (currently 100)
- Adjust max sequence length
- Add data augmentation

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Decrease `max_length` in config
- Use smaller model

### Slow Training
- Reduce `num_epochs`
- Decrease dataset size
- Enable mixed precision if CPU supports it

### Dependencies Issues
- Use `uv` for faster, more reliable installs
- Consider using conda for complex dependencies
- Check Python version compatibility (3.8+)

## Next Steps

1. **Scale Up**: Once working, gradually increase:
   - Dataset size
   - Model size
   - Training epochs

2. **Experiment Tracking**: Use MLflow to:
   - Compare different configurations
   - Track metrics over time
   - Save best models

3. **Model Deployment**: After training:
   - Export model for inference
   - Test on validation data
   - Deploy using the inference pipeline

## Files Created

- `cpu_finetuning_config.yaml` - Training configuration
- `run_cpu_finetuning.py` - Main training script
- `simple_cpu_finetune.py` - Lightweight test script
- `cpu_training_data_simple.json` - Processed training data
- `mlruns/` - MLflow tracking directory

## Support

For issues or questions:
1. Check the logs in `mlruns/` directory
2. Review the framework documentation in `docs/`
3. Ensure all dependencies are correctly installed

---

Generated: 2024-12-05
Framework: Jenga-AI LLM Finetuning Module
Mode: CPU-Optimized Training