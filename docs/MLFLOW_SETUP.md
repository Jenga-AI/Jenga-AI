# MLflow Setup Guide - Jenga-AI

This guide covers the complete setup and usage of MLflow for experiment tracking, model management, and deployment in the Jenga-AI project.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Initialization](#initialization)
4. [Starting the MLflow Server](#starting-the-mlflow-server)
5. [Using MLflow in Training Scripts](#using-mlflow-in-training-scripts)
6. [MLflow UI Guide](#mlflow-ui-guide)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation

### 1. Install Dependencies

MLflow is already included in the project's `requirements.txt`. Install all dependencies:

```bash
cd /home/collins/Documents/Jenga-AI/Jenga-AI
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

Expected output: `MLflow version: 2.9.0` (or higher)

## Configuration

### Environment Variables (Optional)

MLflow can be configured using environment variables. Copy the example file:

```bash
cp .env.example .env
```

Edit `.env` to customize:

```bash
# Tracking URI (local or remote server)
MLFLOW_TRACKING_URI=./mlruns

# Default experiment name
MLFLOW_EXPERIMENT_NAME=jenga-ai-experiments

# Server configuration
MLFLOW_PORT=5000
MLFLOW_HOST=127.0.0.1
```

### Central Configuration File

The project uses `mlflow_config.yaml` for centralized configuration:

```yaml
tracking:
  uri: ./mlruns
  default_experiment: jenga-ai-experiments
  artifact_location: ./mlruns

server:
  host: 127.0.0.1
  port: 5000
  backend_store_uri: ./mlruns

logging:
  auto_log:
    pytorch: true
    transformers: true
```

## Initialization

### Run the Initialization Script

This script validates your MLflow setup, creates default experiments, and tests logging:

```bash
cd /home/collins/Documents/Jenga-AI/Jenga-AI
source .venv/bin/activate
python scripts/initialize_mlflow.py
```

**What it does:**
- ✓ Validates MLflow installation
- ✓ Loads configuration
- ✓ Creates the `mlruns` directory
- ✓ Sets up tracking URI
- ✓ Creates default experiment
- ✓ Tests logging functionality
- ✓ Lists existing experiments

**Expected output:**
```
============================================================
  MLflow Initialization - Jenga-AI Project
============================================================

============================================================
  Validating MLflow Installation
============================================================
✓ MLflow version: 2.9.0

============================================================
  Loading Configuration
============================================================
✓ Configuration loaded from: /path/to/mlflow_config.yaml

...

✓ MLflow initialization completed successfully!
```

## Starting the MLflow Server

### Using the Startup Script

The project includes a convenient bash script to start the MLflow UI:

```bash
cd /home/collins/Documents/Jenga-AI/Jenga-AI
source .venv/bin/activate
bash scripts/start_mlflow_server.sh
```

**What happens:**
- Reads configuration from `mlflow_config.yaml`
- Checks if the port is available
- Starts the MLflow server
- Opens access at `http://localhost:5000`

### Manual Startup

Alternatively, start the server manually:

```bash
mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlruns
```

### Access the UI

Open your browser and navigate to:
```
http://localhost:5000
```

## Using MLflow in Training Scripts

### Basic Example

```python
from multitask_bert.utils.mlflow_utils import (
    initialize_mlflow,
    log_experiment_config,
    log_metrics_dict,
    log_dataset_info,
    set_tags
)
import mlflow

# Initialize MLflow
initialize_mlflow(experiment_name="my-experiment")

# Start a run
with mlflow.start_run(run_name="training-run-1"):
    # Log configuration
    config = {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "num_epochs": 10
    }
    log_experiment_config(config)
    
    # Log dataset info
    log_dataset_info(
        dataset_name="qa_dataset",
        train_size=1000,
        val_size=200,
        test_size=100
    )
    
    # Set tags
    set_tags({
        "model_type": "distilbert",
        "task": "qa_scoring"
    })
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        # ... your training code ...
        
        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }
        log_metrics_dict(metrics, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="QA_Model_v1"
    )
```

### Run the Example Script

Test MLflow integration with the provided example:

```bash
cd /home/collins/Documents/Jenga-AI/Jenga-AI
source .venv/bin/activate
python examples/mlflow_example.py
```

This demonstrates:
- Initializing MLflow
- Logging parameters and configuration
- Logging metrics during training
- Logging models and artifacts
- Setting tags for organization

### Integration with Existing Scripts

Several training scripts already use MLflow:

- **Quality Assurance**: [`scripts/quality_assurance/train.py`](file:///home/collins/Documents/Jenga-AI/Jenga-AI/scripts/quality_assurance/train.py)
- **Classification**: [`scripts/classification/trainer.py`](file:///home/collins/Documents/Jenga-AI/Jenga-AI/scripts/classification/trainer.py)
- **Test Training**: [`tests/train_test.py`](file:///home/collins/Documents/Jenga-AI/Jenga-AI/tests/train_test.py)

These scripts load MLflow configuration from their respective YAML files.

## MLflow UI Guide

For detailed instructions on using the MLflow UI, see:
- [MLFLOW_GUIDE.md](file:///home/collins/Documents/Jenga-AI/Jenga-AI/tests/MLFLOW_GUIDE.md) - Comprehensive UI guide

### Quick UI Navigation

1. **Experiments Page**: View all experiments and their runs
2. **Run Details**: Click a run to see parameters, metrics, and artifacts
3. **Metrics Tab**: View training metrics as charts
4. **Compare Runs**: Select multiple runs to compare performance
5. **Model Registry**: Manage and version your models

## Best Practices

### 1. Experiment Organization

```python
# Use descriptive experiment names
experiment_name = "qa-model-distilbert-v1"

# Use meaningful run names
run_name = f"qa_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### 2. Comprehensive Logging

Log everything needed to reproduce your experiment:

```python
# Configuration parameters
log_experiment_config(config)

# Dataset information
log_dataset_info(dataset_name, train_size, val_size, test_size)

# Model architecture details
mlflow.log_param("model_architecture", "distilbert-base")
mlflow.log_param("num_layers", 6)

# Training hyperparameters
mlflow.log_params({
    "learning_rate": lr,
    "batch_size": batch_size,
    "optimizer": "AdamW"
})
```

### 3. Metric Logging

```python
# Log metrics at each epoch/step
for epoch in range(num_epochs):
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }
    log_metrics_dict(metrics, step=epoch)
```

### 4. Model Artifacts

```python
# Log model with signature
from mlflow.models import infer_signature

signature = infer_signature(input_example, output_example)
mlflow.pytorch.log_model(
    model,
    "model",
    signature=signature,
    registered_model_name="MyModel_v1"
)

# Log additional artifacts
mlflow.log_artifact("config.yaml")
mlflow.log_artifact("training_log.txt")
```

### 5. Tagging

Use tags to organize and filter experiments:

```python
set_tags({
    "project": "jenga-ai",
    "task": "qa_scoring",
    "model_type": "transformer",
    "stage": "production"
})
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

**Error**: `Port 5000 is already in use`

**Solution**:
```bash
# Find and kill the process
kill $(lsof -t -i:5000)

# Or use a different port
export MLFLOW_PORT=5001
bash scripts/start_mlflow_server.sh
```

#### 2. MLflow Not Found

**Error**: `ModuleNotFoundError: No module named 'mlflow'`

**Solution**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### 3. Experiment Not Showing in UI

**Solution**:
- Refresh the browser page (F5)
- Check the tracking URI matches
- Verify the experiment was created:
  ```python
  from mlflow.tracking import MlflowClient
  client = MlflowClient()
  print(client.search_experiments())
  ```

#### 4. Runs Not Logging

**Solution**:
- Ensure you're inside a `with mlflow.start_run():` block
- Check tracking URI is set correctly
- Verify write permissions on `mlruns` directory

### Getting Help

- **Official Documentation**: https://mlflow.org/docs/latest/
- **API Reference**: https://mlflow.org/docs/latest/python_api/index.html
- **Project Guide**: [MLFLOW_GUIDE.md](file:///home/collins/Documents/Jenga-AI/Jenga-AI/tests/MLFLOW_GUIDE.md)

## Advanced Usage

### Remote Tracking Server

To use a remote MLflow server:

1. Update `mlflow_config.yaml`:
```yaml
tracking:
  uri: http://remote-server:5000
```

2. Or set environment variable:
```bash
export MLFLOW_TRACKING_URI=http://remote-server:5000
```

### Cloud Storage for Artifacts

Configure S3/GCS for artifact storage:

```yaml
tracking:
  artifact_location: s3://my-bucket/mlflow-artifacts
```

Set AWS credentials in `.env`:
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Model Registry

Register and version models:

```python
# Log with registration
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="QA_Scorer"
)

# Promote to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="QA_Scorer",
    version=1,
    stage="Production"
)
```

## Summary

You now have MLflow fully set up for the Jenga-AI project! 

**Key files:**
- **Configuration**: `mlflow_config.yaml`, `.env.example`
- **Utilities**: `multitask_bert/utils/mlflow_utils.py`
- **Scripts**: `scripts/initialize_mlflow.py`, `scripts/start_mlflow_server.sh`
- **Example**: `examples/mlflow_example.py`

**Next steps:**
1. Run the initialization script
2. Start the MLflow server
3. Run the example script
4. Integrate MLflow into your training workflows
