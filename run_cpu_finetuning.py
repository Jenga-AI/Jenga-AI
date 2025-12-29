#!/usr/bin/env python3
"""
CPU-Friendly LLM Finetuning Script with MLflow Integration
Uses synthetic test data for efficient CPU training
"""

import os
import sys
import json
import yaml
import torch
import mlflow
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_finetuning.core.config import (
    LLMFinetuningConfig, ModelConfig, DataConfig, 
    TrainingConfig, PeftConfig, LoggingConfig
)
from llm_finetuning.pipeline import FinetuningPipeline
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')

def initialize_mlflow():
    """Initialize MLflow tracking"""
    print("\n=== Initializing MLflow ===")
    
    # Set tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    experiment_name = "cpu-llm-finetuning"
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Could not create/get experiment: {e}")
        experiment_id = None
    
    if experiment_id:
        mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow experiment set: {experiment_name}")
    
    return experiment_name

def prepare_synthetic_data(data_path):
    """Process synthetic data for training"""
    print("\n=== Preparing Synthetic Data ===")
    
    # Read the synthetic data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to the format expected by the framework
    processed_data = []
    for item in data[:100]:  # Limit to 100 samples for CPU efficiency
        # Extract text for language modeling
        text = item.get('text', '')
        if text:
            processed_data.append({
                'text': text,
                'labels': item.get('labels', {}),
                'sample_id': item.get('sample_id', 'unknown'),
                'quality': item.get('quality_level', 'unknown')
            })
    
    # Save processed data
    output_path = project_root / "cpu_training_data.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"✓ Processed {len(processed_data)} samples")
    print(f"✓ Saved to: {output_path}")
    
    return str(output_path)

def setup_cpu_environment():
    """Configure environment for CPU training"""
    print("\n=== Setting up CPU Environment ===")
    
    # Disable CUDA even if available
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set threading for CPU efficiency
    torch.set_num_threads(4)
    
    # Check device
    device = torch.device('cpu')
    print(f"✓ Using device: {device}")
    print(f"✓ Number of CPU threads: {torch.get_num_threads()}")
    
    return device

def create_training_config(config_path):
    """Load and adjust configuration for CPU training"""
    print("\n=== Loading Configuration ===")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Process the synthetic data
    data_path = prepare_synthetic_data(config_dict['data']['path'])
    
    # Update data path to processed data
    config_dict['data']['path'] = data_path
    
    # Create configuration objects
    peft_config = None
    if config_dict['model'].get('peft_config'):
        peft_config = PeftConfig(**config_dict['model']['peft_config'])
    
    model_config = ModelConfig(
        name=config_dict['model']['name'],
        quantization=config_dict['model'].get('quantization'),
        peft_config=peft_config
    )
    
    data_configs = [DataConfig(**config_dict['data'])]
    
    logging_config = None
    if config_dict['training'].get('logging_config'):
        logging_config = LoggingConfig(**config_dict['training']['logging_config'])
    
    training_config = TrainingConfig(
        output_dir=config_dict['training']['output_dir'],
        learning_rate=float(config_dict['training']['learning_rate']),
        batch_size=config_dict['training']['batch_size'],
        num_epochs=config_dict['training']['num_epochs'],
        gradient_accumulation_steps=config_dict['training']['gradient_accumulation_steps'],
        logging_steps=config_dict['training']['logging_steps'],
        save_steps=config_dict['training']['save_steps'],
        logging_config=logging_config
    )
    
    config = LLMFinetuningConfig(
        model=model_config,
        data=data_configs,
        training=training_config
    )
    
    print("✓ Configuration loaded and adjusted for CPU")
    return config

def log_training_metrics(config):
    """Log training configuration to MLflow"""
    print("\n=== Logging to MLflow ===")
    
    try:
        # Log parameters
        mlflow.log_param("model_name", config.model.name)
        mlflow.log_param("batch_size", config.training.batch_size)
        mlflow.log_param("learning_rate", config.training.learning_rate)
        mlflow.log_param("num_epochs", config.training.num_epochs)
        mlflow.log_param("gradient_accumulation", config.training.gradient_accumulation_steps)
        
        if config.model.peft_config:
            mlflow.log_param("peft_type", config.model.peft_config.peft_type)
            mlflow.log_param("lora_r", config.model.peft_config.r)
            mlflow.log_param("lora_alpha", config.model.peft_config.lora_alpha)
        
        # Log system info
        mlflow.log_param("device", "cpu")
        mlflow.log_param("num_threads", torch.get_num_threads())
        
        print("✓ Parameters logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not log to MLflow: {e}")

def main():
    parser = argparse.ArgumentParser(description="CPU-Friendly LLM Finetuning")
    parser.add_argument(
        "--config", 
        default="cpu_finetuning_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        default="cpu-llm-finetuning",
        help="MLflow experiment name"
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  CPU-Friendly LLM Finetuning with Synthetic Data")
    print("="*60)
    
    # Initialize MLflow
    experiment_name = initialize_mlflow()
    
    # Setup CPU environment
    device = setup_cpu_environment()
    
    # Load configuration
    config = create_training_config(args.config)
    
    # Start MLflow run
    run_name = f"cpu-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        with mlflow.start_run(run_name=run_name):
            print(f"\n✓ Started MLflow run: {run_name}")
            
            # Log configuration to MLflow
            log_training_metrics(config)
            
            # Create and run pipeline
            print("\n=== Starting Training Pipeline ===")
            pipeline = FinetuningPipeline(config)
            
            # Run training
            pipeline.run()
            
            # Log completion
            mlflow.log_metric("training_completed", 1.0)
            print("\n✓ Training completed successfully!")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        mlflow.log_metric("training_failed", 1.0)
        mlflow.log_param("error_message", str(e))
        raise
    
    print("\n=== Training Summary ===")
    print(f"✓ Model: {config.model.name}")
    print(f"✓ Output directory: {config.training.output_dir}")
    print(f"✓ MLflow experiment: {experiment_name}")
    print("\nView results in MLflow UI:")
    print("  mlflow ui --host 0.0.0.0 --port 5000")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()