#!/usr/bin/env python3
"""
MLflow Example - Jenga-AI Project

This example demonstrates how to use MLflow for experiment tracking,
parameter logging, metric logging, and model management.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from multitask_bert.utils.mlflow_utils import (
    initialize_mlflow,
    log_experiment_config,
    log_metrics_dict,
    log_dataset_info,
    set_tags,
    end_run
)
import mlflow


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_dummy_data(num_samples=100):
    """Generate dummy data for demonstration."""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return X, y


def train_epoch(model, X, y, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).float().mean().item()
    
    return loss.item(), accuracy


def main():
    """Main example function."""
    
    print("\n" + "=" * 60)
    print("  MLflow Example - Jenga-AI")
    print("=" * 60 + "\n")
    
    # Initialize MLflow
    print("1. Initializing MLflow...")
    experiment_id = initialize_mlflow(
        config_path="mlflow_config.yaml",
        experiment_name="mlflow-example"
    )
    
    # Configuration for the experiment
    config = {
        "model": {
            "input_size": 10,
            "hidden_size": 20,
            "output_size": 2,
        },
        "training": {
            "learning_rate": 0.01,
            "num_epochs": 5,
            "batch_size": 32,
        },
        "optimizer": "SGD"
    }
    
    # Generate dummy data
    print("\n2. Generating dummy data...")
    X_train, y_train = generate_dummy_data(100)
    X_val, y_val = generate_dummy_data(30)
    
    # Create model
    print("\n3. Creating model...")
    model = SimpleModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        output_size=config["model"]["output_size"]
    )
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()
    
    # Start MLflow run
    print("\n4. Starting MLflow run...")
    with mlflow.start_run(run_name="simple_model_example") as run:
        
        print(f"   Run ID: {run.info.run_id}")
        
        # Log configuration
        print("\n5. Logging configuration...")
        log_experiment_config(config)
        
        # Log dataset information
        print("\n6. Logging dataset info...")
        log_dataset_info(
            dataset_name="dummy_data",
            train_size=len(X_train),
            val_size=len(X_val)
        )
        
        # Set tags
        print("\n7. Setting tags...")
        set_tags({
            "model_type": "simple_nn",
            "purpose": "example",
            "framework": "pytorch"
        })
        
        # Training loop
        print("\n8. Training model...")
        for epoch in range(config["training"]["num_epochs"]):
            # Train
            train_loss, train_acc = train_epoch(
                model, X_train, y_train, optimizer, criterion
            )
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val).float().mean().item()
            
            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
            log_metrics_dict(metrics, step=epoch)
            
            print(f"   Epoch {epoch+1}/{config['training']['num_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}")
        
        # Log model
        print("\n9. Logging model...")
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="SimpleModel_Example"
        )
        
        # Log a sample artifact
        print("\n10. Logging artifacts...")
        artifact_file = project_root / "model_summary.txt"
        with open(artifact_file, 'w') as f:
            f.write(f"Model Summary\n")
            f.write(f"=============\n\n")
            f.write(f"Input Size: {config['model']['input_size']}\n")
            f.write(f"Hidden Size: {config['model']['hidden_size']}\n")
            f.write(f"Output Size: {config['model']['output_size']}\n")
            f.write(f"\nFinal Validation Accuracy: {val_acc:.4f}\n")
        
        mlflow.log_artifact(str(artifact_file))
        artifact_file.unlink()  # Clean up
        
        print("\n" + "=" * 60)
        print("  Run completed successfully!")
        print("=" * 60)
        print(f"\n  Run ID: {run.info.run_id}")
        print(f"  Artifact URI: {run.info.artifact_uri}")
        print(f"\n  View in MLflow UI:")
        print(f"  http://localhost:5000/#/experiments/{experiment_id}/runs/{run.info.run_id}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
        end_run(status="KILLED")
    except Exception as e:
        print(f"\n\nError: {e}")
        end_run(status="FAILED")
        raise
