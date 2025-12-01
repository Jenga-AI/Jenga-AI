#!/usr/bin/env python3
"""
Inference Example - Jenga-AI Project

This example demonstrates how to load a trained model from MLflow
and use it for inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import mlflow
import pandas as pd
from multitask_bert.utils.mlflow_utils import initialize_mlflow

def generate_inference_data(num_samples=5):
    """Generate dummy data for inference."""
    # Matches the input size of SimpleModel (10)
    X = torch.randn(num_samples, 10)
    return X

def main():
    print("\n" + "=" * 60)
    print("  MLflow Inference Example - Jenga-AI")
    print("=" * 60 + "\n")

    # 1. Initialize MLflow to ensure we point to the correct tracking URI
    print("1. Initializing MLflow...")
    initialize_mlflow(
        config_path="mlflow_config.yaml",
        experiment_name="inference-test"
    )

    # 2. Load the model
    # We attempt to load the latest version of 'SimpleModel_Example' from the registry
    model_name = "SimpleModel_Example"
    model_stage = "None" # Or "Production", "Staging", "None" (for latest version)
    
    # Construct URI. If stage is None, we can use "latest" alias or just fetch by name
    # For simplicity, let's try to get the latest version.
    # Note: 'models:/<name>/<stage>' or 'models:/<name>/<version>'
    
    # If you want the absolute latest version regardless of stage:
    model_uri = f"models:/{model_name}/latest"
    
    print(f"\n2. Loading model from {model_uri}...")
    try:
        model = mlflow.pytorch.load_model(model_uri)
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model from registry: {e}")
        print("   Make sure you have run 'examples/mlflow_example.py' at least once to register the model.")
        return

    # 3. Prepare Data
    print("\n3. Generating inference data...")
    X_new = generate_inference_data(5)
    print(f"   Input shape: {X_new.shape}")

    # 4. Run Inference
    print("\n4. Running inference...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_new)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    # 5. Show Results
    print("\n5. Inference Results:")
    print("-" * 40)
    for i in range(len(X_new)):
        print(f"   Sample {i+1}: Class {predicted_classes[i].item()} "
              f"(Prob: {probabilities[i][predicted_classes[i]].item():.4f})")
    print("-" * 40)

    print("\n" + "=" * 60)
    print("  Inference completed successfully!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
