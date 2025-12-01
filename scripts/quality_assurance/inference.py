#!/usr/bin/env python3
"""
QA Model Inference Script

This script loads the trained QA multi-head classifier model,
logs it to MLflow, and performs inference on test data.
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import json
import mlflow
import mlflow.pytorch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Heads and Labels (same as training) ---
HEAD_SUBMETRIC_LABELS = {
    "opening": ["Use of call opening phrase"],
    "listening": ["Caller was not interrupted", "Empathizes with the caller", "Paraphrases or rephrases the issue", "Uses 'please' and 'thank you'", "Does not hesitate or sound unsure"],
    "proactiveness": ["Willing to solve extra issues", "Confirms satisfaction with action points", "Follows up on case updates"],
    "resolution": ["Gives accurate information", "Correct language use", "Consults if unsure", "Follows correct steps", "Explains solution process clearly"],
    "hold": ["Explains before placing on hold", "Thanks caller for holding"],
    "closing": ["Proper call closing phrase used"]
}

qa_heads_config = {head: len(labels) for head, labels in HEAD_SUBMETRIC_LABELS.items()}

# --- Model Class (same as training) ---
class MultiHeadQAClassifier(nn.Module):
    def __init__(self, model_name, heads_config, dropout):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({head: nn.Linear(hidden_size, output_dim) for head, output_dim in heads_config.items()})

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(output.last_hidden_state[:, 0])
        logits = {}
        for head_name, head_layer in self.heads.items():
            out = head_layer(pooled_output)
            logits[head_name] = torch.sigmoid(out)
        return {"logits": logits}

def load_model(model_path, base_model="distilbert-base-uncased", dropout=0.1):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Initialize model architecture
    model = MultiHeadQAClassifier(base_model, qa_heads_config, dropout)
    
    # Load weights
    checkpoint_path = Path(model_path) / "pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    print("✓ Model loaded successfully")
    return model

def load_tokenizer(model_path):
    """Load the tokenizer from checkpoint."""
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return tokenizer

def predict(model, tokenizer, text, max_length=128, threshold=0.5, device='cpu'):
    """Perform inference on a single text."""
    model.eval()
    model.to(device)
    
    # Tokenize
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Process predictions
    predictions = {}
    for head, logits in outputs["logits"].items():
        probs = logits.cpu().numpy()[0]
        binary_preds = (probs > threshold).astype(int)
        predictions[head] = {
            "probabilities": probs.tolist(),
            "predictions": binary_preds.tolist(),
            "labels": HEAD_SUBMETRIC_LABELS[head]
        }
    
    return predictions

def log_model_to_mlflow(model, model_path, experiment_name="QA_Model_Inference"):
    """Log the model to MLflow."""
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="QA_Model_Manual_Log") as run:
        # Log parameters
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("base_model", "distilbert-base-uncased")
        mlflow.log_param("heads", list(qa_heads_config.keys()))
        
        # Log the model
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name="QA_MultiHead_Model"
        )
        
        # Log artifacts
        mlflow.log_artifacts(str(model_path), "model_files")
        
        print(f"✓ Model logged to MLflow")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Experiment: {experiment_name}")
        
        return run.info.run_id

def main():
    print("="*60)
    print("  QA Model Inference")
    print("="*60)
    
    # Configuration
    model_path = project_root / "tests/outputs/qa_model_versions/qa_model_final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    
    # Log to MLflow
    print("\nLogging model to MLflow...")
    run_id = log_model_to_mlflow(model, model_path)
    
    # Test inference
    print("\n" + "="*60)
    print("  Testing Inference")
    print("="*60)
    
    test_text = """Welcome to our helpline. I'm here to support you in times of distress. 
Can you tell me what's been happening? I understand your situation. 
It takes great courage to reach out for help. Let's find a way to ensure your safety."""
    
    print(f"\nTest Text: {test_text[:100]}...")
    print("\nPredictions:")
    
    predictions = predict(model, tokenizer, test_text, device=device)
    
    for head, pred_data in predictions.items():
        print(f"\n{head.upper()}:")
        for i, (label, prob, pred) in enumerate(zip(
            pred_data['labels'], 
            pred_data['probabilities'], 
            pred_data['predictions']
        )):
            status = "✓" if pred == 1 else "✗"
            print(f"  {status} {label}: {prob:.3f}")
    
    # Save predictions to file
    output_file = project_root / "tests/outputs/qa_inference_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "text": test_text,
            "predictions": predictions
        }, f, indent=2)
    
    print(f"\n✓ Predictions saved to {output_file}")
    print("\n" + "="*60)
    print(f"Model ready for inference!")
    print(f"MLflow Run ID: {run_id}")
    print("="*60)

if __name__ == "__main__":
    main()
