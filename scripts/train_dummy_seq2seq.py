#!/usr/bin/env python3
"""
Train Dummy Seq2Seq Model

This script trains a simple Seq2Seq model using dummy data to demonstrate
MLflow integration and CPU-mindful execution.
"""

import sys
import json
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multitask_bert.utils.mlflow_utils import initialize_mlflow, log_metrics_dict

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def prepare_data(filepath):
    """Load and prepare data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    pairs = []
    vocab = set()
    
    for item in data:
        src = item['translation']['en']
        tgt = item['translation']['fr']
        pairs.append((src, tgt))
        
        for word in src.split():
            vocab.add(word)
        for word in tgt.split():
            vocab.add(word)
            
    # Simple vocabulary mapping
    word2idx = {word: i+2 for i, word in enumerate(vocab)} # 0 for SOS, 1 for EOS
    word2idx['SOS'] = 0
    word2idx['EOS'] = 1
    
    return pairs, word2idx

def tensorFromSentence(word2idx, sentence, device):
    indexes = [word2idx[word] for word in sentence.split()]
    indexes.append(1) # EOS
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    _, encoder_hidden = encoder(input_tensor[0], encoder_hidden) # Simple pass for demo
    # In a real seq2seq we'd iterate over input length

    decoder_input = torch.tensor([[0]], device=device) # SOS

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def main():
    print("="*60)
    print("  Dummy Seq2Seq Training")
    print("="*60)

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize MLflow
    experiment_id = initialize_mlflow(experiment_name="dummy_seq2seq")
    
    # 3. Load Data
    data_path = project_root / "tests/dummy_seq2seq_data.json"
    print(f"Loading data from: {data_path}")
    pairs, word2idx = prepare_data(data_path)
    print(f"Loaded {len(pairs)} pairs")
    print(f"Vocabulary size: {len(word2idx)}")

    # 4. Model Setup
    hidden_size = 128
    encoder = Encoder(len(word2idx), hidden_size).to(device)
    decoder = Decoder(hidden_size, len(word2idx)).to(device)

    learning_rate = 0.01
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # 5. Training Loop
    n_epochs = 20
    print(f"Starting training for {n_epochs} epochs...")
    
    with mlflow.start_run(run_name="dummy_seq2seq_run") as run:
        mlflow.log_param("device", str(device))
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", n_epochs)
        
        for epoch in range(1, n_epochs + 1):
            total_loss = 0
            for pair in pairs:
                input_tensor = tensorFromSentence(word2idx, pair[0], device)
                target_tensor = tensorFromSentence(word2idx, pair[1], device)
                
                loss = train(input_tensor, target_tensor, encoder, decoder, 
                           encoder_optimizer, decoder_optimizer, criterion, device)
                total_loss += loss
            
            avg_loss = total_loss / len(pairs)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{n_epochs} - Loss: {avg_loss:.4f}")
            
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
        print("Training complete!")
        
        # Log model
        mlflow.pytorch.log_model(encoder, "encoder")
        mlflow.pytorch.log_model(decoder, "decoder")
        print("Models logged to MLflow")

if __name__ == "__main__":
    main()
