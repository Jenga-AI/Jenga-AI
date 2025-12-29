#!/usr/bin/env python3
"""Debug data processing."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multitask_bert.core.config import load_experiment_config
from multitask_bert.data.data_processing import DataProcessor
from transformers import AutoTokenizer

def debug_data_processing():
    print("=== DEBUGGING DATA PROCESSING ===")
    
    # Load config
    config_path = "tests/configs/single_classification_cpu.yaml"
    config = load_experiment_config(config_path)
    print(f"Config loaded. Tasks: {[t.name for t in config.tasks]}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.tokenizer.pad_token_id = tokenizer.pad_token_id
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Process data
    data_processor = DataProcessor(config, tokenizer)
    print(f"DataProcessor created: {data_processor}")
    
    try:
        train_datasets, eval_datasets, updated_config = data_processor.process()
        print(f"Processing complete!")
        print(f"Train datasets: {len(train_datasets) if train_datasets else 0}")
        print(f"Eval datasets: {len(eval_datasets) if eval_datasets else 0}")
        
        if train_datasets:
            print(f"Train datasets type: {type(train_datasets)}")
            print(f"Train datasets keys/contents: {train_datasets}")
            
            # If it's a dict, get the first value
            if isinstance(train_datasets, dict):
                first_key = next(iter(train_datasets.keys()))
                train_dataset = train_datasets[first_key]
                print(f"Using key '{first_key}' for train dataset")
            else:
                train_dataset = train_datasets[0]
                
            print(f"First train dataset type: {type(train_dataset)}")
            print(f"First train dataset: {train_dataset}")
            
            # Try different approaches to get length
            try:
                length = len(train_dataset)
                print(f"Dataset length: {length}")
            except Exception as e:
                print(f"Error getting length: {e}")
                
            try:
                # Try accessing first element
                if hasattr(train_dataset, '__getitem__'):
                    first_item = train_dataset[0]
                    print(f"First item: {first_item}")
            except Exception as e:
                print(f"Error getting first item: {e}")
                
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_processing()