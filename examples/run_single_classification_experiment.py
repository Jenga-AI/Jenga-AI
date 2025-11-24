import os
import torch
from transformers import AutoTokenizer, AutoConfig

from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer

def run_experiment(config_path: str):
    # 1. Load experiment configuration
    config = load_experiment_config(config_path)

    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Update model's embedding layer to account for new pad token
        # This would typically be handled by the model's resize_token_embeddings
        # but for a simple example, we'll assume the base model handles it or it's not critical.

    # 3. Process data
    data_processor = DataProcessor(config, tokenizer)
    train_datasets, eval_datasets, updated_config = data_processor.process()
    config = updated_config # Use the updated config (e.g., with num_labels for NER)

    # 4. Initialize model
    model_config = AutoConfig.from_pretrained(config.model.base_model)
    model = MultiTaskModel(model_config, config.model, config.tasks)

    # 5. Initialize and run trainer
    trainer = Trainer(config, model, tokenizer, train_datasets, eval_datasets)
    trainer.train()
    trainer.close()

    print(f"Experiment '{config.project_name}' completed successfully!")

if __name__ == "__main__":
    # Ensure the dummy data path is correct relative to where the script is run
    # For this example, we assume it's run from the project root or 'examples' directory
    # If run from project root: config_path = "examples/single_classification_experiment.yaml"
    # If run from examples directory: config_path = "single_classification_experiment.yaml"
    
    # Let's assume it's run from the project root for consistency
    config_path = "examples/single_classification_experiment.yaml"
    run_experiment(config_path)
