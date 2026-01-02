import argparse
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel  # ADDED AutoModel
import dataclasses
from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import Trainer
from multitask_bert.tasks.base import BaseTask
from multitask_bert.tasks.classification import MultiHeadSingleLabelClassificationTask, MultiLabelClassificationTask
from multitask_bert.tasks.ner import NERTask

def get_task_class(task_type: str) -> BaseTask:
    """Maps a task type string to its corresponding class."""
    if task_type == "classification": # Changed to "classification"
        return MultiHeadSingleLabelClassificationTask # Changed to MultiHeadSingleLabelClassificationTask
    elif task_type == "multi_label_classification":
        return MultiLabelClassificationTask
    elif task_type == "ner":
        return NERTask
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def main(config_path: str):
    """
    Main function to run a multi-task experiment from a config file.
    """
    # 1. Load Config
    print("Loading experiment configuration...")
    config = load_experiment_config(config_path)

    # 2. Load Tokenizer (only for text-based models)
    tokenizer = None
    if config.model.backbone_type == "text":
        print(f"Loading tokenizer: {config.model.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config.tokenizer.pad_token_id = tokenizer.pad_token_id
    else:
        print(f"Skipping tokenizer (backbone_type={config.model.backbone_type})")

    # 3. Process Data
    print("Processing data for all tasks...")
    train_datasets, eval_datasets, updated_config = DataProcessor(config, tokenizer).process()
    config = updated_config

    # 4. Instantiate Tasks and Model
    print("Instantiating tasks and model...")
    
    # Load the base model configuration to get hidden_size
    # For text models, load from HuggingFace. For others, create a minimal config
    if config.model.backbone_type == "text":
        model_config = AutoConfig.from_pretrained(config.model.base_model)
    else:
        # For non-text backbones (sequential, audio), create a minimal config
        from types import SimpleNamespace
        model_config = SimpleNamespace(hidden_size=768)  # Default hidden size
    
    hidden_size = model_config.hidden_size

    tasks = [get_task_class(t.type)(config=t, hidden_size=hidden_size) for t in config.tasks] # Pass hidden_size
    
    # Create model with proper initialization
    model = MultiTaskModel(
        config=model_config,
        model_config=config.model,
        task_configs=config.tasks # Pass task_configs instead of instantiated tasks
    )
    
    
    # NOTE: The backbone is already loaded inside MultiTaskModel.__init__()
    # via BackboneManager.create(). We don't need to reload it here.
    
    # Resize embeddings if tokenizer was expanded (only for text models)
    if config.model.backbone_type == "text":
        model.resize_token_embeddings(len(tokenizer))


    # 5. Instantiate Trainer
    print("Instantiating trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )

    # 6. Start Training
    print("Starting training...")
    try:
        trainer.train()
        print("Training complete.")

        # 7. Final Evaluation
        print("Running final evaluation...")
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:")
        print(final_metrics)
    finally:
        # 8. Close the logger
        trainer.close()

    # 9. Save the updated config
    import os
    import yaml
    output_config_path = os.path.join(config.training.output_dir, "experiment_config.yaml")
    os.makedirs(config.training.output_dir, exist_ok=True)
    with open(output_config_path, 'w') as f:
        yaml.dump(dataclasses.asdict(config), f, indent=2)
    print(f"Updated experiment config saved to: {output_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-task experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/experiment.yaml",
        help="Path to the experiment YAML file."
    )
    args = parser.parse_args()
    main(args.config)