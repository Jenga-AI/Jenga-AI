"""
JengaAI Security Model Training Script
Dedicated script for training security models (anomaly detection, fraud detection, etc.)
using the Sequential backbone for tabular data.
"""
import argparse
import torch
from types import SimpleNamespace
import dataclasses
from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.security_trainer import SecurityTrainer
from multitask_bert.training.callbacks import SecuritySentinelCallback
from multitask_bert.tasks import TASK_REGISTRY

def main(config_path: str):
    """
    Main function to run a security experiment from a config file.
    """
    # 1. Load Config
    print("Loading security experiment configuration...")
    config = load_experiment_config(config_path)
    
    # 2. No Tokenizer for Security Models (tabular data)
    print(f"Skipping tokenizer (backbone_type={config.model.backbone_type})")
    tokenizer = None

    # 3. Process Data
    print("Processing security data...")
    train_datasets, eval_datasets, updated_config = DataProcessor(config, tokenizer).process()
    config = updated_config

    # 4. Create Model Config for Sequential Backbone
    print("Creating model configuration for security backbone...")
    # For sequential/tabular models, we need a PretrainedConfig (required by HuggingFace)
    from transformers import PretrainedConfig
    
    class SecurityConfig(PretrainedConfig):
        model_type = "security_mlp"
        def __init__(self, hidden_size=768, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = hidden_size
    
    model_config = SecurityConfig(hidden_size=768)
    hidden_size = model_config.hidden_size

    # 5. Instantiate Model
    print("Instantiating security model...")
    model = MultiTaskModel(
        config=model_config,
        model_config=config.model,
        task_configs=config.tasks
    )
    
    # Note: No need to resize embeddings for non-text models

    # 6. Instantiate Security Trainer with Sentinel
    print("Instantiating security trainer with sentinel...")
    sentinel = SecuritySentinelCallback(threshold=0.9, action_target="firewall")
    trainer = SecurityTrainer(
        config=config,
        model=model,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        callbacks=[sentinel] # The Trainer will add NestedLearning internally if missing, but SecurityTrainer handles it in __init__
    )

    # 7. Start Training
    print("Starting security model training...")
    try:
        trainer.train()
        print("Training complete.")

        # 8. Final Evaluation
        print("Running final evaluation...")
        final_metrics = trainer.evaluate()
        print("Final evaluation metrics:")
        print(final_metrics)
    finally:
        # 9. Close the logger
        trainer.close()

    # 10. Save the updated config
    import os
    import yaml
    output_config_path = os.path.join(config.training.output_dir, "experiment_config.yaml")
    os.makedirs(config.training.output_dir, exist_ok=True)
    with open(output_config_path, 'w') as f:
        yaml.dump(dataclasses.asdict(config), f, indent=2)
    print(f"Updated experiment config saved to: {output_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a security model experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/experiment_security.yaml",
        help="Path to the security experiment YAML file."
    )
    args = parser.parse_args()
    main(args.config)
