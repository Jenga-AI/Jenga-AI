import argparse
from transformers import AutoTokenizer
from multitask_bert.core.config import load_experiment_config
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.training.trainer import Trainer

def main(config_path: str):
    """
    Main function to run a multi-task training experiment.
    """
    # 1. Load configuration
    print("Loading experiment configuration...")
    experiment_config = load_experiment_config(config_path)

    # 2. Initialize tokenizer
    print(f"Loading tokenizer: {experiment_config.model.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(experiment_config.model.base_model)
    # Set pad_token_id if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        experiment_config.tokenizer.pad_token_id = tokenizer.pad_token_id


    # 3. Process data
    print("Processing data for all tasks...")
    data_processor = DataProcessor(experiment_config, tokenizer)
    train_datasets, eval_datasets, updated_config = data_processor.process()
    
    # The data processor might update the config (e.g., num_labels for NER)
    experiment_config = updated_config 

    # 4. Instantiate model
    print("Instantiating the multi-task model...")
    model = MultiTaskModel(
        config=AutoTokenizer.from_pretrained(experiment_config.model.base_model).get_config(),
        model_config=experiment_config.model,
        tasks=experiment_config.tasks
    )

    # 5. Initialize Trainer
    print("Initializing the trainer...")
    trainer = Trainer(
        config=experiment_config,
        model=model,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )

    # 6. Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # 7. Final evaluation
    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    print("Final Evaluation Metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")
        
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a JengaAI multi-task experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="hackathon_mvp.yaml",
        help="Path to the experiment YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
