
import os
import sys
import argparse
import logging
import torch
import multiprocessing
from transformers import set_seed
from seq2seq_models.core.config import TranslationConfig
from seq2seq_models.data.data_processing import DatasetProcessor
from seq2seq_models.model.seq2seq_model import Seq2SeqModel
from seq2seq_models.training.trainer import TranslationTrainer
from seq2seq_models.utils.mlflow_manager import MLflowManager
from seq2seq_models.utils.domain_evaluator import DomainEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("translation_training.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Jenga-AI Translation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--domain_src", type=str, help="Path to domain source file for evaluation")
    parser.add_argument("--domain_tgt", type=str, help="Path to domain target file for evaluation")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode (fast, minimal data)")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    logger.info("🚀 Starting Jenga-AI Translation Pipeline")

    # 1. Load Configuration
    try:
        config = TranslationConfig.from_json(args.config)
        logger.info(f"✅ Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # 2. Setup System
    if args.test_mode:
        logger.info("⚠️ RUNNING IN TEST MODE")
        config.training_config.num_epochs = 1
        config.training_config.max_steps = 10
        config.training_config.save_steps = 5
        config.training_config.eval_steps = 5
        config.training_config.logging_steps = 1
        config.dataset_config.max_samples = 100
        config.mlflow_config.enabled = False # Disable MLflow for test mode by default

    set_seed(config.system_config.seed)
    
    if torch.cuda.is_available() and config.system_config.use_cuda:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # 3. Initialize MLflow
    mlflow_manager = MLflowManager(config.mlflow_config)
    mlflow_manager.setup_tracking()
    mlflow_manager.start_run()
    mlflow_manager.log_params(config.__dict__) # Log top-level config

    # 4. Load Model & Tokenizer
    logger.info("📦 Loading model and tokenizer...")
    model_factory = Seq2SeqModel(config)
    model, tokenizer = model_factory.create_model_and_tokenizer()
    
    # 5. Prepare Dataset
    logger.info("📚 Preparing dataset...")
    dataset_processor = DatasetProcessor(config)
    dataset_dict = dataset_processor.load_datasets()
    
    # Tokenize
    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]
        model_inputs = tokenizer(
            inputs, 
            max_length=config.training_config.max_length, 
            truncation=True
        )
        labels = tokenizer(
            text_target=targets, 
            max_length=config.training_config.max_length, 
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        num_proc=config.dataset_config.tokenization.num_proc,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    dataset_processor.validate_dataset(dataset_dict, tokenizer)

    # 6. Initialize Trainer
    trainer = TranslationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        config=config,
        mlflow_manager=mlflow_manager
    )

    # 7. Train
    logger.info("🏋️‍♂️ Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. Final Evaluation
    logger.info("📊 Running final evaluation...")
    metrics = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Final Test Metrics: {metrics}")
    mlflow_manager.log_metrics(metrics)

    # 9. Domain Evaluation (Optional)
    if args.domain_src and args.domain_tgt:
        domain_evaluator = DomainEvaluator(args.domain_src, args.domain_tgt)
        domain_metrics = domain_evaluator.evaluate(
            model, 
            tokenizer, 
            trainer.bleu, 
            trainer.chrf, 
            trainer.comet_qe
        )
        mlflow_manager.log_metrics(domain_metrics)

    # 10. Finish
    mlflow_manager.end_run()
    logger.info("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
