
import os
import json
import logging
import numpy as np
import torch
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from seq2seq_models.core.config import TranslationConfig
from seq2seq_models.training.callbacks import (
    TranslationMLflowCallback,
    EarlyFailureCallback,
    HuggingFaceHubCallback
)
from seq2seq_models.utils.mlflow_manager import MLflowManager
from seq2seq_models.utils.oov_restoration import restore_oov_words_in_translation

logger = logging.getLogger(__name__)

class TranslationTrainer:
    """
    A wrapper class around the Hugging Face `Seq2SeqTrainer` for Translation fine-tuning.
    """
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config: TranslationConfig, mlflow_manager: MLflowManager):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.mlflow_manager = mlflow_manager
        
        self.bleu = evaluate.load('bleu')
        self.chrf = evaluate.load('chrf')
        
        # Initialize COMET if requested
        self.comet_qe = None
        if "comet_qe" in self.config.evaluation_config.metrics:
            try:
                from comet import download_model, load_from_checkpoint
                logger.info("🔧 Initializing COMET-QE model...")
                model_path = download_model("wmt20-comet-qe-da")
                self.comet_qe = load_from_checkpoint(model_path)
                logger.info("✅ COMET-QE model loaded successfully")
            except ImportError:
                logger.warning("⚠️ COMET not installed. Install with: pip install unbabel-comet")
            except Exception as e:
                logger.warning(f"⚠️ Could not load COMET-QE model: {e}")

    def compute_metrics(self, eval_pred):
        """Compute translation metrics"""
        predictions, labels = eval_pred
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        result = {}
        
        if "bleu" in self.config.evaluation_config.metrics:
            try:
                bleu_score = self.bleu.compute(
                    predictions=decoded_preds,
                    references=[[label] for label in decoded_labels]
                )
                result['bleu'] = bleu_score['bleu']
            except Exception as e:
                logger.warning(f"⚠️ BLEU computation failed: {e}")
                result['bleu'] = 0.0
        
        if "chrf" in self.config.evaluation_config.metrics:
            try:
                chrf_score = self.chrf.compute(
                    predictions=decoded_preds,
                    references=decoded_labels
                )
                result['chrf'] = chrf_score['score']
            except Exception as e:
                logger.warning(f"⚠️ chrF computation failed: {e}")
                result['chrf'] = 0.0
            
        return result

    def train(self, resume_from_checkpoint=None):
        """
        Configures and executes the training process.
        """
        training_config = self.config.training_config
        generation_config = self.config.generation_config
        
        # Setup reporting backends
        report_to = []
        if training_config.tensorboard_enabled:
            report_to.append('tensorboard')
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            adam_beta1=training_config.adam_beta1,
            adam_beta2=training_config.adam_beta2,
            adam_epsilon=training_config.adam_epsilon,
            max_grad_norm=training_config.max_grad_norm,
            
            warmup_steps=training_config.warmup_steps,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.lr_scheduler,
            
            logging_dir=f"{training_config.output_dir}/logs",
            logging_steps=training_config.logging_steps,
            
            eval_strategy=training_config.eval_strategy,
            eval_steps=training_config.eval_steps,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            
            predict_with_generate=True,
            generation_max_length=generation_config.max_length,
            generation_num_beams=generation_config.num_beams,
            
            fp16=training_config.mixed_precision == 'fp16',
            gradient_checkpointing=training_config.gradient_checkpointing,
            
            dataloader_num_workers=training_config.dataloader_num_workers,
            
            save_total_limit=3,
            push_to_hub=False, # We handle this manually via callback
            
            label_smoothing_factor=training_config.label_smoothing,
            
            report_to=report_to
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        # Add callbacks
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_threshold=training_config.early_stopping_threshold
        )
        self.trainer.add_callback(early_stopping)
        self.trainer.add_callback(EarlyFailureCallback(patience=3))
        
        if self.mlflow_manager.enabled:
            self.trainer.add_callback(TranslationMLflowCallback(self.mlflow_manager))
            
        if self.config.deployment.push_to_hub and self.config.deployment.push_strategy == 'checkpoint':
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                self.trainer.add_callback(HuggingFaceHubCallback(
                    repo_id=self.config.deployment.hub_model_id,
                    token=hf_token,
                    output_dir=training_config.output_dir
                ))
            else:
                logger.warning("⚠️ HF_TOKEN not found, skipping HuggingFace Hub callback")

        logger.info("🚀 Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info(f"💾 Saving model to {training_config.output_dir}")
        self.trainer.save_model()
        
        # If PEFT model, save_model only saves adapters. We might want to ensure tokenizer is saved too.
        self.tokenizer.save_pretrained(training_config.output_dir)
        
        # If using PEFT, we might want to merge and save full model for easier deployment later, 
        # but for now standard save_model is fine as it saves adapter config.

        
        return train_result

    def evaluate(self, dataset=None):
        """Evaluate the model"""
        return self.trainer.evaluate(eval_dataset=dataset)
