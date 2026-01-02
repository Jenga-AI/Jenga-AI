
import torch
import logging
from transformers import (
    TrainingArguments, 
    Trainer as HuggingFaceTrainer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer as HuggingFaceSeq2SeqTrainer, 
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM
)
from typing import Optional

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_config,
                 peft_config=None, freezing_config=None, distillation_config=None):
        """
        Initialize base trainer with anti-catastrophic forgetting support
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_config: Training configuration
            peft_config: Optional PEFT configuration (LoRA, etc.)
            freezing_config: Optional layer freezing configuration
            distillation_config: Optional knowledge distillation configuration
        """
        
        # Apply freezing first (if enabled)
        if freezing_config and freezing_config.enabled:
            try:
                from jenga_ai.core.peft import freeze_layers
                model = freeze_layers(model, freezing_config)
            except ImportError:
                logger.warning("⚠️ jenga_ai.core.peft not available, skipping layer freezing")
        
        # Apply PEFT (if enabled)
        if peft_config and peft_config.enabled:
            try:
                from jenga_ai.core.peft import apply_peft
                model = apply_peft(model, peft_config, model_type="auto")
            except ImportError:
                logger.warning("⚠️ jenga_ai.core.peft not available, skipping PEFT")
        
        # Wrap in distillation (if enabled)
        if distillation_config and distillation_config.enabled:
            try:
                from jenga_ai.core.distillation.teacher_student import TeacherStudentWrapper
                
                # Load teacher model
                logger.info(f"📚 Loading teacher model: {distillation_config.teacher_model}")
                teacher_model = AutoModelForCausalLM.from_pretrained(distillation_config.teacher_model)
                
                model = TeacherStudentWrapper(
                    student_model=model,
                    teacher_model=teacher_model,
                    distillation_alpha=distillation_config.distillation_alpha,
                    temperature=distillation_config.temperature
                )
            except ImportError:
                logger.warning("⚠️ jenga_ai.core.distillation not available, skipping distillation")
        
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # For causal LM

    def _get_training_arguments(self, is_seq2seq: bool = False):
        report_to = []
        run_name = None
        if self.training_config.logging_config:
            if self.training_config.logging_config.report_to:
                report_to.append(self.training_config.logging_config.report_to)
            run_name = self.training_config.logging_config.run_name

        common_args = {
            "output_dir": self.training_config.output_dir,
            "learning_rate": self.training_config.learning_rate,
            "per_device_train_batch_size": self.training_config.batch_size,
            "per_device_eval_batch_size": self.training_config.batch_size,
            "num_train_epochs": self.training_config.num_epochs,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "logging_steps": self.training_config.logging_steps,
            "save_steps": self.training_config.save_steps,
            "eval_strategy": "steps" if self.eval_dataset else "no",
            "eval_steps": self.training_config.save_steps if self.eval_dataset else None,
            "fp16": torch.cuda.is_available(),
            "report_to": report_to if report_to else None,
            "run_name": run_name,
        }

        if is_seq2seq:
            return Seq2SeqTrainingArguments(**common_args)
        else:
            return TrainingArguments(**common_args)

    def train(self, is_seq2seq: bool = False):
        training_args = self._get_training_arguments(is_seq2seq)

        if is_seq2seq:
            trainer_class = HuggingFaceSeq2SeqTrainer
        else:
            trainer_class = HuggingFaceTrainer
        
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.train()
