"""
Test suite for updated BaseTrainer with anti-forgetting support
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_base_trainer_imports():
    """Test that BaseTrainer can import with new parameters"""
    from llm_finetuning.training.base_trainer import BaseTrainer
    print("✅ BaseTrainer imports successfully")

def test_base_trainer_without_configs():
    """Test BaseTrainer works without anti-forgetting configs (backward compatibility)"""
    from llm_finetuning.training.base_trainer import BaseTrainer
    from llm_finetuning.core.config import TrainingConfig, LoggingConfig
    
    # Create minimal training config
    training_config = TrainingConfig(
        output_dir="tests/output/test_base_trainer",
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=10,
        logging_config=LoggingConfig()
    )
    
    # Create dummy model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy dataset
    dataset = Dataset.from_dict({
        "text": ["Hello world", "Test sentence"]
    })
    
    # Create trainer without anti-forgetting configs
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        training_config=training_config
    )
    
    assert trainer.model is not None
    print("✅ BaseTrainer backward compatibility works")

def test_base_trainer_with_peft():
    """Test BaseTrainer with PEFT config"""
    from llm_finetuning.training.base_trainer import BaseTrainer
    from llm_finetuning.core.config import TrainingConfig, LoggingConfig
    from jenga_ai.core.peft import PEFTConfig
    
    training_config = TrainingConfig(
        output_dir="tests/output/test_base_trainer_peft",
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=10,
        logging_config=LoggingConfig()
    )
    
    peft_config = PEFTConfig(enabled=True, r=4, lora_alpha=8)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = Dataset.from_dict({"text": ["Hello", "World"]})
    
    # Create trainer with PEFT
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        training_config=training_config,
        peft_config=peft_config
    )
    
    # Check that PEFT was applied
    assert hasattr(trainer.model, 'print_trainable_parameters')
    
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    
    assert trainable < total * 0.1
    print(f"✅ BaseTrainer with PEFT works: {trainable}/{total} trainable")

def test_base_trainer_with_freezing():
    """Test BaseTrainer with freezing config"""
    from llm_finetuning.training.base_trainer import BaseTrainer
    from llm_finetuning.core.config import TrainingConfig, LoggingConfig
    from jenga_ai.core.peft import FreezingConfig
    
    training_config = TrainingConfig(
        output_dir="tests/output/test_base_trainer_freeze",
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=10,
        logging_config=LoggingConfig()
    )
    
    freezing_config = FreezingConfig(enabled=True, freeze_embeddings=True)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = Dataset.from_dict({"text": ["Test"]})
    
    trainer = BaseTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        training_config=training_config,
        freezing_config=freezing_config
    )
    
    # Check that some params are frozen
    frozen = sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)
    assert frozen > 0
    print(f"✅ BaseTrainer with freezing works: {frozen} frozen params")

if __name__ == "__main__":
    print("🧪 Running BaseTrainer tests...\n")
    
    test_base_trainer_imports()
    
    test_base_trainer_without_configs()
    
    test_base_trainer_with_peft()
    
    test_base_trainer_with_freezing()
    
    print("\n✅ All BaseTrainer tests passed!")
