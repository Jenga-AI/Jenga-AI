#!/usr/bin/env python3
"""
Sentiment Classification Training Test
=====================================
Tests single-task training with sentiment classification using bert-tiny.

Usage:
    python tests/integration/test_sentiment_training.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.memory_monitor import MemoryMonitor, check_memory_safety


def test_sentiment_training():
    """Test sentiment classification training with memory monitoring."""
    
    print("=" * 70)
    print("  SENTIMENT CLASSIFICATION TRAINING TEST")
    print("=" * 70)
    
    # Initialize memory monitor
    monitor = MemoryMonitor(interval=2.0, name="Sentiment Training Test")
    
    try:
        # Check memory safety first
        print("\n1. Memory Safety Check:")
        model_size_mb = 20  # bert-tiny is ~20MB
        is_safe = check_memory_safety(
            model_size_mb=model_size_mb,
            batch_size=2,
            sequence_length=64,
            safety_margin_gb=1.0  # Reduced safety margin for testing
        )
        
        if not is_safe:
            print("\n⚠️  WARNING: Memory might be tight, but proceeding with test...")
        
        # Start memory monitoring
        monitor.start()
        monitor.log_checkpoint("Test Start")
        
        # Import training modules (this loads dependencies)
        print("\n2. Loading training modules...")
        monitor.log_checkpoint("Before imports")
        
        from multitask_bert.core.config import load_experiment_config
        from multitask_bert.core.model import MultiTaskModel
        from multitask_bert.data.data_processing import DataProcessor
        from multitask_bert.training.trainer import Trainer
        from multitask_bert.tasks.classification import SingleLabelClassificationTask
        from transformers import AutoTokenizer, AutoConfig
        
        monitor.log_checkpoint("After imports")
        
        # Load configuration
        print("\n3. Loading configuration...")
        config_path = "tests/configs/single_classification_cpu.yaml"
        config = load_experiment_config(config_path)
        monitor.log_checkpoint("After config load")
        
        # Load tokenizer
        print(f"\n4. Loading tokenizer: {config.model.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config.tokenizer.pad_token_id = tokenizer.pad_token_id
        monitor.log_checkpoint("After tokenizer")
        
        # Process data
        print("\n5. Processing data...")
        try:
            data_processor = DataProcessor(config, tokenizer)
            train_datasets, eval_datasets, updated_config = data_processor.process()
            config = updated_config
            monitor.log_checkpoint("After data processing")
        except Exception as e:
            print(f"   Data processing error: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        # Check if we have data
        if not train_datasets:
            raise ValueError("No training datasets loaded!")
        
        print(f"   Number of training datasets: {len(train_datasets)}")
        print(f"   Number of eval datasets: {len(eval_datasets) if eval_datasets else 0}")
        
        # Get datasets by task name (they're dictionaries, not lists)
        task_name = config.tasks[0].name  # "SentimentClassification"
        train_dataset = train_datasets[task_name]
        eval_dataset = eval_datasets[task_name] if eval_datasets and task_name in eval_datasets else None
        
        print(f"   Training dataset type: {type(train_dataset)}")
        print(f"   Eval dataset type: {type(eval_dataset) if eval_dataset else 'None'}")
        
        try:
            print(f"   Training samples: {len(train_dataset)}")
        except Exception as e:
            print(f"   Error getting training dataset length: {e}")
            # Let's see what's in the dataset
            print(f"   Train dataset: {train_dataset}")
            raise
            
        try:
            print(f"   Eval samples: {len(eval_dataset) if eval_dataset else 0}")
        except Exception as e:
            print(f"   Error getting eval dataset length: {e}")
        
        # Check first few training samples
        try:
            print(f"   Sample training data:")
            for i in range(min(3, len(train_dataset))):
                sample = train_dataset[i]
                print(f"     {i}: {sample}")
        except Exception as e:
            print(f"   Error accessing training samples: {e}")
        
        # Create task and model
        print("\n6. Creating model...")
        tasks = [SingleLabelClassificationTask(t) for t in config.tasks]
        model_config = AutoConfig.from_pretrained(config.model.base_model)
        
        model = MultiTaskModel.from_pretrained(
            config.model.base_model,
            config=model_config,
            model_config=config.model,
            tasks=tasks
        )
        model.resize_token_embeddings(len(tokenizer))
        monitor.log_checkpoint("After model creation")
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create trainer
        print("\n7. Creating trainer...")
        trainer = Trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_datasets=train_datasets,
            eval_datasets=eval_datasets
        )
        monitor.log_checkpoint("After trainer creation")
        
        # Run a few training steps (not full training)
        print("\n8. Running training test (5 steps only)...")
        print("   This is a connectivity test, not full training")
        
        # Modify config for quick test
        original_epochs = config.training.num_epochs
        original_steps = getattr(config.training, 'max_steps', None)
        
        config.training.num_epochs = 1
        config.training.max_steps = 5  # Just 5 steps for testing
        config.training.eval_strategy = "no"  # Skip evaluation for quick test
        config.training.save_strategy = "no"   # Skip saving for quick test
        
        start_time = time.time()
        try:
            trainer.train()
            training_time = time.time() - start_time
            print(f"   Training test completed in {training_time:.2f} seconds")
            monitor.log_checkpoint("After training")
            
            # Test a forward pass
            print("\n9. Testing forward pass...")
            model.eval()
            with torch.no_grad():
                sample_input = train_dataset[0]
                if hasattr(sample_input, 'keys'):
                    # Remove labels for inference
                    input_dict = {k: v.unsqueeze(0) for k, v in sample_input.items() 
                                if k in ['input_ids', 'attention_mask']}
                    outputs = model(**input_dict)
                    print(f"   Forward pass successful. Output shape: {outputs.logits.shape}")
                    monitor.log_checkpoint("After forward pass")
            
            print("\n✅ SENTIMENT TRAINING TEST PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            monitor.log_checkpoint("Training failed")
            return False
            
        finally:
            trainer.close() if hasattr(trainer, 'close') else None
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        monitor.log_checkpoint("Test failed")
        return False
        
    finally:
        monitor.stop()
        monitor.print_report()
        monitor.save_report("tests/outputs/sentiment_training_memory_report.json")


if __name__ == "__main__":
    # Add missing import for torch
    import torch
    
    # Ensure output directory exists
    os.makedirs("tests/outputs", exist_ok=True)
    
    success = test_sentiment_training()
    exit_code = 0 if success else 1
    
    print(f"\n{'='*70}")
    print(f"  TEST {'PASSED' if success else 'FAILED'}")
    print(f"{'='*70}")
    
    sys.exit(exit_code)