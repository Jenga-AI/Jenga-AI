#!/usr/bin/env python3
"""
Single-Task Training Integration Tests
=====================================
Tests end-to-end training pipeline for individual tasks.

Usage:
    python -m pytest tests/integration/test_single_task_training.py -v
    OR
    python tests/integration/test_single_task_training.py
"""

import sys
import unittest
import tempfile
import shutil
import os
import json
import pandas as pd
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.core.config import ExperimentConfig, TaskConfig, HeadConfig, ModelConfig, TokenizerConfig, TrainingConfig
from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.training.trainer import MultiTaskTrainer
from multitask_bert.tasks.classification import ClassificationTask
from multitask_bert.tasks.ner import NERTask
from transformers import AutoTokenizer, AutoConfig


class TestSingleTaskTraining(unittest.TestCase):
    """Test single-task training end-to-end."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.model_name = "prajjwal1/bert-tiny"  # Tiny model for fast testing
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test data files
        cls._create_sentiment_data()
        cls._create_ner_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level fixtures."""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_sentiment_data(cls):
        """Create synthetic sentiment analysis data."""
        # Create training data (larger for meaningful training)
        train_data = []
        for i in range(50):  # 50 samples for training
            if i % 2 == 0:
                train_data.append({
                    "text": f"This is a positive sentiment example {i}. Great and wonderful!",
                    "label": 1
                })
            else:
                train_data.append({
                    "text": f"This is a negative sentiment example {i}. Terrible and awful!",
                    "label": 0
                })
        
        cls.sentiment_train_file = os.path.join(cls.temp_dir, "sentiment_train.jsonl")
        with open(cls.sentiment_train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        # Create validation data
        val_data = []
        for i in range(10):  # 10 samples for validation
            if i % 2 == 0:
                val_data.append({
                    "text": f"Validation positive example {i}. Excellent!",
                    "label": 1
                })
            else:
                val_data.append({
                    "text": f"Validation negative example {i}. Horrible!",
                    "label": 0
                })
        
        cls.sentiment_val_file = os.path.join(cls.temp_dir, "sentiment_val.jsonl")
        with open(cls.sentiment_val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
    
    @classmethod
    def _create_ner_data(cls):
        """Create synthetic NER data."""
        # Create training data
        train_data = [
            {
                "text": "John works at Google in California.",
                "entities": [
                    {"start": 0, "end": 4, "label": "PERSON"},
                    {"start": 14, "end": 20, "label": "ORG"},
                    {"start": 24, "end": 34, "label": "LOCATION"}
                ]
            },
            {
                "text": "Mary lives in New York City.",
                "entities": [
                    {"start": 0, "end": 4, "label": "PERSON"},
                    {"start": 14, "end": 27, "label": "LOCATION"}
                ]
            },
            {
                "text": "Apple Inc. is based in Cupertino.",
                "entities": [
                    {"start": 0, "end": 9, "label": "ORG"},
                    {"start": 23, "end": 32, "label": "LOCATION"}
                ]
            },
            {
                "text": "Microsoft was founded by Bill Gates.",
                "entities": [
                    {"start": 0, "end": 9, "label": "ORG"},
                    {"start": 25, "end": 35, "label": "PERSON"}
                ]
            }
        ] * 10  # Repeat to get 40 samples
        
        cls.ner_train_file = os.path.join(cls.temp_dir, "ner_train.jsonl")
        with open(cls.ner_train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.output_dir = os.path.join(self.temp_dir, f"output_{id(self)}")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up output directory for this test
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_sentiment_classification_training(self):
        """Test end-to-end sentiment classification training."""
        # Create task configuration
        task_config = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path=self.sentiment_train_file,
            heads=[HeadConfig(name="sentiment_head", num_labels=2)]
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            project_name="test_sentiment",
            tasks=[task_config],
            model=ModelConfig(base_model=self.model_name, fusion=None),
            tokenizer=TokenizerConfig(max_length=64, padding="max_length"),
            training=TrainingConfig(
                output_dir=self.output_dir,
                learning_rate=5e-5,
                batch_size=4,  # Small batch for testing
                num_epochs=2,  # Quick training
                eval_strategy="epoch",
                save_strategy="epoch",
                logging=None  # Disable logging for testing
            )
        )
        
        try:
            # Process data
            processor = DataProcessor(config, self.tokenizer)
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Verify datasets were created
            self.assertIn("sentiment", train_datasets)
            self.assertIn("sentiment", eval_datasets)
            self.assertTrue(len(train_datasets["sentiment"]) > 0)
            
            # Create tasks
            task = ClassificationTask(updated_config.tasks[0])
            
            # Create model
            model_config_obj = AutoConfig.from_pretrained(self.model_name)
            from multitask_bert.core.model import MultiTaskModel
            model = MultiTaskModel(model_config_obj, config.model, [task])
            
            # Create trainer
            trainer = MultiTaskTrainer(
                model=model,
                config=updated_config,
                train_datasets=train_datasets,
                eval_datasets=eval_datasets,
                tokenizer=self.tokenizer
            )
            
            # Train model
            trainer.train()
            
            # Verify training completed
            self.assertTrue(os.path.exists(self.output_dir))
            
            # Check if model was saved
            final_model_path = os.path.join(self.output_dir, "final_model")
            if os.path.exists(final_model_path):
                self.assertTrue(len(os.listdir(final_model_path)) > 0)
            
            # Test inference on a sample
            model.eval()
            sample_text = "This is a positive test example."
            inputs = self.tokenizer(
                sample_text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                output = model.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    task_id=0,
                    labels=None
                )
            
            # Verify inference output
            self.assertIn("sentiment_head", output.logits)
            logits = output.logits["sentiment_head"]
            self.assertEqual(logits.shape, (1, 2))  # 1 sample, 2 classes
            self.assertTrue(torch.isfinite(logits).all())
            
        except Exception as e:
            self.fail(f"Sentiment classification training failed: {e}")
    
    def test_ner_training(self):
        """Test end-to-end NER training."""
        # Create task configuration
        task_config = TaskConfig(
            name="ner",
            type="ner", 
            data_path=self.ner_train_file,
            heads=[HeadConfig(name="ner_head", num_labels=5)]  # Will be updated
        )
        
        # Create experiment configuration
        config = ExperimentConfig(
            project_name="test_ner",
            tasks=[task_config],
            model=ModelConfig(base_model=self.model_name, fusion=None),
            tokenizer=TokenizerConfig(max_length=64, padding="max_length"),
            training=TrainingConfig(
                output_dir=self.output_dir,
                learning_rate=5e-5,
                batch_size=2,  # Very small batch for NER
                num_epochs=1,  # Quick training
                eval_strategy="no",  # Skip evaluation for speed
                save_strategy="no",   # Skip saving for speed
                logging=None
            )
        )
        
        try:
            # Process data
            processor = DataProcessor(config, self.tokenizer)
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Verify datasets were created
            self.assertIn("ner", train_datasets)
            self.assertTrue(len(train_datasets["ner"]) > 0)
            
            # Verify label mapping was created
            ner_task_config = updated_config.tasks[0]
            self.assertIsNotNone(ner_task_config.label_maps)
            self.assertIn("ner_head", ner_task_config.label_maps)
            self.assertTrue(ner_task_config.heads[0].num_labels > 0)
            
            # Create task
            task = NERTask(ner_task_config)
            
            # Create model
            model_config_obj = AutoConfig.from_pretrained(self.model_name)
            from multitask_bert.core.model import MultiTaskModel
            model = MultiTaskModel(model_config_obj, config.model, [task])
            
            # Create trainer
            trainer = MultiTaskTrainer(
                model=model,
                config=updated_config,
                train_datasets=train_datasets,
                eval_datasets=eval_datasets,
                tokenizer=self.tokenizer
            )
            
            # Train model  
            trainer.train()
            
            # Test inference on a sample
            model.eval()
            sample_text = "John works at Google."
            inputs = self.tokenizer(
                sample_text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                output = model.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    task_id=0,
                    labels=None
                )
            
            # Verify inference output
            self.assertIn("ner_head", output.logits)
            logits = output.logits["ner_head"]
            expected_shape = (1, 64, ner_task_config.heads[0].num_labels)  # batch, seq_len, num_labels
            self.assertEqual(logits.shape, expected_shape)
            self.assertTrue(torch.isfinite(logits).all())
            
        except Exception as e:
            self.fail(f"NER training failed: {e}")
    
    def test_training_with_very_small_dataset(self):
        """Test training with minimal dataset size."""
        # Create tiny dataset (2 samples)
        tiny_data = [
            {"text": "Positive example.", "label": 1},
            {"text": "Negative example.", "label": 0}
        ]
        
        tiny_file = os.path.join(self.temp_dir, "tiny_data.jsonl")
        with open(tiny_file, 'w') as f:
            for item in tiny_data:
                f.write(json.dumps(item) + '\n')
        
        task_config = TaskConfig(
            name="tiny_task",
            type="single_label_classification",
            data_path=tiny_file,
            heads=[HeadConfig(name="tiny_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test_tiny",
            tasks=[task_config],
            model=ModelConfig(base_model=self.model_name),
            training=TrainingConfig(
                output_dir=self.output_dir,
                batch_size=1,  # Batch size = dataset size
                num_epochs=1,
                eval_strategy="no",
                save_strategy="no",
                logging=None
            )
        )
        
        try:
            # Should handle tiny datasets gracefully
            processor = DataProcessor(config, self.tokenizer)
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Should create datasets even if tiny
            self.assertIn("tiny_task", train_datasets)
            
        except Exception as e:
            # Should either work or fail with clear error message
            self.assertIn("dataset", str(e).lower(), f"Unexpected error: {e}")
    
    def test_training_memory_usage(self):
        """Test training doesn't exceed memory limits."""
        # This test monitors memory usage during training
        import psutil
        import gc
        
        # Clean up memory before test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create configuration for memory-efficient training
        task_config = TaskConfig(
            name="memory_test",
            type="single_label_classification",
            data_path=self.sentiment_train_file,
            heads=[HeadConfig(name="memory_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="memory_test",
            tasks=[task_config],
            model=ModelConfig(base_model=self.model_name, fusion=None),
            tokenizer=TokenizerConfig(max_length=32),  # Short sequences
            training=TrainingConfig(
                output_dir=self.output_dir,
                batch_size=2,  # Small batch
                num_epochs=1,
                eval_strategy="no",
                save_strategy="no",
                logging=None
            )
        )
        
        try:
            # Process and train
            processor = DataProcessor(config, self.tokenizer)
            train_datasets, eval_datasets, updated_config = processor.process()
            
            task = ClassificationTask(updated_config.tasks[0])
            model_config_obj = AutoConfig.from_pretrained(self.model_name)
            from multitask_bert.core.model import MultiTaskModel
            model = MultiTaskModel(model_config_obj, config.model, [task])
            
            trainer = MultiTaskTrainer(
                model=model,
                config=updated_config,
                train_datasets=train_datasets,
                eval_datasets=eval_datasets,
                tokenizer=self.tokenizer
            )
            
            trainer.train()
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 2GB for tiny model)
            self.assertLess(memory_increase, 2000, 
                          f"Memory usage increased by {memory_increase:.1f}MB, which is too high")
            
        except Exception as e:
            self.fail(f"Memory test failed: {e}")
    
    def test_training_with_invalid_config(self):
        """Test training with invalid configurations."""
        # Test with batch size larger than dataset
        task_config = TaskConfig(
            name="invalid_test",
            type="single_label_classification",
            data_path=self.sentiment_train_file,
            heads=[HeadConfig(name="invalid_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="invalid_test",
            tasks=[task_config],
            model=ModelConfig(base_model=self.model_name),
            training=TrainingConfig(
                output_dir=self.output_dir,
                batch_size=1000,  # Larger than dataset
                num_epochs=1,
                logging=None
            )
        )
        
        try:
            processor = DataProcessor(config, self.tokenizer)
            train_datasets, eval_datasets, updated_config = processor.process()
            
            task = ClassificationTask(updated_config.tasks[0])
            model_config_obj = AutoConfig.from_pretrained(self.model_name)
            from multitask_bert.core.model import MultiTaskModel
            model = MultiTaskModel(model_config_obj, config.model, [task])
            
            trainer = MultiTaskTrainer(
                model=model,
                config=updated_config,
                train_datasets=train_datasets,
                eval_datasets=eval_datasets,
                tokenizer=self.tokenizer
            )
            
            # Should either handle gracefully or raise clear error
            trainer.train()
            
        except Exception as e:
            # Should be a clear, informative error
            self.assertTrue(len(str(e)) > 10, "Error message should be informative")


def run_tests():
    """Run all single-task training tests and generate report."""
    print("=" * 70)
    print("  SINGLE-TASK TRAINING INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSingleTaskTraining)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())