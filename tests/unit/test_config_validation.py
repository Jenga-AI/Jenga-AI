#!/usr/bin/env python3
"""
Config Validation Unit Tests
===========================
Tests YAML config parsing, validation, and error handling.

Usage:
    python -m pytest tests/unit/test_config_validation.py -v
    OR
    python tests/unit/test_config_validation.py
"""

import sys
import unittest
import tempfile
import os
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.core.config import (
    ExperimentConfig, TaskConfig, HeadConfig, ModelConfig, 
    TokenizerConfig, TrainingConfig, LoggingConfig, FusionConfig,
    load_experiment_config
)


class TestConfigDataclasses(unittest.TestCase):
    """Test config dataclass creation and validation."""
    
    def test_head_config_creation(self):
        """Test HeadConfig creation and defaults."""
        head = HeadConfig(name="test_head", num_labels=5)
        
        self.assertEqual(head.name, "test_head")
        self.assertEqual(head.num_labels, 5)
        self.assertEqual(head.weight, 1.0)  # Default weight
        
        # Test with custom weight
        head_custom = HeadConfig(name="custom_head", num_labels=3, weight=0.7)
        self.assertEqual(head_custom.weight, 0.7)
    
    def test_task_config_creation(self):
        """Test TaskConfig creation and structure."""
        heads = [HeadConfig(name="classification_head", num_labels=2)]
        
        task = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path="/path/to/data.csv",
            heads=heads
        )
        
        self.assertEqual(task.name, "sentiment")
        self.assertEqual(task.type, "single_label_classification") 
        self.assertEqual(task.data_path, "/path/to/data.csv")
        self.assertEqual(len(task.heads), 1)
        self.assertIsNone(task.label_maps)  # Default None
        
        # Test with label maps
        label_maps = {"classification_head": {0: "negative", 1: "positive"}}
        task_with_maps = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path="/path/to/data.csv",
            heads=heads,
            label_maps=label_maps
        )
        self.assertEqual(task_with_maps.label_maps, label_maps)
    
    def test_fusion_config_creation(self):
        """Test FusionConfig creation and defaults."""
        # Test with defaults
        fusion_default = FusionConfig()
        self.assertEqual(fusion_default.type, "attention")
        self.assertEqual(fusion_default.hidden_size, 768)
        
        # Test with custom values
        fusion_custom = FusionConfig(type="custom", hidden_size=512)
        self.assertEqual(fusion_custom.type, "custom")
        self.assertEqual(fusion_custom.hidden_size, 512)
    
    def test_model_config_creation(self):
        """Test ModelConfig creation and post_init."""
        # Test without fusion
        model = ModelConfig(
            base_model="bert-base-uncased",
            dropout=0.2
        )
        self.assertEqual(model.base_model, "bert-base-uncased")
        self.assertEqual(model.dropout, 0.2)
        self.assertIsNone(model.fusion)
        
        # Test with fusion dict (should be converted to FusionConfig)
        model_with_fusion = ModelConfig(
            base_model="bert-base-uncased",
            fusion={"type": "attention", "hidden_size": 512}
        )
        self.assertIsInstance(model_with_fusion.fusion, FusionConfig)
        self.assertEqual(model_with_fusion.fusion.type, "attention")
        self.assertEqual(model_with_fusion.fusion.hidden_size, 512)
        
        # Test with FusionConfig object
        fusion_config = FusionConfig(type="test", hidden_size=256)
        model_with_fusion_obj = ModelConfig(
            base_model="bert-base-uncased",
            fusion=fusion_config
        )
        self.assertEqual(model_with_fusion_obj.fusion, fusion_config)
    
    def test_tokenizer_config_creation(self):
        """Test TokenizerConfig creation and defaults."""
        tokenizer = TokenizerConfig()
        
        self.assertEqual(tokenizer.max_length, 128)
        self.assertEqual(tokenizer.padding, "max_length")
        self.assertTrue(tokenizer.truncation)
        self.assertIsNone(tokenizer.pad_token_id)
        
        # Test with custom values
        tokenizer_custom = TokenizerConfig(
            max_length=256,
            padding="longest",
            truncation=False,
            pad_token_id=0
        )
        self.assertEqual(tokenizer_custom.max_length, 256)
        self.assertEqual(tokenizer_custom.padding, "longest")
        self.assertFalse(tokenizer_custom.truncation)
        self.assertEqual(tokenizer_custom.pad_token_id, 0)
    
    def test_logging_config_creation(self):
        """Test LoggingConfig creation and defaults."""
        logging = LoggingConfig()
        
        self.assertEqual(logging.service, "tensorboard")
        self.assertEqual(logging.experiment_name, "multitask_experiment")
        self.assertIsNone(logging.tracking_uri)
        
        # Test with custom values
        logging_custom = LoggingConfig(
            service="mlflow",
            experiment_name="custom_experiment",
            tracking_uri="http://localhost:5000"
        )
        self.assertEqual(logging_custom.service, "mlflow")
        self.assertEqual(logging_custom.experiment_name, "custom_experiment")
        self.assertEqual(logging_custom.tracking_uri, "http://localhost:5000")
    
    def test_training_config_creation(self):
        """Test TrainingConfig creation and post_init."""
        # Test with defaults
        training = TrainingConfig()
        
        self.assertEqual(training.output_dir, "./results")
        self.assertEqual(training.learning_rate, 2.0e-5)
        self.assertEqual(training.batch_size, 16)
        self.assertEqual(training.num_epochs, 3)
        self.assertIsNone(training.logging)
        
        # Test with logging dict (should be converted)
        training_with_logging = TrainingConfig(
            logging={"service": "mlflow", "experiment_name": "test"}
        )
        self.assertIsInstance(training_with_logging.logging, LoggingConfig)
        self.assertEqual(training_with_logging.logging.service, "mlflow")
    
    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation and post_init."""
        # Create task config
        task_dict = {
            "name": "sentiment",
            "type": "single_label_classification",
            "data_path": "/path/to/data.csv",
            "heads": [{"name": "sentiment_head", "num_labels": 2}]
        }
        
        # Test with dict inputs (should be converted)
        experiment = ExperimentConfig(
            project_name="test_project",
            tasks=[task_dict],
            model={"base_model": "bert-base-uncased"},
            tokenizer={"max_length": 256},
            training={"batch_size": 32}
        )
        
        # Verify conversions
        self.assertEqual(experiment.project_name, "test_project")
        self.assertEqual(len(experiment.tasks), 1)
        self.assertIsInstance(experiment.tasks[0], TaskConfig)
        self.assertIsInstance(experiment.model, ModelConfig)
        self.assertIsInstance(experiment.tokenizer, TokenizerConfig)
        self.assertIsInstance(experiment.training, TrainingConfig)
        
        # Verify nested conversions
        task = experiment.tasks[0]
        self.assertEqual(task.name, "sentiment")
        self.assertEqual(len(task.heads), 1)
        self.assertIsInstance(task.heads[0], HeadConfig)
        self.assertEqual(task.heads[0].name, "sentiment_head")


class TestConfigLoading(unittest.TestCase):
    """Test YAML config file loading and parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_yaml_config(self, config_dict, filename="test_config.yaml"):
        """Helper to create YAML config file."""
        config_path = os.path.join(self.temp_dir, filename)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return config_path
    
    def test_valid_config_loading(self):
        """Test loading valid YAML config."""
        config_dict = {
            "project_name": "test_project",
            "tasks": [
                {
                    "name": "sentiment",
                    "type": "single_label_classification",
                    "data_path": "/path/to/sentiment.csv",
                    "heads": [
                        {"name": "sentiment_head", "num_labels": 3, "weight": 1.0}
                    ]
                },
                {
                    "name": "ner",
                    "type": "ner",
                    "data_path": "/path/to/ner.jsonl",
                    "heads": [
                        {"name": "ner_head", "num_labels": 9}
                    ],
                    "label_maps": {
                        "ner_head": {
                            0: "O", 1: "B-PER", 2: "I-PER",
                            3: "B-LOC", 4: "I-LOC"
                        }
                    }
                }
            ],
            "model": {
                "base_model": "prajjwal1/bert-tiny",
                "dropout": 0.1,
                "fusion": {
                    "type": "attention",
                    "hidden_size": 128
                }
            },
            "tokenizer": {
                "max_length": 64,
                "padding": "max_length",
                "truncation": True
            },
            "training": {
                "output_dir": "./test_output",
                "learning_rate": 5e-5,
                "batch_size": 8,
                "num_epochs": 2,
                "logging": {
                    "service": "tensorboard",
                    "experiment_name": "test_experiment"
                }
            }
        }
        
        config_path = self._create_yaml_config(config_dict)
        
        # Load and validate config
        loaded_config = load_experiment_config(config_path)
        
        # Verify top-level structure
        self.assertIsInstance(loaded_config, ExperimentConfig)
        self.assertEqual(loaded_config.project_name, "test_project")
        self.assertEqual(len(loaded_config.tasks), 2)
        
        # Verify tasks
        sentiment_task = loaded_config.tasks[0]
        ner_task = loaded_config.tasks[1]
        
        self.assertEqual(sentiment_task.name, "sentiment")
        self.assertEqual(sentiment_task.type, "single_label_classification")
        self.assertEqual(len(sentiment_task.heads), 1)
        self.assertEqual(sentiment_task.heads[0].num_labels, 3)
        
        self.assertEqual(ner_task.name, "ner")
        self.assertEqual(ner_task.type, "ner")
        self.assertIsNotNone(ner_task.label_maps)
        
        # Verify model config
        self.assertEqual(loaded_config.model.base_model, "prajjwal1/bert-tiny")
        self.assertIsNotNone(loaded_config.model.fusion)
        self.assertEqual(loaded_config.model.fusion.hidden_size, 128)
        
        # Verify tokenizer config
        self.assertEqual(loaded_config.tokenizer.max_length, 64)
        
        # Verify training config
        self.assertEqual(loaded_config.training.batch_size, 8)
        self.assertEqual(loaded_config.training.num_epochs, 2)
        self.assertIsNotNone(loaded_config.training.logging)
        self.assertEqual(loaded_config.training.logging.service, "tensorboard")
    
    def test_minimal_config_loading(self):
        """Test loading minimal valid config."""
        config_dict = {
            "project_name": "minimal_test",
            "tasks": [
                {
                    "name": "simple_task",
                    "type": "single_label_classification",
                    "data_path": "/path/to/data.csv",
                    "heads": [
                        {"name": "simple_head", "num_labels": 2}
                    ]
                }
            ]
        }
        
        config_path = self._create_yaml_config(config_dict)
        loaded_config = load_experiment_config(config_path)
        
        # Verify defaults are applied
        self.assertEqual(loaded_config.model.base_model, "distilbert-base-uncased")  # Default
        self.assertEqual(loaded_config.tokenizer.max_length, 128)  # Default
        self.assertEqual(loaded_config.training.batch_size, 16)  # Default
    
    def test_malformed_yaml_handling(self):
        """Test handling of malformed YAML files."""
        malformed_yaml = """
        project_name: "test"
        tasks:
          - name: "task1"
            invalid_yaml: [unclosed_bracket
        """
        
        config_path = os.path.join(self.temp_dir, "malformed.yaml")
        with open(config_path, 'w') as f:
            f.write(malformed_yaml)
        
        # Should raise yaml parsing error
        with self.assertRaises(yaml.YAMLError):
            load_experiment_config(config_path)
    
    def test_missing_required_fields(self):
        """Test handling of configs with missing required fields."""
        # Missing project_name
        config_missing_project = {
            "tasks": [
                {
                    "name": "task1",
                    "type": "single_label_classification", 
                    "data_path": "/path/to/data.csv",
                    "heads": [{"name": "head1", "num_labels": 2}]
                }
            ]
        }
        
        config_path = self._create_yaml_config(config_missing_project, "missing_project.yaml")
        
        with self.assertRaises(TypeError):
            load_experiment_config(config_path)
        
        # Missing tasks
        config_missing_tasks = {"project_name": "test"}
        
        config_path = self._create_yaml_config(config_missing_tasks, "missing_tasks.yaml")
        
        with self.assertRaises(TypeError):
            load_experiment_config(config_path)
    
    def test_invalid_task_configuration(self):
        """Test handling of invalid task configurations."""
        # Task missing required fields
        config_dict = {
            "project_name": "test",
            "tasks": [
                {
                    "name": "incomplete_task",
                    # Missing: type, data_path, heads
                }
            ]
        }
        
        config_path = self._create_yaml_config(config_dict)
        
        with self.assertRaises(TypeError):
            load_experiment_config(config_path)
    
    def test_invalid_head_configuration(self):
        """Test handling of invalid head configurations."""
        # Head with invalid num_labels
        config_dict = {
            "project_name": "test",
            "tasks": [
                {
                    "name": "task",
                    "type": "single_label_classification",
                    "data_path": "/path/to/data.csv",
                    "heads": [
                        {"name": "head", "num_labels": 0}  # Invalid
                    ]
                }
            ]
        }
        
        config_path = self._create_yaml_config(config_dict)
        
        # Should load but might be caught during validation
        try:
            loaded_config = load_experiment_config(config_path)
            # If it loads, verify the invalid value is preserved
            self.assertEqual(loaded_config.tasks[0].heads[0].num_labels, 0)
        except (ValueError, TypeError):
            # Expected for invalid configurations
            pass
    
    def test_nonexistent_file_handling(self):
        """Test handling of non-existent config files."""
        nonexistent_path = "/path/that/does/not/exist.yaml"
        
        with self.assertRaises(FileNotFoundError):
            load_experiment_config(nonexistent_path)
    
    def test_config_with_extra_fields(self):
        """Test handling of configs with extra/unknown fields."""
        config_dict = {
            "project_name": "test",
            "tasks": [
                {
                    "name": "task",
                    "type": "single_label_classification",
                    "data_path": "/path/to/data.csv",
                    "heads": [{"name": "head", "num_labels": 2}],
                    "extra_field": "should be ignored"  # Extra field
                }
            ],
            "unknown_section": "should be ignored"  # Unknown section
        }
        
        config_path = self._create_yaml_config(config_dict)
        
        # Should load successfully, ignoring unknown fields
        loaded_config = load_experiment_config(config_path)
        
        self.assertEqual(loaded_config.project_name, "test")
        self.assertEqual(len(loaded_config.tasks), 1)
    
    def test_config_type_conversions(self):
        """Test automatic type conversions in configs."""
        config_dict = {
            "project_name": "test",
            "tasks": [
                {
                    "name": "task",
                    "type": "single_label_classification",
                    "data_path": "/path/to/data.csv",
                    "heads": [
                        {
                            "name": "head",
                            "num_labels": "3",  # String that should convert to int
                            "weight": "0.5"     # String that should convert to float
                        }
                    ]
                }
            ],
            "training": {
                "batch_size": "8",        # String that should convert to int
                "learning_rate": "1e-4",  # String that should convert to float
                "num_epochs": "5"         # String that should convert to int
            }
        }
        
        config_path = self._create_yaml_config(config_dict)
        loaded_config = load_experiment_config(config_path)
        
        # Verify type conversions
        head = loaded_config.tasks[0].heads[0]
        self.assertIsInstance(head.num_labels, int)
        self.assertEqual(head.num_labels, 3)
        self.assertIsInstance(head.weight, float)
        self.assertEqual(head.weight, 0.5)
        
        training = loaded_config.training
        self.assertIsInstance(training.batch_size, int)
        self.assertEqual(training.batch_size, 8)
        self.assertIsInstance(training.learning_rate, float)
        self.assertEqual(training.learning_rate, 1e-4)


def run_tests():
    """Run all config validation tests and generate report."""
    print("=" * 70)
    print("  CONFIG VALIDATION UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigDataclasses))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    
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