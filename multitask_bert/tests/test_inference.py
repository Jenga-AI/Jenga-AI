import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, AutoConfig, PreTrainedModel
from multitask_bert.deployment.inference import InferenceHandler
from multitask_bert.core.config import ExperimentConfig, ModelConfig, TaskConfig, HeadConfig, TrainingConfig, TokenizerConfig
from multitask_bert.tasks.classification import MultiHeadSingleLabelClassificationTask, MultiLabelClassificationTask
from multitask_bert.tasks.ner import NERTask
from multitask_bert.core.model import MultiTaskModel

@pytest.fixture
def mock_experiment_config():
    return ExperimentConfig(
        project_name="test_project",
        model=ModelConfig(base_model="bert-base-uncased"),
        tokenizer=TokenizerConfig(max_length=128, padding="max_length", truncation=True),
        tasks=[
            TaskConfig(name="cls_task", type="classification", data_path="dummy.jsonl", heads=[HeadConfig(name="cls_head", num_labels=3, weight=1.0)]),
            TaskConfig(name="mlc_task", type="multi_label_classification", data_path="dummy.jsonl", heads=[HeadConfig(name="mlc_head", num_labels=2, weight=1.0)]),
            TaskConfig(name="ner_task", type="ner", data_path="dummy.jsonl", heads=[HeadConfig(name="ner_head", num_labels=5, weight=1.0)], label_maps={'ner_head': {0: "O", 1: "LOC", 2: "PER", 3: "ORG", 4: "MISC"}})
        ],
        training=TrainingConfig(output_dir="output", learning_rate=1e-5, batch_size=8, num_epochs=1, device="cpu")
    )

@patch('multitask_bert.deployment.inference.AutoTokenizer.from_pretrained')
@patch('multitask_bert.deployment.inference.AutoConfig.from_pretrained')
# Remove the patch for MultiTaskModel.__init__
def test_inference_handler_init(mock_auto_config_from_pretrained, mock_auto_tokenizer_from_pretrained, mock_experiment_config):
    mock_tokenizer_instance = MagicMock()
    mock_auto_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

    mock_model_config_instance = MagicMock(spec=AutoConfig)
    mock_model_config_instance.hidden_size = 768
    mock_auto_config_from_pretrained.return_value = mock_model_config_instance

    # Mock the MultiTaskModel class itself
    with patch('multitask_bert.deployment.inference.MultiTaskModel') as MockMultiTaskModel:
        mock_multi_task_model_instance = MagicMock(spec=MultiTaskModel)
        mock_multi_task_model_instance.eval.return_value = None
        mock_multi_task_model_instance.to.return_value = None
        MockMultiTaskModel.return_value = mock_multi_task_model_instance

        handler = InferenceHandler("dummy_path", mock_experiment_config)

        mock_auto_tokenizer_from_pretrained.assert_called_once_with("dummy_path")
        mock_auto_config_from_pretrained.assert_called_once_with("dummy_path")
        MockMultiTaskModel.assert_called_once() # Assert MultiTaskModel constructor was called
        
        assert handler.tokenizer == mock_tokenizer_instance
        assert handler.model == mock_multi_task_model_instance # Now this assertion is valid
        mock_multi_task_model_instance.eval.assert_called_once()
        mock_multi_task_model_instance.to.assert_called_once_with("cpu")
        assert handler.task_map == {"cls_task": 0, "mlc_task": 1, "ner_task": 2}

def test_inference_handler_get_task_class():
    handler = InferenceHandler.__new__(InferenceHandler) # Create instance without calling __init__
    assert handler._get_task_class("classification") == MultiHeadSingleLabelClassificationTask # Changed
    assert handler._get_task_class("multi_label_classification") == MultiLabelClassificationTask
    assert handler._get_task_class("ner") == NERTask
    with pytest.raises(ValueError):
        handler._get_task_class("unknown_task")

@patch('multitask_bert.deployment.inference.AutoTokenizer.from_pretrained')
@patch('multitask_bert.deployment.inference.AutoConfig.from_pretrained')
# Remove the patch for MultiTaskModel.__init__
def test_inference_handler_predict(mock_auto_config_from_pretrained, mock_auto_tokenizer_from_pretrained, mock_experiment_config):
    # Mock tokenizer behavior
    mock_tokenized_inputs = MagicMock()
    mock_tokenized_inputs.input_ids = torch.tensor([[101, 2054, 2003, 102]])
    mock_tokenized_inputs.attention_mask = torch.tensor([[1, 1, 1, 1]])
    mock_tokenized_inputs.to.return_value = mock_tokenized_inputs # Chain .to()
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = mock_tokenized_inputs
    mock_tokenizer_instance.convert_ids_to_tokens.return_value = ['[CLS]', 'hello', 'world', '[SEP]']
    mock_tokenizer_instance.convert_tokens_to_string.side_effect = lambda x: " ".join(x)
    mock_tokenizer_instance.all_special_tokens = ['[CLS]', '[SEP]']
    mock_auto_tokenizer_from_pretrained.return_value = mock_tokenizer_instance

    # Mock model config behavior
    mock_model_config_instance = MagicMock(spec=AutoConfig)
    mock_model_config_instance.hidden_size = 768
    mock_auto_config_from_pretrained.return_value = mock_model_config_instance

    # Mock outputs for each task
    # Single-label classification
    mock_cls_output_logits = {"cls_head": torch.tensor([[0.1, 0.8, 0.1]])} # Predicts class 1
    
    # Multi-label classification
    mock_mlc_output_logits = {"mlc_head": torch.tensor([[0.2, 0.7]])} # Predicts [0, 1]
    
    # NER
    # Example: [CLS] hello world [SEP] -> O LOC O O
    mock_ner_output_logits = {"ner_head": torch.tensor([[[0.9, 0.05, 0.05, 0.0, 0.0], # CLS
                                                         [0.1, 0.8, 0.05, 0.05, 0.0], # hello -> LOC
                                                         [0.9, 0.05, 0.05, 0.0, 0.0], # world -> O
                                                         [0.9, 0.05, 0.05, 0.0, 0.0]]])} # SEP
    
    # Configure side_effect for model calls based on task_id
    def model_side_effect(*args, **kwargs):
        task_id = kwargs.get('task_id')
        if task_id == 0: # cls_task
            return {"loss": torch.tensor(0.1), "logits": mock_cls_output_logits}
        elif task_id == 1: # mlc_task
            return {"loss": torch.tensor(0.1), "logits": mock_mlc_output_logits}
        elif task_id == 2: # ner_task
            return {"loss": torch.tensor(0.1), "logits": mock_ner_output_logits}
        return {"loss": torch.tensor(0.1), "logits": {}}

    # Mock the MultiTaskModel class itself
    with patch('multitask_bert.deployment.inference.MultiTaskModel') as MockMultiTaskModel:
        mock_multi_task_model_instance = MagicMock(spec=MultiTaskModel)
        mock_multi_task_model_instance.eval.return_value = None
        mock_multi_task_model_instance.to.return_value = None
        mock_multi_task_model_instance.side_effect = model_side_effect # Set side_effect on the instance
        MockMultiTaskModel.return_value = mock_multi_task_model_instance

        handler = InferenceHandler("dummy_path", mock_experiment_config)

        text_to_predict = "Hello world"
        predictions = handler.predict(text_to_predict)

        # Assertions for single-label classification
        assert "cls_task" in predictions
        assert "cls_head" in predictions["cls_task"]
        assert predictions["cls_task"]["cls_head"] == 1 # Predicted class ID

        # Assertions for multi-label classification
        assert "mlc_task" in predictions
        assert "mlc_head" in predictions["mlc_task"]
        assert predictions["mlc_task"]["mlc_head"] == [[0, 1]] # Predicted binary labels

        # Assertions for NER
        assert "ner_task" in predictions
        assert "ner_head" in predictions["ner_task"]
        # Expected: [{'entity': 'hello', 'label': 'LOC'}]
        assert len(predictions["ner_task"]["ner_head"]) == 1
        assert predictions["ner_task"]["ner_head"][0]["entity"] == "hello"
        assert predictions["ner_task"]["ner_head"][0]["label"] == "LOC"
