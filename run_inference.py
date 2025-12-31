#!/usr/bin/env python3
"""
Inference script for the trained Jenga-AI multi-task model.
This script loads the trained model and runs predictions on sample texts.
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from multitask_bert.core.config import load_experiment_config
from multitask_bert.core.model import MultiTaskModel
import json


def load_trained_model(model_path: str, config_path: str):
    """
    Load the trained model and tokenizer.
    
    Args:
        model_path: Path to the saved model checkpoint
        config_path: Path to the experiment configuration YAML file
    
    Returns:
        model, tokenizer, config
    """
    print(f"Loading experiment configuration from: {config_path}")
    config = load_experiment_config(config_path)
    
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from: {model_path}")
    model_auto_config = AutoConfig.from_pretrained(model_path)
    
    # Load state dict first to inspect the actual model structure
    model_state_path = f"{model_path}/model.safetensors"
    print(f"Inspecting model weights from: {model_state_path}")
    
    from safetensors.torch import load_file
    state_dict = load_file(model_state_path)
    
    # Extract the actual number of NER labels from the saved weights
    # The NER head output layer has shape [num_labels, hidden_size]
    ner_weight_key = "tasks.1.heads.ner_head.1.weight"  # Task 1 is NER
    if ner_weight_key in state_dict:
        actual_num_ner_labels = state_dict[ner_weight_key].shape[0]
        print(f"Detected {actual_num_ner_labels} NER labels in saved model")
        
        # Update the config to match the trained model
        config.tasks[1].heads[0].num_labels = actual_num_ner_labels
        
        # Create a dummy label map if not present
        if not config.tasks[1].label_maps or "ner_head" not in config.tasks[1].label_maps:
            config.tasks[1].label_maps = {
                "ner_head": {str(i): f"LABEL_{i}" for i in range(actual_num_ner_labels)}
            }
    
    # Instantiate the model with corrected config
    model = MultiTaskModel(
        config=model_auto_config,
        model_config=config.model,
        task_configs=config.tasks
    )
    
    # Load the trained weights
    print(f"Loading model weights...")
    model.load_state_dict(state_dict)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✅ Model loaded successfully on device: {device}")
    
    return model, tokenizer, config, device


def predict_classification(text: str, model, tokenizer, config, device):
    """
    Run threat classification prediction on a text.
    
    Args:
        text: Input text to classify
        model: Trained model
        tokenizer: Tokenizer
        config: Experiment configuration
        device: Device to run inference on
    
    Returns:
        Prediction results
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer.max_length,
        return_tensors="pt"
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task_id=0  # ThreatClassification is the first task
        )
    
    # Get the classification task config
    task_config = config.tasks[0]  # ThreatClassification
    head_config = task_config.heads[0]  # threat_head
    
    # Get logits and prediction
    logits = outputs["logits"]["threat_head"]
    predicted_id = logits.argmax(dim=-1).item()
    
    # Map to label
    id_to_label = task_config.label_maps.get("threat_head", {})
    predicted_label = id_to_label.get(str(predicted_id), f"ID_{predicted_id}")
    
    # Get probabilities
    probs = torch.softmax(logits, dim=-1)[0]
    
    return {
        "predicted_label": predicted_label,
        "predicted_id": predicted_id,
        "confidence": probs[predicted_id].item(),
        "all_probabilities": {id_to_label.get(str(i), f"ID_{i}"): probs[i].item() for i in range(len(probs))}
    }


def predict_ner(text: str, model, tokenizer, config, device):
    """
    Run NER prediction on a text.
    
    Args:
        text: Input text for entity extraction
        model: Trained model
        tokenizer: Tokenizer
        config: Experiment configuration
        device: Device to run inference on
    
    Returns:
        List of detected entities
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.tokenizer.max_length,
        return_tensors="pt"
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task_id=1  # ThreatNER is the second task
        )
    
    # Get the NER task config
    task_config = config.tasks[1]  # ThreatNER
    
    # Get logits and predictions
    logits = outputs["logits"]["ner_head"]
    predicted_ids = logits.argmax(dim=-1)[0].cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Decode entities
    id_to_label = task_config.label_maps.get("ner_head", {})
    
    entities = []
    current_entity = []
    current_label = None
    
    for token, pred_id in zip(tokens, predicted_ids):
        label = id_to_label.get(str(pred_id), 'O')
        
        # Skip special tokens
        if token in tokenizer.all_special_tokens or token == '[PAD]':
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity)
                entities.append({"text": entity_text, "label": current_label})
                current_entity = []
                current_label = None
            continue
        
        if label != 'O':
            if current_label is None:  # Start of new entity
                current_entity.append(token)
                current_label = label
            elif label == current_label:  # Continuation
                current_entity.append(token)
            else:  # New entity with different label
                entity_text = tokenizer.convert_tokens_to_string(current_entity)
                entities.append({"text": entity_text, "label": current_label})
                current_entity = [token]
                current_label = label
        else:  # 'O' label
            if current_entity:
                entity_text = tokenizer.convert_tokens_to_string(current_entity)
                entities.append({"text": entity_text, "label": current_label})
                current_entity = []
                current_label = None
    
    # Add last entity if any
    if current_entity:
        entity_text = tokenizer.convert_tokens_to_string(current_entity)
        entities.append({"text": entity_text, "label": current_label})
    
    return entities


def main():
    """Main inference function."""
    
    # Paths
    model_path = "./unified_results/JengaAI_MVP/best_model"
    config_path = "./hackathon_mvp.yaml"
    
    # Load model
    model, tokenizer, config, device = load_trained_model(model_path, config_path)
    
    print("\n" + "="*80)
    print("JENGA-AI MULTI-TASK INFERENCE")
    print("="*80)
    
    # Sample texts for classification
    print("\n📊 THREAT CLASSIFICATION EXAMPLES:")
    print("-" * 80)
    
    classification_samples = [
        "There is a bomb threat at the airport terminal.",
        "The weather is nice today and I'm going for a walk.",
        "Armed militants attacked the village last night.",
        "I love spending time with my family on weekends."
    ]
    
    for text in classification_samples:
        result = predict_classification(text, model, tokenizer, config, device)
        print(f"\n📝 Text: {text}")
        print(f"   🏷️  Prediction: {result['predicted_label']}")
        print(f"   📈 Confidence: {result['confidence']:.3f}")
    
    # Sample texts for NER
    print("\n\n🔍 NAMED ENTITY RECOGNITION EXAMPLES:")
    print("-" * 80)
    
    ner_samples = [
        "John Smith reported a security incident in Nairobi involving armed robbers.",
        "The victim, Sarah Johnson, was attacked near the market in Mombasa.",
        "Police arrested the perpetrator Ahmed Hassan in connection with the theft."
    ]
    
    for text in ner_samples:
        entities = predict_ner(text, model, tokenizer, config, device)
        print(f"\n📝 Text: {text}")
        if entities:
            print("   Entities found:")
            for entity in entities:
                print(f"   🏷️  '{entity['text']}' → {entity['label']}")
        else:
            print("   No entities detected.")
    
    print("\n" + "="*80)
    print("✅ Inference completed!")
    print("="*80)


if __name__ == "__main__":
    main()
