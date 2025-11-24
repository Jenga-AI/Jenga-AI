import torch
import yaml
import os
from transformers import AutoTokenizer
from multitask_bert.core.config import ExperimentConfig
from multitask_bert.core.model import MultiTaskModel

def simple_ner_inference():
    """Simple NER inference that should work with your saved model."""
    
    # Paths
    model_dir = "/Users/naynek/Desktop/MultiClassifier/Jenga-AI/unified_results_ner"
    best_model_path = os.path.join(model_dir, "best_model")
    config_path = os.path.join(model_dir, "experiment_config.yaml")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ExperimentConfig(**config_dict)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel.from_pretrained(best_model_path)
    model.to(device)
    model.eval()
    
    # Get label map
    label_map = config.tasks[0].label_maps['ner_head']
    id_to_label = {v: k for k, v in label_map.items()}
    
    print(f"✅ Model loaded with {len(label_map)} labels")
    print(f"📊 Labels: {list(label_map.keys())}")
    
    # Test text
    test_text = "Hello, I'm Vincent from Dar es Salaam. I need help with my 10-year-old daughter."
    
    # Tokenize
    inputs = tokenizer(
        test_text,
        truncation=True,
        padding='max_length',
        max_length=config.tokenizer.max_length,
        return_tensors='pt'
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            task_id=0
        )
    
    # Process results
    logits = outputs.logits['ner_head']
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Display results
    print(f"\n📝 Text: {test_text}")
    print("\n🔍 Token-level predictions:")
    for token, pred in zip(tokens, predictions):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            label = id_to_label.get(pred, 'O')
            if label != 'O':
                print(f"   🏷️  {token} → {label}")
            else:
                print(f"   {token} → {label}")

if __name__ == "__main__":
    simple_ner_inference()