import torch
import yaml
import os
from transformers import AutoTokenizer, AutoConfig
from multitask_bert.core.config import ExperimentConfig
from multitask_bert.core.model import MultiTaskModel
import json
from typing import List, Dict, Any

class QAInference:
    def __init__(self, model_dir: str):
        """
        Initialize the QA inference with trained model.
        
        Args:
            model_dir: Path to the directory containing best_model and experiment_config.yaml
        """
        self.model_dir = model_dir
        self.best_model_path = os.path.join(model_dir, "best_model")
        self.device = torch.device("cpu") # Default to CPU for stability on this Mac
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.best_model_path)
        self.model = self._load_model()
        
        # Get label map
        self.id_to_label = self._get_label_map()
        
        print(f"Loaded model with labels: {self.id_to_label}")
    
    def _load_config(self) -> ExperimentConfig:
        """Load experiment configuration from saved YAML."""
        config_path = os.path.join(self.model_dir, "experiment_config.yaml")
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    
    def _get_label_map(self):
        """Get the correct label mapping from config."""
        task_config = self.config.tasks[0]
        if task_config.label_maps and 'quality_head' in task_config.label_maps:
            return task_config.label_maps['quality_head']
        
        # Fallback
        return {0: "excellent", 1: "good", 2: "poor"}
    
    def _load_model(self):
        """Load the trained model."""
        model_config = AutoConfig.from_pretrained(self.best_model_path)
        
        print(f"Loading model from: {self.best_model_path}")
        model = MultiTaskModel(
            config=model_config,
            model_config=self.config.model,
            task_configs=self.config.tasks
        )
        
        # Load weights
        model_weights_path = os.path.join(self.best_model_path, "model.safetensors")
        if os.path.exists(model_weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(model_weights_path)
        else:
            model_weights_path = os.path.join(self.best_model_path, "pytorch_model.bin")
            state_dict = torch.load(model_weights_path, map_location=self.device)
        
        # Prepare state_dict mapping
        # Since the model was trained before the refactor, we need to map 'encoder' to 'backbone.encoder'
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', 'backbone.encoder.', 1)
                new_state_dict[new_key] = value
            elif key.startswith('tasks.'):
                # Ensure tasks are mapped correctly if necessary
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
                
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict the quality level of the text."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.tokenizer.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=0
            )
        
        logits = outputs["logits"]['quality_head']
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        
        confidence = probs[pred_idx].item()
        label = self.id_to_label.get(pred_idx, str(pred_idx))
        
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {self.id_to_label[i]: p.item() for i, p in enumerate(probs)}
        }

def main():
    model_dir = "./qa_training_results"
    inference = QAInference(model_dir)
    
    test_texts = [
        "Welcome to our helpline. I'm here to support you. How can I help today?",
        "Whatevever man, what do you want? I'm busy here.",
        "Yo, what's up, it's your friendly case follow-up dude! Listen, I got your previous case right here - remember that whole mess? Well, guess what, not much has happened...\nUh huh, yeah, I know it sucks. But hey, at least we tried, right? So, you still wanna do anything about it or should I just forget it and move on?\nYeah, no worries, I'm here if you need me. Just give a shout, kay?",
        "Alrighty! You've reached our help line here, bud! Whatcha need to know?\nYup, I'm curious about yer services... (rest of transcript)...\nAight then. We'll get on that right away. Need any other help with stuff?\nNah, this is all I needed. Cool!\nBye!",
        "Hello, I'm here to support you at our domestic violence hotline. How can I assist you today?\nI've been scared and I don't know what to do. My partner has been abusive and I fear for my safety... (rest of transcript)...\nI understand your situation and I'm truly sorry that you're going through this. It's important to remember that you're not alone and there are resources available to help.\nYes, we can discuss some options. There are shelters nearby where you can stay temporarily. We can also connect you with local law enforcement if you wish to report the incident...\nI appreciate your courage in reaching out. I want to make sure you're comfortable with the next steps. Are there any specific actions you'd prefer us to take?\nWe will follow up on your case and keep you updated. Is there anything else you'd like to discuss today?",


    ]
    
    print("\n" + "="*50)
    print("QA Quality Level Inference")
    print("="*50)
    
    for text in test_texts:
        result = inference.predict(text)
        print(f"\nText: {text}")
        print(f"Result: {result['label']} ({result['confidence']:.2%})")
        print(f"Breakdown: {result['probabilities']}")

if __name__ == "__main__":
    main()
