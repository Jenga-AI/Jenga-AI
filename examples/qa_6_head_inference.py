import torch
import yaml
import os
from transformers import AutoTokenizer, AutoConfig
from multitask_bert.core.config import ExperimentConfig
from multitask_bert.core.model import MultiTaskModel
from typing import List, Dict, Any

class QA6HeadInference:
    def __init__(self, model_dir: str):
        """
        Initialize the 6-head QA inference with trained model.
        
        Args:
            model_dir: Path to the directory containing best_model and experiment_config.yaml
        """
        self.model_dir = model_dir
        self.best_model_path = os.path.join(model_dir, "best_model")
        self.device = torch.device("cpu") # Default to CPU for stability on Mac
        
        # Load configuration
        self.config = self._load_config()
        
        # Load tokenizer and model
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"❌ ERROR: The 'best_model' folder was not found at {self.best_model_path}. "
                                    f"This usually means the training finished without saving because the "
                                    f"metric_for_best_model was incorrect.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.best_model_path)
        self.model = self._load_model()
        
        # Define metric descriptions for the report
        self.head_info = {
            "opening": ["Appropriate Greeting"],
            "listening": ["Attentive", "Patient", "No Interruption", "Clarification", "Empathy", "Tone"],
            "proactiveness": ["Identifying Needs", "Anticipating Questions", "Upselling/Added Value"],
            "resolution": ["Problem Solved", "Clear Instructions", "Correct Information", "Follow-up Set", "Resource Sharing"],
            "hold": ["Proper Hold Procedure", "Hold Duration"],
            "closing": ["Professional Closing"]
        }
        
        print(f"JengaAI 2.0: 6-Head QA Auditor Loaded Successfully.")
    
    def _load_config(self) -> ExperimentConfig:
        """Load experiment configuration from saved YAML."""
        config_path = os.path.join(self.model_dir, "experiment_config.yaml")
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig(**config_dict)
    
    def _load_model(self):
        """Load the trained model and handle layer mapping."""
        model_config = AutoConfig.from_pretrained(self.best_model_path)
        
        model = MultiTaskModel(
            config=model_config,
            model_config=self.config.model,
            task_configs=self.config.tasks
        )
        
        # Load weights
        weights_path = os.path.join(self.best_model_path, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            weights_path = os.path.join(self.best_model_path, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location=self.device)
        
        # JengaAI 2.0 Mapping: Ensure 'backbone.encoder' keys align
        # If model was trained during the refactor, keys might already be correct
        # This mapping handles both cases
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.') and not hasattr(model, 'encoder'):
                new_key = key.replace('encoder.', 'backbone.encoder.', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
                
        model.load_state_dict(new_state_dict)
        model.to(self.device).eval()
        return model
    
    def audit_transcript(self, text: str) -> Dict[str, Any]:
        """Perform a full multi-point audit of the transcript."""
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
        
        all_logits = outputs["logits"]
        audit_results = {}
        
        for head_name, logits in all_logits.items():
            # Apply sigmoid since these are independent binary checks
            scores = torch.sigmoid(logits[0]).tolist()
            descriptions = self.head_info.get(head_name, [f"Metric {i+1}" for i in range(len(scores))])
            
            audit_results[head_name] = {
                "score": sum(scores) / len(scores), # Overall head score
                "details": {desc: score for desc, score in zip(descriptions, scores)}
            }
            
        return audit_results

def print_audit_report(text: str, results: Dict[str, Any]):
    print("\n" + "="*60)
    print("JENGA-AI: MULTI-TASK QA AUDIT REPORT")
    print("="*60)
    print(f"Transcript: {text[:100]}...")
    print("-" * 60)
    
    total_score = 0
    for category, content in results.items():
        score_pct = content['score'] * 100
        status = "✅ PASS" if score_pct > 70 else "⚠️ REVIEW" if score_pct > 40 else "❌ FAIL"
        print(f"\n[{category.upper()}] Status: {status} ({score_pct:.1f}%)")
        
        for detail, val in content['details'].items():
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            print(f"  - {detail.ljust(25)}: {bar} ({val:.2f})")
        total_score += score_pct
        
    final_score = total_score / len(results)
    print("\n" + "="*60)
    print(f"FINAL AUDIT SCORE: {final_score:.1f}%")
    print("="*60)

if __name__ == "__main__":
    model_dir = "./qa_training_results_v2"
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found. Training might still be in progress.")
    else:
        auditor = QA6HeadInference(model_dir)
        
        # sample_text = "Welcome to our helpline. I understand you're upset about the delay. Let's fix this right now."
        sample_text = "I can see you're in a difficult situation. Please tell me what's been happening,\nA heavy silence fell over the line as the caller began to share their story... (rest of transcript) "
  
        report = auditor.audit_transcript(sample_text)
        print_audit_report(sample_text, report)
