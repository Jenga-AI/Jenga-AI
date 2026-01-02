"""
Test suite for jenga_ai.core.peft module
Tests universal PEFT functionality across model types
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel, AutoTokenizer


def test_imports():
    """Test that core modules can be imported"""
    from jenga_ai.core.peft import PEFTConfig, FreezingConfig, apply_peft, freeze_layers, detect_model_type
    from jenga_ai.core.distillation import TeacherStudentWrapper
    assert True

def test_peft_config_creation():
    """Test PEFT config creation with defaults"""
    from jenga_ai.core.peft import PEFTConfig
    
    config = PEFTConfig()
    assert config.enabled == False
    assert config.r == 8
    assert config.lora_alpha == 16
    
    config_enabled = PEFTConfig(enabled=True, r=4)
    assert config_enabled.enabled == True
    assert config_enabled.r == 4

def test_freezing_config_creation():
    """Test freezing config creation"""
    from jenga_ai.core.peft import FreezingConfig
    
    config = FreezingConfig()
    assert config.enabled == False
    
    config_enabled = FreezingConfig(enabled=True, freeze_encoder=True)
    assert config_enabled.enabled == True
    assert config_enabled.freeze_encoder == True

def test_model_type_detection():
    """Test auto-detection of model types"""
    from jenga_ai.core.peft import detect_model_type
    
    # Test with seq2seq model
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", use_safetensors=True)
    assert detect_model_type(seq2seq_model) == "seq2seq"
    
    # # Test with causal LM (commented out to avoid download)
    # causal_model = AutoModelForCausalLM.from_pretrained("gpt2", use_safetensors=True)
    # assert detect_model_type(causal_model) == "causal_lm"
    
    print("✅ Model type detection works")

def test_peft_application_seq2seq():
    """Test applying PEFT to seq2seq model"""
    from jenga_ai.core.peft import PEFTConfig, apply_peft
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", use_safetensors=True)
    
    config = PEFTConfig(enabled=True, r=4, lora_alpha=8)
    peft_model = apply_peft(model, config, model_type="seq2seq")
    
    # Check that model is wrapped
    assert hasattr(peft_model, 'print_trainable_parameters')
    
    # Check that most parameters are frozen
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    
    assert trainable_params < total_params * 0.1  # Less than 10% trainable
    print(f"✅ PEFT applied: {trainable_params}/{total_params} trainable ({trainable_params/total_params*100:.2f}%)")

def test_layer_freezing():
    """Test layer freezing functionality"""
    from jenga_ai.core.peft import FreezingConfig, freeze_layers
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", use_safetensors=True)
    
    # Count trainable params before
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    config = FreezingConfig(enabled=True, freeze_encoder=True)
    freeze_layers(model, config, model_type="seq2seq")
    
    # Count trainable params after
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert trainable_after < trainable_before
    print(f"✅ Froze encoder: {trainable_before} -> {trainable_after} trainable params")

def test_combined_peft_and_freezing():
    """Test combining PEFT with layer freezing"""
    from jenga_ai.core.peft import PEFTConfig, FreezingConfig, apply_peft, freeze_layers
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", use_safetensors=True)
    
    # Apply freezing first
    freeze_config = FreezingConfig(enabled=True, freeze_embeddings=True)
    model = freeze_layers(model, freeze_config)
    
    # Then apply PEFT
    peft_config = PEFTConfig(enabled=True, r=4)
    model = apply_peft(model, peft_config, model_type="seq2seq")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    assert trainable_params < total_params * 0.05  # Very few trainable params
    print(f"✅ Combined: {trainable_params}/{total_params} trainable ({trainable_params/total_params*100:.2f}%)")

if __name__ == "__main__":
    print("🧪 Running jenga_ai.core.peft tests...\n")
    
    test_imports()
    print("✅ Imports successful")
    
    test_peft_config_creation()
    print("✅ Config creation works")
    
    test_freezing_config_creation()
    print("✅ Freezing config works")
    
    test_model_type_detection()
    
    test_peft_application_seq2seq()
    
    test_layer_freezing()
    
    test_combined_peft_and_freezing()
    
    print("\n✅ All tests passed!")
