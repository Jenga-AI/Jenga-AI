"""
Universal model wrapper utilities for applying PEFT and freezing to any HuggingFace model
"""

import logging
from typing import Optional, List
from jenga_ai.core.peft.config import PEFTConfig, FreezingConfig

logger = logging.getLogger(__name__)

def detect_model_type(model) -> str:
    """
    Auto-detect model type from HuggingFace model architecture
    
    Args:
        model: HuggingFace model instance
        
    Returns:
        str: One of "seq2seq", "causal_lm", "encoder", "encoder-decoder"
    """
    # Check for seq2seq (encoder-decoder) models
    if hasattr(model, 'get_encoder') and hasattr(model, 'get_decoder'):
        return "seq2seq"
    
    # Check for causal LM (GPT-style)
    if hasattr(model, 'generate'):
        # Has generation but no encoder/decoder split = causal LM
        return "causal_lm"
    
    # Encoder-only models (BERT-style)
    return "encoder"

def get_default_target_modules(model, model_type: str) -> List[str]:
    """
    Get sensible default target modules for LoRA based on model type
    
    Args:
        model: HuggingFace model
        model_type: "seq2seq", "causal_lm", or "encoder"
        
    Returns:
        List of module names to target for LoRA
    """
    # Common attention projection targets
    defaults = {
        "seq2seq": ["q_proj", "v_proj"],
        "causal_lm": ["q_proj", "v_proj"],
        "encoder": ["query", "value"]  # BERT-style naming
    }
    
    base_targets = defaults.get(model_type, ["q_proj", "v_proj"])
    
    # Try to detect actual module names in the model
    actual_modules = set()
    for name, _ in model.named_modules():
        for target in base_targets:
            if target in name:
                actual_modules.add(target)
    
    # If we found actual modules, use them; otherwise use defaults
    return list(actual_modules) if actual_modules else base_targets

def apply_peft(model, peft_config: PEFTConfig, model_type: str = "auto"):
    """
    Apply PEFT (Parameter-Efficient Fine-Tuning) to any HuggingFace model
    
    Args:
        model: HuggingFace model (Seq2Seq, CausalLM, or Encoder)
        peft_config: PEFT configuration
        model_type: "seq2seq", "causal_lm", "encoder", or "auto"
    
    Returns:
        PEFT-wrapped model
        
    Raises:
        ImportError: If peft library is not installed
        ValueError: If unsupported model type
    """
    if not peft_config.enabled:
        return model
    
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        logger.error("❌ PEFT library not installed. Install with: pip install peft")
        raise ImportError("peft library required for PEFT functionality")
    
    # Auto-detect model type if requested
    if model_type == "auto":
        model_type = detect_model_type(model)
        logger.info(f"🔍 Auto-detected model type: {model_type}")
    
    # Map to PEFT task types
    task_type_map = {
        "seq2seq": TaskType.SEQ_2_SEQ_LM,
        "causal_lm": TaskType.CAUSAL_LM,
        "encoder": TaskType.FEATURE_EXTRACTION
    }
    
    if model_type not in task_type_map:
        raise ValueError(f"Unsupported model type: {model_type}. Must be one of {list(task_type_map.keys())}")
    
    # Auto-detect target modules if not specified
    if peft_config.target_modules is None:
        peft_config.target_modules = get_default_target_modules(model, model_type)
        logger.info(f"🎯 Auto-detected target modules: {peft_config.target_modules}")
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=task_type_map[model_type],
        inference_mode=False,
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.target_modules,
        bias=peft_config.bias,
        modules_to_save=peft_config.modules_to_save,
        fan_in_fan_out=peft_config.fan_in_fan_out
    )
    
    # Apply PEFT to model
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    logger.info(f"✅ Applied LoRA with r={peft_config.r}, alpha={peft_config.lora_alpha}")
    
    return peft_model

def freeze_layers(model, freezing_config: FreezingConfig, model_type: str = "auto"):
    """
    Apply layer freezing to any HuggingFace model
    
    Args:
        model: HuggingFace model
        freezing_config: Freezing configuration
        model_type: "seq2seq", "causal_lm", "encoder", or "auto"
        
    Returns:
        Model with frozen layers (in-place modification)
    """
    if not freezing_config.enabled:
        return model
    
    # Auto-detect model type if requested
    if model_type == "auto":
        model_type = detect_model_type(model)
    
    frozen_params = 0
    total_params = 0
    
    # Freeze embeddings if requested
    if freezing_config.freeze_embeddings:
        for name, param in model.named_parameters():
            if 'embed' in name.lower():
                param.requires_grad = False
                frozen_params += param.numel()
            total_params += param.numel()
        logger.info(f"❄️ Froze embedding layers")
    
    # Freeze encoder for seq2seq models
    if freezing_config.freeze_encoder and model_type in ["seq2seq"]:
        if hasattr(model, 'get_encoder'):
            for param in model.get_encoder().parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            logger.info(f"❄️ Froze encoder layers")
        else:
            logger.warning("⚠️ Model does not have get_encoder() method")
    
    # Freeze decoder for seq2seq models
    if freezing_config.freeze_decoder and model_type in ["seq2seq"]:
        if hasattr(model, 'get_decoder'):
            for param in model.get_decoder().parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            logger.info(f"❄️ Froze decoder layers")
        else:
            logger.warning("⚠️ Model does not have get_decoder() method")
    
    # Freeze first N layers
    if freezing_config.num_layers_to_freeze is not None:
        n = freezing_config.num_layers_to_freeze
        layer_count = 0
        
        for name, param in model.named_parameters():
            # Common layer naming patterns
            if 'layer.' in name or 'layers.' in name or 'block.' in name:
                # Extract layer number
                import re
                match = re.search(r'(?:layer|layers|block)\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    if layer_num < n:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        layer_count = max(layer_count, layer_num + 1)
        
        if layer_count > 0:
            logger.info(f"❄️ Froze first {layer_count} layers")
    
    # Freeze specific layers
    if freezing_config.layers_to_freeze:
        for name, param in model.named_parameters():
            import re
            match = re.search(r'(?:layer|layers|block)\.(\d+)', name)
            if match:
                layer_num = int(match.group(1))
                if layer_num in freezing_config.layers_to_freeze:
                    param.requires_grad = False
                    frozen_params += param.numel()
        logger.info(f"❄️ Froze specific layers: {freezing_config.layers_to_freeze}")
    
    # Count and log total frozen parameters
    for param in model.parameters():
        if not param.requires_grad:
            frozen_params += param.numel()
        total_params += param.numel()
    
    frozen_pct = (frozen_params / total_params) * 100 if total_params > 0 else 0
    logger.info(f"❄️ Froze {frozen_params:,} / {total_params:,} parameters ({frozen_pct:.1f}%)")
    
    return model
