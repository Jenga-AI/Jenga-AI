# Catastrophic Forgetting Prevention Guide

This guide explains how to use the Jenga-AI translation module's built-in strategies to prevent catastrophic forgetting during fine-tuning, based on best practices from recent research.

## Quick Start

### 1. LoRA (Low-Rank Adaptation) - **Recommended**

LoRA is the most efficient and effective method for large models. It adds small trainable matrices while keeping the base model frozen.

**Config Example:**
```json
{
  "model_config": {
    "name": "Helsinki-NLP/opus-mt-en-fr",
    "peft_config": {
      "enabled": true,
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj"]
    }
  }
}
```

**Benefits:**
- ✅ Preserves 99%+ of original model knowledge
- ✅ Reduces trainable parameters by 90%+
- ✅ Faster training and lower memory usage
- ✅ Can train multiple task-specific adapters

**When to use:** Always recommended for large models (>100M parameters)

### 2. Encoder Freezing

Freeze the encoder layers to protect general linguistic knowledge while fine-tuning the decoder.

**Config Example:**
```json
{
  "model_config": {
    "name": "Helsinki-NLP/opus-mt-en-fr",
    "freeze_encoder": true
  }
}
```

**Benefits:**
- ✅ Protects source language understanding
- ✅ Reduces forgetting by 50-70%
- ✅ Faster training

**When to use:** When your new data is in the same source language, just different domain/style

### 3. Knowledge Distillation

Use the original model as a "teacher" to guide the fine-tuned "student" model.

**Config Example:**
```json
{
  "model_config": {
    "name": "Helsinki-NLP/opus-mt-en-fr",
    "teacher_student_config": {
      "teacher_model": "Helsinki-NLP/opus-mt-en-fr",
      "distillation_alpha": 0.5,
      "temperature": 2.0
    }
  }
}
```

**Benefits:**
- ✅ Maintains output distribution similarity
- ✅ Soft regularization approach
- ✅ Good for domain adaptation

**When to use:** When you want the model to maintain similar behavior patterns

### 4. Rehearsal (Data Mixing)

Mix old data with new data during training. Already supported via multi-dataset config!

**Config Example:**
```json
{
  "dataset_config": {
    "custom_datasets": [
      {
        "name": "new_domain_data",
        "path": "data/new_domain.jsonl",
        "weight": 1.0
      },
      {
        "name": "original_data_sample",
        "path": "data/original_sample.jsonl",
        "weight": 0.3
      }
    ]
  }
}
```

**Benefits:**
- ✅ Most intuitive and effective
- ✅ Directly reinforces old knowledge
- ✅ No code changes needed

**When to use:** When you have access to a sample of original training data

## Combining Strategies

For best results, combine multiple approaches:

### Recommended Configuration for Production

```json
{
  "language_pair": "en-fr",
  "language_name": "French",
  "model_config": {
    "name": "Helsinki-NLP/opus-mt-en-fr",
    "freeze_encoder": true,
    "peft_config": {
      "enabled": true,
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
    }
  },
  "dataset_config": {
    "custom_datasets": [
      {
        "name": "specialized_domain",
        "path": "data/medical_translations.jsonl",
        "weight": 1.0
      },
      {
        "name": "general_sample",
        "path": "data/general_translations_sample.jsonl",
        "weight": 0.2
      }
    ]
  },
  "training_config": {
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "early_stopping_patience": 3
  }
}
```

**This combination:**
- Uses LoRA to preserve base model (99% of parameters frozen)
- Freezes encoder to protect source language understanding
- Mixes 20% old data via rehearsal
- Uses conservative learning rate

## Parameter Guide

### LoRA Parameters

- **`r`** (rank): Lower = fewer params, less expressiveness. Start with 8.
- **`lora_alpha`**: Controls scaling. Typically 2× `r` (e.g., `r=8`, `alpha=16`).
- **`lora_dropout`**: Regularization. 0.1 is standard.
- **`target_modules`**: Which layers to adapt. For seq2seq:
  - Minimal: `["q_proj", "v_proj"]`
  - Balanced: `["q_proj", "v_proj", "k_proj", "out_proj"]`
  - Aggressive: `["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]`

### Teacher-Student Parameters

- **`distillation_alpha`**: Balance between task loss and distillation loss. 0.5 = equal weight.
- **`temperature`**: Softness of teacher outputs. Higher = softer. 2.0 is standard.

## Verification

Always evaluate on old tasks to detect forgetting:

1. Keep a small validation set from original training data
2. Evaluate before and after fine-tuning
3. Monitor metrics like BLEU drop on original domain

```python
# Example evaluation workflow
python scripts/train_translation.py \
  --config configs/my_finetuning.json \
  --domain_src data/original_src.txt \
  --domain_tgt data/original_tgt.txt
```

## Troubleshooting

**Q: Model still forgetting despite using LoRA?**
- Increase rehearsal ratio (more old data)
- Lower learning rate
- Add encoder freezing

**Q: LoRA not installing?**
```bash
pip install peft
```

**Q: Training too slow with LoRA?**
- LoRA should be faster! Check if you accidentally enabled full fine-tuning
- Reduce `r` (rank) to 4 or lower

**Q: Need to merge LoRA adapters back into base model?**
```python
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "path/to/adapters")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("path/to/merged_model")
```

## Real-World Examples

### Medical Translation (High Forgetting Risk)
```json
{
  "model_config": {
    "peft_config": {"enabled": true, "r": 16},
    "freeze_encoder": true
  },
  "dataset_config": {
    "custom_datasets": [
      {"name": "medical", "path": "medical.jsonl", "weight": 1.0},
      {"name": "general", "path": "general_sample.jsonl", "weight": 0.3}
    ]
  }
}
```

### Domain Adaptation (Low Forgetting Risk)
```json
{
  "model_config": {
    "freeze_encoder": true
  },
  "training_config": {
    "learning_rate": 5e-5,
    "num_epochs": 2
  }
}
```

## References

- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Rehearsal: Experience Replay in continual learning
- Knowledge Distillation: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
