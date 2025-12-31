
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import json
import os

@dataclass
class PEFTConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class TeacherStudentConfig:
    teacher_model: str
    distillation_alpha: float = 0.5
    temperature: float = 2.0

@dataclass
class ModelConfig:
    name: str = "Helsinki-NLP/opus-mt-mul-en"
    freeze_encoder: bool = False
    peft_config: PEFTConfig = field(default_factory=PEFTConfig)
    teacher_student_config: Optional[TeacherStudentConfig] = None

@dataclass
class TokenizationConfig:
    batch_size: int = 1000
    num_proc: int = 4
    use_cache: bool = True

@dataclass
class DatasetConfig:
    primary_dataset: str = "custom"
    custom_datasets: List[Dict[str, Any]] = field(default_factory=list)
    validation_split: float = 0.1
    test_split: float = 0.05
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    max_length: int = 512
    filter_length_ratio: bool = True
    max_length_ratio: float = 2.5
    min_length_ratio: float = 0.4
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    num_epochs: int = 8
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    logging_steps: int = 50
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    tensorboard_enabled: bool = True
    output_dir: str = "models/finetuned-model" # Added to match BaseTrainer expectation

@dataclass
class GenerationConfig:
    max_length: int = 256
    min_length: int = 1
    num_beams: int = 5
    length_penalty: float = 0.6
    early_stopping: bool = True
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 1.5
    do_sample: bool = False

@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["bleu", "chrf", "comet_qe"])
    compute_comet_during_training: bool = False
    test_size: int = 1000
    max_eval_samples: int = 5000
    save_predictions: bool = True
    batch_size: int = 32

@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "translation-training"
    run_name: Optional[str] = None
    log_models: bool = True
    log_artifacts: bool = True
    log_metrics: bool = True

@dataclass
class DeploymentConfig:
    model_output_dir: str = "models/finetuned-sw-en"
    save_tokenizer: bool = True
    save_config: bool = True
    create_model_card: bool = True
    push_to_hub: bool = False
    hub_model_id: str = "openchs-sw-en-translation"
    push_strategy: str = "end"

@dataclass
class SystemConfig:
    cache_dir: str = ".cache"
    use_cuda: bool = True
    cuda_device: str = "cuda:0"
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 4

@dataclass
class TranslationConfig:
    language_pair: str
    language_name: str
    model_config: ModelConfig
    dataset_config: DatasetConfig
    training_config: TrainingConfig
    generation_config: GenerationConfig
    evaluation_config: EvaluationConfig
    mlflow_config: MLflowConfig
    deployment: DeploymentConfig
    system_config: SystemConfig

    @classmethod
    def from_json(cls, json_path: str) -> 'TranslationConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        # Helper to safely create nested dataclasses
        def create_dataclass(dc_cls, data):
            if not data:
                return dc_cls()
            # Correct way to get field names:
            from dataclasses import fields
            valid_keys = {f.name for f in fields(dc_cls)}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return dc_cls(**filtered_data)

        # Recursively create config objects
        dataset_conf = create_dataclass(DatasetConfig, config_dict.get('dataset_config'))
        if 'tokenization' in config_dict.get('dataset_config', {}):
             dataset_conf.tokenization = create_dataclass(TokenizationConfig, config_dict['dataset_config']['tokenization'])
        
        model_conf = create_dataclass(ModelConfig, config_dict.get('model_config'))
        # Handle nested PEFT and TeacherStudent configs manually if needed, or rely on create_dataclass if structure matches
        if 'peft_config' in config_dict.get('model_config', {}):
            model_conf.peft_config = create_dataclass(PEFTConfig, config_dict['model_config']['peft_config'])
        if 'teacher_student_config' in config_dict.get('model_config', {}):
            model_conf.teacher_student_config = create_dataclass(TeacherStudentConfig, config_dict['model_config']['teacher_student_config'])
        
        # Backwards compatibility for flat model_name
        if 'model_name' in config_dict:
            model_conf.name = config_dict['model_name']

        return cls(
            language_pair=config_dict.get('language_pair', 'sw-en'),
            language_name=config_dict.get('language_name', 'Swahili'),
            model_config=model_conf,
            dataset_config=dataset_conf,
            training_config=create_dataclass(TrainingConfig, config_dict.get('training_config')),
            generation_config=create_dataclass(GenerationConfig, config_dict.get('generation_config')),
            evaluation_config=create_dataclass(EvaluationConfig, config_dict.get('evaluation_config')),
            mlflow_config=create_dataclass(MLflowConfig, config_dict.get('mlflow_config')),
            deployment=create_dataclass(DeploymentConfig, config_dict.get('deployment')),
            system_config=create_dataclass(SystemConfig, config_dict.get('system_config'))
        )
