
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from seq2seq_models.core.config import TranslationConfig

class Seq2SeqModel:
    """
    A factory class for creating Sequence-to-Sequence (Seq2Seq) models for fine-tuning.
    """
    def __init__(self, config: TranslationConfig):
        """
        Initializes the Seq2SeqModel factory with a given model configuration.

        Args:
            config (TranslationConfig): The configuration object defining the
                                        base model and its modifications.
        """
        self.config = config

    def create_model_and_tokenizer(self):
        """
        Creates and configures the Seq2Seq model and its tokenizer.

        Returns:
            tuple: A tuple containing the configured model and its tokenizer.
        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        model_config = self.config.model_config
        model_name = model_config.name
        cache_dir = self.config.system_config.cache_dir
        
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, use_safetensors=True)
        except OSError as e:
            raise ValueError(f"Failed to load model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except OSError as e:
            raise ValueError(f"Failed to load tokenizer for model '{model_name}'. Check model name, network connection, or Hugging Face credentials. Error: {e}")

        # 1. Freeze Encoder if requested
        if model_config.freeze_encoder:
            print("❄️ Freezing encoder layers...")
            for param in model.get_encoder().parameters():
                param.requires_grad = False

        # 2. Apply PEFT (LoRA) if enabled
        if model_config.peft_config.enabled:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                print("🚀 Applying PEFT (LoRA)...")
                
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=model_config.peft_config.r,
                    lora_alpha=model_config.peft_config.lora_alpha,
                    lora_dropout=model_config.peft_config.lora_dropout,
                    bias=model_config.peft_config.bias,
                    target_modules=model_config.peft_config.target_modules
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            except ImportError:
                print("⚠️ PEFT library not installed. Skipping LoRA.")
            except Exception as e:
                print(f"⚠️ Failed to apply PEFT: {e}")

        # 3. Setup Teacher-Student Distillation if configured
        if model_config.teacher_student_config:
            from seq2seq_models.model.teacher_student import TeacherStudentModel
            teacher_name = model_config.teacher_student_config.teacher_model
            print(f"👨‍🏫 Setting up Teacher-Student Distillation with teacher: {teacher_name}")
            
            try:
                teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_name, cache_dir=cache_dir, use_safetensors=True)
                # Freeze teacher
                for param in teacher_model.parameters():
                    param.requires_grad = False
                
                model = TeacherStudentModel(student_model=model, teacher_model=teacher_model)
            except Exception as e:
                print(f"⚠️ Failed to load teacher model: {e}")

        return model, tokenizer
