"""
Unified JengaHub Model Architecture

This module combines all JengaHub components into a cohesive multimodal
architecture that supports African languages, code-switching, nested LoRA,
and hierarchical memory systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from transformers import AutoModel, AutoTokenizer, WhisperModel

from .config import MultiModalConfig
from .memory import ContinuumMemorySystem
from .nested_lora import NestedLoRALinear, NestedLoRAConverter
from .code_switching import MultimodalCodeSwitchingBridge
from ..data.processor import ProcessedSample


class AudioTextBridge(nn.Module):
    """
    Bridge module that connects audio and text representations
    through shared embedding space and cross-modal attention.
    """
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        bridge_dim: int = 768,
        num_heads: int = 8,
        enable_cross_attention: bool = True
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.bridge_dim = bridge_dim
        self.enable_cross_attention = enable_cross_attention
        
        # Projection layers to common embedding space
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-modal attention layers
        if enable_cross_attention:
            self.audio_to_text_attention = nn.MultiheadAttention(
                embed_dim=bridge_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            
            self.text_to_audio_attention = nn.MultiheadAttention(
                embed_dim=bridge_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
        
        # Fusion mechanisms
        self.fusion_gate = nn.Sequential(
            nn.Linear(bridge_dim * 2, bridge_dim),
            nn.Sigmoid()
        )
        
        self.output_projection = nn.Linear(bridge_dim, bridge_dim)
        
        # Language detection heads for code-switching
        self.audio_language_head = nn.Linear(bridge_dim, 10)  # Support 10 languages
        self.text_language_head = nn.Linear(bridge_dim, 10)
        
    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_cross_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the audio-text bridge.
        
        Args:
            audio_features: [batch_size, audio_seq_len, audio_dim]
            text_features: [batch_size, text_seq_len, text_dim]
            audio_mask: [batch_size, audio_seq_len]
            text_mask: [batch_size, text_seq_len]
            return_cross_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - fused_audio: Enhanced audio features
            - fused_text: Enhanced text features
            - bridge_representation: Unified multimodal representation
            - language_predictions: Language detection outputs
        """
        outputs = {}
        attention_weights = {}
        
        # Project to common space
        if audio_features is not None:
            audio_projected = self.audio_projector(audio_features)
            outputs["audio_projected"] = audio_projected
            
            # Language detection for audio
            audio_lang_logits = self.audio_language_head(audio_projected.mean(dim=1))
            outputs["audio_language_logits"] = audio_lang_logits
        
        if text_features is not None:
            text_projected = self.text_projector(text_features)
            outputs["text_projected"] = text_projected
            
            # Language detection for text
            text_lang_logits = self.text_language_head(text_projected.mean(dim=1))
            outputs["text_language_logits"] = text_lang_logits
        
        # Cross-modal attention if both modalities present
        if (audio_features is not None and text_features is not None and 
            self.enable_cross_attention):
            
            # Audio attending to text
            audio_enhanced, audio_to_text_attn = self.audio_to_text_attention(
                query=audio_projected,
                key=text_projected,
                value=text_projected,
                key_padding_mask=~text_mask if text_mask is not None else None
            )
            
            # Text attending to audio  
            text_enhanced, text_to_audio_attn = self.text_to_audio_attention(
                query=text_projected,
                key=audio_projected,
                value=audio_projected,
                key_padding_mask=~audio_mask if audio_mask is not None else None
            )
            
            if return_cross_attention:
                attention_weights["audio_to_text"] = audio_to_text_attn
                attention_weights["text_to_audio"] = text_to_audio_attn
            
            # Fusion with gating mechanism
            audio_concat = torch.cat([audio_projected, audio_enhanced], dim=-1)
            audio_gate = self.fusion_gate(audio_concat)
            fused_audio = audio_gate * audio_projected + (1 - audio_gate) * audio_enhanced
            
            text_concat = torch.cat([text_projected, text_enhanced], dim=-1)
            text_gate = self.fusion_gate(text_concat)
            fused_text = text_gate * text_projected + (1 - text_gate) * text_enhanced
            
            outputs["fused_audio"] = self.output_projection(fused_audio)
            outputs["fused_text"] = self.output_projection(fused_text)
            
            # Create unified bridge representation
            # Pool and combine both modalities
            audio_pooled = fused_audio.mean(dim=1)  # [batch_size, bridge_dim]
            text_pooled = fused_text.mean(dim=1)    # [batch_size, bridge_dim]
            
            bridge_representation = (audio_pooled + text_pooled) / 2
            outputs["bridge_representation"] = bridge_representation
            
        elif audio_features is not None:
            outputs["fused_audio"] = self.output_projection(audio_projected)
            outputs["bridge_representation"] = audio_projected.mean(dim=1)
            
        elif text_features is not None:
            outputs["fused_text"] = self.output_projection(text_projected)
            outputs["bridge_representation"] = text_projected.mean(dim=1)
        
        if return_cross_attention:
            outputs["attention_weights"] = attention_weights
            
        return outputs


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion mechanism that combines Jenga-AI's attention fusion
    with NestedWhisper's multi-timescale architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_levels: int,
        num_tasks: int,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        
        # Task embeddings for each level
        self.task_embeddings = nn.ModuleList([
            nn.Embedding(num_tasks, hidden_dim)
            for _ in range(num_levels)
        ])
        
        # Level-specific attention mechanisms
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_levels)
        ])
        
        # Cross-level fusion
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projections
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        level_representations: List[torch.Tensor],
        task_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical fusion.
        
        Args:
            level_representations: List of tensors from each memory level
            task_id: Task identifier
            attention_mask: Attention mask
            
        Returns:
            fused_representation: Hierarchically fused output
            attention_info: Attention weights and metrics
        """
        batch_size = level_representations[0].size(0)
        seq_len = level_representations[0].size(1)
        
        # Apply task-specific attention at each level
        task_enhanced_levels = []
        attention_info = {}
        
        for level_idx, level_repr in enumerate(level_representations):
            # Get task embedding for this level
            task_emb = self.task_embeddings[level_idx](
                torch.tensor(task_id, device=level_repr.device)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            
            # Broadcast task embedding
            task_emb = task_emb.expand(batch_size, seq_len, -1)
            
            # Combine with level representation
            combined = level_repr + task_emb
            
            # Apply attention
            attended, attn_weights = self.level_attentions[level_idx](
                query=combined,
                key=combined,
                value=combined,
                key_padding_mask=~attention_mask if attention_mask is not None else None
            )
            
            # Project output
            projected = self.output_projections[level_idx](attended)
            task_enhanced_levels.append(projected)
            
            attention_info[f"level_{level_idx}_attention"] = attn_weights
        
        # Cross-level attention
        if len(task_enhanced_levels) > 1:
            # Stack all levels
            stacked_levels = torch.stack(task_enhanced_levels, dim=1)  # [batch, num_levels, seq, hidden]
            stacked_levels = stacked_levels.view(batch_size, -1, self.hidden_dim)  # [batch, num_levels*seq, hidden]
            
            # Apply cross-level attention
            cross_attended, cross_attn_weights = self.cross_level_attention(
                query=stacked_levels,
                key=stacked_levels,
                value=stacked_levels
            )
            
            # Reshape back
            cross_attended = cross_attended.view(batch_size, self.num_levels, seq_len, self.hidden_dim)
            
            attention_info["cross_level_attention"] = cross_attn_weights
            
            # Update enhanced levels
            for i in range(self.num_levels):
                task_enhanced_levels[i] = cross_attended[:, i, :, :]
        
        # Final fusion
        concatenated = torch.cat(task_enhanced_levels, dim=-1)  # [batch, seq, hidden*num_levels]
        fused_representation = self.final_fusion(concatenated)
        
        return fused_representation, attention_info


class JengaHubMultiModalModel(nn.Module):
    """
    Main JengaHub model that integrates all components:
    - Multimodal processing (audio + text)
    - Hierarchical memory (CMS)
    - Nested LoRA adaptation
    - Code-switching awareness
    - Multi-task learning
    """
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.text.hidden_size
        
        # Base encoders
        self.text_encoder = AutoModel.from_pretrained(config.text.base_model)
        self.audio_encoder = WhisperModel.from_pretrained(config.audio.base_model).encoder
        
        # Apply NestedLoRA to encoders
        if config.nested_lora.n_levels > 0:
            self.text_encoder = NestedLoRAConverter.convert_model(
                self.text_encoder,
                config.nested_lora,
                target_modules=config.nested_lora.target_modules
            )
            
            self.audio_encoder = NestedLoRAConverter.convert_model(
                self.audio_encoder,
                config.nested_lora,
                target_modules=config.nested_lora.target_modules
            )
        
        # Continuum Memory System
        self.memory_system = ContinuumMemorySystem(
            config=config.nested_lora,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            language_bridges=config.language_bridges
        )
        
        # Code-switching bridge
        self.code_switching_bridge = MultimodalCodeSwitchingBridge(
            audio_config=config.audio,
            text_config=config.text,
            bridge_specs=config.language_bridges,
            hidden_dim=self.hidden_dim
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        self._create_task_heads()
        
        # Cross-modal fusion
        self.cross_modal_fusion = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Language-adaptive layer norm
        self.adaptive_layer_norm = nn.ModuleDict({
            lang: nn.LayerNorm(self.hidden_dim)
            for lang in config.language_bridges.keys()
        })
        
        # Task embeddings for multi-task learning
        self.task_embeddings = nn.Embedding(
            len(config.text.tasks), 
            self.hidden_dim
        )
        
    def _create_task_heads(self):
        """Create task-specific output heads."""
        for task in self.config.text.tasks:
            if task in ['classification', 'sentiment_analysis']:
                # Classification head
                self.task_heads[task] = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim // 2, 3)  # Default 3 classes
                )
            
            elif task == 'ner':
                # Token-level classification
                self.task_heads[task] = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, 9)  # Standard BIO NER tags
                )
            
            elif task == 'qa':
                # Question answering head
                self.task_heads[task] = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, 2)  # Start/end positions
                )
            
            else:
                # Generic head
                self.task_heads[task] = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, 1)
                )
        
    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the complete JengaHub architecture.
        
        Args:
            audio_features: Audio mel-spectrogram features [batch, time, n_mels]
            audio_attention_mask: Audio attention mask [batch, time]
            text_input_ids: Text token IDs [batch, seq_len]
            text_attention_mask: Text attention mask [batch, seq_len]
            task_ids: Task identifiers [batch]
            language_ids: Language identifiers [batch]
            labels: Ground truth labels [batch] or [batch, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs (logits, losses, metrics)
        """
        batch_size = (
            audio_features.size(0) if audio_features is not None 
            else text_input_ids.size(0)
        )
        
        outputs = {}
        
        # Process audio modality
        audio_representations = None
        if audio_features is not None:
            audio_outputs = self.audio_encoder(
                input_features=audio_features,
                attention_mask=audio_attention_mask,
                return_dict=True
            )
            audio_representations = audio_outputs.last_hidden_state
        
        # Process text modality
        text_representations = None
        if text_input_ids is not None:
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True
            )
            text_representations = text_outputs.last_hidden_state
        
        # Code-switching analysis
        cs_results = self.code_switching_bridge(
            audio_features=audio_representations,
            text_features=text_representations,
            audio_attention_mask=audio_attention_mask,
            text_attention_mask=text_attention_mask
        )
        
        outputs['code_switching'] = cs_results
        
        # Get primary representation for memory processing
        if audio_representations is not None and text_representations is not None:
            # Multimodal case - use fused representation
            primary_repr = cs_results.get('fused_features', text_representations)
            primary_language = self._detect_primary_language(language_ids, batch_size)
        elif text_representations is not None:
            primary_repr = text_representations
            primary_language = self._detect_primary_language(language_ids, batch_size)
        elif audio_representations is not None:
            primary_repr = audio_representations
            primary_language = self._detect_primary_language(language_ids, batch_size)
        else:
            raise ValueError("At least one modality (audio or text) must be provided")
        
        # Process through Continuum Memory System
        memory_output, memory_metrics = self.memory_system(
            x=primary_repr,
            task_id=task_ids[0].item() if task_ids is not None else None,
            language=primary_language,
            modality_type="multimodal" if audio_representations is not None and text_representations is not None else "text"
        )
        
        outputs['memory_metrics'] = memory_metrics
        
        # Apply language-adaptive layer normalization
        if primary_language and primary_language in self.adaptive_layer_norm:
            memory_output = self.adaptive_layer_norm[primary_language](memory_output)
        
        # Task-specific processing
        task_outputs = {}
        
        if task_ids is not None:
            for i, task_id in enumerate(task_ids):
                task_name = self.config.text.tasks[task_id.item()]
                
                # Add task embedding
                task_emb = self.task_embeddings(task_id).unsqueeze(1)  # [1, hidden_dim]
                task_enhanced = memory_output[i:i+1] + task_emb  # [1, seq_len, hidden_dim]
                
                # Apply task head
                if task_name in self.task_heads:
                    if task_name in ['classification', 'sentiment_analysis']:
                        # Use [CLS] token or global pooling
                        pooled_repr = task_enhanced.mean(dim=1)  # [1, hidden_dim]
                        task_logits = self.task_heads[task_name](pooled_repr)
                    else:
                        # Token-level prediction
                        task_logits = self.task_heads[task_name](task_enhanced)
                    
                    task_outputs[task_name] = task_logits
        else:
            # Default classification
            pooled_repr = memory_output.mean(dim=1)
            default_logits = self.task_heads.get(
                'classification', 
                list(self.task_heads.values())[0]
            )(pooled_repr)
            task_outputs['default'] = default_logits
        
        outputs['task_outputs'] = task_outputs
        
        # Compute losses if labels provided
        if labels is not None:
            losses = self._compute_losses(task_outputs, labels, task_ids)
            outputs['losses'] = losses
            outputs['loss'] = sum(losses.values())
        
        # Main prediction logits (for compatibility)
        if task_outputs:
            outputs['logits'] = list(task_outputs.values())[0]
        
        # Enhanced representations
        outputs['memory_enhanced_repr'] = memory_output
        outputs['audio_repr'] = audio_representations
        outputs['text_repr'] = text_representations
        
        if return_dict:
            return outputs
        else:
            return outputs.get('logits', memory_output)
    
    def _detect_primary_language(
        self, 
        language_ids: Optional[torch.Tensor], 
        batch_size: int
    ) -> Optional[str]:
        """Detect primary language for the batch."""
        if language_ids is not None and len(language_ids) > 0:
            # Use most common language in batch
            primary_id = language_ids.mode().values.item()
            languages = list(self.config.language_bridges.keys())
            if primary_id < len(languages):
                return languages[primary_id]
        
        # Default to primary language from config
        return self.config.audio.primary_language
    
    def _compute_losses(
        self,
        task_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute task-specific losses."""
        losses = {}
        
        for task_name, logits in task_outputs.items():
            if task_name in ['classification', 'sentiment_analysis', 'default']:
                # Cross-entropy loss for classification
                loss = F.cross_entropy(logits, labels)
            
            elif task_name == 'ner':
                # Token-level cross-entropy (ignore -100 labels)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            elif task_name == 'qa':
                # For QA, labels should contain start/end positions
                if labels.dim() == 2 and labels.size(1) == 2:
                    start_logits, end_logits = logits.split(1, dim=-1)
                    start_loss = F.cross_entropy(start_logits.squeeze(-1), labels[:, 0])
                    end_loss = F.cross_entropy(end_logits.squeeze(-1), labels[:, 1])
                    loss = (start_loss + end_loss) / 2
                else:
                    # Fallback to classification loss
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            else:
                # Default MSE loss for regression tasks
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            
            losses[task_name] = loss
        
        return losses
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # LoRA statistics
        lora_stats = NestedLoRAConverter.get_model_statistics(self)
        stats.update({f'lora_{k}': v for k, v in lora_stats.items()})
        
        # Memory statistics
        memory_stats = self.memory_system.get_memory_statistics()
        stats.update({f'memory_{k}': v for k, v in memory_stats.items()})
        
        # Task head statistics
        stats['task_heads'] = {
            task: {
                'parameters': sum(p.numel() for p in head.parameters()),
                'layers': len(list(head.modules())) - 1  # Exclude container
            }
            for task, head in self.task_heads.items()
        }
        
        return stats
    
    def add_task(self, task_name: str, num_classes: int = 2):
        """Dynamically add a new task head."""
        if task_name in ['classification', 'sentiment_analysis']:
            head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, num_classes)
            )
        else:
            head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, num_classes)
            )
        
        self.task_heads[task_name] = head
        
        # Move to same device as model
        device = next(self.parameters()).device
        head.to(device)
    
    def freeze_backbone(self):
        """Freeze the backbone encoders for transfer learning."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone encoders."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    
    def get_language_representations(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get language-specific representations for analysis."""
        
        # Get base text representations
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True
        )
        
        base_repr = text_outputs.last_hidden_state
        
        # Apply language-specific adaptations
        lang_representations = {}
        
        for lang_name, layer_norm in self.adaptive_layer_norm.items():
            adapted_repr = layer_norm(base_repr)
            lang_representations[lang_name] = adapted_repr
        
        return lang_representations


class JengaHubModelFactory:
    """Factory for creating JengaHub models with different configurations."""
    
    @staticmethod
    def create_base_model(config: MultiModalConfig) -> JengaHubMultiModalModel:
        """Create base JengaHub model."""
        return JengaHubMultiModalModel(config)
    
    @staticmethod
    def create_text_only_model(config: MultiModalConfig) -> JengaHubMultiModalModel:
        """Create text-only model (audio components disabled)."""
        model = JengaHubMultiModalModel(config)
        
        # Disable audio processing in forward pass
        original_forward = model.forward
        
        def text_only_forward(*args, audio_features=None, audio_attention_mask=None, **kwargs):
            return original_forward(*args, audio_features=None, audio_attention_mask=None, **kwargs)
        
        model.forward = text_only_forward
        return model
    
    @staticmethod
    def create_specialized_model(
        config: MultiModalConfig,
        target_language: str,
        target_tasks: List[str]
    ) -> JengaHubMultiModalModel:
        """Create language and task-specialized model."""
        
        # Modify config for specialization
        specialized_config = config.__class__.from_dict({
            **config.to_dict(),
            'audio': {
                **config.audio.__dict__,
                'primary_language': target_language
            },
            'text': {
                **config.text.__dict__,
                'tasks': target_tasks
            }
        })
        
        return JengaHubMultiModalModel(specialized_config)
    
    @staticmethod
    def create_lightweight_model(config: MultiModalConfig) -> JengaHubMultiModalModel:
        """Create lightweight model for mobile/edge deployment."""
        
        # Reduce model complexity
        lightweight_config = config.__class__.from_dict({
            **config.to_dict(),
            'text': {
                **config.text.__dict__,
                'hidden_size': config.text.hidden_size // 2
            },
            'nested_lora': {
                **config.nested_lora.__dict__,
                'n_levels': 3,  # Reduce memory levels
                'base_rank': config.nested_lora.base_rank // 2
            }
        })
        
        return JengaHubMultiModalModel(lightweight_config)


# Utility functions
def load_pretrained_jengahub(
    model_path: str,
    config: Optional[MultiModalConfig] = None
) -> JengaHubMultiModalModel:
    """Load pretrained JengaHub model."""
    
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Extract config if not provided
    if config is None:
        if 'config' in state_dict:
            config = MultiModalConfig.from_dict(state_dict['config'])
        else:
            raise ValueError("Config must be provided when loading model without embedded config")
    
    # Create model
    model = JengaHubMultiModalModel(config)
    
    # Load weights
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    return model


def save_jengahub_model(
    model: JengaHubMultiModalModel,
    save_path: str,
    include_config: bool = True
):
    """Save JengaHub model with configuration."""
    
    save_dict = {
        'model_state_dict': model.state_dict()
    }
    
    if include_config:
        save_dict['config'] = model.config.to_dict()
        save_dict['model_statistics'] = model.get_model_statistics()
    
    torch.save(save_dict, save_path)