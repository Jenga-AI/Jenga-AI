"""
Unified Continuum Memory System for JengaHub

This module integrates NestedWhisper's hierarchical memory architecture 
with Jenga-AI's attention fusion mechanisms to create a powerful 
multi-modal memory system that works across speech and text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from .config import NestedLoRAConfig, LanguageBridgeSpec


class NestedMemoryLevel(nn.Module):
    """
    Individual memory level in the hierarchical CMS architecture.
    Combines NestedWhisper's audio memory blocks with text attention fusion.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        update_frequency: int,
        memory_size: int = 512,
        level_id: int = 0,
        modality: str = "multimodal",  # audio, text, multimodal
        device: str = "cuda"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.update_frequency = update_frequency
        self.memory_size = memory_size
        self.level_id = level_id
        self.modality = modality
        self.device = device
        self.step_count = 0
        
        # Cross-modal memory bank
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, hidden_dim, device=device) * 0.02
        )
        
        # Attention mechanism for memory retrieval
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-modal fusion
        if modality == "multimodal":
            self.audio_proj = nn.Linear(input_dim, hidden_dim)
            self.text_proj = nn.Linear(input_dim, hidden_dim)
            self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Task-specific projections (for multi-task learning)
        self.task_projections = nn.ModuleDict()
        
        # Adaptive threshold for memory updates
        self.surprise_threshold = nn.Parameter(
            torch.tensor(0.5, device=device)
        )
        
        # Memory consolidation weights
        self.consolidation_weight = nn.Parameter(
            torch.tensor(0.9, device=device)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        task_id: Optional[int] = None,
        modality_type: str = "multimodal",
        force_update: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through memory level.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            task_id: Task identifier for multi-task learning
            modality_type: "audio", "text", or "multimodal"
            force_update: Force memory update regardless of frequency
            
        Returns:
            output: Memory-enhanced representation
            metrics: Dictionary of memory metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        if self.modality == "multimodal" and modality_type in ["audio", "text"]:
            if modality_type == "audio":
                query = self.audio_proj(x)
            else:
                query = self.text_proj(x)
        else:
            query = self.query_proj(x)
        
        # Memory retrieval via attention
        memory_keys = self.key_proj(self.memory_bank)  # [memory_size, hidden_dim]
        memory_values = self.value_proj(self.memory_bank)
        
        # Compute attention scores
        attention_scores = torch.matmul(
            query.view(-1, self.hidden_dim),  # [batch*seq, hidden_dim]
            memory_keys.transpose(-2, -1)     # [hidden_dim, memory_size]
        ) / math.sqrt(self.hidden_dim)
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch*seq, memory_size]
        
        # Retrieve memory
        retrieved_memory = torch.matmul(
            attention_weights,  # [batch*seq, memory_size]
            memory_values      # [memory_size, hidden_dim]
        )  # [batch*seq, hidden_dim]
        
        retrieved_memory = retrieved_memory.view(batch_size, seq_len, self.hidden_dim)
        
        # Combine input with retrieved memory
        if self.modality == "multimodal":
            combined = torch.cat([query, retrieved_memory], dim=-1)
            output = torch.tanh(self.fusion_gate(combined))
        else:
            output = query + retrieved_memory
        
        # Task-specific adaptation
        if task_id is not None and str(task_id) in self.task_projections:
            task_adaptation = self.task_projections[str(task_id)](output)
            output = output + task_adaptation
        
        # Memory update (based on frequency and surprise)
        should_update = (
            force_update or 
            (self.step_count % self.update_frequency == 0)
        )
        
        metrics = {}
        if should_update and self.training:
            surprise = self._compute_surprise(query, attention_weights)
            if surprise > self.surprise_threshold:
                self._update_memory(query, surprise)
                metrics["memory_updated"] = True
                metrics["surprise"] = surprise
            else:
                metrics["memory_updated"] = False
                metrics["surprise"] = surprise
        
        metrics["attention_entropy"] = self._compute_attention_entropy(attention_weights)
        metrics["memory_utilization"] = self._compute_memory_utilization(attention_weights)
        
        self.step_count += 1
        return output, metrics
    
    def _compute_surprise(
        self, 
        query: torch.Tensor, 
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute surprise signal for memory updates."""
        # High surprise = low attention to existing memories
        max_attention = attention_weights.max(dim=-1)[0]
        surprise = 1.0 - max_attention.mean()
        return surprise
    
    def _update_memory(self, query: torch.Tensor, surprise: torch.Tensor):
        """Update memory bank with new patterns."""
        batch_size, seq_len, _ = query.shape
        
        # Select most surprising tokens for memory update
        flat_query = query.view(-1, self.hidden_dim)
        n_updates = min(batch_size, self.memory_size // 10)  # Update 10% max
        
        # Use surprise-weighted sampling
        if flat_query.size(0) > n_updates:
            indices = torch.randperm(flat_query.size(0), device=self.device)[:n_updates]
            selected_patterns = flat_query[indices]
        else:
            selected_patterns = flat_query
        
        # Consolidate with existing memory (weighted update)
        memory_indices = torch.randint(
            0, self.memory_size, 
            (selected_patterns.size(0),), 
            device=self.device
        )
        
        for i, mem_idx in enumerate(memory_indices):
            self.memory_bank.data[mem_idx] = (
                self.consolidation_weight * self.memory_bank.data[mem_idx] +
                (1 - self.consolidation_weight) * selected_patterns[i]
            )
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        # Add small epsilon for numerical stability
        eps = 1e-8
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + eps), 
            dim=-1
        ).mean()
        return entropy
    
    def _compute_memory_utilization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute how evenly memory is being utilized."""
        # Number of memories receiving significant attention (>1/memory_size)
        threshold = 1.0 / self.memory_size
        utilized_memories = (attention_weights > threshold).float().sum(dim=-1).mean()
        utilization = utilized_memories / self.memory_size
        return utilization
    
    def add_task_projection(self, task_id: int, projection: nn.Module):
        """Add task-specific projection layer."""
        self.task_projections[str(task_id)] = projection
    
    def reset_memory(self):
        """Reset memory bank to random initialization."""
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)


class LanguageFamilyHub(nn.Module):
    """
    Language family hub that shares information across related languages
    using Linguistic Bridge Specifications (LBS).
    """
    
    def __init__(
        self,
        family: str,
        languages: List[str],
        hidden_dim: int,
        bridge_specs: Dict[str, LanguageBridgeSpec]
    ):
        super().__init__()
        
        self.family = family
        self.languages = languages
        self.hidden_dim = hidden_dim
        self.bridge_specs = bridge_specs
        
        # Shared family representation
        self.family_embedding = nn.Parameter(
            torch.randn(hidden_dim) * 0.02
        )
        
        # Language-specific adaptations
        self.language_adaptations = nn.ModuleDict({
            lang: nn.Linear(hidden_dim, hidden_dim)
            for lang in languages
        })
        
        # Cross-language attention
        self.cross_lang_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Phonetic similarity matrix (from LBS)
        self.similarity_matrix = self._build_similarity_matrix()
    
    def _build_similarity_matrix(self) -> torch.Tensor:
        """Build similarity matrix from LBS specifications."""
        n_langs = len(self.languages)
        similarity = torch.eye(n_langs)  # Identity matrix
        
        lang_to_idx = {lang: i for i, lang in enumerate(self.languages)}
        
        for i, lang1 in enumerate(self.languages):
            for j, lang2 in enumerate(self.languages):
                if i != j and lang1 in self.bridge_specs and lang2 in self.bridge_specs:
                    spec1 = self.bridge_specs[lang1]
                    spec2 = self.bridge_specs[lang2]
                    
                    # Compute similarity based on shared phonemes
                    shared_phones = set(spec1.shared_phones) & set(spec2.shared_phones)
                    total_phones = set(spec1.shared_phones) | set(spec2.shared_phones)
                    
                    if total_phones:
                        sim = len(shared_phones) / len(total_phones)
                        similarity[i, j] = sim
        
        return nn.Parameter(similarity, requires_grad=False)
    
    def forward(
        self, 
        representations: Dict[str, torch.Tensor],
        active_language: str
    ) -> torch.Tensor:
        """
        Fuse representations across languages in the family.
        
        Args:
            representations: Dict of language -> representation tensors
            active_language: Primary language being processed
            
        Returns:
            Family-enhanced representation
        """
        if active_language not in self.languages:
            raise ValueError(f"Language {active_language} not in family {self.family}")
        
        # Get active language representation
        active_repr = representations.get(active_language)
        if active_repr is None:
            # Use family embedding as fallback
            batch_size = list(representations.values())[0].size(0)
            active_repr = self.family_embedding.unsqueeze(0).expand(
                batch_size, -1
            )
        
        # Apply language-specific adaptation
        if active_language in self.language_adaptations:
            active_repr = self.language_adaptations[active_language](active_repr)
        
        # Cross-language attention if multiple languages available
        if len(representations) > 1:
            # Stack all available representations
            lang_reprs = []
            lang_indices = []
            
            for lang, repr_tensor in representations.items():
                if lang in self.languages:
                    lang_reprs.append(repr_tensor)
                    lang_indices.append(self.languages.index(lang))
            
            if len(lang_reprs) > 1:
                stacked_reprs = torch.stack(lang_reprs, dim=1)  # [batch, n_langs, hidden]
                
                # Apply cross-language attention
                enhanced_repr, attention_weights = self.cross_lang_attention(
                    query=active_repr.unsqueeze(1),  # [batch, 1, hidden]
                    key=stacked_reprs,               # [batch, n_langs, hidden] 
                    value=stacked_reprs              # [batch, n_langs, hidden]
                )
                
                active_repr = enhanced_repr.squeeze(1)  # [batch, hidden]
        
        # Add family-level information
        family_enhanced = active_repr + self.family_embedding
        
        return family_enhanced


class ContinuumMemorySystem(nn.Module):
    """
    Complete hierarchical memory system combining NestedWhisper's CMS
    with Jenga-AI's multi-task capabilities.
    """
    
    def __init__(
        self,
        config: NestedLoRAConfig,
        input_dim: int,
        hidden_dim: int,
        language_bridges: Dict[str, LanguageBridgeSpec],
        device: str = "cuda"
    ):
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_levels = config.n_levels
        self.device = device
        
        # Create nested memory levels
        self.memory_levels = nn.ModuleList([
            NestedMemoryLevel(
                input_dim=hidden_dim if i > 0 else input_dim,
                hidden_dim=hidden_dim,
                update_frequency=config.update_frequencies[i],
                memory_size=max(512 // (2 ** i), 32),  # Decreasing memory size
                level_id=i,
                modality="multimodal",
                device=device
            )
            for i in range(self.n_levels)
        ])
        
        # Language family hubs
        self.language_bridges = language_bridges
        self.family_hubs = self._build_family_hubs()
        
        # Cross-level attention
        self.level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * self.n_levels, hidden_dim)
        
        # Level importance weights (learnable)
        self.level_weights = nn.Parameter(
            torch.ones(self.n_levels, device=device) / self.n_levels
        )
    
    def _build_family_hubs(self) -> nn.ModuleDict:
        """Build language family hubs from bridge specifications."""
        families = {}
        for lang, spec in self.language_bridges.items():
            family = spec.family
            if family not in families:
                families[family] = []
            families[family].append(lang)
        
        hubs = nn.ModuleDict()
        for family, languages in families.items():
            if len(languages) > 1:  # Only create hub if multiple languages
                hubs[family] = LanguageFamilyHub(
                    family=family,
                    languages=languages,
                    hidden_dim=self.hidden_dim,
                    bridge_specs=self.language_bridges
                )
        
        return hubs
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        language: Optional[str] = None,
        modality_type: str = "multimodal"
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Forward pass through complete memory hierarchy.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            task_id: Task identifier
            language: Language identifier for family hub routing
            modality_type: "audio", "text", or "multimodal"
            
        Returns:
            output: Hierarchically processed representation
            metrics: Comprehensive memory metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through each memory level
        level_outputs = []
        level_metrics = []
        current_input = x
        
        for i, memory_level in enumerate(self.memory_levels):
            level_output, metrics = memory_level(
                current_input, 
                task_id=task_id,
                modality_type=modality_type
            )
            
            level_outputs.append(level_output)
            level_metrics.append(metrics)
            current_input = level_output
        
        # Apply language family processing if applicable
        if language and self.language_bridges.get(language):
            spec = self.language_bridges[language]
            family = spec.family
            
            if family in self.family_hubs:
                # Create representations dict (simplified - in practice would have multiple)
                representations = {language: level_outputs[-1].mean(dim=1)}  # Pool over sequence
                
                family_enhanced = self.family_hubs[family](representations, language)
                
                # Broadcast back to sequence length
                family_enhanced = family_enhanced.unsqueeze(1).expand(-1, seq_len, -1)
                level_outputs[-1] = level_outputs[-1] + family_enhanced
        
        # Combine levels with learnable weights
        weighted_outputs = []
        weights = F.softmax(self.level_weights, dim=0)
        
        for i, level_output in enumerate(level_outputs):
            weighted_outputs.append(weights[i] * level_output)
        
        # Concatenate all levels
        concatenated = torch.cat(weighted_outputs, dim=-1)  # [batch, seq, hidden*n_levels]
        
        # Final projection
        final_output = self.output_proj(concatenated)
        
        # Aggregate metrics
        aggregated_metrics = {
            "level_metrics": level_metrics,
            "level_weights": weights.detach().cpu().tolist(),
            "memory_hierarchy_depth": self.n_levels,
            "active_language": language,
            "active_family": self.language_bridges.get(language, {}).get("family") if language else None
        }
        
        return final_output, aggregated_metrics
    
    def add_language_bridge(self, language: str, spec: LanguageBridgeSpec):
        """Add new language bridge specification."""
        self.language_bridges[language] = spec
        
        # Rebuild family hubs if needed
        family = spec.family
        if family not in self.family_hubs:
            family_languages = [
                lang for lang, bridge in self.language_bridges.items() 
                if bridge.family == family
            ]
            
            if len(family_languages) > 1:
                self.family_hubs[family] = LanguageFamilyHub(
                    family=family,
                    languages=family_languages,
                    hidden_dim=self.hidden_dim,
                    bridge_specs=self.language_bridges
                )
    
    def get_memory_statistics(self) -> Dict[str, any]:
        """Get comprehensive memory system statistics."""
        stats = {
            "total_memory_parameters": sum(
                level.memory_bank.numel() for level in self.memory_levels
            ),
            "memory_sizes": [
                level.memory_size for level in self.memory_levels
            ],
            "update_frequencies": [
                level.update_frequency for level in self.memory_levels
            ],
            "family_hubs": list(self.family_hubs.keys()),
            "supported_languages": list(self.language_bridges.keys())
        }
        
        return stats