"""
Code-Switching Bridge Architecture for JengaHub

This module implements advanced code-switching detection and adaptation
mechanisms that work across both speech and text modalities, enabling
seamless multilingual processing with African language specialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import re
from dataclasses import dataclass
from enum import Enum

from .config import LanguageBridgeSpec, AudioConfig, TextConfig


class SwitchType(Enum):
    """Types of code-switching patterns."""
    INTER_SENTENTIAL = "inter_sentential"  # Between sentences
    INTRA_SENTENTIAL = "intra_sentential"  # Within sentences
    INTRA_WORD = "intra_word"             # Within words (morphological)
    TAG_SWITCHING = "tag_switching"        # Discourse markers
    EMBLEMATIC = "emblematic"            # Cultural expressions


@dataclass
class SwitchPoint:
    """Information about a detected code-switching point."""
    
    position: int  # Frame/token position
    from_language: str
    to_language: str
    switch_type: SwitchType
    confidence: float
    linguistic_context: Optional[str] = None
    phonetic_trigger: Optional[str] = None


class LanguageIdentificationHead(nn.Module):
    """
    Frame/token-level language identification for code-switching detection.
    Works for both audio frames and text tokens.
    """
    
    def __init__(
        self,
        input_dim: int,
        languages: List[str],
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.languages = languages
        self.num_languages = len(languages)
        self.lang_to_id = {lang: i for i, lang in enumerate(languages)}
        self.id_to_lang = {i: lang for lang, i in self.lang_to_id.items()}
        
        # Bidirectional LSTM for context modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Language classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_languages)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for language identification.
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            language_logits: Language predictions [batch_size, seq_len, num_languages]
            confidence_scores: Confidence scores [batch_size, seq_len, 1]
            hidden_states: LSTM hidden states [batch_size, seq_len, hidden_dim*2]
        """
        batch_size, seq_len, _ = features.shape
        
        # Apply LSTM
        lstm_output, _ = self.lstm(features)  # [batch, seq, hidden*2]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(lstm_output)
            lstm_output = lstm_output * mask_expanded
        
        # Language classification
        language_logits = self.classifier(lstm_output)  # [batch, seq, num_languages]
        
        # Confidence estimation
        confidence_scores = self.confidence_head(lstm_output)  # [batch, seq, 1]
        
        return language_logits, confidence_scores, lstm_output


class CodeSwitchingDetector(nn.Module):
    """
    Advanced code-switching detection that identifies switch points
    and classifies switch types based on linguistic patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        languages: List[str],
        bridge_specs: Dict[str, LanguageBridgeSpec],
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.languages = languages
        self.bridge_specs = bridge_specs
        self.hidden_dim = hidden_dim
        
        # Language identification
        self.lid_head = LanguageIdentificationHead(
            input_dim=input_dim,
            languages=languages,
            hidden_dim=hidden_dim
        )
        
        # Switch point detection
        self.switch_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Switch / No switch
        )
        
        # Switch type classification
        self.switch_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + len(SwitchType), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(SwitchType))
        )
        
        # Linguistic context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Build similarity matrix from bridge specs
        self.language_similarity = self._build_similarity_matrix()
    
    def _build_similarity_matrix(self) -> torch.Tensor:
        """Build language similarity matrix from bridge specifications."""
        n_langs = len(self.languages)
        similarity = torch.eye(n_langs)
        
        for i, lang1 in enumerate(self.languages):
            for j, lang2 in enumerate(self.languages):
                if i != j and lang1 in self.bridge_specs and lang2 in self.bridge_specs:
                    spec1 = self.bridge_specs[lang1]
                    spec2 = self.bridge_specs[lang2]
                    
                    # Family similarity
                    family_sim = 1.0 if spec1.family == spec2.family else 0.0
                    
                    # Phonetic similarity
                    phonetic_sim = spec1.phonetic_similarity if hasattr(spec1, 'phonetic_similarity') else 0.0
                    
                    # Combined similarity
                    combined_sim = 0.7 * family_sim + 0.3 * phonetic_sim
                    similarity[i, j] = combined_sim
        
        return nn.Parameter(similarity, requires_grad=False)
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        linguistic_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect code-switching in input features.
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            linguistic_features: Additional linguistic features
            
        Returns:
            Dictionary with detection results
        """
        batch_size, seq_len, _ = features.shape
        
        # Language identification
        lang_logits, confidence, hidden_states = self.lid_head(features, attention_mask)
        
        # Enhanced context modeling
        context_features = self.context_encoder(hidden_states)
        
        # Switch point detection (compare adjacent frames/tokens)
        if seq_len > 1:
            # Compute differences between adjacent timesteps
            diff_features = context_features[:, 1:] - context_features[:, :-1]
            
            # Pad to maintain sequence length
            padding = torch.zeros(batch_size, 1, context_features.size(-1), device=features.device)
            diff_features = torch.cat([padding, diff_features], dim=1)
            
            # Detect switch points
            switch_logits = self.switch_detector(diff_features)
        else:
            switch_logits = torch.zeros(batch_size, seq_len, 2, device=features.device)
        
        # Switch type classification (only at detected switch points)
        switch_probs = F.softmax(switch_logits, dim=-1)
        is_switch = switch_probs[:, :, 1] > 0.5  # Threshold for switch detection
        
        # For switch points, classify the type
        switch_type_logits = torch.zeros(
            batch_size, seq_len, len(SwitchType), 
            device=features.device
        )
        
        if is_switch.any():
            # Get linguistic context features
            if linguistic_features is not None:
                context_with_ling = torch.cat([context_features, linguistic_features], dim=-1)
            else:
                # Create dummy linguistic features
                ling_dummy = torch.zeros(
                    batch_size, seq_len, len(SwitchType), 
                    device=features.device
                )
                context_with_ling = torch.cat([context_features, ling_dummy], dim=-1)
            
            switch_type_logits = self.switch_type_classifier(context_with_ling)
        
        return {
            'language_logits': lang_logits,
            'language_probs': F.softmax(lang_logits, dim=-1),
            'confidence_scores': confidence,
            'switch_logits': switch_logits,
            'switch_probs': switch_probs,
            'switch_type_logits': switch_type_logits,
            'hidden_states': hidden_states,
            'context_features': context_features
        }
    
    def extract_switch_points(
        self,
        detection_output: Dict[str, torch.Tensor],
        threshold: float = 0.5,
        min_confidence: float = 0.3
    ) -> List[List[SwitchPoint]]:
        """
        Extract switch points from detection output.
        
        Args:
            detection_output: Output from forward pass
            threshold: Threshold for switch detection
            min_confidence: Minimum confidence for switch points
            
        Returns:
            List of switch points for each sample in batch
        """
        batch_switch_points = []
        
        language_probs = detection_output['language_probs']
        switch_probs = detection_output['switch_probs']
        confidence_scores = detection_output['confidence_scores']
        switch_type_logits = detection_output['switch_type_logits']
        
        batch_size, seq_len = language_probs.shape[:2]
        
        for batch_idx in range(batch_size):
            sample_switches = []
            
            current_lang = None
            
            for pos in range(seq_len):
                # Check if this is a switch point
                is_switch = switch_probs[batch_idx, pos, 1].item() > threshold
                confidence = confidence_scores[batch_idx, pos, 0].item()
                
                if confidence < min_confidence:
                    continue
                
                # Get predicted language
                lang_idx = language_probs[batch_idx, pos].argmax().item()
                predicted_lang = self.lid_head.id_to_lang[lang_idx]
                
                if is_switch and current_lang is not None and current_lang != predicted_lang:
                    # Classify switch type
                    switch_type_idx = switch_type_logits[batch_idx, pos].argmax().item()
                    switch_type = list(SwitchType)[switch_type_idx]
                    
                    switch_point = SwitchPoint(
                        position=pos,
                        from_language=current_lang,
                        to_language=predicted_lang,
                        switch_type=switch_type,
                        confidence=confidence
                    )
                    
                    sample_switches.append(switch_point)
                
                current_lang = predicted_lang
            
            batch_switch_points.append(sample_switches)
        
        return batch_switch_points


class MultimodalCodeSwitchingBridge(nn.Module):
    """
    Main bridge architecture for handling code-switching across
    both speech and text modalities with shared representations.
    """
    
    def __init__(
        self,
        audio_config: AudioConfig,
        text_config: TextConfig,
        bridge_specs: Dict[str, LanguageBridgeSpec],
        hidden_dim: int = 768
    ):
        super().__init__()
        
        self.audio_config = audio_config
        self.text_config = text_config
        self.bridge_specs = bridge_specs
        self.hidden_dim = hidden_dim
        
        # Collect all languages
        audio_languages = [audio_config.primary_language] + audio_config.secondary_languages
        all_languages = list(set(audio_languages))
        
        # Audio processing components
        self.audio_feature_extractor = nn.Sequential(
            nn.Linear(audio_config.n_mels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Text processing components  
        self.text_feature_extractor = nn.Sequential(
            nn.Linear(text_config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Code-switching detectors for each modality
        self.audio_cs_detector = CodeSwitchingDetector(
            input_dim=hidden_dim,
            languages=all_languages,
            bridge_specs=bridge_specs,
            hidden_dim=hidden_dim
        )
        
        self.text_cs_detector = CodeSwitchingDetector(
            input_dim=hidden_dim,
            languages=all_languages,
            bridge_specs=bridge_specs,
            hidden_dim=hidden_dim
        )
        
        # Cross-modal alignment
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Language-specific adaptation layers
        self.language_adapters = nn.ModuleDict()
        for lang in all_languages:
            self.language_adapters[lang] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Switch-aware fusion
        self.switch_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # audio + text + switch context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """
        Process multimodal input with code-switching awareness.
        
        Args:
            audio_features: Audio mel-spectrograms [batch, time, n_mels]
            text_features: Text embeddings [batch, seq_len, hidden_size]
            audio_attention_mask: Audio attention mask
            text_attention_mask: Text attention mask
            
        Returns:
            Dictionary with processing results and switch information
        """
        results = {}
        
        # Process audio modality
        if audio_features is not None:
            audio_repr = self.audio_feature_extractor(audio_features)
            audio_cs_output = self.audio_cs_detector(
                audio_repr, 
                audio_attention_mask
            )
            results['audio'] = audio_cs_output
            results['audio_features'] = audio_repr
        
        # Process text modality
        if text_features is not None:
            text_repr = self.text_feature_extractor(text_features)
            text_cs_output = self.text_cs_detector(
                text_repr,
                text_attention_mask
            )
            results['text'] = text_cs_output
            results['text_features'] = text_repr
        
        # Cross-modal alignment and fusion
        if audio_features is not None and text_features is not None:
            # Align audio and text representations
            audio_aligned, _ = self.cross_modal_attention(
                query=results['audio_features'],
                key=results['text_features'],
                value=results['text_features']
            )
            
            text_aligned, _ = self.cross_modal_attention(
                query=results['text_features'],
                key=results['audio_features'],
                value=results['audio_features']
            )
            
            # Create switch-aware context
            if results['audio']['switch_probs'].shape[1] == results['text']['switch_probs'].shape[1]:
                # Same sequence length, can combine directly
                switch_context = (
                    results['audio']['context_features'] + 
                    results['text']['context_features']
                ) / 2
            else:
                # Different lengths, use interpolation
                switch_context = results['audio']['context_features']  # Fallback
            
            # Fuse representations
            combined_features = torch.cat([
                audio_aligned,
                text_aligned,
                switch_context
            ], dim=-1)
            
            fused_features = self.switch_fusion(combined_features)
            results['fused_features'] = fused_features
        
        # Apply language-specific adaptations
        for modality in ['audio', 'text']:
            if modality in results:
                lang_probs = results[modality]['language_probs']
                features = results[f'{modality}_features']
                
                # Weighted combination of language-specific adaptations
                adapted_features = torch.zeros_like(features)
                
                for lang_name, adapter in self.language_adapters.items():
                    if lang_name in self.audio_cs_detector.lid_head.lang_to_id:
                        lang_idx = self.audio_cs_detector.lid_head.lang_to_id[lang_name]
                        lang_weight = lang_probs[:, :, lang_idx:lang_idx+1]
                        
                        lang_adapted = adapter(features)
                        adapted_features = adapted_features + (lang_weight * lang_adapted)
                
                results[f'{modality}_adapted'] = adapted_features
        
        return results
    
    def get_switch_analysis(
        self,
        results: Dict[str, any],
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """Get detailed code-switching analysis from results."""
        analysis = {}
        
        for modality in ['audio', 'text']:
            if modality in results:
                # Extract switch points
                switch_points = self.audio_cs_detector.extract_switch_points(
                    results[modality], 
                    threshold=threshold
                )
                
                # Language distribution
                lang_probs = results[modality]['language_probs']
                avg_lang_probs = lang_probs.mean(dim=(0, 1))  # Average over batch and sequence
                
                lang_distribution = {
                    self.audio_cs_detector.lid_head.id_to_lang[i]: prob.item()
                    for i, prob in enumerate(avg_lang_probs)
                }
                
                analysis[modality] = {
                    'switch_points': switch_points,
                    'language_distribution': lang_distribution,
                    'avg_confidence': results[modality]['confidence_scores'].mean().item(),
                    'total_switches': sum(len(sp) for sp in switch_points)
                }
        
        return analysis


# Utility functions for linguistic analysis
def detect_linguistic_triggers(text: str, target_languages: List[str]) -> Dict[str, List[str]]:
    """
    Detect linguistic triggers that commonly lead to code-switching.
    
    Args:
        text: Input text
        target_languages: Languages to analyze
        
    Returns:
        Dictionary of triggers for each language
    """
    triggers = {lang: [] for lang in target_languages}
    
    # Common triggers for African languages
    trigger_patterns = {
        'swahili': [
            r'\b(kwanza|sasa|lakini|ama|kwa sababu)\b',  # Discourse markers
            r'\b(pole|asante|karibu)\b',  # Social expressions
            r'\b(haya|sawa|bado)\b'  # Discourse particles
        ],
        'kikuyu': [
            r'\b(nowe|uria|mwega)\b',  # Discourse markers
            r'\b(ngatho|ageni)\b',  # Social expressions
        ],
        'english': [
            r'\b(you know|I mean|like|actually)\b',  # Discourse markers
            r'\b(okay|alright|so)\b'  # Transition words
        ]
    }
    
    for lang, patterns in trigger_patterns.items():
        if lang in target_languages:
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                triggers[lang].extend(matches)
    
    return triggers