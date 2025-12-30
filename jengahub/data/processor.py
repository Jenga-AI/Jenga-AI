"""
Unified Data Processing for JengaHub

This module provides comprehensive data processing capabilities that handle
both audio and text data, supporting multi-modal training and African language
processing with code-switching support.
"""

import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json
from dataclasses import dataclass
from transformers import AutoTokenizer, WhisperFeatureExtractor
from datasets import Dataset, load_dataset
import re

from ..core.config import MultiModalConfig, AudioConfig, TextConfig


@dataclass
class ProcessedSample:
    """Container for a processed multimodal sample."""
    
    # Audio features
    audio_features: Optional[torch.Tensor] = None
    audio_length: Optional[int] = None
    
    # Text features  
    text_input_ids: Optional[torch.Tensor] = None
    text_attention_mask: Optional[torch.Tensor] = None
    text_length: Optional[int] = None
    
    # Labels and metadata
    labels: Optional[torch.Tensor] = None
    task_id: Optional[int] = None
    language: Optional[str] = None
    language_labels: Optional[torch.Tensor] = None  # For frame-level LID
    
    # Code-switching information
    switch_points: Optional[List[int]] = None
    segment_languages: Optional[List[str]] = None
    
    # Original data
    audio_path: Optional[str] = None
    text: Optional[str] = None
    transcript: Optional[str] = None


class AudioProcessor:
    """Handles audio preprocessing for speech recognition and analysis."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            config.base_model
        ) if config.base_model else None
        
        # Audio parameters
        self.sampling_rate = config.sampling_rate
        self.n_mels = config.n_mels
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        
        # Language mapping for frame-level LID
        all_languages = [config.primary_language] + config.secondary_languages
        self.language_to_id = {lang: i for i, lang in enumerate(all_languages)}
        self.id_to_language = {i: lang for lang, i in self.language_to_id.items()}
        
    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load and resample audio file."""
        try:
            # Try with librosa first (more robust)
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        except Exception:
            # Fallback to torchaudio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                audio = resampler(audio)
            audio = audio.squeeze().numpy()
        
        return audio
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract mel-spectrogram features from audio."""
        if self.feature_extractor:
            # Use Whisper's feature extractor
            features = self.feature_extractor(
                audio, 
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )
            return features.input_features.squeeze(0)  # Remove batch dimension
        else:
            # Manual mel-spectrogram extraction
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sampling_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            
            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec)
            
            return torch.from_numpy(log_mel).transpose(0, 1).float()
    
    def detect_language_segments(
        self, 
        transcript: str,
        audio_length: int
    ) -> Tuple[List[str], List[int], torch.Tensor]:
        """
        Detect language segments in transcript and create frame-level labels.
        
        Args:
            transcript: Transcribed text with language tags
            audio_length: Length of audio in frames
            
        Returns:
            segment_languages: List of languages per segment
            switch_points: Frame indices where language switches occur
            frame_labels: Frame-level language labels [audio_length]
        """
        # Parse language-tagged transcript like "<sw>Habari</sw> <en>hello</en>"
        pattern = r'<(\w+)>(.*?)</\w+>'
        matches = re.findall(pattern, transcript)
        
        if not matches:
            # No language tags, assume primary language
            segment_languages = [self.config.primary_language]
            switch_points = [0]
            frame_labels = torch.full(
                (audio_length,), 
                self.language_to_id[self.config.primary_language]
            )
        else:
            segment_languages = []
            switch_points = [0]
            
            # Estimate timing based on character positions
            total_chars = sum(len(text) for _, text in matches)
            current_frame = 0
            
            for lang, text in matches:
                segment_languages.append(lang)
                
                # Estimate frames for this segment
                char_ratio = len(text) / total_chars
                segment_frames = int(char_ratio * audio_length)
                
                if current_frame + segment_frames < audio_length:
                    switch_points.append(current_frame + segment_frames)
                current_frame += segment_frames
            
            # Create frame-level labels
            frame_labels = torch.zeros(audio_length, dtype=torch.long)
            for i, (start, end) in enumerate(zip(switch_points[:-1], switch_points[1:])):
                lang = segment_languages[i]
                lang_id = self.language_to_id.get(lang, 0)
                frame_labels[start:end] = lang_id
            
            # Fill remaining frames with last language
            if switch_points:
                last_lang = segment_languages[-1]
                last_lang_id = self.language_to_id.get(last_lang, 0)
                frame_labels[switch_points[-1]:] = last_lang_id
        
        return segment_languages, switch_points[1:], frame_labels  # Remove initial 0
    
    def process_sample(
        self, 
        audio_path: str,
        transcript: Optional[str] = None
    ) -> ProcessedSample:
        """Process a single audio sample."""
        # Load and extract features
        audio = self.load_audio(audio_path)
        features = self.extract_features(audio)
        
        # Initialize sample
        sample = ProcessedSample(
            audio_features=features,
            audio_length=features.size(0),
            audio_path=audio_path,
            transcript=transcript
        )
        
        # Process language information if transcript available
        if transcript and self.config.enable_frame_lid:
            segment_langs, switch_points, frame_labels = self.detect_language_segments(
                transcript, features.size(0)
            )
            
            sample.segment_languages = segment_langs
            sample.switch_points = switch_points
            sample.language_labels = frame_labels
            
            # Primary language for sample
            sample.language = segment_langs[0] if segment_langs else self.config.primary_language
        else:
            sample.language = self.config.primary_language
        
        return sample


class TextProcessor:
    """Handles text preprocessing for NLP tasks."""
    
    def __init__(self, config: TextConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Task mapping
        self.task_to_id = {task: i for i, task in enumerate(config.tasks)}
        self.id_to_task = {i: task for task, i in self.task_to_id.items()}
        
    def detect_code_switching(self, text: str) -> Dict[str, Any]:
        """
        Detect code-switching patterns in text.
        
        Returns:
            Dictionary with switching information
        """
        # Simple pattern-based detection (can be enhanced with ML models)
        
        # Common Swahili words
        swahili_words = {
            'na', 'ya', 'wa', 'kwa', 'ni', 'la', 'za', 'cha', 'ndi', 'tu',
            'habari', 'karibu', 'asante', 'pole', 'sawa', 'bado', 'leo'
        }
        
        # Common Kikuyu words  
        kikuyu_words = {
            'ni', 'wa', 'ta', 'ma', 'ka', 'na', 'ku', 'no', 'riu', 'guo',
            'muthenya', 'uria', 'nowe', 'ageni', 'mwega', 'ngatho'
        }
        
        words = text.lower().split()
        language_predictions = []
        
        for word in words:
            if word in swahili_words:
                language_predictions.append('swahili')
            elif word in kikuyu_words:
                language_predictions.append('kikuyu')
            elif re.match(r'^[a-zA-Z]+$', word):
                language_predictions.append('english')
            else:
                language_predictions.append('unknown')
        
        # Detect switches
        switches = []
        for i in range(1, len(language_predictions)):
            if language_predictions[i] != language_predictions[i-1]:
                switches.append(i)
        
        return {
            'word_languages': language_predictions,
            'switch_points': switches,
            'has_code_switching': len(switches) > 0,
            'dominant_language': max(set(language_predictions), key=language_predictions.count)
        }
    
    def process_for_task(
        self, 
        text: str, 
        task: str,
        label: Optional[Union[str, int, List]] = None
    ) -> Dict[str, torch.Tensor]:
        """Process text for a specific task."""
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
        
        # Process labels based on task type
        if label is not None:
            if task in ['classification', 'sentiment_analysis']:
                if isinstance(label, str):
                    # Convert string label to integer
                    label_map = {'negative': 0, 'positive': 1, 'neutral': 2}
                    result['labels'] = torch.tensor(label_map.get(label.lower(), 0))
                else:
                    result['labels'] = torch.tensor(label)
                    
            elif task == 'ner':
                # For NER, labels should align with tokens
                if isinstance(label, list):
                    # Align labels with tokenized input
                    word_ids = encoded.word_ids()
                    label_ids = []
                    
                    for word_id in word_ids:
                        if word_id is None:
                            label_ids.append(-100)  # Ignore special tokens
                        elif word_id < len(label):
                            label_ids.append(label[word_id])
                        else:
                            label_ids.append(0)  # Default to O tag
                    
                    result['labels'] = torch.tensor(label_ids)
                    
            elif task == 'qa':
                # For QA, label should be start and end positions
                if isinstance(label, dict):
                    result['start_positions'] = torch.tensor(label.get('start', 0))
                    result['end_positions'] = torch.tensor(label.get('end', 0))
        
        return result
    
    def process_sample(
        self,
        text: str,
        task: str,
        label: Optional[Any] = None
    ) -> ProcessedSample:
        """Process a single text sample."""
        # Detect code-switching
        cs_info = self.detect_code_switching(text)
        
        # Process for specific task
        processed = self.process_for_task(text, task, label)
        
        # Create sample
        sample = ProcessedSample(
            text_input_ids=processed['input_ids'],
            text_attention_mask=processed['attention_mask'],
            text_length=processed['attention_mask'].sum().item(),
            labels=processed.get('labels'),
            task_id=self.task_to_id.get(task, 0),
            language=cs_info['dominant_language'],
            text=text
        )
        
        # Add code-switching information
        if cs_info['has_code_switching']:
            sample.switch_points = cs_info['switch_points']
            sample.segment_languages = cs_info['word_languages']
        
        return sample


class UnifiedDataProcessor:
    """
    Main data processor that handles both audio and text data
    for multimodal training.
    """
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.audio_processor = AudioProcessor(config.audio)
        self.text_processor = TextProcessor(config.text)
        
    def process_multimodal_sample(
        self,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        transcript: Optional[str] = None,
        task: str = "classification",
        label: Optional[Any] = None
    ) -> ProcessedSample:
        """
        Process a multimodal sample with both audio and text.
        
        Args:
            audio_path: Path to audio file
            text: Text content for NLP tasks
            transcript: Transcript of audio (for alignment)
            task: Task type
            label: Labels for the task
            
        Returns:
            ProcessedSample with both audio and text features
        """
        # Process audio if provided
        audio_sample = None
        if audio_path:
            audio_sample = self.audio_processor.process_sample(
                audio_path, transcript
            )
        
        # Process text if provided
        text_sample = None
        if text:
            text_sample = self.text_processor.process_sample(text, task, label)
        
        # Combine samples
        if audio_sample and text_sample:
            # Multimodal sample
            combined_sample = ProcessedSample(
                # Audio features
                audio_features=audio_sample.audio_features,
                audio_length=audio_sample.audio_length,
                
                # Text features
                text_input_ids=text_sample.text_input_ids,
                text_attention_mask=text_sample.text_attention_mask,
                text_length=text_sample.text_length,
                
                # Combined metadata
                labels=text_sample.labels or audio_sample.labels,
                task_id=text_sample.task_id,
                language=text_sample.language or audio_sample.language,
                
                # Original data
                audio_path=audio_path,
                text=text,
                transcript=transcript
            )
            
            # Merge code-switching information
            if (hasattr(audio_sample, 'switch_points') and 
                hasattr(text_sample, 'switch_points')):
                combined_sample.switch_points = (
                    audio_sample.switch_points or text_sample.switch_points
                )
                combined_sample.segment_languages = (
                    audio_sample.segment_languages or text_sample.segment_languages
                )
            
            return combined_sample
            
        elif audio_sample:
            return audio_sample
        elif text_sample:
            return text_sample
        else:
            raise ValueError("At least one of audio_path or text must be provided")
    
    def load_dataset(
        self,
        data_path: Union[str, Dict[str, str]],
        task: str,
        split: str = "train"
    ) -> List[ProcessedSample]:
        """
        Load and process dataset from various formats.
        
        Args:
            data_path: Path to data file or dict of paths
            task: Task type
            split: Dataset split
            
        Returns:
            List of processed samples
        """
        samples = []
        
        if isinstance(data_path, dict):
            # Multiple data sources
            for source, path in data_path.items():
                source_samples = self._load_single_source(path, task, split)
                samples.extend(source_samples)
        else:
            # Single data source
            samples = self._load_single_source(data_path, task, split)
        
        return samples
    
    def _load_single_source(
        self,
        data_path: str,
        task: str,
        split: str
    ) -> List[ProcessedSample]:
        """Load data from a single source file."""
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            return self._load_json_dataset(data_path, task, split)
        elif data_path.suffix == '.jsonl':
            return self._load_jsonl_dataset(data_path, task, split)
        elif data_path.suffix == '.csv':
            return self._load_csv_dataset(data_path, task, split)
        else:
            # Try HuggingFace datasets
            return self._load_hf_dataset(str(data_path), task, split)
    
    def _load_json_dataset(self, path: Path, task: str, split: str) -> List[ProcessedSample]:
        """Load JSON dataset."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and split in data:
            data = data[split]
        
        samples = []
        for item in data:
            sample = self.process_multimodal_sample(
                audio_path=item.get('audio_path'),
                text=item.get('text'),
                transcript=item.get('transcript'),
                task=task,
                label=item.get('label')
            )
            samples.append(sample)
        
        return samples
    
    def _load_jsonl_dataset(self, path: Path, task: str, split: str) -> List[ProcessedSample]:
        """Load JSONL dataset."""
        samples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                sample = self.process_multimodal_sample(
                    audio_path=item.get('audio_path'),
                    text=item.get('text'),
                    transcript=item.get('transcript'),
                    task=task,
                    label=item.get('label')
                )
                samples.append(sample)
        
        return samples
    
    def _load_csv_dataset(self, path: Path, task: str, split: str) -> List[ProcessedSample]:
        """Load CSV dataset."""
        df = pd.read_csv(path)
        samples = []
        
        for _, row in df.iterrows():
            sample = self.process_multimodal_sample(
                audio_path=row.get('audio_path'),
                text=row.get('text'),
                transcript=row.get('transcript'),
                task=task,
                label=row.get('label')
            )
            samples.append(sample)
        
        return samples
    
    def _load_hf_dataset(self, dataset_name: str, task: str, split: str) -> List[ProcessedSample]:
        """Load HuggingFace dataset."""
        try:
            dataset = load_dataset(dataset_name, split=split)
            samples = []
            
            for item in dataset:
                sample = self.process_multimodal_sample(
                    audio_path=item.get('audio_path') or item.get('file'),
                    text=item.get('text') or item.get('sentence'),
                    transcript=item.get('transcript'),
                    task=task,
                    label=item.get('label')
                )
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")
    
    def create_balanced_dataset(
        self,
        samples: List[ProcessedSample],
        balance_by: str = "language"
    ) -> List[ProcessedSample]:
        """Create a balanced dataset by specified attribute."""
        if balance_by == "language":
            # Group by language
            language_groups = {}
            for sample in samples:
                lang = sample.language or "unknown"
                if lang not in language_groups:
                    language_groups[lang] = []
                language_groups[lang].append(sample)
            
            # Find minimum group size
            min_size = min(len(group) for group in language_groups.values())
            
            # Sample equally from each group
            balanced_samples = []
            for group in language_groups.values():
                balanced_samples.extend(group[:min_size])
            
            return balanced_samples
        
        elif balance_by == "task":
            # Group by task
            task_groups = {}
            for sample in samples:
                task_id = sample.task_id or 0
                if task_id not in task_groups:
                    task_groups[task_id] = []
                task_groups[task_id].append(sample)
            
            # Find minimum group size
            min_size = min(len(group) for group in task_groups.values())
            
            # Sample equally from each group
            balanced_samples = []
            for group in task_groups.values():
                balanced_samples.extend(group[:min_size])
            
            return balanced_samples
        
        else:
            return samples
    
    def create_batch_collator(self) -> callable:
        """
        Create a batch collation function for DataLoader.
        
        Returns:
            Collation function for batching samples
        """
        def collate_fn(samples: List[ProcessedSample]) -> Dict[str, torch.Tensor]:
            """Collate function for batching multimodal samples."""
            batch = {}
            
            # Separate audio and text samples
            audio_samples = [s for s in samples if s.audio_features is not None]
            text_samples = [s for s in samples if s.text_input_ids is not None]
            
            # Audio features batching
            if audio_samples:
                # Pad audio features to same length
                max_audio_len = max(s.audio_length for s in audio_samples)
                audio_features = []
                audio_masks = []
                
                for sample in audio_samples:
                    features = sample.audio_features
                    pad_length = max_audio_len - features.size(0)
                    
                    if pad_length > 0:
                        padding = torch.zeros(pad_length, features.size(1))
                        features = torch.cat([features, padding], dim=0)
                    
                    audio_features.append(features)
                    
                    # Create attention mask
                    mask = torch.ones(max_audio_len)
                    if pad_length > 0:
                        mask[-pad_length:] = 0
                    audio_masks.append(mask)
                
                batch['audio_features'] = torch.stack(audio_features)
                batch['audio_attention_mask'] = torch.stack(audio_masks)
                
                # Language labels for frame-level LID
                if audio_samples[0].language_labels is not None:
                    language_labels = []
                    for sample in audio_samples:
                        labels = sample.language_labels
                        pad_length = max_audio_len - len(labels)
                        
                        if pad_length > 0:
                            padding = torch.full((pad_length,), -100)  # Ignore padding in loss
                            labels = torch.cat([labels, padding])
                        
                        language_labels.append(labels)
                    
                    batch['language_labels'] = torch.stack(language_labels)
            
            # Text features batching
            if text_samples:
                batch['text_input_ids'] = torch.stack([s.text_input_ids for s in text_samples])
                batch['text_attention_mask'] = torch.stack([s.text_attention_mask for s in text_samples])
            
            # Labels and metadata
            if samples[0].labels is not None:
                if isinstance(samples[0].labels, torch.Tensor):
                    if samples[0].labels.dim() == 0:  # Scalar labels
                        batch['labels'] = torch.stack([s.labels for s in samples])
                    else:  # Sequence labels (e.g., NER)
                        batch['labels'] = torch.stack([s.labels for s in samples])
                else:
                    batch['labels'] = torch.tensor([s.labels for s in samples])
            
            # Task IDs
            if samples[0].task_id is not None:
                batch['task_ids'] = torch.tensor([s.task_id for s in samples])
            
            # Languages (encoded as IDs)
            language_ids = []
            unique_languages = list(set(s.language for s in samples if s.language))
            lang_to_id = {lang: i for i, lang in enumerate(unique_languages)}
            
            for sample in samples:
                lang_id = lang_to_id.get(sample.language, 0)
                language_ids.append(lang_id)
            
            if language_ids:
                batch['language_ids'] = torch.tensor(language_ids)
            
            # Sample indices for tracking
            batch['sample_indices'] = torch.arange(len(samples))
            
            return batch
            
        return collate_fn
    
    def get_data_statistics(self, samples: List[ProcessedSample]) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        stats = {
            'total_samples': len(samples),
            'audio_samples': sum(1 for s in samples if s.audio_features is not None),
            'text_samples': sum(1 for s in samples if s.text_input_ids is not None),
            'multimodal_samples': sum(1 for s in samples 
                                    if s.audio_features is not None and s.text_input_ids is not None)
        }
        
        # Language distribution
        languages = [s.language for s in samples if s.language]
        if languages:
            from collections import Counter
            lang_counts = Counter(languages)
            stats['language_distribution'] = dict(lang_counts)
        
        # Task distribution
        task_ids = [s.task_id for s in samples if s.task_id is not None]
        if task_ids:
            from collections import Counter
            task_counts = Counter(task_ids)
            stats['task_distribution'] = dict(task_counts)
        
        # Code-switching statistics
        cs_samples = sum(1 for s in samples if s.switch_points is not None)
        stats['code_switching_samples'] = cs_samples
        stats['code_switching_ratio'] = cs_samples / len(samples) if samples else 0
        
        # Audio statistics
        audio_lengths = [s.audio_length for s in samples if s.audio_length is not None]
        if audio_lengths:
            stats['audio_length_stats'] = {
                'mean': sum(audio_lengths) / len(audio_lengths),
                'min': min(audio_lengths),
                'max': max(audio_lengths),
                'total_hours': sum(audio_lengths) * 0.02 / 3600  # Assuming 20ms frames
            }
        
        # Text statistics
        text_lengths = [s.text_length for s in samples if s.text_length is not None]
        if text_lengths:
            stats['text_length_stats'] = {
                'mean': sum(text_lengths) / len(text_lengths),
                'min': min(text_lengths),
                'max': max(text_lengths)
            }
        
        return stats