"""
JengaHub Data Processing Module

This module provides unified data processing capabilities for both audio and text,
supporting multi-modal training and African language processing.
"""

from .processor import (
    UnifiedDataProcessor,
    AudioProcessor,
    TextProcessor
)

from .dataset import (
    AudioTextDataset,
    MultiModalDataLoader,
    CodeSwitchingDataset
)

from .bridge import (
    LanguageBridgeSpec,
    LBSParser,
    LanguageFamilyMapper
)

from .utils import (
    AudioUtils,
    TextUtils,
    LanguageDetector
)

__all__ = [
    "UnifiedDataProcessor",
    "AudioProcessor", 
    "TextProcessor",
    "AudioTextDataset",
    "MultiModalDataLoader",
    "CodeSwitchingDataset",
    "LanguageBridgeSpec",
    "LBSParser",
    "LanguageFamilyMapper",
    "AudioUtils",
    "TextUtils",
    "LanguageDetector"
]