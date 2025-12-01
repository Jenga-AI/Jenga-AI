"""
Universal Data Validator & Guardrails System for Jenga-AI

This module provides comprehensive data validation, sanitization, and safety checks
for all data inputs across the framework. It addresses critical security and data
quality issues identified in the framework analysis.

Features:
- Schema validation
- Input sanitization
- Content safety checks
- File path validation
- Size limits enforcement
- Data quality scoring
- Malicious content detection
- Type coercion

Usage:
    from multitask_bert.core.data_validators import UniversalDataValidator, DataType
    
    validator = UniversalDataValidator(data_type=DataType.QA_METRICS)
    result = validator.validate_file('data.json')
    
    if result.is_valid:
        clean_data = result.cleaned_data
    else:
        print(result.errors)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib


# ===== Data Types =====

class DataType(Enum):
    """Supported data types for validation."""
    QA_METRICS = "qa_metrics"
    CLASSIFICATION = "classification"
    NER = "ner"
    MULTI_LABEL = "multi_label"
    SEQ2SEQ = "seq2seq"
    LLM_FINETUNING = "llm_finetuning"
    GENERIC = "generic"


# ===== Validation Result =====

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[Any] = None
    quality_score: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


# ===== Security Validators =====

class SecurityValidator:
    """Validates inputs for security threats."""
    
    # Patterns for malicious content
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # Eval calls
        r'exec\s*\(',  # Exec calls
        r'\$\{.*?\}',  # Template injection
        r'{{.*?}}',  # Template injection (alternative)
    ]
    
    # Suspicious file patterns
    SUSPICIOUS_EXTENSIONS = ['.exe', '.dll', '.so', '.sh', '.bat', '.cmd']
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate file path for security issues.
        
        Args:
            file_path: Path to validate
            allowed_dirs: List of allowed directory paths
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Convert to Path object
            path = Path(file_path).resolve()
            
            # Check if path traversal attempt
            if '..' in file_path:
                return False, "Path traversal detected (..)"
            
            # Check if file exists
            if not path.exists():
                return False, f"File does not exist: {file_path}"
            
            # Check if it's a file (not directory)
            if not path.is_file():
                return False, f"Not a file: {file_path}"
            
            # Check if symlink (potential security issue)
            if path.is_symlink():
                return False, "Symbolic links not allowed"
            
            # Check file extension
            if path.suffix.lower() in SecurityValidator.SUSPICIOUS_EXTENSIONS:
                return False, f"Suspicious file extension: {path.suffix}"
            
            # Validate against allowed directories
            if allowed_dirs:
                path_str = str(path)
                allowed = any(path_str.startswith(str(Path(d).resolve())) for d in allowed_dirs)
                if not allowed:
                    return False, f"File not in allowed directories"
            
            return True, None
            
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    @staticmethod
    def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize text input to remove potentially malicious content.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
        
        # Trim if needed
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        # Remove dangerous patterns
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def check_malicious_content(text: str) -> List[str]:
        """
        Check for malicious content patterns.
        
        Args:
            text: Text to check
            
        Returns:
            List of detected issues
        """
        issues = []
        
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(f"Suspicious pattern detected: {pattern}")
        
        return issues


# ===== Schema Validators =====

class SchemaValidator:
    """Validates data against expected schemas."""
    
    # Schema definitions for different data types
    SCHEMAS = {
        DataType.QA_METRICS: {
            "required_fields": ["text", "labels", "sample_id"],
            "optional_fields": ["scenario", "quality_level"],
            "label_structure": {
                "opening": 1,
                "listening": 5,
                "proactiveness": 3,
                "resolution": 5,
                "hold": 2,
                "closing": 1
            },
            "text_max_length": 10000,
            "text_min_length": 10
        },
        DataType.CLASSIFICATION: {
            "required_fields": ["text", "label"],
            "optional_fields": ["sample_id", "metadata"],
            "text_max_length": 5000,
            "text_min_length": 5
        },
        DataType.NER: {
            "required_fields": ["text", "entities"],
            "optional_fields": ["sample_id"],
            "entity_fields": ["start", "end", "label"],
            "text_max_length": 5000
        },
        DataType.MULTI_LABEL: {
            "required_fields": ["text", "labels"],
            "optional_fields": ["sample_id"],
            "text_max_length": 5000
        }
    }
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], data_type: DataType) -> List[str]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            data_type: Type of data
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if data_type not in SchemaValidator.SCHEMAS:
            return [f"No schema defined for {data_type}"]
        
        schema = SchemaValidator.SCHEMAS[data_type]
        
        # Check required fields
        for field in schema["required_fields"]:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate text field if present
        if "text" in data:
            text = data["text"]
            if not isinstance(text, str):
                errors.append(f"'text' must be string, got {type(text)}")
            elif len(text) < schema.get("text_min_length", 0):
                errors.append(f"'text' too short (min: {schema.get('text_min_length')} chars)")
            elif len(text) > schema.get("text_max_length", float('inf')):
                errors.append(f"'text' too long (max: {schema.get('text_max_length')} chars)")
        
        # Data type specific validation
        if data_type == DataType.QA_METRICS and "labels" in data:
            label_errors = SchemaValidator._validate_qa_labels(data["labels"], schema["label_structure"])
            errors.extend(label_errors)
        
        elif data_type == DataType.NER and "entities" in data:
            entity_errors = SchemaValidator._validate_ner_entities(data["entities"], schema.get("entity_fields", []))
            errors.extend(entity_errors)
        
        return errors
    
    @staticmethod
    def _validate_qa_labels(labels: Any, expected_structure: Dict[str, int]) -> List[str]:
        """Validate QA labels structure."""
        errors = []
        
        if not isinstance(labels, dict):
            return [f"Labels must be dict, got {type(labels)}"]
        
        # Check all required heads present
        for head, expected_count in expected_structure.items():
            if head not in labels:
                errors.append(f"Missing label head: {head}")
            else:
                label_array = labels[head]
                if not isinstance(label_array, list):
                    errors.append(f"{head} labels must be list, got {type(label_array)}")
                elif len(label_array) != expected_count:
                    errors.append(f"{head} has {len(label_array)} labels, expected {expected_count}")
                elif not all(isinstance(v, int) and v in [0, 1] for v in label_array):
                    errors.append(f"{head} labels must be binary integers (0 or 1)")
        
        return errors
    
    @staticmethod
    def _validate_ner_entities(entities: Any, required_fields: List[str]) -> List[str]:
        """Validate NER entities structure."""
        errors = []
        
        if not isinstance(entities, list):
            return [f"Entities must be list, got {type(entities)}"]
        
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"Entity {i} must be dict, got {type(entity)}")
                continue
            
            for field in required_fields:
                if field not in entity:
                    errors.append(f"Entity {i} missing field: {field}")
        
        return errors


# ===== Universal Data Validator =====

class UniversalDataValidator:
    """Main validator class for all data types."""
    
    def __init__(
        self,
        data_type: DataType,
        max_file_size_mb: float = 100.0,
        allowed_dirs: Optional[List[str]] = None,
        enable_sanitization: bool = True,
        enable_security_checks: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            data_type: Type of data to validate
            max_file_size_mb: Maximum file size in MB
            allowed_dirs: Allowed directories for file paths
            enable_sanitization: Whether to sanitize text
            enable_security_checks: Whether to run security checks
        """
        self.data_type = data_type
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.allowed_dirs = allowed_dirs
        self.enable_sanitization = enable_sanitization
        self.enable_security_checks = enable_security_checks
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate and clean data from a file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            ValidationResult with cleaned data or errors
        """
        result = ValidationResult(is_valid=True)
        
        # Step 1: Validate file path
        if self.enable_security_checks:
            is_valid, error = SecurityValidator.validate_file_path(file_path, self.allowed_dirs)
            if not is_valid:
                result.add_error(f"File path validation failed: {error}")
                return result
        
        # Step 2: Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                result.add_error(f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {self.max_file_size / 1024 / 1024}MB)")
                return result
            result.stats['file_size_mb'] = file_size / 1024 / 1024
        except Exception as e:
            result.add_error(f"Failed to check file size: {str(e)}")
            return result
        
        # Step 3: Load data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                elif file_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    result.add_error(f"Unsupported file format: {Path(file_path).suffix}")
                    return result
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON: {str(e)}")
            return result
        except Exception as e:
            result.add_error(f"Failed to load file: {str(e)}")
            return result
        
        # Step 4: Validate and clean data
        if isinstance(data, list):
            result = self._validate_dataset(data, result)
        else:
            result.add_error("Data must be a list of samples")
        
        return result
    
    def _validate_dataset(self, dataset: List[Dict], result: ValidationResult) -> ValidationResult:
        """Validate entire dataset."""
        cleaned_data = []
        result.stats['total_samples'] = len(dataset)
        result.stats['valid_samples'] = 0
        result.stats['invalid_samples'] = 0
        
        for idx, sample in enumerate(dataset):
            sample_result = self._validate_sample(sample, idx)
            
            if sample_result.is_valid:
                cleaned_data.append(sample_result.cleaned_data)
                result.stats['valid_samples'] += 1
            else:
                result.stats['invalid_samples'] += 1
                # Add sample-specific errors to main result
                for error in sample_result.errors:
                    result.add_error(f"Sample {idx}: {error}")
                # Don't fail entire dataset for one bad sample
                if len(result.errors) < 10:  # Limit error reporting
                    result.add_warning(f"Sample {idx} invalid (see errors)")
        
        result.cleaned_data = cleaned_data
        result.quality_score = self._calculate_quality_score(result.stats)
        
        # Dataset is valid if at least some samples are valid
        result.is_valid = result.stats['valid_samples'] > 0
        
        return result
    
    def _validate_sample(self, sample: Dict[str, Any], index: int) -> ValidationResult:
        """Validate a single sample."""
        result = ValidationResult(is_valid=True)
        
        # Schema validation
        schema_errors = SchemaValidator.validate_schema(sample, self.data_type)
        for error in schema_errors:
            result.add_error(error)
        
        if not result.is_valid:
            return result
        
        # Create cleaned copy
        cleaned_sample = sample.copy()
        
        # Sanitize text if enabled
        if self.enable_sanitization and "text" in cleaned_sample:
            original_text = cleaned_sample["text"]
            cleaned_sample["text"] = SecurityValidator.sanitize_text(original_text)
            
            if len(cleaned_sample["text"]) < len(original_text) * 0.5:
                result.add_warning(f"Text heavily sanitized (removed {len(original_text) - len(cleaned_sample['text'])} chars)")
        
        # Security checks
        if self.enable_security_checks and "text" in cleaned_sample:
            security_issues = SecurityValidator.check_malicious_content(cleaned_sample["text"])
            for issue in security_issues:
                result.add_warning(issue)
        
        result.cleaned_data = cleaned_sample
        return result
    
    def _calculate_quality_score(self, stats: Dict[str, Any]) -> float:
        """Calculate data quality score (0-1)."""
        if stats['total_samples'] == 0:
            return 0.0
        
        valid_ratio = stats['valid_samples'] / stats['total_samples']
        
        # Quality score based on valid sample ratio
        return valid_ratio


# ===== Convenience Functions =====

def validate_qa_data(file_path: str, **kwargs) -> ValidationResult:
    """Validate QA metrics data."""
    validator = UniversalDataValidator(DataType.QA_METRICS, **kwargs)
    return validator.validate_file(file_path)


def validate_classification_data(file_path: str, **kwargs) -> ValidationResult:
    """Validate classification data."""
    validator = UniversalDataValidator(DataType.CLASSIFICATION, **kwargs)
    return validator.validate_file(file_path)


def validate_ner_data(file_path: str, **kwargs) -> ValidationResult:
    """Validate NER data."""
    validator = UniversalDataValidator(DataType.NER, **kwargs)
    return validator.validate_file(file_path)
