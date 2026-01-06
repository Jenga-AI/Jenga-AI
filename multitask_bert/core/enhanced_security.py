"""
Enhanced Security System for Jenga-AI

This module provides enhanced security features to prevent misuse of the framework
for malicious purposes. It extends the existing data validators with additional
security layers.

Features:
- Advanced malicious content detection
- Training objective validation
- Harmful content filtering
- Enhanced path traversal protection
- Improved file size enforcement
- Content safety scoring
- Audit logging
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time

# Import base validators
from .data_validators import SecurityValidator, ValidationResult, DataType


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAnalysis:
    """Result of security analysis."""
    threat_level: ThreatLevel
    detected_threats: List[str] = field(default_factory=list)
    content_safety_score: float = 0.0
    recommendation: str = ""
    blocked: bool = False


class HarmfulContentDetector:
    """Advanced detector for harmful content and training objectives."""
    
    # Harmful content categories and patterns
    HARMFUL_PATTERNS = {
        "violence": [
            r"\b(?:kill|murder|assassinate|torture|bomb|explode|attack|weapon|gun|knife)\b",
            r"\b(?:terrorism|terrorist|radical|extremist|jihad)\b",
            r"\b(?:suicide.*bomb|mass.*shooting|school.*shooting)\b"
        ],
        "hate_speech": [
            r"\b(?:hate|racist|nazi|supremacist|bigot|discrimination)\b",
            r"\b(?:inferior.*race|ethnic.*cleansing|genocide)\b",
            r"amplify.*bias|promote.*stereotypes"
        ],
        "illegal_activities": [
            r"\b(?:drug.*trafficking|money.*laundering|fraud|scam|hack)\b",
            r"\b(?:identity.*theft|credit.*card.*fraud|phishing)\b",
            r"bypass.*security|exploit.*vulnerability"
        ],
        "sexual_content": [
            r"\b(?:child.*exploitation|underage|minor.*sexual)\b",
            r"\b(?:pornography|explicit.*content|sexual.*abuse)\b"
        ],
        "privacy_violation": [
            r"\b(?:extract.*personal|private.*information|surveillance)\b",
            r"\b(?:stalking|doxxing|identity.*exposure)\b",
            r"collect.*private.*data|scrape.*personal"
        ],
        "misinformation": [
            r"\b(?:fake.*news|propaganda|disinformation|conspiracy)\b",
            r"\b(?:deepfake|manipulated.*media|false.*narrative)\b",
            r"generate.*false.*information|spread.*rumors"
        ],
        "malware": [
            r"\b(?:virus|malware|trojan|ransomware|spyware|keylogger)\b",
            r"\b(?:backdoor|rootkit|botnet|ddos)\b",
            r"inject.*code|execute.*payload"
        ]
    }
    
    # Training objective patterns that should be blocked
    BLOCKED_OBJECTIVES = [
        "create explosives", "make bomb", "build weapon",
        "generate hate speech", "amplify bias", "promote discrimination", 
        "create deepfakes", "fake videos", "manipulate media",
        "extract personal information", "violate privacy", "steal data",
        "generate malware", "create virus", "hack system",
        "bypass security", "exploit vulnerability", "break encryption",
        "child exploitation", "underage content", "sexual abuse",
        "terrorist propaganda", "radical content", "extremist material",
        "drug trafficking", "illegal activities", "criminal behavior",
        "spread misinformation", "create propaganda", "false information",
        "phishing emails", "scam messages", "fraud content"
    ]
    
    @staticmethod
    def analyze_content(text: str) -> SecurityAnalysis:
        """
        Analyze content for harmful patterns and security threats.
        
        Args:
            text: Text content to analyze
            
        Returns:
            SecurityAnalysis with threat assessment
        """
        analysis = SecurityAnalysis(threat_level=ThreatLevel.LOW)
        text_lower = text.lower()
        
        # Check for harmful content patterns
        total_matches = 0
        category_matches = {}
        
        for category, patterns in HarmfulContentDetector.HARMFUL_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches += 1
                    total_matches += 1
                    analysis.detected_threats.append(f"Harmful content ({category}): {pattern}")
            
            if matches > 0:
                category_matches[category] = matches
        
        # Check for blocked training objectives
        for objective in HarmfulContentDetector.BLOCKED_OBJECTIVES:
            if objective.lower() in text_lower:
                analysis.detected_threats.append(f"Blocked training objective: {objective}")
                total_matches += 2  # Weight objectives higher
        
        # Calculate threat level based on matches
        if total_matches == 0:
            analysis.threat_level = ThreatLevel.LOW
            analysis.content_safety_score = 1.0
        elif total_matches <= 2:
            analysis.threat_level = ThreatLevel.MEDIUM
            analysis.content_safety_score = 0.6
        elif total_matches <= 5:
            analysis.threat_level = ThreatLevel.HIGH
            analysis.content_safety_score = 0.3
        else:
            analysis.threat_level = ThreatLevel.CRITICAL
            analysis.content_safety_score = 0.0
        
        # Determine if content should be blocked
        analysis.blocked = (
            analysis.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] or
            len([t for t in analysis.detected_threats if "Blocked training objective" in t]) > 0
        )
        
        # Generate recommendation
        if analysis.blocked:
            analysis.recommendation = "BLOCK: Content contains harmful patterns or blocked objectives"
        elif analysis.threat_level == ThreatLevel.MEDIUM:
            analysis.recommendation = "REVIEW: Content may contain sensitive material"
        else:
            analysis.recommendation = "ALLOW: Content appears safe"
        
        return analysis


class EnhancedSecurityValidator(SecurityValidator):
    """Enhanced security validator with additional protections."""
    
    @staticmethod
    def validate_file_path_enhanced(file_path: str, allowed_dirs: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """
        Enhanced file path validation with improved security checks.
        
        Args:
            file_path: Path to validate
            allowed_dirs: List of allowed directory paths
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic validation first
            is_valid, error = SecurityValidator.validate_file_path(file_path, allowed_dirs)
            if not is_valid:
                return is_valid, error
            
            # Additional enhanced checks
            path = Path(file_path).resolve()
            
            # Check for hidden files (potential security risk)
            if any(part.startswith('.') and part != '.' and part != '..' for part in path.parts):
                return False, "Hidden files not allowed"
            
            # Check for suspicious patterns in filename
            filename = path.name.lower()
            suspicious_patterns = [
                'passwd', 'shadow', 'hosts', 'config', 'secret', 'key',
                'token', 'credential', 'private', 'admin', 'root'
            ]
            
            for pattern in suspicious_patterns:
                if pattern in filename:
                    return False, f"Suspicious filename pattern: {pattern}"
            
            # Enhanced directory traversal checks
            path_str = str(path)
            original_path = file_path.lower()
            
            # Check for various directory traversal patterns
            dangerous_patterns = [
                '../', '..\\', '%2e%2e%2f', '%2e%2e%5c', '%2e%2e/', '%2e%2e\\',
                '..%2f', '..%5c', '..%c0%af', '..%c1%9c', '..../', '....\\',
                '..;/', '..;\\', '..//', '..\\\\', '~/', '$HOME/', '${HOME}/',
                '..%252f', '..%255c', '0x2e0x2e/', '0x2e0x2e\\', '%252e%252e/',
                '%c0%ae%c0%ae/', '%%32%65%%32%65/', '%uff0e%uff0e%uff0f'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in original_path:
                    return False, f"Directory traversal detected: {pattern}"
            
            # Check for multiple consecutive dots
            if '..' in file_path:
                return False, "Path traversal detected (..)"
            
            # Check for absolute paths to critical system directories
            system_dirs = [
                '/etc', '/proc', '/sys', '/dev', '/boot', '/root', 
                'C:\\Windows', 'C:\\System32', 'C:\\Program Files',
                '/var/log', '/usr/bin', '/bin', '/sbin'
            ]
            
            for sys_dir in system_dirs:
                if path_str.startswith(sys_dir) or original_path.startswith(sys_dir.lower()):
                    return False, f"Access to system directory denied: {sys_dir}"
            
            # Check for tilde expansion attempts
            if file_path.startswith('~') or '${HOME}' in file_path or '$HOME' in file_path:
                return False, "Home directory access not allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"Enhanced path validation error: {str(e)}"
    
    @staticmethod
    def check_file_size_strict(file_path: str, max_size_mb: float) -> Tuple[bool, Optional[str]]:
        """
        Strict file size validation.
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum allowed size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"File does not exist: {file_path}"
            
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                actual_mb = file_size / 1024 / 1024
                return False, f"File size too large: {actual_mb:.2f}MB (max: {max_size_mb}MB)"
            
            # Additional check for suspicious file sizes
            if file_size == 0:
                return False, "Empty file not allowed"
            
            return True, None
            
        except Exception as e:
            return False, f"File size check error: {str(e)}"


class TrainingObjectiveValidator:
    """Validator for training objectives and use cases."""
    
    # Approved use cases
    APPROVED_OBJECTIVES = {
        "sentiment_analysis", "text_classification", "named_entity_recognition",
        "question_answering", "text_summarization", "language_translation",
        "topic_modeling", "document_classification", "intent_detection",
        "chatbot_training", "language_understanding", "information_extraction",
        "content_moderation", "spam_detection", "fraud_detection"
    }
    
    # Requires approval (sensitive but potentially legitimate)
    APPROVAL_REQUIRED = {
        "medical_diagnosis", "legal_analysis", "financial_advice",
        "psychological_assessment", "educational_content", "news_generation",
        "social_media_analysis", "personality_analysis", "behavior_prediction"
    }
    
    @staticmethod
    def validate_objective(objective: str, context: Optional[str] = None) -> Tuple[bool, str, ThreatLevel]:
        """
        Validate training objective for safety and appropriateness.
        
        Args:
            objective: Training objective description
            context: Additional context about the use case
            
        Returns:
            Tuple of (is_approved, message, threat_level)
        """
        objective_lower = objective.lower()
        
        # Check for blocked objectives
        for blocked in HarmfulContentDetector.BLOCKED_OBJECTIVES:
            if blocked.lower() in objective_lower:
                return False, f"Blocked objective detected: {blocked}", ThreatLevel.CRITICAL
        
        # Analyze content for harmful patterns
        analysis = HarmfulContentDetector.analyze_content(objective)
        if analysis.blocked:
            return False, f"Harmful content detected: {', '.join(analysis.detected_threats[:3])}", analysis.threat_level
        
        # Check if objective requires approval
        for sensitive in TrainingObjectiveValidator.APPROVAL_REQUIRED:
            if sensitive in objective_lower:
                return False, f"Objective requires approval: {sensitive}", ThreatLevel.MEDIUM
        
        # Check if objective is in approved list
        for approved in TrainingObjectiveValidator.APPROVED_OBJECTIVES:
            if approved in objective_lower:
                return True, f"Approved objective: {approved}", ThreatLevel.LOW
        
        # Default to requiring review for unknown objectives
        return False, "Objective requires review (not in approved list)", ThreatLevel.MEDIUM


class SecurityAuditLogger:
    """Audit logger for security events."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        """Initialize audit logger."""
        self.log_file = log_file
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], threat_level: ThreatLevel = ThreatLevel.LOW):
        """Log security event."""
        log_entry = {
            "event_type": event_type,
            "threat_level": threat_level.value,
            "timestamp": time.time(),
            "details": details
        }
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.error(f"SECURITY_ALERT: {json.dumps(log_entry)}")
        else:
            self.logger.info(f"SECURITY_EVENT: {json.dumps(log_entry)}")
    
    def log_blocked_content(self, content: str, threats: List[str], file_path: str = None):
        """Log blocked malicious content."""
        details = {
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "content_length": len(content),
            "detected_threats": threats,
            "file_path": file_path
        }
        self.log_security_event("BLOCKED_CONTENT", details, ThreatLevel.HIGH)
    
    def log_failed_validation(self, file_path: str, errors: List[str], validator_type: str):
        """Log validation failures."""
        details = {
            "file_path": file_path,
            "errors": errors,
            "validator_type": validator_type
        }
        self.log_security_event("VALIDATION_FAILED", details, ThreatLevel.MEDIUM)


# Global audit logger instance
security_logger = SecurityAuditLogger()


def enhanced_content_analysis(text: str, context: Optional[str] = None) -> SecurityAnalysis:
    """
    Perform enhanced content analysis with multiple security layers.
    
    Args:
        text: Content to analyze
        context: Additional context (training objective, use case)
        
    Returns:
        SecurityAnalysis with comprehensive threat assessment
    """
    # Base content analysis
    analysis = HarmfulContentDetector.analyze_content(text)
    
    # Additional context analysis if provided
    if context:
        context_analysis = HarmfulContentDetector.analyze_content(context)
        
        # Combine threat assessments
        analysis.detected_threats.extend(context_analysis.detected_threats)
        analysis.content_safety_score = min(analysis.content_safety_score, context_analysis.content_safety_score)
        
        if context_analysis.threat_level.value > analysis.threat_level.value:
            analysis.threat_level = context_analysis.threat_level
        
        analysis.blocked = analysis.blocked or context_analysis.blocked
    
    # Log security events
    if analysis.blocked:
        security_logger.log_blocked_content(text, analysis.detected_threats)
    
    return analysis


# Export key functions and classes
__all__ = [
    'HarmfulContentDetector',
    'EnhancedSecurityValidator', 
    'TrainingObjectiveValidator',
    'SecurityAuditLogger',
    'enhanced_content_analysis',
    'ThreatLevel',
    'SecurityAnalysis'
]