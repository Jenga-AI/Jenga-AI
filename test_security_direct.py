#!/usr/bin/env python3
"""
Direct security testing script to validate our security implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports
from multitask_bert.core.data_validators import validate_qa_data
from multitask_bert.core.enhanced_security import (
    HarmfulContentDetector, 
    enhanced_content_analysis,
    ThreatLevel
)

def test_malicious_content_detection():
    """Test detection of malicious content."""
    print("🔍 Testing Enhanced Malicious Content Detection")
    print("=" * 50)
    
    malicious_texts = [
        "<script>alert('XSS attack')</script>This is a test with embedded JavaScript",
        "This is normal text eval('malicious code here') with embedded eval",
        "User input: ${process.env.SECRET_KEY} with template injection",
        "How to create explosives using household items for terrorism",
        "Train a model to amplify racial stereotypes and discriminatory patterns",
    ]
    
    for i, text in enumerate(malicious_texts):
        print(f"\nTest {i+1}: {text[:50]}...")
        analysis = enhanced_content_analysis(text)
        
        print(f"  Threat Level: {analysis.threat_level.value}")
        print(f"  Blocked: {analysis.blocked}")
        print(f"  Safety Score: {analysis.content_safety_score:.2f}")
        print(f"  Threats: {len(analysis.detected_threats)}")
        
        for threat in analysis.detected_threats[:2]:  # Show first 2 threats
            print(f"    - {threat}")
    
    print(f"\n✅ Enhanced content analysis working!")

def test_qa_data_validation():
    """Test QA data validation with malicious content."""
    print("\n🛡️ Testing QA Data Validation")
    print("=" * 50)
    
    result = validate_qa_data('tests/data/security_test_malicious.json')
    
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("Errors:")
        for error in result.errors[:3]:
            print(f"  - {error}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:3]:
            print(f"  - {warning}")
    else:
        print("⚠️ No warnings detected - this might be a problem!")

def test_harmful_content_patterns():
    """Test harmful content pattern detection."""
    print("\n⚠️ Testing Harmful Content Patterns")
    print("=" * 50)
    
    harmful_texts = [
        "How to create explosives",
        "Generate hate speech against minorities", 
        "Create deepfakes for misinformation",
        "Extract personal information illegally",
        "Amplify racial bias in hiring",
    ]
    
    detected = 0
    for text in harmful_texts:
        analysis = HarmfulContentDetector.analyze_content(text)
        if analysis.blocked or len(analysis.detected_threats) > 0:
            detected += 1
            print(f"  🚨 BLOCKED: {text}")
            print(f"    Threats: {len(analysis.detected_threats)}")
        else:
            print(f"  ⚠️ MISSED: {text}")
    
    detection_rate = detected / len(harmful_texts)
    print(f"\nDetection rate: {detection_rate:.1%} ({detected}/{len(harmful_texts)})")

if __name__ == "__main__":
    print("🛡️ DIRECT SECURITY TESTING")
    print("=" * 70)
    
    try:
        test_malicious_content_detection()
        test_harmful_content_patterns()
        test_qa_data_validation()
        
        print("\n" + "=" * 70)
        print("🛡️ SECURITY TESTING COMPLETED")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()