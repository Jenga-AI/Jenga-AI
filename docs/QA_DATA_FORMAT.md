# QA Data Format Specification

## Overview

This document defines the expected format for QA (Quality Assurance) metrics training data. All data must conform to this specification before being used for model training.

## File Format

**Format**: JSON (JavaScript Object Notation)  
**Encoding**: UTF-8  
**Structure**: Array of objects

## Sample Structure

```json
[
  {
    "text": "Full conversation transcript...",
    "labels": {
      "opening": [0],
      "listening": [1, 0, 1, 1, 0],
      "proactiveness": [1, 1, 0],
      "resolution": [1, 1, 0, 1, 1],
      "hold": [0, 0],
      "closing": [1]
    },
    "sample_id": "qa_1",
    "scenario": "customer_support",
    "quality_level": "good"
  }
]
```

## Required Fields

### 1. `text` (Required)
- **Type**: String
- **Description**: The full conversation transcript or text to be evaluated
- **Constraints**:
  - Must not be empty
  - Should be meaningful text (not just whitespace)
- **Example**: `"Welcome to support. How can I help you today?..."`

### 2. `labels` (Required)
- **Type**: Object (dictionary)
- **Description**: QA metric scores for different evaluation heads
- **Constraints**:
  - Must contain all 6 heads (see Label Structure below)
  - Each head must have correct number of binary labels
  - Labels must be integers (0 or 1)
  - **NOT** JSON strings - must be native objects/arrays

#### Label Structure

| Head | # of Labels | Sub-metrics |
|------|-------------|-------------|
| `opening` | 1 | Use of call opening phrase |
| `listening` | 5 | Not interrupted, Empathizes, Paraphrases, Uses please/thank you, No hesitation |
| `proactiveness` | 3 | Willing to solve extra issues, Confirms satisfaction, Follows up |
| `resolution` | 5 | Accurate info, Correct language, Consults if unsure, Correct steps, Clear explanation |
| `hold` | 2 | Explains before hold, Thanks for holding |
| `closing` | 1 | Proper closing phrase |

**Example**:
```json
{
  "opening": [1],
  "listening": [1, 1, 0, 1, 0],
  "proactiveness": [1, 0, 1],
  "resolution": [1, 1, 1, 1, 0],
  "hold": [1, 1],
  "closing": [1]
}
```

### 3. `sample_id` (Required)
- **Type**: String
- **Description**: Unique identifier for the sample
- **Constraints**:
  - Must be unique across the entire dataset
  - Recommended format: `qa_{number}` or similar
- **Example**: `"qa_1234"`

## Optional Fields

### 4. `scenario` (Optional)
- **Type**: String
- **Description**: The type of conversation or scenario
- **Examples**: `"customer_support"`, `"technical_help"`, `"complaint_handling"`

### 5. `quality_level` (Optional)
- **Type**: String
- **Description**: Overall quality assessment
- **Valid values**: `"poor"`, `"fair"`, `"good"`, `"excellent"`

## Common Data Quality Issues

### ❌ Issue 1: Labels as JSON Strings
**BAD**:
```json
{
  "labels": "{\"opening\": [1], \"listening\": [1,0,1,1,0]...}"
}
```

**GOOD**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1, 1, 0],
    ...
  }
}
```

### ❌ Issue 2: Incorrect Label Dimensions
**BAD**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1],  // Should be 5, not 3!
    ...
  }
}
```

**GOOD**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1, 0, 0],  // Correct: 5 labels
    ...
  }
}
```

### ❌ Issue 3: String Values in Label Arrays
**BAD**:
```json
{
  "labels": {
    "hold": ["No hold occurred", "N/A"]
  }
}
```

**GOOD**:
```json
{
  "labels": {
    "hold": [0, 0]  // Use 0 if metric doesn't apply
  }
}
```

### ❌ Issue 4: Missing Heads
**BAD**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1, 1, 0]
    // Missing other heads!
  }
}
```

**GOOD**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1, 1, 0],
    "proactiveness": [1, 0, 1],
    "resolution": [1, 1, 1, 1, 0],
    "hold": [0, 0],
    "closing": [1]
  }
}
```

## Validation Checklist

Before using data for training, verify:

- [ ] File is valid JSON
- [ ] Root structure is an array
- [ ] Each sample has `text`, `labels`, and `sample_id` fields
- [ ] All `text` fields are non-empty strings
- [ ] All `labels` are native objects (not JSON strings)
- [ ] All 6 heads present in every sample
- [ ] Each head has correct number of labels:
  - [ ] `opening`: 1
  - [ ] `listening`: 5
  - [ ] `proactiveness`: 3
  - [ ] `resolution`: 5
  - [ ] `hold`: 2
  - [ ] `closing`: 1
- [ ] All label values are integers (0 or 1)
- [ ] All `sample_id` values are unique

## Data Cleaning

If your data doesn't meet this specification, use the cleaning script:

```bash
python tests/clean_qa_data.py \\
    --input your_data.json \\
    --output your_data_cleaned.json
```

The cleaner will:
- Parse JSON string labels to objects
- Normalize label dimensions (pad with zeros or truncate)
- Remove invalid samples
- Generate quality report

## Size Recommendations

| Dataset Type | Recommended Size |
|--------------|------------------|
| Testing/Debug | 50-100 samples |
| Development | 500-1000 samples |
| Training (Small) | 1000-5000 samples |
| Training (Medium) | 5000-10000 samples |
| Training (Large) | 10000+ samples |

## Example: Complete Valid Sample

```json
{
  "text": "Welcome to Child Helpline support. Thank you for reaching out. I understand you're concerned about a situation. Can you please tell me more about what's happening? I'm here to listen and help. Let me make sure I understand correctly - you mentioned issues at school. I appreciate you sharing this with me. Would you like me to help you explore some options for addressing this? That's a great question. Let me explain the process clearly. First, we'll need to document what you've shared. I want to make sure you're satisfied with our discussion today. Is there anything else you'd like to talk about? Thank you for trusting us with this. Take care.",
  "labels": {
    "opening": [1],
    "listening": [1, 1, 1, 1, 1],
    "proactiveness": [1, 1, 1],
    "resolution": [1, 1, 1, 1, 1],
    "hold": [0, 0],
    "closing": [1]
  },
  "sample_id": "qa_example_1",
  "scenario": "child_helpline",
  "quality_level": "excellent"
}
```

---

**Version**: 1.0  
**Last Updated**: 2025-11-25  
**Maintained by**: Jenga-AI Team
