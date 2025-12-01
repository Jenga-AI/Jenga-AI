# Data Guardrails System - Usage Guide

## Overview

The Data Guardrails System provides comprehensive validation, sanitization, and safety checks for all data inputs in the Jenga-AI framework. It addresses critical security vulnerabilities and data quality issues.

## Quick Start

```python
from multitask_bert.core.data_validators import (
    UniversalDataValidator,
    DataType,
    validate_qa_data
)

# Validate QA data
result = validate_qa_data('data.json')

if result.is_valid:
    print(f"✅ Data valid! Quality score: {result.quality_score:.2%}")
    print(f"Valid samples: {result.stats['valid_samples']}")
    use_data(result.cleaned_data)
else:
    print(f"❌ Validation failed!")
    for error in result.errors:
        print(f"  - {error}")
```

## Features

### 1. Security Guardrails

**File Path Validation**:
- Prevents path traversal attacks (`../../etc/passwd`)
- Blocks symbolic links
- Validates file existence and permissions
- Restricts to allowed directories

**Input Sanitization**:
- Removes malicious patterns (script tags, eval, exec)
- Removes null bytes
- Normalizes whitespace
- Enforces length limits

**Content Safety**:
- Detects JavaScript injection
- Detects template injection
- Detects event handlers
- Warns on suspicious content

### 2. Schema Validation

Validates data structure for each task type:

**QA Metrics**:
- Required: `text`, `labels`, `sample_id`
- Label structure validation (6 heads with correct dimensions)
- Binary label values (0 or 1)

**Classification**:
- Required: `text`, `label`
- Label type validation

**NER**:
- Required: `text`, `entities`
- Entity structure validation (`start`, `end`, `label`)

**Multi-Label**:
- Required: `text`, `labels`
- Label array validation

### 3. Data Quality Scoring

Automatic quality assessment based on:
- Valid sample ratio
- Schema compliance
- Text quality
- Label consistency

Score: 0.0 (poor) to 1.0 (excellent)

## Usage Examples

### Basic Validation

```python
from multitask_bert.core.data_validators import UniversalDataValidator, DataType

# Create validator
validator = UniversalDataValidator(
    data_type=DataType.QA_METRICS,
    max_file_size_mb=50.0,  # Max 50MB
    enable_sanitization=True,
    enable_security_checks=True
)

# Validate file
result = validator.validate_file('tests/my_data.json')

# Check result
if result.is_valid:
    # Use cleaned data
    clean_data = result.cleaned_data
    print(f"Loaded {len(clean_data)} samples")
    print(f"Quality score: {result.quality_score:.1%}")
else:
    # Handle errors
    print("Validation failed:")
    for error in result.errors[:5]:  #  Show first 5 errors
        print(f"  ❌ {error}")
```

### With Allowed Directories

```python
validator = UniversalDataValidator(
    data_type=DataType.CLASSIFICATION,
    allowed_dirs=[
        '/home/user/project/data',
        '/var/data/staging'
    ]
)

# Only files in allowed directories will be accepted
result = validator.validate_file('/home/user/project/data/train.json')
```

### Disable Security Features (Development Only!)

```python
# WARNING: Only for development/testing
validator = UniversalDataValidator(
    data_type=DataType.NER,
    enable_security_checks=False,  # Disable path validation
    enable_sanitization=False      # Disable text sanitization
)
```

### Convenience Functions

```python
from multitask_bert.core.data_validators import (
    validate_qa_data,
    validate_classification_data,
    validate_ner_data
)

# Quick validation
qa_result = validate_qa_data('qa_data.json')
clf_result = validate_classification_data('clf_data.json')
ner_result = validate_ner_data('ner_data.json')
```

## Validation Result

```python
@dataclass
class ValidationResult:
    is_valid: bool                    # Overall validity
    errors: List[str]                 # Validation errors
    warnings: List[str]               # Non-fatal warnings
    cleaned_data: Optional[Any]       # Sanitized data
    quality_score: float             # Quality score (0-1)
    stats: Dict[str, Any]            # Statistics
```

**Stats Include**:
- `file_size_mb`: File size in MB
- `total_samples`: Total samples in dataset
- `valid_samples`: Number of valid samples
- `invalid_samples`: Number of invalid samples

## Integration with Training Pipeline

### Before Training

```python
from multitask_bert.core.data_validators import validate_qa_data

# Validate data before training
result = validate_qa_data('training_data.json')

if not result.is_valid:
    print(f"❌ Data validation failed! Cannot train.")
    for error in result.errors:
        print(f"  - {error}")
    sys.exit(1)

if result.quality_score < 0.8:
    print(f"⚠️ Low quality score: {result.quality_score:.1%}")
    print("Consider cleaning data first")

# Use cleaned data for training
train_with_data(result.cleaned_data)
```

### In Data Cleaning Script

```python
from multitask_bert.core.data_validators import UniversalDataValidator, DataType

def clean_and_validate(input_file, output_file):
    """Clean data and validate."""
    
    # Step 1: Run your cleaning
    cleaned_data = your_cleaning_function(input_file)
    
    # Step 2: Save to temp file
    temp_file = '/tmp/cleaned_temp.json'
    save_json(cleaned_data, temp_file)
    
    # Step 3: Validate cleaned data
    validator = UniversalDataValidator(DataType.QA_METRICS)
    result = validator.validate_file(temp_file)
    
    if result.quality_score < 0.95:
        print(f"⚠️ Quality check: {result.quality_score:.1%}")
        print(f"Warnings: {len(result.warnings)}")
    
    # Step 4: Save validated data
    save_json(result.cleaned_data, output_file)
    return result
```

## Error Handling

```python
result = validator.validate_file('data.json')

# Check specific error types
path_errors = [e for e in result.errors if 'path' in e.lower()]
schema_errors = [e for e in result.errors if 'field' in e.lower()]
security_errors = [e for e in result.warnings if 'suspicious' in e.lower()]

print(f"Path errors: {len(path_errors)}")
print(f"Schema errors: {len(schema_errors)}")
print(f"Security warnings: {len(security_errors)}")

# Detailed error reporting
for i, error in enumerate(result.errors, 1):
    print(f"{i}. {error}")
```

## Custom Validators

Extend for custom data types:

```python
from multitask_bert.core.data_validators import (
    UniversalDataValidator,
    SchemaValidator,
    DataType
)

# Add custom schema
SchemaValidator.SCHEMAS[DataType.GENERIC] = {
    "required_fields": ["text", "metadata"],
    "text_max_length": 2000
}

# Use custom validator
validator = UniversalDataValidator(DataType.GENERIC)
result = validator.validate_file('custom_data.json')
```

## Security Best Practices

### ✅ DO

1. **Always validate external data**
   ```python
   result = validate_qa_data(user_provided_file)
   if not result.is_valid:
       reject_and_notify_user(result.errors)
   ```

2. **Restrict file paths**
   ```python
   validator = UniversalDataValidator(
       data_type=DataType.QA_METRICS,
       allowed_dirs=['/app/data/uploads']
   )
   ```

3. **Set reasonable limits**
   ```python
   validator = UniversalDataValidator(
       data_type=DataType.QA_METRICS,
       max_file_size_mb=10.0  # Don't allow huge files
   )
   ```

4. **Review warnings**
   ```python
   if result.warnings:
       for warning in result.warnings:
           logger.warning(f"Data validation: {warning}")
   ```

### ❌ DON'T

1. **Don't skip validation**
   ```python
   # BAD: Loading data without validation
   data = json.load(open(user_file))
   ```

2. **Don't disable security in production**
   ```python
   # BAD: Disabling security checks
   validator = UniversalDataValidator(
       data_type=DataType.QA_METRICS,
       enable_security_checks=False  # NEVER in production!
   )
   ```

3. **Don't ignore quality scores**
   ```python
   # BAD: Using low-quality data
   if result.quality_score < 0.5:
       # This is bad data, don't use it!
       raise ValueError("Data quality too low")
   ```

## Troubleshooting

### "File path validation failed"
- Check file exists
- Verify file permissions
- Ensure no `..` in path
- Check if file is in allowed directories

### "Text too long"
- Reduce text length
- Split into multiple samples
- Increase `text_max_length` in schema (if appropriate)

### "Missing required field"
- Check schema requirements
- Add missing fields
- Verify field names match exactly

### "Low quality score"
- Review validation errors
- Check data format
- Run data cleaning script
- Inspect individual samples

## Performance Tips

1. **Batch validation**: Validate entire dataset at once (done automatically)
2. **Disable sanitization**: If data is trusted, disable for speed (dev only!)
3. **Limit error reporting**: Only first N errors collected
4. **Stream large files**: For very large datasets, process in chunks

## API Reference

### UniversalDataValidator

```python
UniversalDataValidator(
    data_type: DataType,
    max_file_size_mb: float = 100.0,
    allowed_dirs: Optional[List[str]] = None,
    enable_sanitization: bool = True,
    enable_security_checks: bool = True
)
```

**Methods**:
- `validate_file(file_path: str) -> ValidationResult`

### DataType Enum

- `DataType.QA_METRICS`
- `DataType.CLASSIFICATION`
- `DataType.NER`
- `DataType.MULTI_LABEL`
- `DataType.SEQ2SEQ`
- `DataType.LLM_FINETUNING`
- `DataType.GENERIC`

### Convenience Functions

- `validate_qa_data(file_path, **kwargs) -> ValidationResult`
- `validate_classification_data(file_path, **kwargs) -> ValidationResult`
- `validate_ner_data(file_path, **kwargs) -> ValidationResult`

---

## See Also

- [Framework Analysis](FRAMEWORK_ANALYSIS.md) - Security issues and recommendations
- [Data Cleaning Guide](DATA_CLEANING_GUIDE.md) - How to clean data
- [QA Data Format](QA_DATA_FORMAT.md) - Format specification
