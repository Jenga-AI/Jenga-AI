# Data Cleaning Guide for QA Metrics

## Quick Start

**Problem**: Your QA data has quality issues and training crashes?

**Solution**: Clean your data before training!

```bash
# 1. Clean the data
python tests/clean_qa_data.py \\
    --input your_data.json \\
    --output your_data_cleaned.json

# 2. Update config to use cleaned data
# Edit your config.yaml:
#   data:
#     train_data_path: "path/to/your_data_cleaned.json"

# 3. Train with clean data
python tests/train_test.py --config your_config.yaml
```

## Common Data Quality Issues

### 1. **JSON String Labels** (Most Common)
**Problem**: Labels stored as JSON strings instead of native objects

**Symptoms**:
- `ValueError: too many dimensions 'str'`
- Training crashes during data loading

**Example Bad Data**:
```json
{
  "labels": "{\"opening\": [1], \"listening\": [1,0,1,1,0]}"
          //  ^^^^ JSON string - BAD!
}
```

**Fix**: Use the cleaning script (automatically parses JSON strings)

---

### 2. **Incorrect Label Dimensions**
**Problem**: Label arrays have wrong number of elements

**Expected Sizes**:
- `opening`: 1
- `listening`: 5
- `proactiveness`: 3
- `resolution`: 5
- `hold`: 2
- `closing`: 1

**Example Bad Data**:
```json
{
  "listening": [1, 0, 1]  // Should be 5, not 3!
}
```

**Fix**: Cleaner pads with zeros or truncates to correct size

---

### 3. **String Values in Arrays**
**Problem**: Non-numeric strings in label arrays

**Example Bad Data**:
```json
{
  "hold": ["No hold occurred", "N/A"]
}
```

**Fix**: Cleaner replaces with zero array `[0, 0]`

---

### 4. **Missing Heads**
**Problem**: Some QA heads missing from labels dict

**Example Bad Data**:
```json
{
  "labels": {
    "opening": [1],
    "listening": [1,0,1,1,0]
    // Missing: proactiveness, resolution, hold, closing
  }
}
```

**Fix**: Cleaner adds missing heads with zero arrays

---

## Using the Data Cleaner

### Basic Usage

```bash
python tests/clean_qa_data.py \\
    --input tests/synthetic_qa_metrics_data_v01x.json \\
    --output tests/synthetic_qa_metrics_data_cleaned.json
```

### Output Example

```
Loading data from tests/synthetic_qa_metrics_data_v01x.json...
Loaded 4996 samples
Cleaning data...

Cleaning complete!
Valid samples: 4989/4996
Dimension fixes: 3083
Label parse errors: 0
Missing fields: 0

Warnings: 7 (showing first 5)
  - Sample 376: invalid literal for int() with base 10: 'No hold occurred'
  - Sample 1246: invalid literal for int() with base 10: ''
  
✅ Saved 4989 cleaned samples to tests/synthetic_qa_metrics_data_cleaned.json
```

### What the Cleaner Does

1. **Parses JSON strings** → Native Python objects
2. **Normalizes dimensions** → Pads or truncates to correct sizes
3. **Validates fields** → Ensures required fields exist
4. **Removes invalid samples** → Skips samples that can't be fixed
5. **Generates report** → Shows what was fixed

---

## Data Ingestion Best Practices

### Before Training: Data Quality Checklist

```bash
# 1. Validate format
python tests/clean_qa_data.py --input your_data.json --output /tmp/test_clean.json

# 2. Check the report
# Look for:
#   - High valid sample rate (>95%)
#   - Low dimension fixes (<10%)
#   - Zero parse errors

# 3. Inspect cleaned data
python -c "
import json
with open('/tmp/test_clean.json') as f:
    data = json.load(f)
    print(f'Samples: {len(data)}'.
    print(f'Sample labels type: {type(data[0][\"labels\"])}')
    # Should be <class 'dict'>, NOT <class 'str'>
"

# 4. Use cleaned data for training
```

### When Creating New Datasets

**✅ DO**:
- Use native Python data structures, not JSON strings
- Ensure all 6 heads are present
- Use integers (0/1) for labels
- Validate data before saving
- Run cleaner as final step

**❌ DON'T**:
- Store labels as JSON strings
- Use inconsistent label dimensions
- Mix data types (strings with integers)
- Skip validation

### Code Example: Generating Clean Data

```python
import json

def create_qa_sample(text, labels_dict, sample_id):
    """Create a properly formatted QA sample."""
    # Ensure labels are native dicts, not strings!
    return {
        "text": text,
        "labels": labels_dict,  # Already a dict
        "sample_id": sample_id,
        "scenario": "example",
        "quality_level": "good"
    }

# Generate samples
samples = []
for i in range(100):
    labels = {
        "opening": [1],
        "listening": [1, 0, 1, 1, 0],  # Native list
        "proactiveness": [1, 1, 0],
        "resolution": [1, 1, 1, 0, 1],
        "hold": [0, 0],
        "closing": [1]
    }
    sample = create_qa_sample(
        text=f"Sample conversation {i}...",
        labels_dict=labels,  # Pass dict, not JSON string!
        sample_id=f"qa_{i}"
    )
    samples.append(sample)

# Save as JSON
with open('clean_dataset.json', 'w') as f:
    json.dump(samples, f, indent=2)
```

---

## Troubleshooting

### "ValueError: too many dimensions 'str'"

**Cause**: Labels are stored as JSON strings

**Fix**: Run data cleaner

```bash
python tests/clean_qa_data.py --input your_data.json --output your_data_cleaned.json
```

### "RuntimeError: stack expects each tensor to be equal size"

**Cause**: Inconsistent label array dimensions

**Fix**: Run data cleaner (normalizes all dimensions)

### "All samples removed during cleaning"

**Causes**:
1. Wrong JSON format (not an array of objects)
2. Missing required fields
3. Corrupted data

**Debug**:
```python
import json

# Check structure
with open('your_data.json') as f:
    data = json.load(f)
    print(f"Type: {type(data)}")  # Should be list
    if isinstance(data, list) and len(data) > 0:
        print(f"First sample: {data[0].keys()}")  # Should have text, labels, sample_id
```

---

## Data Quality Metrics

### Good Dataset
- ✅ Valid samples: >95%
- ✅ Dimension fixes: <5%
- ✅ Parse errors: 0
- ✅ Missing fields: 0

### Acceptable Dataset
- ⚠️ Valid samples: 85-95%
- ⚠️ Dimension fixes: 5-20%
- ⚠️ Parse errors: <10
- ⚠️ Missing fields: <5

### Poor Dataset (Needs Review)
- ❌ Valid samples: <85%
- ❌ Dimension fixes: >20%
- ❌ Parse errors: >10
- ❌ Missing fields: >5

---

## Advanced: Custom Cleaning

need different cleaning logic? Extend the `QADataCleaner` class:

```python
from tests.clean_qa_data import QADataCleaner

class CustomCleaner(QADataCleaner):
    def clean_labels(self, labels):
        # Your custom cleaning logic
        labels = super().clean_labels(labels)
        
        # Additional processing
        # ...
        
        return labels

# Use it
cleaner = CustomCleaner()
cleaner.clean_dataset('input.json', 'output.json')
```

---

## References

- [QA Data Format Specification](QA_DATA_FORMAT.md)
- [Training Guide](../tests/MLFLOW_GUIDE.md)
- Cleaning Script: `tests/clean_qa_data.py`
