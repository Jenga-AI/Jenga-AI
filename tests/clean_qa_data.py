"""
QA Data Cleaning Script

This script cleans and validates QA metrics data to ensure it meets the
required format for training. It handles common data quality issues like:
- JSON string labels (converts to native Python objects)
- Incorrect label dimensions (pads/truncates to expected sizes)
- Missing or invalid fields
- Data type inconsistencies

Usage:
    python clean_qa_data.py --input <input_file> --output <output_file>
    
Example:
    python clean_qa_data.py \\
        --input tests/synthetic_qa_metrics_data_v01x.json \\
        --output tests/synthetic_qa_metrics_data_cleaned.json
"""

import pandas as pd
import json
import argparse
from typing import Dict, List, Any
import sys
from pathlib import Path

# Expected label configuration for QA heads
QA_HEADS_CONFIG = {
    "opening": 1,
    "listening": 5,
    "proactiveness": 3,
    "resolution": 5,
    "hold": 2,
    "closing": 1
}

# Required fields in each sample
REQUIRED_FIELDS = ["text", "labels", "sample_id"]
OPTIONAL_FIELDS = ["scenario", "quality_level"]


class QADataCleaner:
    """Cleans and validates QA metrics data."""
    
    def __init__(self):
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "label_parse_errors": 0,
            "dimension_fixes": 0,
            "missing_fields": 0,
            "warnings": []
        }
    
    def clean_labels(self, labels: Any) -> Dict[str, List[int]]:
        """
        Clean and normalize label data.
        
        Args:
            labels: Raw label data (could be string or dict)
            
        Returns:
            Dictionary with properly formatted labels
        """
        # Parse JSON string if needed
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError as e:
                self.stats["label_parse_errors"] += 1
                raise ValueError(f"Failed to parse labels JSON: {e}")
        
        if not isinstance(labels, dict):
            raise ValueError(f"Labels must be dict or JSON string, got {type(labels)}")
        
        # Normalize each head's labels
        cleaned_labels = {}
        for head, expected_size in QA_HEADS_CONFIG.items():
            if head not in labels:
                # Missing head - create zero array
                cleaned_labels[head] = [0] * expected_size
                self.stats["dimension_fixes"] += 1
                self.stats["warnings"].append(f"Missing head '{head}', filled with zeros")
                continue
            
            label_array = labels[head]
            
            # Parse if it's a JSON string
            if isinstance(label_array, str):
                try:
                    label_array = json.loads(label_array)
                except json.JSONDecodeError:
                    # Can't parse, create zero array
                    cleaned_labels[head] = [0] * expected_size
                    self.stats["dimension_fixes"] += 1
                    self.stats["warnings"].append(f"Failed to parse {head} labels, filled with zeros")
                    continue
            
            # Ensure it's a list
            if not isinstance(label_array, list):
                cleaned_labels[head] = [0] * expected_size
                self.stats["dimension_fixes"] += 1
                continue
            
            actual_size = len(label_array)
            
            if actual_size < expected_size:
                # Pad with zeros
                cleaned_labels[head] = label_array + [0] * (expected_size - actual_size)
                self.stats["dimension_fixes"] += 1
            elif actual_size > expected_size:
                # Truncate
                cleaned_labels[head] = label_array[:expected_size]
                self.stats["dimension_fixes"] += 1
            else:
                # Correct size
                cleaned_labels[head] = label_array
            
            # Ensure all values are integers (not floats or strings)
            cleaned_labels[head] = [int(v) for v in cleaned_labels[head]]
        
        return cleaned_labels
    
    def validate_sample(self, sample: Dict[str, Any], index: int) -> bool:
        """
        Validate a single sample.
        
        Args:
            sample: Sample dictionary
            index: Sample index for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in sample or sample[field] is None:
                self.stats["missing_fields"] += 1
                self.stats["warnings"].append(f"Sample {index}: Missing required field '{field}'")
                return False
        
        # Check text is not empty
        if not sample["text"] or not isinstance(sample["text"], str):
            self.stats["warnings"].append(f"Sample {index}: Invalid text field")
            return False
        
        return True
    
    def clean_dataset(self, input_file: str, output_file: str, verbose: bool = True):
        """
        Clean entire dataset and save to file.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
            verbose: Print progress messages
        """
        if verbose:
            print(f"Loading data from {input_file}...")
        
        # Load data
        try:
            df = pd.read_json(input_file)
        except Exception as e:
            print(f"ERROR: Failed to load input file: {e}")
            sys.exit(1)
        
        self.stats["total_samples"] = len(df)
        
        if verbose:
            print(f"Loaded {len(df)} samples")
            print("Cleaning data...")
        
        cleaned_data = []
        
        for idx, row in df.iterrows():
            # Convert to dict
            sample = row.to_dict()
            
            # Validate
            if not self.validate_sample(sample, idx):
                continue
            
            # Clean labels
            try:
                sample["labels"] = self.clean_labels(sample["labels"])
            except Exception as e:
                self.stats["warnings"].append(f"Sample {idx}: {str(e)}")
                continue
            
            cleaned_data.append(sample)
            self.stats["valid_samples"] += 1
        
        if verbose:
            print(f"\nCleaning complete!")
            print(f"Valid samples: {self.stats['valid_samples']}/{self.stats['total_samples']}")
            print(f"Dimension fixes: {self.stats['dimension_fixes']}")
            print(f"Label parse errors: {self.stats['label_parse_errors']}")
            print(f"Missing fields: {self.stats['missing_fields']}")
            
            if self.stats["warnings"]:
                print(f"\nWarnings: {len(self.stats['warnings'])} (showing first 5)")
                for warning in self.stats["warnings"][:5]:
                    print(f"  - {warning}")
        
        # Save cleaned data
        if verbose:
            print(f"\nSaving cleaned data to {output_file}...")
        
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON with proper formatting
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        if verbose:
            print(f"✅ Saved {len(cleaned_data)} cleaned samples to {output_file}")
        
        return cleaned_data


def main():
    parser = argparse.ArgumentParser(description='Clean QA metrics data for training')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    
    args = parser.parse_args()
    
    cleaner = QADataCleaner()
    cleaner.clean_dataset(args.input, args.output, verbose=not args.quiet)
    
    # Exit with error code if no valid samples
    if cleaner.stats["valid_samples"] == 0:
        print("ERROR: No valid samples after cleaning!")
        sys.exit(1)


if __name__ == "__main__":
    main()
