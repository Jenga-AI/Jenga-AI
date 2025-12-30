#!/usr/bin/env python3
"""
Create tiny datasets for rapid CPU testing
"""

import csv
import json
from pathlib import Path

def create_tiny_sentiment_data():
    """Create tiny sentiment dataset"""
    tiny_data = [
        {"text": "This is excellent!", "label": 1},
        {"text": "Very disappointing", "label": 0}, 
        {"text": "Amazing work here", "label": 1},
        {"text": "Really bad quality", "label": 0},
        {"text": "Love this feature", "label": 1},
        {"text": "Completely broken", "label": 0},
        {"text": "Perfect solution", "label": 1},
        {"text": "Waste of time", "label": 0},
        {"text": "Brilliant idea", "label": 1},
        {"text": "Total failure", "label": 0}
    ]
    
    # Save as CSV
    csv_path = Path("tests/data/sentiment_tiny.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(tiny_data)
    
    # Save as JSONL
    jsonl_path = Path("tests/data/sentiment_tiny.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in tiny_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created tiny sentiment dataset: {len(tiny_data)} samples")
    return csv_path, jsonl_path

def create_tiny_ner_data():
    """Create tiny NER dataset"""
    tiny_ner_data = [
        {
            "text": "John lives in New York",
            "labels": ["B-PER", "O", "O", "B-LOC", "I-LOC"]
        },
        {
            "text": "Apple Inc is headquartered",
            "labels": ["B-ORG", "I-ORG", "O", "O"]
        },
        {
            "text": "Mary visited London yesterday",
            "labels": ["B-PER", "O", "B-LOC", "O"]
        },
        {
            "text": "Microsoft develops software products",
            "labels": ["B-ORG", "O", "O", "O"]
        },
        {
            "text": "Paris is beautiful",
            "labels": ["B-LOC", "O", "O"]
        }
    ]
    
    # Save as JSONL
    jsonl_path = Path("tests/data/ner_tiny.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in tiny_ner_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created tiny NER dataset: {len(tiny_ner_data)} samples")
    return jsonl_path

def create_tiny_qa_data():
    """Create tiny QA scoring data"""
    tiny_qa_data = [
        {
            "text": "Hello, how can I help you today?",
            "quality_scores": {"opening": 1, "listening": 3, "resolution": 2}
        },
        {
            "text": "I understand your concern and will help you.",
            "quality_scores": {"opening": 0, "listening": 4, "resolution": 3}
        },
        {
            "text": "Thank you for calling, goodbye!",
            "quality_scores": {"opening": 0, "listening": 1, "resolution": 1}
        },
        {
            "text": "Let me check that information for you.",
            "quality_scores": {"opening": 0, "listening": 3, "resolution": 2}
        },
        {
            "text": "What seems to be the problem?",
            "quality_scores": {"opening": 1, "listening": 2, "resolution": 1}
        }
    ]
    
    # Save as JSON
    json_path = Path("tests/data/qa_tiny.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tiny_qa_data, f, indent=2)
    
    print(f"✓ Created tiny QA dataset: {len(tiny_qa_data)} samples")
    return json_path

def create_tiny_llm_data():
    """Create tiny dataset for LLM fine-tuning"""
    tiny_llm_data = [
        {
            "input": "Hello",
            "output": "Hi there! How can I help you today?"
        },
        {
            "input": "Thank you",
            "output": "You're welcome! Happy to help."
        },
        {
            "input": "Goodbye",
            "output": "Have a great day! Feel free to return anytime."
        },
        {
            "input": "How are you?",
            "output": "I'm doing well, thank you for asking!"
        },
        {
            "input": "Help me",
            "output": "Of course! What do you need assistance with?"
        }
    ]
    
    # Save as JSON
    json_path = Path("tests/data/llm_tiny.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tiny_llm_data, f, indent=2)
    
    print(f"✓ Created tiny LLM dataset: {len(tiny_llm_data)} samples")
    return json_path

def main():
    """Create all tiny datasets"""
    print("\n" + "="*50)
    print("  CREATING TINY DATASETS FOR RAPID TESTING")
    print("="*50)
    
    # Create datasets
    sentiment_csv, sentiment_jsonl = create_tiny_sentiment_data()
    ner_jsonl = create_tiny_ner_data()
    qa_json = create_tiny_qa_data()
    llm_json = create_tiny_llm_data()
    
    print(f"\n✓ All datasets created in tests/data/")
    print(f"   Sentiment: {sentiment_csv}")
    print(f"   NER: {ner_jsonl}")
    print(f"   QA: {qa_json}")
    print(f"   LLM: {llm_json}")
    
    return True

if __name__ == "__main__":
    main()