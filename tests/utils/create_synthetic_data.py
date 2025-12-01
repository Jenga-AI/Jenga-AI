#!/usr/bin/env python3
"""
Synthetic Data Generator for Testing
=====================================
Creates small, controlled datasets for testing the Jenga-AI framework.

Usage:
    python tests/utils/create_synthetic_data.py --all
    python tests/utils/create_synthetic_data.py --sentiment
    python tests/utils/create_synthetic_data.py --ner
    python tests/utils/create_synthetic_data.py --agriculture
    python tests/utils/create_synthetic_data.py --qa
"""

import json
import csv
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Swahili and English sample texts for sentiment analysis
POSITIVE_TEXTS_SWAHILI = [
    "Ninafurahi sana leo!",
    "Hii ni habari njema kabisa!",
    "Napenda sana kazi hii.",
    "Matunda haya ni mazuri.",
    "Elimu ni muhimu kwa maendeleo.",
    "Shamba letu limezaa vizuri mwaka huu.",
    "Wananchi wanafurahi na huduma za afya.",
    "Biashara inaendelea vizuri.",
    "Mvua imenyesha vizuri msimu huu.",
    "Watoto wanasoma vizuri shuleni.",
]

NEGATIVE_TEXTS_SWAHILI = [
    "Hii ni mbaya sana.",
    "Sina furaha leo.",
    "Mazao yameharibiwa.",
    "Hakuna maji ya kunywa.",
    "Ugonjwa umezidi sana.",
    "Njaa imetanda kijijini.",
    "Barabara ni mbaya.",
    "Elimu haipatikani kwa wote.",
    "Usalama hauridhishi.",
    "Bei za chakula zimepanda.",
]

POSITIVE_TEXTS_ENGLISH = [
    "This is excellent news!",
    "I am very happy with the results.",
    "The harvest was successful this year.",
    "Education access has improved greatly.",
    "Healthcare services are working well.",
    "The economy is growing steadily.",
    "Security has been enhanced.",
    "Infrastructure development is impressive.",
    "Youth employment opportunities have increased.",
    "Clean water is now accessible.",
]

NEGATIVE_TEXTS_ENGLISH = [
    "This is very disappointing.",
    "The situation is getting worse.",
    "Crop diseases are spreading.",
    "Water shortage is critical.",
    "Healthcare facilities are inadequate.",
    "Unemployment rate is high.",
    "Security concerns are rising.",
    "Infrastructure is deteriorating.",
    "Education quality has declined.",
    "Food insecurity is increasing.",
]

CODE_SWITCHED_TEXTS = [
    "Nina feel good leo, job imenitafutia pesa nzuri.",
    "Hii system inafanya kazi properly sana.",
    "Mazao ya organic farming ni better kuliko conventional.",
    "Healthcare services should be accessible kwa kila mtu.",
    "Security operations zimefanya improvements kubwa.",
]

# NER sample texts
NER_SAMPLES = [
    {
        "text": "President Ruto visited Nairobi yesterday to discuss security.",
        "entities": [
            {"text": "President Ruto", "start": 0, "end": 14, "label": "PERSON"},
            {"text": "Nairobi", "start": 23, "end": 30, "label": "LOCATION"},
        ]
    },
    {
        "text": "Cyber attack on KRA systems detected by security team.",
        "entities": [
            {"text": "Cyber attack", "start": 0, "end": 12, "label": "THREAT"},
            {"text": "KRA", "start": 16, "end": 19, "label": "ORGANIZATION"},
        ]
    },
    {
        "text": "John Doe reported ransomware infection in Mombasa office.",
        "entities": [
            {"text": "John Doe", "start": 0, "end": 8, "label": "PERSON"},
            {"text": "ransomware infection", "start": 18, "end": 38, "label": "THREAT"},
            {"text": "Mombasa", "start": 42, "end": 49, "label": "LOCATION"},
        ]
    },
    {
        "text": "Safaricom detected phishing attempts targeting customers.",
        "entities": [
            {"text": "Safaricom", "start": 0, "end": 9, "label": "ORGANIZATION"},
            {"text": "phishing attempts", "start": 19, "end": 36, "label": "THREAT"},
        ]
    },
    {
        "text": "Lake Victoria region faces environmental threats.",
        "entities": [
            {"text": "Lake Victoria", "start": 0, "end": 13, "label": "LOCATION"},
        ]
    },
]

# Agricultural disease descriptions
AGRICULTURE_HEALTHY = [
    "Green maize leaves with no spots or discoloration.",
    "Healthy cassava plants showing vigorous growth.",
    "Normal wheat stalks with golden heads.",
    "Tomato plants with bright green foliage.",
    "Bean plants showing strong development.",
    "Rice paddies with uniform green color.",
    "Potato plants with healthy leaf structure.",
    "Banana plants with no visible damage.",
    "Coffee plants with dark green leaves.",
    "Tea bushes showing normal growth patterns.",
]

AGRICULTURE_DISEASE = [
    "Maize leaves with brown spots and yellowing (maize blight).",
    "Cassava showing mosaic patterns on leaves (mosaic virus).",
    "Wheat stalks with rust-colored pustules (wheat rust).",
    "Tomato plants with wilting and brown stems (bacterial wilt).",
    "Bean leaves with circular brown lesions (anthracnose).",
    "Rice plants with blast lesions on leaves (rice blast).",
    "Potato leaves with dark brown patches (late blight).",
    "Banana plants with yellowing leaves (Panama disease).",
    "Coffee leaves with orange rust spots (coffee rust).",
    "Tea leaves with blister-like symptoms (blister blight).",
]


def create_sentiment_dataset(output_dir: Path, num_samples: int = 100):
    """Create sentiment analysis dataset (CSV and JSONL)."""
    print(f"Creating sentiment analysis dataset ({num_samples} samples)...")
    
    # Create balanced dataset
    half = num_samples // 2
    
    # Combine all text sources
    all_positive = POSITIVE_TEXTS_SWAHILI + POSITIVE_TEXTS_ENGLISH + [
        t for t in CODE_SWITCHED_TEXTS if random.random() > 0.5
    ]
    all_negative = NEGATIVE_TEXTS_SWAHILI + NEGATIVE_TEXTS_ENGLISH
    
    # Sample with replacement if needed
    positive_samples = random.choices(all_positive, k=half)
    negative_samples = random.choices(all_negative, k=half)
    
    # Combine and shuffle
    data = [(text, 1) for text in positive_samples] + [(text, 0) for text in negative_samples]
    random.shuffle(data)
    
    # Save as CSV
    csv_path = output_dir / "sentiment_mini.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        writer.writerows(data)
    print(f"  ✓ Created {csv_path}")
    
    # Save as JSONL
    jsonl_path = output_dir / "sentiment_mini.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for text, label in data:
            json.dump({'text': text, 'label': label}, f)
            f.write('\n')
    print(f"  ✓ Created {jsonl_path}")
    
    return csv_path, jsonl_path


def create_ner_dataset(output_dir: Path, num_samples: int = 50):
    """Create NER dataset (JSONL format)."""
    print(f"Creating NER dataset ({num_samples} samples)...")
    
    label_map = {
        "O": 0,
        "LOCATION": 1,
        "PERSON": 2,
        "THREAT": 3,
        "ORGANIZATION": 4,
    }
    
    def create_ner_sample(template):
        """Convert entity template to token-level labels."""
        text = template["text"]
        tokens = text.split()
        labels = [0] * len(tokens)  # Start with all "O" labels
        
        # Simple tokenization - match entities to tokens
        for entity in template["entities"]:
            entity_text = entity["text"]
            entity_tokens = entity_text.split()
            label = label_map[entity["label"]]
            
            # Find the entity in the token list (simplified)
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if " ".join(tokens[i:i+len(entity_tokens)]).lower() == entity_text.lower():
                    labels[i] = label
                    break
        
        return {
            "text": text,
            "tokens": tokens,
            "labels": labels
        }
    
    # Generate samples with variation
    samples = []
    for _ in range(num_samples):
        template = random.choice(NER_SAMPLES)
        sample = create_ner_sample(template)
        samples.append(sample)
    
    # Save as JSONL
    jsonl_path = output_dir / "ner_mini.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')
    print(f"  ✓ Created {jsonl_path}")
    
    # Also save label map
    label_map_path = output_dir / "ner_label_map.json"
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2)
    print(f"  ✓ Created {label_map_path}")
    
    return jsonl_path


def create_agriculture_dataset(output_dir: Path, num_samples: int = 80):
    """Create agricultural disease classification dataset (CSV)."""
    print(f"Creating agriculture dataset ({num_samples} samples)...")
    
    half = num_samples // 2
    
    # Sample with replacement if needed
    healthy_samples = random.choices(AGRICULTURE_HEALTHY, k=half)
    disease_samples = random.choices(AGRICULTURE_DISEASE, k=half)
    
    # Combine and shuffle
    data = [(text, 0) for text in healthy_samples] + [(text, 1) for text in disease_samples]
    random.shuffle(data)
    
    # Save as CSV
    csv_path = output_dir / "agriculture_mini.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        writer.writerows(data)
    print(f"  ✓ Created {csv_path}")
    
    return csv_path


def create_qa_scoring_dataset(output_dir: Path, num_samples: int = 60):
    """Create QA scoring multi-head classification dataset (JSON)."""
    print(f"Creating QA scoring dataset ({num_samples} samples)...")
    
    qa_conversations = [
        "Agent: Hello, how can I help you? Customer: I need help with my account.",
        "Agent: Good morning! What can I do for you? Customer: My payment didn't go through.",
        "Agent: Hi there! How may I assist you today? Customer: I have a question about charges.",
        "Agent: Welcome! What brings you here? Customer: I need to update my information.",
        "Agent: Hello! How can I support you? Customer: I want to cancel my subscription.",
    ]
    
    samples = []
    for _ in range(num_samples):
        conversation = random.choice(qa_conversations)
        sample = {
            "text": conversation,
            "opening": random.randint(0, 1),
            "listening": random.randint(0, 4),
            "proactiveness": random.randint(0, 2),
            "resolution": random.randint(0, 4),
            "hold": random.randint(0, 1),
            "closing": random.randint(0, 1),
        }
        samples.append(sample)
    
    # Save as JSON
    json_path = output_dir / "qa_scoring_mini.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    print(f"  ✓ Created {json_path}")
    
    return json_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic test datasets")
    parser.add_argument("--all", action="store_true", help="Create all datasets")
    parser.add_argument("--sentiment", action="store_true", help="Create sentiment dataset")
    parser.add_argument("--ner", action="store_true", help="Create NER dataset")
    parser.add_argument("--agriculture", action="store_true", help="Create agriculture dataset")
    parser.add_argument("--qa", action="store_true", help="Create QA scoring dataset")
    parser.add_argument("--output-dir", type=str, default="tests/data",
                       help="Output directory for datasets")
    parser.add_argument("--num-samples", type=int, help="Override default number of samples")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  JENGA-AI SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}\n")
    
    # Determine which datasets to create
    create_all = args.all or not (args.sentiment or args.ner or args.agriculture or args.qa)
    
    created_files = []
    
    if create_all or args.sentiment:
        num_samples = args.num_samples if args.num_samples else 100
        files = create_sentiment_dataset(output_dir, num_samples)
        created_files.extend(files)
    
    if create_all or args.ner:
        num_samples = args.num_samples if args.num_samples else 50
        file = create_ner_dataset(output_dir, num_samples)
        created_files.append(file)
    
    if create_all or args.agriculture:
        num_samples = args.num_samples if args.num_samples else 80
        file = create_agriculture_dataset(output_dir, num_samples)
        created_files.append(file)
    
    if create_all or args.qa:
        num_samples = args.num_samples if args.num_samples else 60
        file = create_qa_scoring_dataset(output_dir, num_samples)
        created_files.append(file)
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Created {len(created_files)} file(s)")
    for file_path in created_files:
        print(f"    • {file_path}")
    print("=" * 70)
    print("\n✅ Synthetic data generation complete!\n")


if __name__ == "__main__":
    main()


