"""
Configuration management for synthetic data generation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "generated_data"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Data files
LOCATIONS_FILE = DATA_DIR / "kenyan_locations.json"
NAMES_FILE = DATA_DIR / "kenyan_names.json"
THREAT_CATEGORIES_FILE = DATA_DIR / "threat_categories.json"
SENTIMENT_LABELS_FILE = DATA_DIR / "sentiment_labels.json"
NER_ENTITY_TYPES_FILE = DATA_DIR / "ner_entity_types.json"
MAIN_CATEGORIES_FILE = DATA_DIR / "main_categories.json"
INTERVENTIONS_FILE = DATA_DIR / "interventions.json"

# Gemini API settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")  # or gemini-1.5-pro-latest or gemini-pro
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

# Generation settings
DEFAULT_NUM_SAMPLES = 100
BATCH_SIZE = 10  # Process in batches to manage rate limits

# Language settings
LANGUAGES = {
    "en": "English",
    "sw": "Swahili",
    "sheng": "Sheng (Kenyan slang)"
}

# Code-switching probability (for mixing languages)
CODE_SWITCH_PROBABILITY = 0.3  # 30% of samples will have code-switching

# Kenyan context phrases (for realistic data)
KENYAN_PHRASES = {
    "greetings": ["Habari", "Mambo", "Niaje", "Sasa", "Vipi"],
    "slang": ["maze", "poa", "sawa", "shida", "kizunguzungu"],
    "locations": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
    "common_words": ["boda boda", "matatu", "sukuma wiki", "ugali", "chai"]
}

# Task-specific settings
TASK_CONFIGS = {
    "classification": {
        "min_length": 50,
        "max_length": 300,
        "include_code_switching": True
    },
    "sentiment": {
        "min_length": 30,
        "max_length": 200,
        "include_code_switching": True
    },
    "ner": {
        "min_length": 50,
        "max_length": 250,
        "min_entities": 2,
        "max_entities": 8
    },
    "qa": {
        "context_min_length": 100,
        "context_max_length": 400,
        "question_min_length": 10,
        "question_max_length": 50
    },
    "translation": {
        "min_length": 20,
        "max_length": 150
    }
}

# Output formats
OUTPUT_FORMATS = {
    "classification": "jsonl",  # One JSON object per line
    "sentiment": "jsonl",
    "ner": "jsonl",
    "qa": "jsonl",
    "translation": "jsonl"
}


def get_output_path(task_name: str, num_samples: int, version: str = "v1") -> Path:
    """
    Get output file path for a task
    
    Args:
        task_name: Name of the task (classification, ner, etc.)
        num_samples: Number of samples generated
        version: Version string
    
    Returns:
        Path to output file
    """
    ext = OUTPUT_FORMATS.get(task_name, "jsonl")
    filename = f"{task_name}_synthetic_{num_samples}samples_{version}.{ext}"
    return OUTPUT_DIR / filename


def load_json_config(file_path: Path) -> Dict:
    """
    Load JSON configuration file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded JSON data
    """
    import json
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
