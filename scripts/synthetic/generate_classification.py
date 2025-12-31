"""
Generate synthetic classification data for threat detection using Gemini API
Supports: Threat vs Non-Threat, Multi-class threat categories
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from gemini_client import create_client
from config import (
    THREAT_CATEGORIES_FILE, LOCATIONS_FILE, NAMES_FILE,
    get_output_path, load_json_config, KENYAN_PHRASES, CODE_SWITCH_PROBABILITY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()


class ThreatClassificationGenerator:
    """Generate synthetic threat classification data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = create_client(api_key)
        self.threat_categories = load_json_config(THREAT_CATEGORIES_FILE)
        
        # Load Kenyan context data
        try:
            self.locations = load_json_config(LOCATIONS_FILE)
            self.names = load_json_config(NAMES_FILE)
        except FileNotFoundError:
            logging.warning("⚠️ Location/names files not found. Using defaults.")
            self.locations = []
            self.names = {"Male_names": {"first": ["John"], "last": ["Doe"]}, 
                         "Female_names": {"first": ["Jane"], "last": ["Doe"]}}
    
    def _create_prompt(self, category: str, use_code_switching: bool = False) -> str:
        """Create prompt for generating threat classification example"""
        
        category_def = self.threat_categories[category]
        
        # Randomly select Kenyan context
        kenyan_context = ""
        if random.random() < 0.7:  # 70% include Kenyan context
            location = random.choice(KENYAN_PHRASES["locations"])
            kenyan_context = f"Set the context in Kenya, potentially mentioning {location}."
        
        code_switch_instruction = ""
        if use_code_switching:
            code_switch_instruction = """
Include natural code-switching between English and Swahili/Sheng. Examples:
- "Maze, kuna shida kubwa hapa" (Man, there's a big problem here)
- "Niaje, tumepata information" (Hey, we got information)
- "Sasa, twende tu" (Now, let's just go)
Use Kenyan slang naturally: poa (cool), sawa (okay), shida (problem), maze (man)
"""
        
        prompt = f"""Generate a realistic text example for threat detection classification.

Category: {category}
Definition: {category_def}

Requirements:
1. Create a SHORT text (50-200 words) that clearly represents this category
2. Make it sound like real communication: social media post, message, or conversation
3. {kenyan_context}
4. {code_switch_instruction if use_code_switching else "Use clear English."}
5. Make it realistic and contextually appropriate for Kenya
6. DO NOT label it or add metadata - just the raw text

Output ONLY the text content, nothing else."""

        return prompt
    
    def generate_sample(self, category: str, use_code_switching: bool = False) -> Dict:
        """
        Generate a single classification sample
        
        Returns:
            {"text": "...", "label": "category"}
        """
        prompt = self._create_prompt(category, use_code_switching)
        
        try:
            text = self.client.generate(prompt, temperature=0.8)
            
            # Clean up the text
            text = text.strip().strip('"').strip("'")
            
            return {
                "text": text,
                "label": category
            }
        except Exception as e:
            logging.error(f"❌ Failed to generate sample for {category}: {e}")
            return {"text": "", "label": category}
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        balance_classes: bool = True,
        code_switch_prob: float = CODE_SWITCH_PROBABILITY
    ) -> List[Dict]:
        """
        Generate a balanced dataset of threat classification examples
        
        Args:
            num_samples: Total number of samples to generate
            balance_classes: Whether to balance samples across categories
            code_switch_prob: Probability of code-switching (0.0-1.0)
        
        Returns:
            List of {"text": "...", "label": "..."} dictionaries
        """
        dataset = []
        categories = list(self.threat_categories.keys())
        
        if balance_classes:
            samples_per_category = num_samples // len(categories)
            remainder = num_samples % len(categories)
        else:
            samples_per_category = None
        
        logging.info(f"🚀 Generating {num_samples} threat classification samples...")
        
        for i, category in enumerate(categories):
            if balance_classes:
                n = samples_per_category + (1 if i < remainder else 0)
            else:
                n = num_samples // len(categories)
            
            logging.info(f"📝 Generating {n} samples for '{category}'...")
            
            for j in range(n):
                use_code_switching = random.random() < code_switch_prob
                sample = self.generate_sample(category, use_code_switching)
                
                if sample["text"]:  # Only add if generation succeeded
                    dataset.append(sample)
                
                if (j + 1) % 5 == 0:
                    logging.info(f"   Progress: {j+1}/{n}")
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logging.info(f"✅ Generated {len(dataset)} samples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: Optional[Path] = None):
        """Save dataset to JSONL file"""
        if output_path is None:
            output_path = get_output_path("threat_classification", len(dataset))
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logging.info(f"💾 Saved dataset to {output_path}")
        
        # Print statistics
        label_counts = {}
        for sample in dataset:
            label = sample["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logging.info("\n📊 Dataset Statistics:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"   {label}: {count} ({count/len(dataset)*100:.1f}%)")


def main():
    """Main function to generate threat classification dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic threat classification data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--no-balance", action="store_true", help="Don't balance classes")
    parser.add_argument("--code-switch-prob", type=float, default=0.3, help="Code-switching probability")
    
    args = parser.parse_args()
    
    generator = ThreatClassificationGenerator()
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        balance_classes=not args.no_balance,
        code_switch_prob=args.code_switch_prob
    )
    
    output_path = Path(args.output) if args.output else None
    generator.save_dataset(dataset, output_path)


if __name__ == "__main__":
    main()
