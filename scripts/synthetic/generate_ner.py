"""
Generate synthetic NER (Named Entity Recognition) data using Gemini API
Supports: PERSON, ORGANIZATION, LOCATION, VICTIM, PERPETRATOR, INCIDENT_TYPE, etc.
"""

import json
import random
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from gemini_client import create_client
from config import (
    NER_ENTITY_TYPES_FILE, get_output_path, load_json_config,
    KENYAN_PHRASES, CODE_SWITCH_PROBABILITY
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class NERGenerator:
    """Generate synthetic NER data"""
    
    def __init__(self, api_key: Optional[str] = None, entity_set: str = "security"):
        """
        Args:
            api_key: Gemini API key
            entity_set: Which entity set to use (general, security, kenyan_specific, or all)
        """
        self.client = create_client(api_key)
        entity_types_data = load_json_config(NER_ENTITY_TYPES_FILE)
        
        # Select entity types based on entity_set
        if entity_set == "all":
            self.entity_types = {}
            for category in entity_types_data.values():
                self.entity_types.update(category)
        elif entity_set in entity_types_data:
            self.entity_types = entity_types_data[entity_set]
        else:
            # Combine general and specified set
            self.entity_types = {**entity_types_data.get("general_entities", {}),
                                **entity_types_data.get(f"{entity_set}_entities", {})}
    
    def _create_prompt(self, entity_types: List[str], use_code_switching: bool = False) -> str:
        """Create prompt for generating NER example"""
        
        entity_descriptions = "\n".join([
            f"- {etype}: {self.entity_types[etype]}"
            for etype in entity_types
        ])
        
        scenarios = [
            "a security incident report",
            "a news article about a crime",
            "a witness statement",
            "a police report",
            "a social media post about an event",
            "a conversation about a local incident"
        ]
        scenario = random.choice(scenarios)
        
        code_switch_instruction = ""
        if use_code_switching:
            code_switch_instruction = "Include natural Swahili-English code-switching."
        
        prompt = f"""Generate a realistic Kenyan text for named entity recognition training.

Scenario: {scenario}
Required Entity Types: {', '.join(entity_types)}

Entity Definitions:
{entity_descriptions}

Requirements:
1. Write 50-200 words of realistic Kenyan text
2. Include AT LEAST one example of EACH required entity type
3. Make entities clearly identifiable but natural
4. {code_switch_instruction if use_code_switching else "Use clear English."}
5. Set the context in Kenya (mention Kenyan locations, names, etc.)

Output Format:
First, output the text.
Then, on new lines, list each entity in this EXACT format:
ENTITY: [entity text] | TYPE: [entity type] | START: [character position] | END: [character position]

Example:
John Kamau reported the incident in Nairobi yesterday.
ENTITY: John Kamau | TYPE: PERSON | START: 0 | END: 10
ENTITY: Nairobi | TYPE: LOCATION | START: 35 | END: 42

Now generate the text and entities:"""

        return prompt
    
    def _parse_entities(self, text: str, entity_lines: List[str]) -> List[Dict]:
        """Parse entity annotations from LLM output"""
        entities = []
        
        for line in entity_lines:
            if not line.strip() or "ENTITY:" not in line:
                continue
            
            try:
                # Parse format: ENTITY: text | TYPE: type | START: pos | END: pos
                parts = line.split("|")
                entity_text = parts[0].split("ENTITY:")[1].strip()
                entity_type = parts[1].split("TYPE:")[1].strip()
                start = int(parts[2].split("START:")[1].strip())
                end = int(parts[3].split("END:")[1].strip())
                
                # Validate that entity exists in text at specified position
                if text[start:end] == entity_text or entity_text in text:
                    # If positions don't match, find correct position
                    if text[start:end] != entity_text:
                        start = text.find(entity_text)
                        if start != -1:
                            end = start + len(entity_text)
                        else:
                            continue  # Skip if entity not found
                    
                    entities.append({
                        "text": entity_text,
                        "label": entity_type,
                        "start": start,
                        "end": end
                    })
            except (IndexError, ValueError) as e:
                logging.warning(f"⚠️ Failed to parse entity line: {line} - {e}")
                continue
        
        return entities
    
    def generate_sample(
        self,
        num_entity_types: int = 3,
        use_code_switching: bool = False
    ) -> Dict:
        """Generate a single NER sample"""
        
        # Randomly select entity types
        available_types = list(self.entity_types.keys())
        selected_types = random.sample(available_types, min(num_entity_types, len(available_types)))
        
        prompt = self._create_prompt(selected_types, use_code_switching)
        
        try:
            response = self.client.generate(prompt, temperature=0.7, max_tokens=1024)
            
            # Split response into text and entity annotations
            lines = response.strip().split("\n")
            
            # Find where entities start (after the text)
            text_lines = []
            entity_lines = []
            in_entities = False
            
            for line in lines:
                if "ENTITY:" in line:
                    in_entities = True
                
                if in_entities:
                    entity_lines.append(line)
                else:
                    text_lines.append(line)
            
            text = "\n".join(text_lines).strip()
            entities = self._parse_entities(text, entity_lines)
            
            return {
                "text": text,
                "entities": entities
            }
            
        except Exception as e:
            logging.error(f"❌ Failed to generate NER sample: {e}")
            return {"text": "", "entities": []}
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        entities_per_sample: int = 4,
        code_switch_prob: float = CODE_SWITCH_PROBABILITY
    ) -> List[Dict]:
        """Generate NER dataset"""
        dataset = []
        
        logging.info(f"🚀 Generating {num_samples} NER samples...")
        
        for i in range(num_samples):
            use_code_switching = random.random() < code_switch_prob
            sample = self.generate_sample(entities_per_sample, use_code_switching)
            
            if sample["text"] and sample["entities"]:
                dataset.append(sample)
            
            if (i + 1) % 5 == 0:
                logging.info(f"   Progress: {i+1}/{num_samples}")
        
        logging.info(f"✅ Generated {len(dataset)} samples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: Optional[Path] = None):
        """Save dataset to JSONL file"""
        if output_path is None:
            output_path = get_output_path("ner", len(dataset))
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        logging.info(f"💾 Saved dataset to {output_path}")
        
        # Statistics
        entity_type_counts = {}
        total_entities = 0
        for sample in dataset:
            for entity in sample["entities"]:
                etype = entity["label"]
                entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1
                total_entities += 1
        
        logging.info(f"\n📊 Dataset Statistics:")
        logging.info(f"   Total entities: {total_entities}")
        logging.info(f"   Avg entities per sample: {total_entities/len(dataset):.1f}")
        logging.info(f"\n   Entity type distribution:")
        for etype, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"      {etype}: {count} ({count/total_entities*100:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic NER data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--entity-set", type=str, default="security", 
                       choices=["general", "security", "kenyan_specific", "all"],
                       help="Which entity set to use")
    parser.add_argument("--entities-per-sample", type=int, default=4, help="Target entities per sample")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--code-switch-prob", type=float, default=0.3, help="Code-switching probability")
    
    args = parser.parse_args()
    
    generator = NERGenerator(entity_set=args.entity_set)
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        entities_per_sample=args.entities_per_sample,
        code_switch_prob=args.code_switch_prob
    )
    
    output_path = Path(args.output) if args.output else None
    generator.save_dataset(dataset, output_path)


if __name__ == "__main__":
    main()
