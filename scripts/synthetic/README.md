# Synthetic Data Generation with Gemini API

Generate high-quality synthetic datasets for multiple NLP tasks using Google's Gemini API.

## Features

- **Multi-Task Support**: Classification, Sentiment, NER, QA, Translation
- **Kenyan Context**: Swahili-English code-switching, Sheng slang, local names and locations
- **Gemini-Powered**: Uses Google's Gemini 1.5 Flash/Pro for generation
- **Configurable**: Easy customization of categories, entity types, and generation parameters

## Setup

### 1. Install Dependencies

```bash
pip install google-generativeai python-dotenv tenacity
```

### 2. Get Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Copy `.env.template` to `.env`
4. Add your API key to `.env`:

```bash
cp .env.template .env
# Edit .env and add: GEMINI_API_KEY=your_actual_key_here
```

## Usage

### Threat Classification

Generate threat detection training data:

```bash
python generate_classification.py --num-samples 200 --code-switch-prob 0.4
```

**Output**: `generated_data/threat_classification_200samples_v1.jsonl`

**Format**:
```json
{"text": "Kuna shida kubwa hapa, someone is planning an attack", "label": "Physical Threat"}
{"text": "Just had the best ugali at Mama Oliech's!", "label": "Non-Threat"}
```

### Sentiment Analysis

Generate sentiment analysis data:

```bash
python generate_sentiment.py --num-samples 150
```

**Output**: `generated_data/sentiment_150samples_v1.jsonl`

**Format**:
```json
{"text": "M-Pesa service ni poa sana! Very reliable", "sentiment": "positive"}
{"text": "Frustrated with this system, haiwezi work", "sentiment": "negative"}
```

### Named Entity Recognition (NER)

Generate NER training data:

```bash
python generate_ner.py --num-samples 100 --entity-set security --entities-per-sample 5
```

**Entity Sets**:
- `general`: PERSON, ORGANIZATION, LOCATION, DATE, etc.
- `security`: VICTIM, PERPETRATOR, INCIDENT_TYPE, WEAPON, etc.
- `kenyan_specific`: COUNTY, SUBCOUNTY, WARD, LANDMARK, etc.
- `all`: All entity types combined

**Output**: `generated_data/ner_100samples_v1.jsonl`

**Format**:
```json
{
  "text": "John Kamau reported a robbery in Nairobi CBD yesterday.",
  "entities": [
    {"text": "John Kamau", "label": "PERSON", "start": 0, "end": 10},
    {"text": "robbery", "label": "INCIDENT_TYPE", "start": 23, "end": 30},
    {"text": "Nairobi CBD", "label": "LOCATION", "start": 34, "end": 45}
  ]
}
```

## Configuration

### Environment Variables (`.env`)

```bash
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=2048
```

### Category Files (`data/`)

- `threat_categories.json`: Threat detection categories
- `sentiment_labels.json`: Sentiment labels
- `ner_entity_types.json`: NER entity type definitions
- `kenyan_locations.json`: Kenyan geographic data
- `kenyan_names.json`: Kenyan names database

## Advanced Usage

### Custom Categories

Edit `data/threat_categories.json` to add your own categories:

```json
{
  "Custom Category": "Description of what this category represents",
  "Another Category": "Another description"
}
```

### Batch Generation

Generate multiple datasets at once:

```bash
# Classification
python generate_classification.py --num-samples 500

# Sentiment
python generate_sentiment.py --num-samples 500

# NER
python generate_ner.py --num-samples 300 --entity-set all
```

### Code-Switching Control

Adjust the probability of Swahili-English code-switching:

```bash
# More code-switching (60%)
python generate_classification.py --code-switch-prob 0.6

# Less code-switching (10%)
python generate_sentiment.py --code-switch-prob 0.1

# No code-switching (0%)
python generate_ner.py --code-switch-prob 0.0
```

## Output Structure

All generated datasets are saved to `generated_data/` in JSONL format (one JSON object per line).

```
generated_data/
├── threat_classification_200samples_v1.jsonl
├── sentiment_150samples_v1.jsonl
├── ner_100samples_v1.jsonl
└── ...
```

## Tips for Best Results

1. **Start Small**: Test with 10-20 samples first to verify quality
2. **Monitor Rate Limits**: Free tier has 15 requests/minute limit
3. **Adjust Temperature**: Lower (0.3-0.5) for consistent output, higher (0.7-0.9) for variety
4. **Review Output**: Always manually review a sample of generated data
5. **Balance Classes**: Use `--no-balance` flag if you want unbalanced datasets

## Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure `.env` file exists and contains your API key
- Check that you're running from the `scripts/synthetic/` directory

### Rate Limit Errors
- The client automatically handles rate limiting
- If you hit limits, reduce `--num-samples` or wait a few minutes

### Poor Quality Output
- Try adjusting `GEMINI_TEMPERATURE` in `.env`
- Use `gemini-1.5-pro` instead of `gemini-1.5-flash` for better quality
- Review and refine category descriptions in JSON files

## Examples

### Generate Balanced Threat Detection Dataset
```bash
python generate_classification.py \
  --num-samples 500 \
  --code-switch-prob 0.3 \
  --output my_threat_data.jsonl
```

### Generate Security-Focused NER Data
```bash
python generate_ner.py \
  --num-samples 200 \
  --entity-set security \
  --entities-per-sample 6 \
  --code-switch-prob 0.2
```

### Generate Mixed-Sentiment Social Media Data
```bash
python generate_sentiment.py \
  --num-samples 300 \
  --code-switch-prob 0.5
```

## Integration with JengaAI

Use generated datasets directly with JengaAI framework:

```yaml
# hackathon_mvp.yaml
tasks:
  - name: "ThreatClassification"
    type: "classification"
    data_path: "scripts/synthetic/generated_data/threat_classification_500samples_v1.jsonl"
    
  - name: "ThreatNER"
    type: "ner"
    data_path: "scripts/synthetic/generated_data/ner_200samples_v1.jsonl"
```

## License

MIT License - Feel free to use and modify for your projects!
