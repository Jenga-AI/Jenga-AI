# Quick Start Guide: Synthetic Data Generation

## What Was Created

A complete synthetic data generation system for JengaAI using Google Gemini API.

### New Files

**Core Infrastructure**:
- `gemini_client.py` - Gemini API client with rate limiting and retry logic
- `config.py` - Centralized configuration management
- `.env.template` - Environment variable template

**Category Definitions** (`data/`):
- `threat_categories.json` - 16 threat categories (Cyber Attack, Terrorism, etc.)
- `sentiment_labels.json` - 4 sentiment types (Positive, Negative, Neutral, Mixed)
- `ner_entity_types.json` - 20+ entity types (PERSON, VICTIM, INCIDENT_TYPE, etc.)

**Generators**:
- `generate_classification.py` - Threat detection classification data
- `generate_sentiment.py` - Sentiment analysis data
- `generate_ner.py` - Named entity recognition data

**Documentation**:
- `README.md` - Complete usage guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
cd /Users/naynek/Desktop/MultiClassifier/Jenga-AI/scripts/synthetic
pip install google-generativeai python-dotenv tenacity
```

### 2. Configure API Key
```bash
# Copy template
cp .env.template .env

# Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
nano .env  # or use any text editor
```

## Quick Test (2 minutes)

Generate 10 samples to test the system:

```bash
# Test classification generator
python generate_classification.py --num-samples 10

# Test sentiment generator
python generate_sentiment.py --num-samples 10

# Test NER generator
python generate_ner.py --num-samples 10 --entity-set security
```

Check output in `generated_data/` folder.

## Production Use

### Generate Training Datasets

```bash
# Threat classification (500 samples, 30% code-switching)
python generate_classification.py --num-samples 500 --code-switch-prob 0.3

# Sentiment analysis (300 samples, 40% code-switching)
python generate_sentiment.py --num-samples 300 --code-switch-prob 0.4

# NER security entities (200 samples, 5 entities per sample)
python generate_ner.py --num-samples 200 --entity-set security --entities-per-sample 5
```

### Use with JengaAI

Update `hackathon_mvp.yaml`:

```yaml
tasks:
  - name: "ThreatClassification"
    type: "classification"
    data_path: "scripts/synthetic/generated_data/threat_classification_500samples_v1.jsonl"
    
  - name: "ThreatNER"
    type: "ner"
    data_path: "scripts/synthetic/generated_data/ner_200samples_v1.jsonl"
```

Then run training:
```bash
python run_hackathon_mvp.py
```

## Features

✅ **Multi-Task Support**: Classification, Sentiment, NER  
✅ **Kenyan Context**: Swahili-English code-switching, Sheng slang  
✅ **Gemini-Powered**: Uses Google's latest Gemini 1.5 models  
✅ **Rate Limiting**: Automatic handling of API limits (15 RPM free tier)  
✅ **Retry Logic**: Automatic retry with exponential backoff  
✅ **Configurable**: Easy customization of all parameters  

## Next Steps

1. **Get API Key**: Visit https://makersuite.google.com/app/apikey
2. **Test System**: Run quick test commands above
3. **Generate Data**: Create production datasets
4. **Train Models**: Use generated data with JengaAI framework

## Troubleshooting

**"GEMINI_API_KEY not found"**
→ Ensure `.env` file exists with your API key

**Rate limit errors**
→ System handles this automatically, just wait

**Poor quality output**
→ Try `gemini-1.5-pro` instead of `gemini-1.5-flash` in `.env`

For full documentation, see `README.md`.
