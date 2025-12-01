# Kenyan YouTube Data Collection Scripts

## 📝 Quick Start

This directory contains scripts for collecting Kenyan YouTube content for training AI models.

## 🔑 Key Finding
**Most Kenyan YouTube content does NOT have subtitles/transcripts** - you'll need actual speech-to-text transcription.

## 🛠️ Scripts Available

### 1. `simple_kenyan_collector.py` - Quick Test
Lightweight script to find Kenyan videos and check for subtitles.
```bash
python3 simple_kenyan_collector.py
```

### 2. `lightweight_transcriber.py` - Single Video
Process one video at a time with multiple fallback methods.
```bash
python3 lightweight_transcriber.py
```

### 3. `kenyan_scraper.py` - Bulk Collection
Selenium-based scraper for large-scale collection.
```bash
python3 kenyan_scraper.py
```

## 💡 Recommended Next Steps

Since Kenyan videos lack subtitles, choose one:

### Option A: Vosk (Lightweight, Offline)
```bash
# Install
pip install vosk

# Download 50MB model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Modify lightweight_transcriber.py to use Vosk
```

### Option B: AssemblyAI (Best Quality, Cheap)
```bash
# Install
pip install assemblyai

# Get API key from  https://www.assemblyai.com/
# Cost: ~$0.25 per audio hour
```

### Option C: Find Channels with Subtitles
Manually curate channels that provide subtitles consistently.

## 📦 Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Test Results
- **Videos tested**: 60+
- **With transcripts**: 0
- **Conclusion**: Need actual speech-to-text

## 🎯 Example Kenyan Queries
- "Kenya podcast"
- "Iko Nini podcast"
- "Nairobi news"
- "Kenyan politics discussion"
- "Kenya current affairs"

## ⚠️ Known Issues
- yt-dlp audio download blocked (403 errors)
- Most videos lack subtitles
- Need authentication/cookies for some downloads
