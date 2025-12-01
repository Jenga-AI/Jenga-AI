#!/usr/bin/env python3
"""
Enhanced YouTube Transcription with Audio Download
Supports multiple lightweight transcription options
"""

import subprocess
import json
import re
import os
from pathlib import Path
from typing import Optional, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import pandas as pd


def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|youtu\.be/|embed/|watch\?v=)([^&\n?#]+)',
        r'youtu\.be/([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_audio(video_url: str, video_id: str, output_dir: str = "./downloads") -> Optional[str]:
    """
    Download audio from YouTube video using yt-dlp.
    Returns path to downloaded audio file.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Download best audio quality in m4a format
        output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio[ext=m4a]/best',  # Best audio quality, prefer m4a
            '-o', output_template,
            '--no-playlist',
            video_url
        ]
        
        print(f"  🔄 Downloading audio...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Find the downloaded file
        audio_file = os.path.join(output_dir, f"{video_id}.m4a")
        if not os.path.exists(audio_file):
            # Try other common extensions
            for ext in ['webm', 'mp4', 'opus']:
                alt_file = os.path.join(output_dir, f"{video_id}.{ext}")
                if os.path.exists(alt_file):
                    audio_file = alt_file
                    break
        
        if os.path.exists(audio_file):
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            print(f"  ✅ Audio downloaded: {audio_file} ({file_size_mb:.2f} MB)")
            return audio_file
        else:
            print(f"  ❌ Audio file not found after download")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Download timeout (>5 minutes)")
        return None
    except Exception as e:
        print(f"  ❌ Download error: {e}")
        return None


def transcribe_with_vosk(audio_file: str, model_path: str = None) -> Optional[Dict]:
    """
    Transcribe audio using Vosk (lightweight offline model).
    Model size: ~50MB for small model, ~1.8GB for large model.
    Install: pip install vosk
    Download model: https://alphacephei.com/vosk/models
    """
    try:
        from vosk import Model, KaldiRecognizer
        import wave
        import subprocess
        
        # Convert to WAV if needed (Vosk requires WAV)
        wav_file = audio_file.replace(os.path.splitext(audio_file)[1], '.wav')
        if not os.path.exists(wav_file):
            print(f"  🔄 Converting to WAV...")
            cmd = ['ffmpeg', '-i', audio_file, '-ar', '16000', '-ac', '1', wav_file, '-y']
            subprocess.run(cmd, capture_output=True, timeout=60)
        
        # Use default model path if not provided
        if model_path is None:
            model_path = "./vosk-model-small-en-us-0.15"
        
        if not os.path.exists(model_path):
            print(f"  ⚠️  Vosk model not found at: {model_path}")
            print(f"      Download from: https://alphacephei.com/vosk/models")
            return None
        
        print(f"  🔄 Transcribing with Vosk...")
        model = Model(model_path)
        
        wf = wave.open(wav_file, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        
        transcript_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcript_text += result.get('text', '') + " "
        
        # Final result
        final_result = json.loads(rec.FinalResult())
        transcript_text += final_result.get('text', '')
        
        wf.close()
        
        return {
            'method': 'vosk',
            'text': transcript_text.strip(),
            'success': True
        }
        
    except ImportError:
        print(f"  ⚠️  Vosk not installed. Run: pip install vosk")
        return None
    except Exception as e:
        print(f"  ❌ Vosk transcription error: {e}")
        return None


def get_transcript_youtube_api(video_id: str) -> Optional[Dict]:
    """Try to get transcript using YouTube Transcript API."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([segment['text'] for segment in transcript_list])
        
        return {
            'method': 'youtube_api',
            'text': full_text,
            'segments': transcript_list,
            'success': True
        }
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"  ⚠️  YouTube API: No subtitles available")
        return None
    except Exception as e:
        print(f"  ❌ YouTube API error: {e}")
        return None


def get_video_metadata(video_url: str) -> Optional[Dict]:
    """Get video metadata using yt-dlp."""
    try:
        cmd = ['yt-dlp', '--dump-json', '--no-download', video_url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            return {
                'title': metadata.get('title', 'Unknown'),
                'channel': metadata.get('uploader', 'Unknown'),
                'duration': metadata.get('duration', 0),
                'upload_date': metadata.get('upload_date', 'Unknown'),
                'view_count': metadata.get('view_count', 0),
                'description': metadata.get('description', '')
            }
    except Exception as e:
        print(f"  ⚠️  Metadata extraction failed: {e}")
    return None


def transcribe_video(video_url: str, output_dir: str = "./downloads", 
                     use_vosk: bool = False, vosk_model_path: str = None) -> Dict:
    """
    Main transcription function.
    1. Try YouTube Transcript API (subtitles)
    2. Download audio
    3. Try Vosk if enabled
    """
    print(f"\n{'='*60}")
    print(f"🎥 Processing: {video_url}")
    print(f"{'='*60}")
    
    video_id = get_video_id(video_url)
    if not video_id:
        return {'url': video_url, 'success': False, 'error': 'Invalid YouTube URL'}
    
    print(f"📹 Video ID: {video_id}")
    
    # Get metadata
    print(f"\n📊 Fetching metadata...")
    metadata = get_video_metadata(video_url)
    if metadata:
        print(f"  ✅ Title: {metadata['title']}")
        print(f"  ✅ Channel: {metadata['channel']}")
        print(f"  ✅ Duration: {metadata['duration']} seconds ({metadata['duration']//60} min)")
    
    # Try subtitle-based transcription first
    print(f"\n🔍 Method 1: YouTube Transcript API (subtitles)...")
    transcript_result = get_transcript_youtube_api(video_id)
    
    audio_file = None
    
    # If no subtitles, download audio
    if not transcript_result:
        print(f"\n📥 No subtitles available. Downloading audio...")
        audio_file = download_audio(video_url, video_id, output_dir)
        
        # Try Vosk if enabled and audio downloaded
        if audio_file and use_vosk:
            print(f"\n🔍 Method 2: Vosk (offline transcription)...")
            transcript_result = transcribe_with_vosk(audio_file, vosk_model_path)
    
    # Prepare result
    if transcript_result and transcript_result.get('success'):
        result = {
            'url': video_url,
            'video_id': video_id,
            'success': True,
            'transcription_method': transcript_result['method'],
            'transcript': transcript_result['text'],
            'word_count': len(transcript_result['text'].split()),
            'audio_file': audio_file,
            **(metadata or {})
        }
        
        print(f"\n✅ SUCCESS!")
        print(f"  Method: {transcript_result['method']}")
        print(f"  Word count: {result['word_count']}")
        print(f"  Preview: {transcript_result['text'][:200]}...")
        
        return result
    else:
        # Audio downloaded but no transcription
        if audio_file:
            print(f"\n⚠️  Audio downloaded but not transcribed")
            print(f"  Audio file: {audio_file}")
            print(f"  You can manually process this audio file")
            
            return {
                'url': video_url,
                'video_id': video_id,
                'success': False,
                'audio_file': audio_file,
                'error': 'Audio downloaded but transcription not available (enable Vosk or use external tool)',
                **(metadata or {})
            }
        else:
            return {
                'url': video_url,
                'video_id': video_id,
                'success': False,
                'error': 'All methods failed',
                **(metadata or {})
            }


def main():
    """Main execution function."""
    test_video = "https://youtu.be/S5ybIhZwY8g?si=clzxS4_Y4JYmzeV6"
    output_dir = "./downloads"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try with Vosk disabled first (just download audio)
    print("="*60)
    print("Configuration:")
    print("  - Vosk transcription: DISABLED (set use_vosk=True to enable)")
    print("  - Audio download: ENABLED")
    print("="*60)
    
    result = transcribe_video(test_video, output_dir, use_vosk=False)
    
    # Save results
    df = pd.DataFrame([result])
    output_csv = os.path.join(output_dir, "kenyan_data.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Results saved to: {output_csv}")
    
    if result.get('audio_file'):
        print(f"\n📁 Audio file ready for processing: {result['audio_file']}")
        print(f"\n💡 Next steps:")
        print(f"   1. Install Vosk: pip install vosk")
        print(f"   2. Download model: https://alphacephei.com/vosk/models")
        print(f"   3. Run with use_vosk=True")
        print(f"\n   OR use online service like Google Speech-to-Text API")


if __name__ == "__main__":
    main()
