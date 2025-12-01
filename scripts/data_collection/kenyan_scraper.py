#!/usr/bin/env python3
"""
Kenyan YouTube Content Scraper
Finds Kenyan podcasts and videos WITH subtitles/transcripts available
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
import re
import random
from typing import List, Dict, Optional


# List of user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Kenyan-focused search queries
KENYAN_QUERIES = [
    "Kenya podcast",
    "Kenyan politics discussion",
    "Nairobi news analysis",
    "Kenya current affairs",
    "Kenyan interviews",
    "Kenya business podcast",
    "East Africa news Kenya",
    "Kenyan media interviews",
    "Kenya parliamentary debates",
    "Kenyan social issues",
]


def get_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    pattern = r'(?:v=|youtu\.be/|embed/|watch\?v=)([^&\n?#]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def check_transcript_available(video_id: str) -> bool:
    """Check if a video has transcripts available."""
    try:
        YouTubeTranscriptApi.get_transcript(video_id)
        return True
    except:
        return False


def get_transcript(video_id: str) -> Optional[str]:
    """Get transcript text for a video."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment['text'] for segment in transcript_list])
    except:
        return None


def scrape_kenyan_videos(
    search_query: str,
    max_videos: int = 20,
    scroll_count: int = 3,
    min_duration: int = 300,  # 5 minutes
    max_duration: int = 1800,  # 30 minutes
    headless: bool = True
) -> List[Dict]:
    """
    Scrape Kenyan YouTube videos and filter for those with transcripts.
    
    Args:
        search_query: YouTube search query
        max_videos: Maximum number of videos to collect
        scroll_count: Number of times to scroll down
        min_duration: Minimum video duration in seconds
        max_duration: Maximum video duration in seconds
        headless: Run browser in headless mode
    
    Returns:
        List of dictionaries containing video data
    """
    print(f"\n🔍 Searching for: '{search_query}'")
    print(f"  Target: {max_videos} videos with transcripts")
    print(f"  Duration filter: {min_duration//60}-{max_duration//60} minutes")
    
    # Setup Chrome driver
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Try to find chromedriver
    try:
        service = Service(executable_path="/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except:
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"❌ Failed to initialize Chrome driver: {e}")
            return []
    
    # Build search URL
    search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    driver.get(search_url)
    time.sleep(3)
    
    # Scroll to load more videos
    print(f"  📜 Scrolling to load videos...")
    for i in range(scroll_count):
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
    
    # Extract video information
    print(f"  🎥 Extracting video data...")
    video_elements = driver.find_elements(By.XPATH, '//a[contains(@href, "/watch?v=") and @id="video-title"]')
    
    collected_videos = []
    checked_count = 0
    
    for element in video_elements:
        if len(collected_videos) >= max_videos:
            break
        
        try:
            video_url = element.get_attribute('href')
            video_title = element.get_attribute('title')
            
            if not video_url or not video_title:
                continue
            
            video_id = get_video_id(video_url)
            if not video_id:
                continue
            
            checked_count += 1
            
            # Check if transcript is available
            if not check_transcript_available(video_id):
                continue
            
            # Get transcript
            transcript = get_transcript(video_id)
            if not transcript or len(transcript) < 100:  # Skip very short transcripts
                continue
            
            # Estimate duration from transcript length (rough estimate)
            word_count = len(transcript.split())
            estimated_duration = word_count * 0.4  # Rough: 150 words/minute = 0.4 sec/word
            
            # Apply duration filter (approximate)
            if estimated_duration < min_duration or estimated_duration > max_duration:
                continue
            
            video_data = {
                'video_id': video_id,
                'url': video_url,
                'title': video_title,
                'search_query': search_query,
                'transcript': transcript,
                'word_count': word_count,
                'estimated_duration_seconds': int(estimated_duration),
                'has_transcript': True
            }
            
            collected_videos.append(video_data)
            print(f"  ✅ [{len(collected_videos)}/{max_videos}] {video_title[:60]}... ({word_count} words)")
            
        except Exception as e:
            print(f"  ⚠️  Error processing video: {e}")
            continue
    
    driver.quit()
    
    print(f"\n📊 Results: Found {len(collected_videos)} videos with transcripts (checked {checked_count} total)")
    return collected_videos


def main():
    """Main execution function."""
    print("="*70)
    print("🇰🇪 KENYAN CONTENT SCRAPER")
    print("="*70)
    print("\nThis script searches for Kenyan YouTube content with available transcripts")
    print("(no heavy transcription needed - uses existing subtitles only)\n")
    
    all_videos = []
    
    # Use first few queries as example
    queries_to_use = KENYAN_QUERIES[:3]  # Start with 3 queries
    
    for query in queries_to_use:
        videos = scrape_kenyan_videos(
            search_query=query,
            max_videos=10,  # 10 videos per query
            scroll_count=3,
            min_duration=600,  # 10 minutes minimum
            max_duration=1800,  # 30 minutes maximum
            headless=True
        )
        all_videos.extend(videos)
        time.sleep(2)  # Be nice to YouTube
    
    # Save to CSV
    if all_videos:
        df = pd.DataFrame(all_videos)
        
        # Remove duplicates by video_id
        df = df.drop_duplicates(subset=['video_id'], keep='first')
        
        output_file = 'kenyan_youtube_data.csv'
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS!")
        print(f"{'='*70}")
        print(f"📊 Total unique videos collected: {len(df)}")
        print(f"📊 Total words: {df['word_count'].sum():,}")
        print(f"💾 Saved to: {output_file}")
        print(f"\n📋 Sample titles:")
        for i, row in df.head(5).iterrows():
            print(f"  {i+1}. {row['title'][:65]}...")
    else:
        print("\n❌ No videos with transcripts found")


if __name__ == "__main__":
    main()
