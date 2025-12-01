#!/usr/bin/env python3
"""
Simple Kenyan YouTube Content Collector
Uses YouTube search API patterns and transcript checking
No Selenium required - lighter and faster
"""

import requests
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
import re
from typing import List, Dict, Optional
from urllib.parse import quote


# Kenyan-focused search queries
KENYAN_QUERIES = [
    "Kenya podcast",
    "Kenyan politics discussion",
    "Nairobi news",
    "Kenya current affairs",
    "Kenyan interviews",
    "Iko Nini podcast",  # The channel from the test video
]


def search_youtube_simple(query: str, max_results: int = 20) -> List[str]:
    """
    Simple YouTube search using RSS/Atom feed approach.
    Returns list of video IDs.
    """
    # Try to use the search results page pattern
    # Note: This is a simplified version. For production, use YouTube API
    search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
    
    try:
        # Make request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        
        # Extract video IDs from the HTML
        video_ids = re.findall(r'"videoId":"([^"]{11})"', response.text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen and len(vid) == 11:
                seen.add(vid)
                unique_ids.append(vid)
                if len(unique_ids) >= max_results:
                    break
        
        return unique_ids
    
    except Exception as e:
        print(f"  ⚠️  Search error: {e}")
        return []


def get_video_info(video_id: str) -> Optional[Dict]:
    """Get video information and check for transcript."""
    try:
        # Try to get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([segment['text'] for segment in transcript_list])
        
        # Get basic info from YouTube
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        return {
            'video_id': video_id,
            'url': video_url,
            'transcript': transcript_text,
            'word_count': len(transcript_text.split()),
            'has_transcript': True
        }
    
    except Exception:
        return None


def collect_kenyan_content(
    queries: List[str] = None,
    videos_per_query: int = 10,
    min_words: int = 500
) -> pd.DataFrame:
    """
    Collect Kenyan YouTube content with transcripts.
    
    Args:
        queries: List of search queries (default: KENYAN_QUERIES)
        videos_per_query: Number of videos to collect per query
        min_words: Minimum word count for transcripts
    
    Returns:
        DataFrame with collected videos
    """
    if queries is None:
        queries = KENYAN_QUERIES[:3]  # Use first 3 by default
    
    print("="*70)
    print("🇰🇪 KENYAN CONTENT COLLECTOR (Simple Version)")
    print("="*70)
    
    all_videos = []
    
    for query in queries:
        print(f"\n🔍 Searching: '{query}'")
        
        # Search for videos
        video_ids = search_youtube_simple(query, max_results=30)
        print(f"  Found {len(video_ids)} video IDs")
        
        collected = 0
        for video_id in video_ids:
            if collected >= videos_per_query:
                break
            
            # Get video info and check transcript
            video_info = get_video_info(video_id)
            
            if video_info and video_info['word_count'] >= min_words:
                video_info['search_query'] = query
                all_videos.append(video_info)
                collected += 1
                print(f"  ✅ [{collected}/{videos_per_query}] Video {video_id}: {video_info['word_count']} words")
            
            time.sleep(0.5)  # Be nice to YouTube
        
        print(f"  📊 Collected {collected} videos from this query")
    
    if all_videos:
        df = pd.DataFrame(all_videos)
        df = df.drop_duplicates(subset=['video_id'], keep='first')
        
        print(f"\n{'='*70}")
        print(f"✅ COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total unique videos: {len(df)}")
        print(f"📊 Total words: {df['word_count'].sum():,}")
        
        return df
    else:
        print("\n❌ No videos collected")
        return pd.DataFrame()


def main():
    """Main execution."""
    df = collect_kenyan_content(
        queries=KENYAN_QUERIES[:2],  # Start with 2 queries
        videos_per_query=10,
        min_words=500
    )
    
    if not df.empty:
        # Save to CSV
        output_file = 'kenyan_youtube_data.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Saved to: {output_file}")
        
        # Save just URLs for reference
        urls_file = 'kenyan_video_urls.txt'
        with open(urls_file, 'w') as f:
            for idx, row in df.iterrows():
                f.write(f"{row['url']}\n")
        print(f"💾 URLs saved to: {urls_file}")
        
        # Show preview of first transcript
        if len(df) > 0:
            first_video = df.iloc[0]
            print(f"\n📝 Sample transcript preview (Video: {first_video['video_id']}):")
            print(f"   {first_video['transcript'][:300]}...")


if __name__ == "__main__":
    main()
