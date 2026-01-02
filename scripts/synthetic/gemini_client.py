"""
Gemini API Client for Synthetic Data Generation
Handles all interactions with Google's Gemini API
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Gemini model to use (gemini-pro, gemini-1.5-pro, etc.)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it as environment variable or pass to constructor.")
        
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Rate limiting (15 RPM for free tier)
        self.requests_per_minute = 15
        self.request_timestamps = []
        
        logging.info(f"✅ Gemini client initialized with model: {model_name}")
    
    def _check_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        # If we've hit the limit, wait
        if len(self.request_timestamps) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logging.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_timestamps = []
        
        self.request_timestamps.append(current_time)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API with retry logic
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        self._check_rate_limit()
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                logging.error("❌ Empty response from Gemini API")
                return ""
            
            logging.info("✅ Generated response from Gemini")
            return response.text.strip()
            
        except Exception as e:
            logging.error(f"❌ Gemini API error: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            show_progress: Show progress logging
        
        Returns:
            List of generated texts
        """
        results = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts, 1):
            if show_progress:
                logging.info(f"📝 Generating {i}/{total}...")
            
            try:
                result = self.generate(prompt, temperature, max_tokens)
                results.append(result)
            except Exception as e:
                logging.error(f"❌ Failed to generate for prompt {i}: {e}")
                results.append("")  # Append empty string on failure
        
        return results
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
        
        Returns:
            Token count
        """
        try:
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            logging.warning(f"⚠️ Token counting failed: {e}")
            # Rough estimate: 1 token ≈ 4 characters
            return len(text) // 4


# Convenience function for quick usage
def create_client(api_key: Optional[str] = None, model: str = "gemini-pro") -> GeminiClient:
    """
    Create a Gemini client
    
    Args:
        api_key: Google API key
        model: Model name
    
    Returns:
        GeminiClient instance
    """
    return GeminiClient(api_key=api_key, model_name=model)
