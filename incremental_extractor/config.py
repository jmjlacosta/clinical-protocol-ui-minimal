"""
Configuration settings for the incremental extractor.
"""

import os
from typing import Optional

class ExtractorConfig:
    """Configuration for the incremental extractor"""
    
    def __init__(self):
        # API Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Model Configuration
        self.extraction_model = "gpt-3.5-turbo"
        self.chunk_analysis_model = "gpt-3.5-turbo-0125"
        
        # Chunking Configuration
        self.use_intelligent_chunking = self._get_bool_env("USE_INTELLIGENT_CHUNKING", False)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "50000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "1000"))
        
        # Extraction Configuration
        self.max_single_field_chars = 48000  # Safe limit for GPT-3.5
        self.max_batch_field_chars = 48000
        self.outcome_extraction_chars = 250000  # More text for outcomes
        
        # Feature Flags
        self.enable_smart_validation = True
        self.enable_filename_extraction = True
        self.enable_comparison = True
        
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, "").lower()
        if value in ["true", "1", "yes", "on"]:
            return True
        elif value in ["false", "0", "no", "off"]:
            return False
        return default
    
    def get_chunk_config(self) -> dict:
        """Get chunking configuration"""
        return {
            "enabled": self.use_intelligent_chunking,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model": self.chunk_analysis_model
        }
    
    def enable_intelligent_chunking(self):
        """Enable intelligent chunking"""
        self.use_intelligent_chunking = True
        
    def disable_intelligent_chunking(self):
        """Disable intelligent chunking"""
        self.use_intelligent_chunking = False

# Global configuration instance
config = ExtractorConfig()