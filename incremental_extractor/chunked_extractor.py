"""Chunked extraction strategy for finding fields in large PDFs"""
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class ChunkedExtractor:
    """Extract fields by searching through document chunks"""
    
    def __init__(self, chunk_size: int = 48000, overlap: int = 2000):
        """
        Initialize chunked extractor
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks to avoid missing info at boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def get_chunks(self, text: str) -> List[Tuple[int, str]]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of (start_position, chunk_text) tuples
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append((start, chunk))
            
            # Move to next chunk with overlap
            start = end - self.overlap
            
            # Don't create tiny final chunks
            if len(text) - start < self.overlap:
                break
                
        return chunks
    
    def extract_with_chunks(self, text: str, field_name: str, 
                          extract_func, max_chunks: int = 5) -> Optional[str]:
        """
        Try to extract field from multiple chunks
        
        Args:
            text: Full document text
            field_name: Field to extract
            extract_func: Function to extract from a chunk
            max_chunks: Maximum chunks to try before giving up
            
        Returns:
            Extracted value or None
        """
        chunks = self.get_chunks(text)
        
        # Special handling for different field types
        if field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
            # Outcomes often appear in middle/later sections
            # Start from chunk 2 (skip intro)
            start_idx = min(1, len(chunks) - 1)
        elif field_name in ["study_title", "sponsor", "nct_number"]:
            # These appear early
            start_idx = 0
        elif field_name in ["locations", "enrollment", "start_date"]:
            # These often appear later
            start_idx = min(2, len(chunks) - 1)
        else:
            # Default: start from beginning
            start_idx = 0
            
        attempts = 0
        for idx in range(start_idx, len(chunks)):
            if attempts >= max_chunks:
                logger.info(f"Reached max chunks ({max_chunks}) for {field_name}")
                break
                
            chunk_start, chunk_text = chunks[idx]
            logger.info(f"Trying chunk {idx+1}/{len(chunks)} for {field_name} "
                       f"(chars {chunk_start}-{chunk_start+len(chunk_text)})")
            
            try:
                result = extract_func(chunk_text)
                if result and result.lower() not in ["not found", "not_found", "none", "n/a"]:
                    logger.info(f"Found {field_name} in chunk {idx+1}")
                    return result
            except Exception as e:
                logger.error(f"Error extracting {field_name} from chunk {idx+1}: {e}")
                
            attempts += 1
            
        logger.info(f"Could not find {field_name} after {attempts} chunks")
        return None