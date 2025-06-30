"""
Intelligent document chunking system for clinical trial PDFs.
Splits documents into manageable chunks while preserving context.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    page_numbers: List[int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'page_numbers': self.page_numbers
        }


class IntelligentChunker:
    """
    Splits documents into overlapping chunks for efficient processing.
    
    Phase 1: Basic chunking with character-based splits
    Future phases will add:
    - Section boundary detection
    - Table preservation
    - Semantic coherence
    """
    
    def __init__(self, chunk_size: int = 50000, overlap_size: int = 1000):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            overlap_size: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
    def chunk_document(self, text: str, page_breaks: Optional[List[int]] = None) -> List[DocumentChunk]:
        """
        Split document into overlapping chunks.
        
        Args:
            text: Full document text
            page_breaks: List of character positions where pages break
            
        Returns:
            List of DocumentChunk objects
        """
        if not text:
            return []
            
        chunks = []
        chunk_id = 0
        start_pos = 0
        
        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # Try to find a good break point (paragraph or sentence)
            if end_pos < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start_pos + self.chunk_size - 500, end_pos)
                if paragraph_break > start_pos:
                    end_pos = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start_pos + self.chunk_size - 200, end_pos)
                    if sentence_break > start_pos:
                        end_pos = sentence_break + 2
            
            # Extract chunk text
            chunk_text = text[start_pos:end_pos]
            
            # Determine which pages this chunk spans
            page_numbers = self._get_page_numbers(start_pos, end_pos, page_breaks)
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start_pos,
                end_char=end_pos,
                page_numbers=page_numbers
            )
            chunks.append(chunk)
            
            logger.debug(f"Created chunk {chunk_id}: chars {start_pos}-{end_pos}, pages {page_numbers}")
            
            # Move to next chunk with overlap
            start_pos = end_pos - self.overlap_size
            chunk_id += 1
            
            # Prevent infinite loop on small documents
            if end_pos >= len(text):
                break
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _get_page_numbers(self, start_char: int, end_char: int, 
                         page_breaks: Optional[List[int]] = None) -> List[int]:
        """
        Determine which pages a chunk spans.
        
        Args:
            start_char: Starting character position
            end_char: Ending character position
            page_breaks: List of character positions where pages break
            
        Returns:
            List of page numbers (1-indexed)
        """
        if not page_breaks:
            return [1]  # Assume single page if no breaks provided
        
        pages = set()
        
        # Find pages that contain this chunk
        current_page = 1
        for i, break_pos in enumerate(page_breaks):
            if start_char < break_pos:
                pages.add(current_page)
            if end_char <= break_pos:
                break
            current_page += 1
        
        # Add last page if chunk extends beyond last break
        if end_char > page_breaks[-1]:
            pages.add(len(page_breaks) + 1)
            
        return sorted(list(pages))
    
    def get_chunk_summary(self, chunks: List[DocumentChunk]) -> Dict:
        """
        Get summary statistics about chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with summary statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_pages': 0
            }
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        all_pages = set()
        for chunk in chunks:
            all_pages.update(chunk.page_numbers)
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_pages': len(all_pages),
            'overlap_size': self.overlap_size,
            'target_chunk_size': self.chunk_size
        }