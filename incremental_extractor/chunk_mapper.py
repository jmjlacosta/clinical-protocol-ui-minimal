"""
Chunk mapping system that analyzes chunks to identify which fields they contain.
Uses LLM to intelligently map chunks to target extraction fields.
"""

import json
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from .intelligent_chunker import DocumentChunk

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

@dataclass
class ChunkMapping:
    """Represents the mapping between a chunk and the fields it contains"""
    chunk_id: int
    identified_fields: List[str]
    confidence_scores: Dict[str, float]
    relevant_sections: List[str]  # e.g., "Study Design", "Eligibility Criteria"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'identified_fields': self.identified_fields,
            'confidence_scores': self.confidence_scores,
            'relevant_sections': self.relevant_sections
        }


class ChunkMapper:
    """
    Maps document chunks to the fields they likely contain.
    Uses GPT-3.5 to analyze chunk content and identify relevant fields.
    """
    
    # Target fields we're looking for
    TARGET_FIELDS = [
        'brief_title',
        'official_title',
        'brief_summary',
        'detailed_description',
        'conditions',
        'interventions',
        'primary_outcome_measures',
        'secondary_outcome_measures',
        'other_outcome_measures',
        'eligibility_criteria',
        'inclusion_criteria',
        'exclusion_criteria',
        'sex',
        'minimum_age',
        'maximum_age',
        'healthy_volunteers',
        'enrollment',
        'study_type',
        'study_design',
        'phase',
        'allocation',
        'intervention_model',
        'primary_purpose',
        'masking',
        'sponsor',
        'collaborators',
        'investigators',
        'start_date',
        'completion_date',
        'study_status'
    ]
    
    def __init__(self, model: str = "gpt-3.5-turbo-0125"):
        """
        Initialize the chunk mapper.
        
        Args:
            model: OpenAI model to use for analysis
        """
        self.model = model
        if openai:
            self.client = openai.OpenAI()
        else:
            self.client = None
            logger.warning("OpenAI not available, chunk mapping will be disabled")
        
    def analyze_chunk(self, chunk: DocumentChunk) -> ChunkMapping:
        """
        Analyze a single chunk to identify which fields it contains.
        
        Args:
            chunk: Document chunk to analyze
            
        Returns:
            ChunkMapping with identified fields
        """
        if not self.client:
            # Return empty mapping if OpenAI not available
            return ChunkMapping(
                chunk_id=chunk.chunk_id,
                identified_fields=[],
                confidence_scores={},
                relevant_sections=[]
            )
        
        # Prepare the analysis prompt
        prompt = self._create_analysis_prompt(chunk.text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical trial document analyzer. Identify which fields from a clinical trial are present in the given text chunk."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Create mapping
            mapping = ChunkMapping(
                chunk_id=chunk.chunk_id,
                identified_fields=result.get('fields', []),
                confidence_scores=result.get('confidence', {}),
                relevant_sections=result.get('sections', [])
            )
            
            logger.debug(f"Chunk {chunk.chunk_id} contains fields: {mapping.identified_fields}")
            return mapping
            
        except Exception as e:
            logger.error(f"Error analyzing chunk {chunk.chunk_id}: {e}")
            # Return empty mapping on error
            return ChunkMapping(
                chunk_id=chunk.chunk_id,
                identified_fields=[],
                confidence_scores={},
                relevant_sections=[]
            )
    
    def analyze_chunks(self, chunks: List[DocumentChunk]) -> List[ChunkMapping]:
        """
        Analyze multiple chunks in batch.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunk mappings
        """
        mappings = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
            mapping = self.analyze_chunk(chunk)
            mappings.append(mapping)
        
        return mappings
    
    def get_best_chunk_for_field(self, field_name: str, 
                                mappings: List[ChunkMapping]) -> Optional[int]:
        """
        Find the best chunk for extracting a specific field.
        
        Args:
            field_name: Name of the field to extract
            mappings: List of chunk mappings
            
        Returns:
            Chunk ID of the best chunk, or None if field not found
        """
        best_chunk_id = None
        best_confidence = 0.0
        
        for mapping in mappings:
            if field_name in mapping.identified_fields:
                confidence = mapping.confidence_scores.get(field_name, 0.5)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_chunk_id = mapping.chunk_id
        
        if best_chunk_id is not None:
            logger.debug(f"Best chunk for {field_name}: chunk {best_chunk_id} (confidence: {best_confidence})")
        else:
            logger.debug(f"No chunk found containing {field_name}")
            
        return best_chunk_id
    
    def get_field_distribution(self, mappings: List[ChunkMapping]) -> Dict[str, List[int]]:
        """
        Get distribution of fields across chunks.
        
        Args:
            mappings: List of chunk mappings
            
        Returns:
            Dictionary mapping field names to list of chunk IDs containing them
        """
        distribution = {}
        
        for mapping in mappings:
            for field in mapping.identified_fields:
                if field not in distribution:
                    distribution[field] = []
                distribution[field].append(mapping.chunk_id)
        
        return distribution
    
    def _create_analysis_prompt(self, chunk_text: str) -> str:
        """
        Create the prompt for chunk analysis.
        
        Args:
            chunk_text: Text of the chunk to analyze
            
        Returns:
            Formatted prompt
        """
        # Truncate chunk if too long
        max_chunk_length = 4000
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        prompt = f"""Analyze this chunk of a clinical trial document and identify which fields it contains.

Target fields to look for:
{json.dumps(self.TARGET_FIELDS, indent=2)}

Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Return a JSON object with:
1. "fields": list of field names found in this chunk
2. "confidence": dictionary mapping each found field to confidence score (0-1)
3. "sections": list of document sections present (e.g., "Study Design", "Eligibility")

Only include fields that are actually present with meaningful content in the chunk.
Focus on fields where the primary/detailed information is in this chunk, not just mentions.

Example response:
{{
  "fields": ["primary_outcome_measures", "secondary_outcome_measures", "enrollment"],
  "confidence": {{
    "primary_outcome_measures": 0.9,
    "secondary_outcome_measures": 0.85,
    "enrollment": 0.7
  }},
  "sections": ["Outcome Measures", "Study Design"]
}}"""
        
        return prompt
    
    def create_extraction_plan(self, mappings: List[ChunkMapping]) -> Dict[str, int]:
        """
        Create a plan for which chunk to use for each field extraction.
        
        Args:
            mappings: List of chunk mappings
            
        Returns:
            Dictionary mapping field names to chunk IDs
        """
        extraction_plan = {}
        
        for field in self.TARGET_FIELDS:
            chunk_id = self.get_best_chunk_for_field(field, mappings)
            if chunk_id is not None:
                extraction_plan[field] = chunk_id
        
        logger.info(f"Created extraction plan for {len(extraction_plan)} fields")
        return extraction_plan