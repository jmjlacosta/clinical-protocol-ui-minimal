"""
Field Equivalence Checker using ChatGPT for intelligent comparison
"""
import os
import re
import json
import hashlib
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

@dataclass
class EquivalenceResult:
    """Result of field equivalence check"""
    match_status: str  # MATCH, NO_MATCH, PARTIAL_MATCH
    confidence: int    # 0-100
    explanation: str
    
class FieldEquivalenceChecker:
    """Check equivalence between extracted and CT.gov field values using ChatGPT"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("No API key found for FieldEquivalenceChecker")
        
        # In-memory cache to avoid repeated API calls
        self._cache: Dict[str, EquivalenceResult] = {}
        
    def check_equivalence(self, field_name: str, value1: str, value2: str) -> Optional[EquivalenceResult]:
        """
        Check if two field values are equivalent
        
        Args:
            field_name: Name of the field being compared
            value1: First value (extracted)
            value2: Second value (CT.gov)
            
        Returns:
            EquivalenceResult or None if comparison cannot be performed
        """
        # Skip if either value is empty/not found
        if not value1 or not value2:
            return None
            
        # Normalize values
        value1_str = str(value1).strip()
        value2_str = str(value2).strip()
        
        # Skip if either is marked as not found
        if value1_str.upper() in ["NOT_FOUND", "NOT FOUND", "NONE", "NULL", ""]:
            return None
        if value2_str.upper() in ["NOT_FOUND", "NOT FOUND", "NONE", "NULL", ""]:
            return None
            
        # Check cache first
        cache_key = self._get_cache_key(field_name, value1_str, value2_str)
        if cache_key in self._cache:
            logger.info(f"Cache hit for field comparison: {field_name}")
            return self._cache[cache_key]
            
        # If no client, return None
        if not self.client:
            return None
            
        try:
            # Prepare values for comparison (truncate if too long)
            val1_truncated = value1_str[:500] + "..." if len(value1_str) > 500 else value1_str
            val2_truncated = value2_str[:500] + "..." if len(value2_str) > 500 else value2_str
            
            # Create the prompt
            prompt = self._create_comparison_prompt(field_name, val1_truncated, val2_truncated)
            
            # Call ChatGPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at comparing clinical trial field values."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=150
            )
            
            # Parse response
            result = self._parse_response(response.choices[0].message.content)
            
            # Cache the result
            if result:
                self._cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error in field equivalence check: {e}")
            return None
    
    def _create_comparison_prompt(self, field_name: str, value1: str, value2: str) -> str:
        """Create the comparison prompt for ChatGPT"""
        
        # Field-specific context
        field_context = {
            "nct_number": "NCT numbers may have different formatting (spaces, dashes, prefixes)",
            "enrollment": "Enrollment numbers may be expressed differently (e.g., '100' vs '100 patients' vs '100 subjects')",
            "sponsor": "Sponsor names may have variations (abbreviations, Inc. vs Incorporated, etc.)",
            "conditions": "Conditions may be listed in different orders or use synonyms",
            "interventions": "Interventions may be described with slight variations",
            "start_date": "Dates may be in different formats",
            "completion_date": "Dates may be in different formats",
            "primary_outcome_measures": "Outcomes may be paraphrased or reordered",
            "secondary_outcome_measures": "Outcomes may be paraphrased or reordered",
            "phases": "Phase descriptions may vary (e.g., 'Phase 2' vs 'Phase II' vs 'P2')",
            "study_type": "Study types may use different terminology for the same concept"
        }
        
        context = field_context.get(field_name, "Consider semantic meaning and common variations")
        
        prompt = f"""Compare these two values for the clinical trial field "{field_name}".

Field context: {context}

Value 1 (Extracted): {value1}
Value 2 (CT.gov): {value2}

Determine if these values are equivalent by considering:
- Semantic meaning (not just exact text match)
- Common formatting variations
- Abbreviations and synonyms
- For numbers: different representations or units
- For dates: different formats
- For lists: items may be in different order

Respond in EXACTLY this format (no other text):
MATCH_STATUS: [Choose one: MATCH|NO_MATCH|PARTIAL_MATCH]
CONFIDENCE: [Number 0-100]
EXPLANATION: [One brief sentence explaining the comparison]

Example response:
MATCH_STATUS: MATCH
CONFIDENCE: 95
EXPLANATION: Same enrollment count with different formatting (100 vs 100 patients)"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Optional[EquivalenceResult]:
        """Parse ChatGPT response into EquivalenceResult"""
        try:
            lines = response_text.strip().split('\n')
            
            match_status = None
            confidence = None
            explanation = None
            
            for line in lines:
                if line.startswith("MATCH_STATUS:"):
                    match_status = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.split(":", 1)[1].strip()
                    confidence = int(confidence_str)
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()
            
            # Validate we got all required fields
            if match_status and confidence is not None and explanation:
                # Validate match_status
                if match_status not in ["MATCH", "NO_MATCH", "PARTIAL_MATCH"]:
                    logger.warning(f"Invalid match status: {match_status}")
                    return None
                    
                return EquivalenceResult(
                    match_status=match_status,
                    confidence=confidence,
                    explanation=explanation
                )
            else:
                logger.warning(f"Could not parse ChatGPT response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing ChatGPT response: {e}")
            return None
    
    def _get_cache_key(self, field_name: str, value1: str, value2: str) -> str:
        """Generate cache key for the comparison"""
        # Create a consistent key regardless of value order
        combined = f"{field_name}|{min(value1, value2)}|{max(value1, value2)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the comparison cache"""
        self._cache.clear()
        logger.info("Field equivalence cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_hits": sum(1 for _ in self._cache.values())  # Simplified for now
        }