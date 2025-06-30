"""Intelligent comparison using LLM for semantic matching"""
import logging
from typing import Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class IntelligentComparator:
    """Use LLM to determine if extracted values match CT.gov values semantically"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def compare_fields(self, field_name: str, extracted_value: str, 
                      ctgov_value: str) -> Tuple[bool, float, str]:
        """
        Compare two field values using LLM for semantic matching.
        
        Returns:
            Tuple of (match: bool, confidence: float, explanation: str)
        """
        if not extracted_value or not ctgov_value:
            return False, 0.0, "One or both values are empty"
        
        # Special handling for exact matches
        if extracted_value.strip().lower() == ctgov_value.strip().lower():
            return True, 1.0, "Exact match"
        
        prompt = f"""You are a clinical trial data comparison expert. Your job is to determine if two values for the same field are semantically equivalent, even if they have different formats or wording.

Field: {field_name}
Value 1 (Extracted from PDF): {extracted_value}
Value 2 (From ClinicalTrials.gov): {ctgov_value}

Consider the following:
1. Different formats (e.g., "Phase 2" vs "PHASE2" vs "Phase II")
2. Abbreviations (e.g., "ICG" vs "Indocyanine Green")
3. Additional details in one value (e.g., "DRUG: ICG" vs "ICG")
4. Status variations (e.g., "Recruiting" vs "ACTIVE_NOT_RECRUITING" - these are different!)
5. Partial information (longer title in one vs shortened in other)
6. Formatting differences (e.g., "PharmaMar" vs "Pharma Mar S.A.")
7. Age equivalencies (e.g., "Children" = "0-17 years" = "CHILD", "Adults" = "18+ years" = "ADULT")
8. Intervention types (e.g., "antibiotic guidelines" = "BEHAVIORAL: antibiotic guidelines")

Respond with ONLY a JSON object in this format:
{{"match": true/false, "confidence": 0.0-1.0, "explanation": "brief explanation"}}

Examples:
- "Phase 2" vs "PHASE2" -> {{"match": true, "confidence": 0.95, "explanation": "Same phase, different format"}}
- "Recruiting" vs "COMPLETED" -> {{"match": false, "confidence": 0.95, "explanation": "Different study statuses"}}
- "ICG" vs "DRUG: ICG" -> {{"match": true, "confidence": 0.9, "explanation": "Same drug, CT.gov adds category prefix"}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise clinical trial data comparison expert. Respond only with the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            try:
                result = json.loads(result_text)
                return result['match'], result['confidence'], result['explanation']
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {result_text}")
                return False, 0.0, "Failed to parse comparison result"
                
        except Exception as e:
            logger.error(f"Comparison error for {field_name}: {e}")
            # Fallback to simple comparison
            return False, 0.0, f"Comparison error: {str(e)}"
    
    def get_match_summary(self, extracted_value: str, ctgov_value: str,
                         match: bool, confidence: float, explanation: str) -> str:
        """Format a nice summary of the comparison"""
        if match:
            return f"MATCH (confidence: {confidence:.0%}) - {explanation}"
        else:
            return f"MISMATCH (confidence: {confidence:.0%}) - {explanation}"