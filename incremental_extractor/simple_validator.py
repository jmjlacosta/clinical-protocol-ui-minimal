"""Simple validation module that just checks if values exist in the document"""
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SimpleValidator:
    """Simple validator that just verifies text exists in document"""
    
    def __init__(self):
        # Fields that can have multiple values separated by semicolons/commas
        self.multi_value_fields = {
            "sponsor", "collaborators", "conditions", "interventions", 
            "locations", "primary_outcome_measures", "secondary_outcome_measures",
            "other_outcome_measures"
        }
    
    def validate_extraction(self, field_name: str, extracted_value: Optional[str], 
                          document_text: str, llm_response: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Simple validation - just check if the value exists in the document
        
        Args:
            field_name: Name of the field being validated
            extracted_value: The extracted value to validate
            document_text: The full document text
            llm_response: The full LLM response (unused in simple validation)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not extracted_value or extracted_value.upper() in ["NOT_FOUND", "NONE", "N/A"]:
            return True, None
        
        # For multi-value fields, check if at least some parts exist
        if field_name in self.multi_value_fields:
            # Split by common separators
            parts = re.split(r'[;,]\s*', extracted_value)
            found_parts = 0
            
            for part in parts:
                part = part.strip()
                if part and self._text_exists_in_document(part, document_text):
                    found_parts += 1
            
            if found_parts == 0:
                return False, f"None of the extracted values found in document"
            elif found_parts < len(parts):
                logger.debug(f"Partial match for {field_name}: {found_parts}/{len(parts)} parts found")
            
            return True, None
        else:
            # For single-value fields, check if it exists
            if self._text_exists_in_document(extracted_value, document_text):
                return True, None
            else:
                return False, f"Value '{extracted_value}' not found in document"
    
    def _text_exists_in_document(self, text: str, document: str) -> bool:
        """Check if text exists in document, with some flexibility"""
        # Direct check
        if text in document:
            return True
        
        # Case-insensitive check
        if text.lower() in document.lower():
            return True
        
        # Check without common prefixes that LLMs add
        prefixes_to_remove = ["Drug:", "Device:", "Procedure:", "Behavioral:", "Other:", "Treatment:"]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                clean_text = text[len(prefix):].strip()
                if clean_text in document or clean_text.lower() in document.lower():
                    return True
                
                # Also check for partial matches for drug names
                # e.g., "Lurbinectedin 3.2 mg/m2" should match if "Lurbinectedin" exists
                parts = clean_text.split()
                if parts and (parts[0] in document or parts[0].lower() in document.lower()):
                    return True
        
        # For NCT numbers specifically, check multiple formats
        if re.match(r'^NCT\d{8}$', text):
            # Check with spaces, dashes, etc.
            variants = [
                text,
                text[:3] + " " + text[3:],  # NCT 12345678
                text[:3] + "-" + text[3:],  # NCT-12345678
            ]
            for variant in variants:
                if variant in document or variant.lower() in document.lower():
                    return True
        
        return False
    
    def batch_validate(self, extracted_values: Dict[str, Optional[str]], 
                      document_text: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Validate multiple extractions at once"""
        results = {}
        for field_name, value in extracted_values.items():
            results[field_name] = self.validate_extraction(field_name, value, document_text)
        return results