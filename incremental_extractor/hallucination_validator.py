"""Hallucination validation module for clinical trial extractions"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class HallucinationValidator:
    """Validates extractions to prevent hallucinations"""
    
    def __init__(self):
        # Known hallucinated values database
        self.known_hallucinations = {
            "nct_number": {
                "NCT00514324",  # Lurbinectedin Phase 1
                "NCT00389610",  # Another lurbinectedin trial
                "NCT01970540",  # Phase 1 lurbinectedin
                "NCT00877474",  # PM01183 trial
            },
            "sponsor": {
                # Remove PharmaMar as it's a legitimate sponsor
                "Jazz Pharmaceuticals": ["Jazz Pharma", "Jazz"],
                "Pfizer": ["Pfizer Inc.", "Pfizer Inc"],
                "Novartis": ["Novartis Pharmaceuticals", "Novartis AG"],
            },
            "drug": {
                "lurbinectedin": ["PM1183", "PM01183", "PM-1183"],
                "pembrolizumab": ["Keytruda", "MK-3475"],
                "nivolumab": ["Opdivo", "BMS-936558"],
            }
        }
        
        # Pattern matchers for common hallucinations
        self.hallucination_patterns = {
            "nct_number": re.compile(r"NCT\d{8}"),
            "date": re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"),
            "enrollment": re.compile(r"\d+"),
        }
        
        # Fields that require strict verbatim extraction
        self.verbatim_fields = {
            "nct_number", "enrollment", "start_date",
            "primary_completion_date", "completion_date"
        }
        
        # Context window for validation (characters before/after)
        self.context_window = 100
        
    def validate_extraction(self, field_name: str, extracted_value: Optional[str], 
                          document_text: str, llm_response: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate an extracted value against hallucination patterns
        
        Args:
            field_name: Name of the field being validated
            extracted_value: The extracted value to validate
            document_text: The full document text
            llm_response: The full LLM response (for source quote checking)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not extracted_value or extracted_value == "NOT_FOUND":
            return True, None
        
        # Check against known hallucinations
        is_valid, error = self._check_known_hallucinations(field_name, extracted_value)
        if not is_valid:
            return False, error
        
        # Verify value exists in document
        is_valid, error = self._verify_in_document(field_name, extracted_value, document_text)
        if not is_valid:
            return False, error
        
        # For verbatim fields, check exact match
        if field_name in self.verbatim_fields:
            is_valid, error = self._verify_verbatim_match(field_name, extracted_value, document_text)
            if not is_valid:
                return False, error
        
        # For NCT numbers, perform additional validation
        if field_name == "nct_number":
            is_valid, error = self._validate_nct_number(extracted_value, document_text, llm_response)
            if not is_valid:
                return False, error
        
        # Check for suspicious patterns
        is_valid, error = self._check_suspicious_patterns(field_name, extracted_value, document_text)
        if not is_valid:
            return False, error
        
        return True, None
    
    def _check_known_hallucinations(self, field_name: str, value: str) -> Tuple[bool, Optional[str]]:
        """Check if value matches known hallucination patterns"""
        if field_name in self.known_hallucinations:
            known_values = self.known_hallucinations[field_name]
            
            if isinstance(known_values, dict):
                # Handle sponsor/drug with variations
                for primary, variations in known_values.items():
                    all_variations = [primary] + variations
                    if value in all_variations:
                        logger.warning(f"Known hallucination detected for {field_name}: {value}")
                        return False, f"Value '{value}' is a known hallucination for {field_name}"
            else:
                # Handle simple sets
                if value in known_values:
                    logger.warning(f"Known hallucination detected for {field_name}: {value}")
                    return False, f"Value '{value}' is a known hallucination for {field_name}"
        
        return True, None
    
    def _verify_in_document(self, field_name: str, value: str, document_text: str) -> Tuple[bool, Optional[str]]:
        """Verify the extracted value exists in the document"""
        # For fields that can have multiple values, check each part
        if field_name in ["sponsor", "collaborators", "conditions", "interventions", "locations"]:
            # Split by common separators
            parts = re.split(r'[;,]\s*', value)
            all_found = True
            missing_parts = []
            
            for part in parts:
                part = part.strip()
                if part and part not in document_text:
                    all_found = False
                    missing_parts.append(part)
            
            if not all_found:
                logger.error(f"Parts not found in document for {field_name}: {missing_parts}")
                # If at least some parts were found, it might still be valid
                if len(missing_parts) < len(parts):
                    logger.warning(f"Partial match for {field_name}: {len(parts) - len(missing_parts)}/{len(parts)} parts found")
                    # Allow partial matches for these fields
                    return True, None
                else:
                    return False, f"Value '{value}' not found in document text"
        else:
            # For single-value fields, require exact match
            if value not in document_text:
                logger.error(f"Value not found in document for {field_name}: {value}")
                return False, f"Value '{value}' not found in document text"
        
        return True, None
    
    def _verify_verbatim_match(self, field_name: str, value: str, document_text: str) -> Tuple[bool, Optional[str]]:
        """For verbatim fields, ensure exact match with context"""
        # Find all occurrences
        occurrences = []
        start = 0
        while True:
            pos = document_text.find(value, start)
            if pos == -1:
                break
            occurrences.append(pos)
            start = pos + 1
        
        if not occurrences:
            return False, f"Verbatim value '{value}' not found in document"
        
        # For NCT numbers and critical fields, verify it's in a reasonable location
        if field_name == "nct_number" and occurrences:
            # NCT should appear in first 50k characters typically
            first_occurrence = occurrences[0]
            if first_occurrence > 50000:
                logger.warning(f"NCT number found very late in document at position {first_occurrence}")
                # Still valid but log for monitoring
        
        return True, None
    
    def _validate_nct_number(self, nct_value: str, document_text: str, 
                           llm_response: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Special validation for NCT numbers"""
        # Check format
        if not re.match(r"^NCT\d{8}$", nct_value):
            return False, f"Invalid NCT format: {nct_value}"
        
        # Extract context around NCT number
        nct_pos = document_text.find(nct_value)
        if nct_pos == -1:
            return False, f"NCT {nct_value} not found in document"
        
        # Get surrounding context
        start = max(0, nct_pos - self.context_window)
        end = min(len(document_text), nct_pos + len(nct_value) + self.context_window)
        context = document_text[start:end]
        
        # Look for indicators this is the right NCT
        nct_indicators = [
            "ClinicalTrials.gov",
            "NCT Number:",
            "NCT ID:",
            "Identifier:",
            "Registry:",
            "Trial Registration:"
        ]
        
        has_indicator = any(indicator.lower() in context.lower() for indicator in nct_indicators)
        if not has_indicator:
            logger.warning(f"NCT {nct_value} found but without typical indicators in context")
            # Still might be valid but suspicious
        
        # If LLM response provided, check for source quote
        if llm_response and "Source quote:" in llm_response:
            has_valid_quote = self._validate_source_quote(nct_value, llm_response, document_text)
            if not has_valid_quote:
                return False, "Source quote validation failed"
        
        return True, None
    
    def _validate_source_quote(self, value: str, llm_response: str, document_text: str) -> bool:
        """Validate that source quote is real and contains the value"""
        lines = llm_response.split('\n')
        for line in lines:
            if "Source quote:" in line:
                quote = line.split("Source quote:", 1)[1].strip()
                # Remove quotes if present
                quote = quote.strip('"\'')
                
                # Check if quote exists in document
                if quote not in document_text:
                    logger.error(f"Source quote not found in document: {quote[:50]}...")
                    return False
                
                # Check if value is in the quote
                if value not in quote:
                    logger.error(f"Value '{value}' not found in provided source quote")
                    return False
                
                return True
        
        # No source quote found when expected
        return False
    
    def _check_suspicious_patterns(self, field_name: str, value: str, 
                                 document_text: str) -> Tuple[bool, Optional[str]]:
        """Check for suspicious patterns that indicate hallucination"""
        # Check if this appears to be from a different trial
        if field_name == "nct_number":
            # Look for other NCT numbers in the document
            all_ncts = re.findall(r"NCT\d{8}", document_text)
            if all_ncts and value not in all_ncts:
                return False, f"NCT {value} not among NCT numbers found in document: {all_ncts}"
        
        # Check for common test/example values
        test_values = {
            "nct_number": ["NCT12345678", "NCT00000000", "NCT99999999"],
            "enrollment": ["100", "1000", "50"],
            "sponsor": ["Sponsor Name", "Company Name", "Test Sponsor"]
        }
        
        if field_name in test_values and value in test_values[field_name]:
            # These might be real but are suspicious
            logger.warning(f"Suspicious test-like value for {field_name}: {value}")
        
        return True, None
    
    def batch_validate(self, extractions: Dict[str, Optional[str]], 
                      document_text: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Validate multiple extractions at once"""
        results = {}
        
        for field_name, value in extractions.items():
            is_valid, error = self.validate_extraction(field_name, value, document_text)
            results[field_name] = (is_valid, error)
            
            if not is_valid:
                logger.error(f"Validation failed for {field_name}: {error}")
        
        return results
    
    def get_validation_report(self, extractions: Dict[str, Optional[str]], 
                            document_text: str) -> Dict:
        """Generate a detailed validation report"""
        validation_results = self.batch_validate(extractions, document_text)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_fields": len(extractions),
            "valid_fields": sum(1 for v, _ in validation_results.values() if v),
            "invalid_fields": sum(1 for v, _ in validation_results.values() if not v),
            "details": {}
        }
        
        for field_name, (is_valid, error) in validation_results.items():
            report["details"][field_name] = {
                "value": extractions[field_name],
                "valid": is_valid,
                "error": error,
                "in_document": extractions[field_name] in document_text if extractions[field_name] else None
            }
        
        report["validation_score"] = (report["valid_fields"] / report["total_fields"] * 100) if report["total_fields"] > 0 else 0
        
        return report