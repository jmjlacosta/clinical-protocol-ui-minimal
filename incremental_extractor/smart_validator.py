"""Smart validation that understands different field types"""
import logging
import re
from typing import Dict, List, Optional, Tuple
from .field_categories import (
    VERBATIM_FIELDS, SUMMARY_FIELDS, INFERRED_FIELDS, 
    REGISTRY_ONLY_FIELDS, TERMINOLOGY_VARIANTS
)

logger = logging.getLogger(__name__)

class SmartValidator:
    """Validator that understands different types of extractions"""
    
    def validate_extraction(self, field_name: str, extracted_value: Optional[str], 
                          document_text: str, llm_response: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Smart validation based on field type
        
        Args:
            field_name: Name of the field being validated
            extracted_value: The extracted value to validate
            document_text: The full document text
            llm_response: The full LLM response (for checking reasoning)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not extracted_value or extracted_value.upper() in ["NOT_FOUND", "NONE", "N/A"]:
            return True, None
        
        # Registry-only fields shouldn't be extracted from protocols
        if field_name in REGISTRY_ONLY_FIELDS:
            logger.debug(f"{field_name} is a registry-only field, not expected in protocols")
            return True, None  # Allow NOT_FOUND for these
        
        # Verbatim fields must exist in document
        if field_name in VERBATIM_FIELDS:
            return self._validate_verbatim(field_name, extracted_value, document_text)
        
        # Summary fields don't need exact matches
        elif field_name in SUMMARY_FIELDS:
            return self._validate_summary(field_name, extracted_value, document_text)
        
        # Inferred fields need loose validation
        elif field_name in INFERRED_FIELDS:
            return self._validate_inferred(field_name, extracted_value, document_text)
        
        # Default to smart validation
        else:
            return self._validate_smart(field_name, extracted_value, document_text)
    
    def _validate_verbatim(self, field_name: str, value: str, document: str) -> Tuple[bool, Optional[str]]:
        """Validate fields that should exist verbatim"""
        # For multi-value fields, check each part
        if ";" in value:
            parts = [p.strip() for p in value.split(";")]
            found = sum(1 for p in parts if p and (p in document or p.lower() in document.lower()))
            if found == 0:
                return False, f"None of the values found in document"
            elif found < len(parts):
                logger.debug(f"Partial match for {field_name}: {found}/{len(parts)} parts found")
            return True, None
        
        # Single value must exist
        if value in document or value.lower() in document.lower():
            return True, None
        
        # Check without common formatting
        clean_value = re.sub(r'[^\w\s]', ' ', value).strip()
        clean_doc = re.sub(r'[^\w\s]', ' ', document)
        if clean_value.lower() in clean_doc.lower():
            return True, None
            
        return False, f"Value '{value}' not found in document"
    
    def _validate_summary(self, field_name: str, value: str, document: str) -> Tuple[bool, Optional[str]]:
        """Validate summary/interpretation fields"""
        # For summaries, just check that key terms exist
        if field_name == "brief_summary":
            # Extract key medical terms from the summary
            key_terms = self._extract_key_terms(value)
            found_terms = sum(1 for term in key_terms if term.lower() in document.lower())
            
            if found_terms < len(key_terms) * 0.3:  # Less than 30% of terms found
                return False, f"Summary contains terms not found in document"
            return True, None
        
        elif field_name == "study_design":
            # Check for design-related terms
            design_terms = ["multicenter", "open-label", "randomized", "controlled", 
                          "phase", "trial", "study", "cohort", "arm"]
            found_any = any(term in value.lower() and term in document.lower() 
                          for term in design_terms)
            if not found_any:
                return False, "Study design terms not found in document"
            return True, None
        
        elif field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
            # For outcomes, check if key outcome terms exist
            outcome_terms = self._extract_medical_terms(value)
            if not outcome_terms:
                return True, None  # Can't validate without terms
            
            found = sum(1 for term in outcome_terms if term.lower() in document.lower())
            if found == 0:
                return False, "No outcome terms found in document"
            return True, None
        
        elif field_name == "interventions":
            # Special handling for interventions
            return self._validate_interventions(value, document)
        
        # Default: accept summaries
        return True, None
    
    def _validate_inferred(self, field_name: str, value: str, document: str) -> Tuple[bool, Optional[str]]:
        """Validate inferred fields with loose matching"""
        if field_name == "study_status":
            # Status might be inferred from protocol version, dates, etc.
            status_terms = ["ongoing", "active", "recruiting", "completed", "terminated"]
            if any(term in value.lower() for term in status_terms):
                return True, None
                
        elif field_name == "study_type":
            # Type might be inferred from design
            if "phase" in value.lower() and "phase" in document.lower():
                return True, None
                
        elif field_name in ["sex", "age"]:
            # These come from eligibility criteria
            if "eligibility" in document.lower() or "inclusion" in document.lower():
                return True, None
                
        return True, None  # Be permissive with inferred fields
    
    def _validate_smart(self, field_name: str, value: str, document: str) -> Tuple[bool, Optional[str]]:
        """Smart validation for uncategorized fields"""
        # Check if we have terminology variants
        if field_name in TERMINOLOGY_VARIANTS:
            variants = TERMINOLOGY_VARIANTS[field_name]
            doc_lower = document.lower()
            if any(variant in doc_lower for variant in variants):
                return True, None  # Document discusses this concept
        
        # For unknown fields, be permissive but check for obvious issues
        if len(value) > 1000:
            return False, "Extracted value unreasonably long"
        
        # Check if at least some key terms exist
        terms = self._extract_key_terms(value)
        if terms:
            found = sum(1 for term in terms if term.lower() in document.lower())
            if found == 0:
                return False, "No relevant terms found in document"
        
        return True, None
    
    def _validate_interventions(self, value: str, document: str) -> Tuple[bool, Optional[str]]:
        """Special validation for interventions field"""
        # Remove common prefixes
        clean_value = re.sub(r'(Drug|Device|Procedure|Behavioral|Treatment):\s*', '', value)
        
        # Split into parts
        parts = re.split(r'[;,]\s*', clean_value)
        
        # Check each intervention
        found = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Extract just the drug/intervention name
            name_match = re.match(r'^([A-Za-z0-9\-]+)', part)
            if name_match:
                name = name_match.group(1)
                if name.lower() in document.lower():
                    found += 1
            elif part.lower() in document.lower():
                found += 1
        
        if found == 0:
            return False, "No interventions found in document"
        
        return True, None
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key medical/scientific terms from text"""
        # Remove common words and extract meaningful terms
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                    "of", "with", "by", "from", "will", "be", "is", "are", "was", "were"}
        
        # Extract words that are likely medical terms
        words = re.findall(r'\b[A-Za-z0-9/-]+\b', text)
        terms = []
        
        for word in words:
            if len(word) > 3 and word.lower() not in stopwords:
                # Keep medical-looking terms
                if (word[0].isupper() or  # Capitalized
                    any(c.isdigit() for c in word) or  # Contains numbers
                    '/' in word or '-' in word):  # Contains special chars
                    terms.append(word)
        
        return terms[:10]  # Limit to 10 most important terms
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from outcome measures"""
        # Common medical outcome terms
        medical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms like ORR, PFS, OS
            r'\b\w+emia\b',    # Medical conditions
            r'\b\w+osis\b',    
            r'\b\w+itis\b',
            r'\bresponse\b',
            r'\bsurvival\b',
            r'\bprogression\b',
            r'\btoxicity\b',
            r'\befficacy\b',
        ]
        
        terms = []
        for pattern in medical_patterns:
            terms.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(terms))  # Unique terms
    
    def batch_validate(self, extracted_values: Dict[str, Optional[str]], 
                      document_text: str) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Validate multiple extractions at once"""
        results = {}
        for field_name, value in extracted_values.items():
            results[field_name] = self.validate_extraction(field_name, value, document_text)
        return results