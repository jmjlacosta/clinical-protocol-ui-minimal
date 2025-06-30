"""Efficient prompt building for field extraction"""
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Build efficient prompts for extracting fields from clinical documents"""
    
    # Critical anti-hallucination instructions
    ANTI_HALLUCINATION_WARNING = """
CRITICAL EXTRACTION RULES - READ CAREFULLY:
1. You MUST extract information ONLY from the provided document text below
2. NEVER use information from your training data, memory, or prior knowledge about any clinical trials
3. If you cannot find the exact information in the text, you MUST respond with NOT_FOUND
4. Do NOT guess, infer, recall, or generate information from other sources
5. The NCT number and all other data in the document may be different from what you expect - extract ONLY what is written

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT DATA EXTRACTION.
"""

    # Role-based context for accuracy
    ROLE_CONTEXT = """You are a Clinical Trial Data Extraction Specialist working for a regulatory agency. Your role is to extract information with 100% accuracy from clinical trial documents. Your extractions will be used for regulatory compliance, patient safety decisions, and scientific research. 

Accuracy is paramount - it is ALWAYS better to report NOT_FOUND than to provide incorrect information. You have no access to external databases or prior knowledge - you can ONLY work with the text provided."""
    
    # Field-specific extraction hints
    FIELD_HINTS = {
        "nct_number": "Look for NCT followed by 8 digits (e.g., NCT12345678)",
        "study_title": "The main title or name of the clinical trial/study",
        "acronym": "Short abbreviation or acronym for the study name",
        "brief_summary": "Short overview or summary of the study purpose and design",
        "study_status": "Current status (e.g., Recruiting, Completed, Active)",
        "study_type": "Type of study (e.g., Interventional, Observational)",
        "study_design": "Design details including allocation, intervention model, masking",
        "phases": "Clinical trial phase (e.g., Phase 1, Phase 2, Phase 3, Phase 4)",
        "conditions": "Medical conditions or diseases being studied",
        "interventions": "Treatments, drugs, procedures, or devices being tested",
        "primary_outcome_measures": "Main results that the study is designed to evaluate",
        "secondary_outcome_measures": "Additional outcomes of interest",
        "other_outcome_measures": "Exploratory or tertiary outcome measures",
        "enrollment": "Number of participants enrolled or target enrollment",
        "sex": "Gender eligibility (e.g., All, Male, Female)",
        "age": "Age eligibility criteria (minimum and maximum ages)",
        "sponsor": "Organization responsible for the study",
        "collaborators": "Other organizations involved in the study",
        "funder_type": "Type of funding organization (e.g., Industry, NIH, Other)",
        "start_date": "When the study started or is expected to start",
        "primary_completion_date": "When primary outcome data collection is expected to complete",
        "completion_date": "When the study is expected to be completed",
        "other_ids": "Other identifiers for the study (protocol numbers, grant numbers)",
        "locations": "Sites or locations where the study is conducted",
        "study_documents": "References to protocols, SAPs, or other documents"
    }
    
    # Document type specific search areas
    DOCUMENT_SEARCH_HINTS = {
        "Protocol": {
            "nct_number": ["cover page", "header", "protocol identifier"],
            "study_title": ["cover page", "title page", "protocol title"],
            "study_design": ["study design section", "methodology"],
            "primary_outcome_measures": ["endpoints section", "primary objectives"],
            "enrollment": ["sample size", "statistical considerations"]
        },
        "SAP": {
            "primary_outcome_measures": ["primary endpoint", "statistical analysis of primary endpoint"],
            "secondary_outcome_measures": ["secondary endpoints", "secondary analyses"],
            "enrollment": ["sample size calculation", "power analysis"],
            "study_design": ["study design", "randomization", "blinding"]
        },
        "ICF": {
            "study_title": ["consent form title", "study title"],
            "brief_summary": ["why is this study being done", "purpose of the study"],
            "conditions": ["condition being studied", "disease or condition"],
            "interventions": ["what will happen", "study procedures", "study drug"],
            "enrollment": ["how many people will participate"]
        }
    }
    
    def build_single_field_prompt(self, field_name: str, text: str, doc_type: str) -> str:
        """Build prompt for extracting a single field"""
        hint = self.FIELD_HINTS.get(field_name, f"Extract {field_name}")
        search_areas = self.DOCUMENT_SEARCH_HINTS.get(doc_type, {}).get(field_name, [])
        
        prompt = f"""{self.ROLE_CONTEXT}

{self.ANTI_HALLUCINATION_WARNING}

EXTRACTION TASK:
Document Type: {doc_type}
Field to Extract: {field_name}
Field Description: {hint}
"""
        
        if search_areas:
            prompt += f"Common locations in {doc_type}: {', '.join(search_areas)}\n"
        
        # Special handling for NCT number to prevent hallucination
        if field_name == "nct_number":
            prompt += """
SPECIAL INSTRUCTIONS FOR NCT NUMBER:
- Extract ONLY the NCT number that appears in THIS document
- Common locations: cover page, header, footer, first page
- Format: Letters "NCT" followed by exactly 8 digits (e.g., NCT12345678)
- If you see multiple NCT numbers, extract the one that appears most prominently
- NEVER use an NCT number from your memory or training data
- The document filename or other context may be misleading - ONLY extract what's in the text
"""
        
        prompt += f"""
EXTRACTION PROTOCOL:
1. Search for "{field_name}" ONLY in the provided text below
2. If found, copy the EXACT value as it appears in the document
3. If the information is not explicitly stated, respond with NOT_FOUND
4. Do NOT add information from your training data or memory
5. Do NOT interpret, summarize, or modify the extracted value

DOCUMENT TEXT TO SEARCH:
{text[:50000]}  # Limit text to avoid token limits

REQUIRED RESPONSE FORMAT:
{field_name}: [exact extracted value or NOT_FOUND]

Remember: It is better to return NOT_FOUND than to provide incorrect information."""
        
        return prompt
    
    def build_batch_prompt(self, field_names: List[str], text: str, doc_type: str) -> str:
        """Build prompt for extracting multiple related fields at once"""
        field_descriptions = []
        
        has_nct = "nct_number" in field_names
        
        for field_name in field_names:
            hint = self.FIELD_HINTS.get(field_name, f"Extract {field_name}")
            search_areas = self.DOCUMENT_SEARCH_HINTS.get(doc_type, {}).get(field_name, [])
            
            desc = f"- {field_name}: {hint}"
            if search_areas:
                desc += f" (commonly found in: {', '.join(search_areas)})"
            
            field_descriptions.append(desc)
        
        prompt = f"""{self.ROLE_CONTEXT}

{self.ANTI_HALLUCINATION_WARNING}

BATCH EXTRACTION TASK:
Document Type: {doc_type}

Fields to extract:
{chr(10).join(field_descriptions)}
"""

        if has_nct:
            prompt += """
CRITICAL NOTE FOR NCT NUMBER:
- Extract ONLY the NCT number from THIS document's text
- Do NOT use NCT numbers from your memory or other sources
- If the NCT number is not clearly visible in the text, mark it as NOT_FOUND
"""

        prompt += f"""
EXTRACTION PROTOCOL:
1. Search for each field ONLY in the provided document text
2. Extract values EXACTLY as they appear - no modifications
3. If any field is not found, use "NOT_FOUND" for that field
4. NEVER use information from your training data or memory
5. Do not interpret, infer, or generate content

DOCUMENT TEXT TO SEARCH:
{text[:50000]}  # Limit text to avoid token limits

REQUIRED RESPONSE FORMAT (one field per line):
field_name: exact_extracted_value
field_name: NOT_FOUND
...

Remember: Accuracy is critical. Only extract what you can find in the document."""
        
        return prompt
    
    def get_optimal_field_groups(self, remaining_fields: List[str], doc_type: str) -> List[List[str]]:
        """
        Group fields optimally for batch extraction.
        Returns list of field groups to extract together.
        """
        from .schema import FIELD_GROUPS
        
        # Start with predefined groups
        grouped_fields = []
        used_fields = set()
        
        # Use predefined groups first
        for group_name, group_fields in FIELD_GROUPS.items():
            # Only include fields that are in remaining_fields
            relevant_fields = [f for f in group_fields if f in remaining_fields and f not in used_fields]
            
            if relevant_fields:
                grouped_fields.append(relevant_fields)
                used_fields.update(relevant_fields)
        
        # Add any ungrouped fields individually
        for field in remaining_fields:
            if field not in used_fields:
                grouped_fields.append([field])
        
        return grouped_fields
    
    def parse_extraction_response(self, response: str, expected_fields: List[str]) -> Dict[str, Optional[str]]:
        """
        Parse the LLM response to extract field values.
        
        Args:
            response: The LLM response text
            expected_fields: List of fields we expect to find
            
        Returns:
            Dictionary mapping field names to extracted values
        """
        results = {}
        
        # Initialize all expected fields as None
        for field in expected_fields:
            results[field] = None
        
        # Parse each line of the response
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Look for field:value pattern
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    field_name = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    
                    # Match against expected fields (case-insensitive)
                    for expected_field in expected_fields:
                        if field_name == expected_field.lower() or field_name.replace('_', '') == expected_field.lower().replace('_', ''):
                            if value and value.upper() != 'NOT_FOUND':
                                results[expected_field] = value
                            break
        
        return results
    
    def create_comparison_prompt(self, field_name: str, extracted_value: str, ctgov_value: str) -> str:
        """Create prompt for comparing extracted value with CT.gov value"""
        prompt = f"""Compare these two values for the field "{field_name}":

Extracted from PDF: {extracted_value}
From ClinicalTrials.gov: {ctgov_value}

Are these values essentially the same? Consider:
1. They may have minor formatting differences
2. One may be more detailed than the other
3. Abbreviations may differ
4. Dates may be in different formats

Response format:
MATCH: [Yes/No]
SIMILARITY: [0-100]%
NOTES: [Brief explanation if they differ]
"""
        
        return prompt