"""Enhanced prompt building with field-specific extraction strategies"""
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedPromptBuilder:
    """Build optimized prompts for extracting fields from clinical documents"""
    
    # Critical anti-hallucination instructions
    ANTI_HALLUCINATION_WARNING = """
⚠️ CRITICAL DATA INTEGRITY WARNING ⚠️
You are extracting data from a SPECIFIC clinical trial document. This is NOT a test of your knowledge.

STRICT RULES:
1. Extract information ONLY from the text provided below
2. NEVER use information from your training data about any clinical trials
3. NEVER recall or generate NCT numbers, drug names, or study details from memory
4. If information is not EXPLICITLY written in the document, return NOT_FOUND
5. The document may contain information that differs from your training data - ALWAYS trust the document

COMMON HALLUCINATION TRAPS TO AVOID:
- Using NCT numbers from similar studies in your training data
- Recalling drug dosages or study designs from memory
- Generating plausible-sounding information when the field is missing
- Completing partial information with your knowledge

REMEMBER: This is a data extraction task, not a knowledge test. Only the document matters.
"""

    # Role-based context for accuracy
    ROLE_CONTEXT = """You are a Clinical Trial Data Extraction Specialist performing regulatory document review. You have been hired specifically because you have NO prior knowledge of any clinical trials - this ensures unbiased extraction.

Your job is to act as a precise scanner that:
1. Reads ONLY the provided document text
2. Extracts information EXACTLY as written
3. Reports NOT_FOUND when information is absent
4. Never adds, infers, or recalls information

Think of yourself as a specialized OCR system that can understand context but has no memory of other documents."""
    
    # Field-specific extraction templates with examples
    FIELD_TEMPLATES = {
        "nct_number": {
            "description": "NCT registry number (8 digits after NCT)",
            "examples": ["NCT12345678", "NCT00123456"],
            "patterns": ["NCT followed by 8 digits", "ClinicalTrials.gov Identifier"],
            "format": "NCT########"
        },
        
        "study_title": {
            "description": "Official title of the clinical trial",
            "examples": [
                "A Phase 2 Study of Drug X in Patients with Condition Y",
                "Randomized Controlled Trial of Treatment A versus Treatment B"
            ],
            "hints": ["Usually on cover page", "May be labeled 'Protocol Title' or 'Study Title'"],
            "avoid": ["Don't include protocol version numbers or dates"]
        },
        
        "acronym": {
            "description": "Study acronym or short name",
            "examples": ["KEYNOTE-001", "CHECKMATE-238", "DESTINY-Breast01"],
            "hints": ["Often in parentheses after full title", "May be empty if no acronym exists"],
            "format": "Letters/numbers with hyphens, typically uppercase"
        },
        
        "brief_summary": {
            "description": "Extract or create a brief summary BASED ONLY ON information in THIS document",
            "hints": [
                "Look for existing summary sections first",
                "If no summary exists, compile ONLY from information in the document",
                "Do NOT add information from your knowledge"
            ],
            "instructions": "IMPORTANT: Create a summary using ONLY information found in THIS document. If the document lacks sufficient information for a summary, return NOT_FOUND. Do NOT use information from your training data about this or similar studies. The summary should include ONLY what you can find about: 1) The study's purpose, 2) The intervention being tested, 3) The target population - all from THIS document only.",
            "examples": [
                "The purpose of this study is to evaluate whether a discharge antibiotic stewardship program can reduce inappropriate antibiotic prescriptions in children. The intervention includes guidelines and feedback to physicians. The study will measure prescription appropriateness and treatment outcomes.",
                "This Phase 2 trial will test the efficacy and safety of lurbinectedin in patients with advanced solid tumors. Participants will receive the study drug and be monitored for tumor response and adverse events. The primary goal is to determine the overall response rate."
            ],
            "verbatim": False  # This field allows synthesis but ONLY from document content
        },
        
        "study_status": {
            "description": "Current recruitment/study status",
            "valid_values": ["Not yet recruiting", "Recruiting", "Active, not recruiting", "Completed", "Terminated", "Suspended", "Withdrawn"],
            "hints": ["May be on cover page", "Look for 'Study Status' or 'Recruitment Status'"],
            "format": "Use exact status name from valid values list"
        },
        
        "study_type": {
            "description": "Type of clinical study",
            "valid_values": ["Interventional", "Observational", "Expanded Access"],
            "hints": ["Usually clearly stated as 'Study Type'", "Interventional = testing treatment"],
            "format": "Must be one of the valid values"
        },
        
        "study_design": {
            "description": "Key design features",
            "components": ["Allocation", "Intervention Model", "Masking", "Primary Purpose"],
            "examples": [
                "Allocation: Randomized, Intervention Model: Parallel Assignment, Masking: Double-blind, Primary Purpose: Treatment",
                "Allocation: Non-randomized, Intervention Model: Single Group, Masking: None, Primary Purpose: Diagnostic"
            ],
            "format": "Component: Value format for each design element"
        },
        
        "phases": {
            "description": "Clinical trial phase",
            "valid_values": ["Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4", "Not Applicable"],
            "examples": ["Phase 2", "Phase 1/Phase 2"],
            "format": "Use 'Phase #' format"
        },
        
        "conditions": {
            "description": "Extract SPECIFIC medical conditions, diseases, or health problems being studied",
            "examples": [
                "Non-Small Cell Lung Cancer",
                "Type 2 Diabetes Mellitus",
                "COVID-19",
                "Pneumonia; Urinary Tract Infections; Skin Infections"
            ],
            "hints": [
                "Look for specific disease names, not generic descriptions",
                "Common patterns: 'patients with...', 'treatment of...', 'children with...'",
                "Medical terms often end in: -itis, -osis, -emia, -opathy, -syndrome",
                "Include ALL conditions mentioned, separated by semicolons"
            ],
            "instructions": "Extract the actual medical conditions, NOT phrases like 'medical conditions being studied'. If the document mentions 'children with pneumonia and UTIs', extract 'Pneumonia; Urinary Tract Infections'",
            "format": "List specific conditions separated by semicolons"
        },
        
        "interventions": {
            "description": "Treatments, procedures, or actions being tested - must categorize by type",
            "categories": {
                "Drug": "medications, pharmaceuticals, chemotherapy",
                "Device": "medical devices, equipment, monitors",
                "Behavioral": "guidelines, education, counseling, lifestyle changes",
                "Procedure": "surgeries, medical procedures",
                "Biological": "vaccines, gene therapy, cell therapy",
                "Other": "anything that doesn't fit above"
            },
            "examples": [
                "Drug: Lurbinectedin 3.2 mg/m2 IV every 3 weeks",
                "Device: Continuous Glucose Monitor",
                "Behavioral: Discharge antibiotic stewardship guidelines",
                "Procedure: Laparoscopic surgery",
                "Drug: Pembrolizumab; Drug: Chemotherapy"
            ],
            "instructions": "MUST format as 'Type: Description'. Identify what's being tested and categorize it. For guidelines/protocols/education, use 'Behavioral'. For multiple interventions, separate with semicolons. Include dosing and frequency for drugs when available.",
            "hints": [
                "'New guidelines' = Behavioral intervention",
                "'Study drug' or medication names = Drug intervention",
                "'Education program' = Behavioral intervention",
                "Look for: treatment, intervention, study procedures, what will be done"
            ],
            "format": "Type: Name/Description; Type: Name/Description"
        },
        
        "enrollment": {
            "description": "Total number of participants",
            "examples": ["100", "250", "1000"],
            "hints": ["Look for 'Sample Size', 'Number of Subjects', 'Enrollment'"],
            "format": "Number only (no text)"
        },
        
        "sex": {
            "description": "Gender eligibility",
            "valid_values": ["All", "Female", "Male"],
            "hints": ["Look in eligibility criteria"],
            "format": "Must be one of: All, Female, Male"
        },
        
        "age": {
            "description": "Age eligibility - extract any age information or descriptors",
            "examples": [
                "18 Years to 65 Years",
                "18 Years and older", 
                "Children",
                "Adults",
                "0-17 years",
                "Pediatric population"
            ],
            "hints": [
                "Look for: age, years old, children, adults, pediatric, adolescent",
                "'Children' typically means under 18 years",
                "'Adults' typically means 18 years and older",
                "May be in eligibility or 'Who can participate' sections"
            ],
            "instructions": "Extract whatever age information is provided - could be ranges, categories (children/adults), or specific ages. Don't convert categories to ranges. If no age information is found, return NOT_FOUND.",
            "format": "Extract as found in document"
        },
        
        "sponsor": {
            "description": "Primary organization responsible for the study",
            "examples": ["Pfizer", "National Cancer Institute", "Johns Hopkins University"],
            "hints": ["Usually on cover page", "Look for 'Sponsor' or 'Sponsored by'"],
            "avoid": ["Don't include CROs or vendors unless they are the primary sponsor"]
        },
        
        "primary_outcome_measures": {
            "description": "Primary endpoints with timeframes",
            "examples": [
                "Overall Response Rate (ORR) [Time Frame: 6 months]",
                "Progression-Free Survival (PFS) [Time Frame: 12 months]"
            ],
            "format": "Outcome name [Time Frame: duration]"
        },
        
        "secondary_outcome_measures": {
            "description": "Secondary endpoints with timeframes",
            "examples": [
                "Change in Quality of Life Score [Time Frame: 12 months]",
                "Adverse Event Rate [Time Frame: Through study completion, average 1 year]",
                "Biomarker Response [Time Frame: 3 months]"
            ],
            "hints": [
                "Look for 'Secondary Outcomes', 'Secondary Endpoints', 'Secondary Objectives'",
                "May be in a separate section from primary outcomes",
                "Include all secondary measures with their timeframes"
            ],
            "format": "Outcome name [Time Frame: duration]; Multiple outcomes separated by semicolons"
        },
        
        "other_outcome_measures": {
            "description": "Other/exploratory outcome measures with timeframes",
            "examples": [
                "Exploratory biomarker analysis [Time Frame: Baseline and 6 months]",
                "Pharmacokinetic parameters [Time Frame: Day 1 and Day 15]"
            ],
            "hints": [
                "Look for 'Other Outcomes', 'Exploratory Endpoints', 'Tertiary Objectives'",
                "These are often optional or exploratory analyses"
            ],
            "format": "Outcome name [Time Frame: duration]; Multiple outcomes separated by semicolons"
        }
    }
    
    def build_single_field_prompt(self, field_name: str, text: str, doc_type: str) -> str:
        """Build enhanced prompt for extracting a single field"""
        template = self.FIELD_TEMPLATES.get(field_name, {})
        
        prompt = f"""{self.ROLE_CONTEXT}

{self.ANTI_HALLUCINATION_WARNING}

EXTRACTION TASK FOR {doc_type.upper()} DOCUMENT:

FIELD TO EXTRACT: {field_name}

FIELD DETAILS:
- Description: {template.get('description', f'Extract {field_name}')}
"""
        
        if 'instructions' in template:
            prompt += f"\nSPECIAL INSTRUCTIONS:\n{template['instructions']}\n"
        
        if 'valid_values' in template:
            prompt += f"- Valid Values: {', '.join(template['valid_values'])}\n"
        
        if 'examples' in template:
            prompt += f"- Examples: {', '.join(template['examples'])}\n"
        
        if 'format' in template:
            prompt += f"- Required Format: {template['format']}\n"
        
        if 'hints' in template:
            prompt += f"- Where to Look: {'; '.join(template['hints'])}\n"
        
        if 'patterns' in template:
            prompt += f"- Patterns: {'; '.join(template['patterns'])}\n"
            
        if 'avoid' in template:
            prompt += f"- Avoid: {'; '.join(template['avoid'])}\n"
        
        # Special anti-hallucination measures for NCT number
        if field_name == "nct_number":
            prompt += """
🚨 NCT NUMBER EXTRACTION - HIGHEST RISK OF HALLUCINATION 🚨
- This document may be about ANY clinical trial, not the one you might expect
- Extract ONLY the NCT number you can see in the text below
- Common trap: The filename or context might mention a different NCT number - IGNORE IT
- If you cannot find "NCT" followed by 8 digits in the text, return NOT_FOUND
- Do NOT generate or recall NCT numbers from your training data
"""
        
        # Check if field requires verbatim extraction
        is_verbatim = template.get('verbatim', True)
        
        prompt += f"""
EXTRACTION RULES:
1. Search for this field ONLY in the document text provided below
2. {'Extract EXACTLY as written - copy verbatim' if is_verbatim else 'Extract or synthesize ONLY from information in this document'}
3. If the field is not found or cannot be determined from the document, respond with NOT_FOUND
4. NEVER use information from your training data, memory, or knowledge of other trials
5. If multiple values exist, include all (separated by semicolons)
6. For fields with valid values, use ONLY if the document explicitly states one of them

DOCUMENT TEXT TO SEARCH:
{text[:15000]}  # Increased limit for better context

REQUIRED RESPONSE FORMAT:
{field_name}: [extracted value or NOT_FOUND]

FINAL REMINDER: You are extracting from THIS SPECIFIC DOCUMENT ONLY. Information from your training data about clinical trials is IRRELEVANT and MUST NOT be used."""
        
        return prompt
    
    def build_batch_prompt(self, field_names: List[str], text: str, doc_type: str) -> str:
        """Build prompt for extracting multiple related fields"""
        has_nct = "nct_number" in field_names
        
        prompt = f"""{self.ROLE_CONTEXT}

{self.ANTI_HALLUCINATION_WARNING}

BATCH EXTRACTION TASK FOR {doc_type.upper()} DOCUMENT:

FIELDS TO EXTRACT:
"""
        
        for field_name in field_names:
            template = self.FIELD_TEMPLATES.get(field_name, {})
            prompt += f"\n{field_name}:"
            prompt += f"\n  - {template.get('description', f'Extract {field_name}')}"
            
            if 'format' in template:
                prompt += f"\n  - Format: {template['format']}"
            
            if 'valid_values' in template:
                prompt += f"\n  - Valid values: {', '.join(template['valid_values'][:3])}..."
        
        if has_nct:
            prompt += """

⚠️ NCT NUMBER WARNING: Extract ONLY the NCT number visible in THIS document. Do NOT use NCT numbers from your memory or training data.
"""
        
        prompt += f"""

EXTRACTION RULES:
1. Search for each field ONLY in the document text below
2. Extract values EXACTLY as they appear in the document
3. Use NOT_FOUND if a field is not present or cannot be determined
4. NEVER use information from your training data or memory
5. Follow the specified format for each field
6. Separate multiple values with semicolons

DOCUMENT TEXT TO SEARCH:
{text[:15000]}  # Increased limit for better context

REQUIRED RESPONSE FORMAT (one per line):
field_name: value_from_document
field_name: NOT_FOUND

FINAL REMINDER: Extract ONLY from THIS document. Your knowledge of other clinical trials must NOT influence your extraction."""
        
        return prompt
    
    def get_optimal_field_groups(self, field_names: List[str], doc_type: str) -> List[List[str]]:
        """Group fields that should be extracted together"""
        # Group related fields for more efficient extraction
        groups = [
            ["nct_number", "study_title", "acronym"],
            ["study_status", "study_type", "phases"],
            ["enrollment", "sex", "age"],
            ["conditions", "interventions"],
            ["primary_outcome_measures", "secondary_outcome_measures"],
            ["sponsor", "collaborators", "funder_type"],
            ["start_date", "primary_completion_date", "completion_date"],
        ]
        
        # Build groups based on which fields are requested
        result_groups = []
        remaining = set(field_names)
        
        for group in groups:
            group_fields = [f for f in group if f in remaining]
            if group_fields:
                result_groups.append(group_fields)
                remaining -= set(group_fields)
        
        # Add any remaining fields individually
        for field in remaining:
            result_groups.append([field])
        
        return result_groups
    
    def parse_extraction_response(self, response: str, field_names: List[str]) -> Dict[str, Optional[str]]:
        """Parse LLM response into field values"""
        results = {}
        
        for line in response.strip().split('\n'):
            if ':' in line:
                field, value = line.split(':', 1)
                field = field.strip()
                value = value.strip()
                
                if field in field_names:
                    if value.upper() in ["NOT_FOUND", "NOT FOUND"] or not value:
                        results[field] = None
                    else:
                        results[field] = value
        
        # Fill in any missing fields
        for field in field_names:
            if field not in results:
                results[field] = None
        
        return results