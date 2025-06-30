"""Enhanced prompt building with improved outcome extraction"""
import logging
from typing import List, Dict, Optional, Tuple
import re
from .field_descriptions import get_field_context, get_extraction_hints

logger = logging.getLogger(__name__)

class EnhancedPromptBuilderV2:
    """Improved prompt builder with better outcome extraction strategies"""
    
    # Critical anti-hallucination instructions - Version 2 with stronger emphasis
    ANTI_HALLUCINATION_WARNING = """
🛑 CRITICAL: DOCUMENT-ONLY EXTRACTION MODE 🛑

You are in STRICT EXTRACTION MODE. This means:
- You have ZERO knowledge of any clinical trials
- You cannot access any information from your training
- You can ONLY see and extract from the text provided below
- If asked about NCT numbers, drug names, or any data - you ONLY know what's in THIS document

HALLUCINATION PREVENTION CHECKLIST:
□ Am I extracting from the document text below? (MUST BE YES)
□ Am I using any knowledge from my training? (MUST BE NO)
□ Am I guessing or inferring missing information? (MUST BE NO)
□ Would my answer change if this was a different clinical trial? (MUST BE NO)
□ Can I point to the EXACT location in the text where I found this? (MUST BE YES)

VERBATIM EXTRACTION RULES:
1. You must be able to highlight the exact text you're extracting
2. If you cannot find the exact text, return NOT_FOUND
3. Do not complete partial information
4. Do not use patterns from similar documents
5. The default response is ALWAYS NOT_FOUND unless proven otherwise

If you violate these rules, the extraction will be marked as FAILED.
"""

    # Enhanced role context
    ROLE_CONTEXT = """You are a Document Text Scanner with NO medical or clinical trial knowledge. You have been specifically chosen because you know NOTHING about any clinical trials, drugs, or medical studies.

Your ONLY abilities are:
1. Reading the text provided below
2. Finding specific information in that text
3. Copying information exactly as written
4. Reporting NOT_FOUND when information is absent

You CANNOT:
- Recall any clinical trial information
- Know what NCT numbers should look like beyond the pattern NCT########
- Remember any drug names or dosages
- Infer missing information
- Complete partial matches
- Use similar values from other documents

IMPORTANT: For EVERY extraction, you must be able to quote the exact sentence or phrase from the document where you found the information. If you cannot do this, the answer is NOT_FOUND.

Think of yourself as a very basic search function that can only find and copy text."""
    
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
        
        "primary_outcome_measures": {
            "description": "Primary endpoints - the main outcomes being measured",
            "examples": [
                "Overall Response Rate (ORR) [Time Frame: 6 months]",
                "Progression-Free Survival (PFS) [Time Frame: 12 months]",
                "Overall Response Rate (ORR) [Time Frame: From start of treatment to progression]"
            ],
            "hints": [
                "Look for sections labeled 'PRIMARY ENDPOINT', 'Primary Outcome', 'Primary Objective'",
                "May include ORR, PFS, OS, DFS, or other clinical measures",
                "Timeframe might be described separately or in the methodology section",
                "If no explicit timeframe, look for assessment schedule (e.g., 'assessed every 2 cycles')"
            ],
            "instructions": """Extract the primary outcome measure(s) including:
1. The outcome name (e.g., "Overall Response Rate (ORR)")
2. The timeframe if specified (e.g., "6 months", "until progression", "every 2 cycles")
3. If timeframe is not explicitly stated with the outcome, look for it in the surrounding text

If the document says "Primary endpoint: Overall Response Rate (ORR)" and elsewhere mentions "assessed until progression", 
format as: "Overall Response Rate (ORR) [Time Frame: Until progression]"

Multiple primary outcomes should be separated by semicolons.""",
            "format": "Outcome name [Time Frame: duration/schedule]"
        },
        
        "secondary_outcome_measures": {
            "description": "Secondary endpoints - additional outcomes being measured",
            "examples": [
                "Duration of Response (DR) [Time Frame: From response to progression]",
                "Clinical Benefit [Time Frame: 4 months]",
                "Progression-free Survival (PFS) [Time Frame: From first infusion to progression or death]",
                "Overall Survival (OS) [Time Frame: From first infusion to death or last follow-up]"
            ],
            "hints": [
                "Look for 'SECONDARY ENDPOINTS', 'Secondary Outcomes', 'Secondary Objectives'",
                "Often listed after primary endpoints",
                "May include DR, CB, PFS, OS, safety measures, QoL",
                "Include all secondary measures listed"
            ],
            "instructions": """Extract ALL secondary outcome measures. For each:
1. Include the full outcome name
2. Add timeframe in brackets if available
3. If timeframe is described separately, combine them
4. Separate multiple outcomes with semicolons

Common abbreviations: DR (Duration of Response), CB (Clinical Benefit), PFS (Progression-free Survival), 
OS (Overall Survival), ORR (Overall Response Rate), AE (Adverse Events)""",
            "format": "Outcome name [Time Frame: duration]; Next outcome [Time Frame: duration]"
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
        }
    }
    
    def _preprocess_text_for_outcomes(self, text: str, field_name: str) -> str:
        """Preprocess text to focus on outcome-relevant sections"""
        if "outcome_measures" not in field_name:
            return text
        
        # Find sections that likely contain outcome information
        outcome_sections = []
        lines = text.split('\n')
        
        # Keywords to search for
        if "primary" in field_name:
            keywords = ['primary endpoint', 'primary outcome', 'primary objective', '3.1 primary']
        else:
            keywords = ['secondary endpoint', 'secondary outcome', 'secondary objective', '3.2 secondary']
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                # Extract context around the match
                start = max(0, i - 5)
                end = min(len(lines), i + 50)  # Get more context for outcomes
                context = '\n'.join(lines[start:end])
                outcome_sections.append(context)
        
        # If we found relevant sections, use them; otherwise use original text
        if outcome_sections:
            # Combine the sections with the beginning of the document for context
            focused_text = text[:5000] + "\n\n[...]\n\n" + "\n\n---\n\n".join(outcome_sections[:3])
            return focused_text[:40000]  # Still respect the limit
        
        return text[:40000]
    
    def build_single_field_prompt(self, field_name: str, text: str, doc_type: str) -> str:
        """Build enhanced prompt for extracting a single field"""
        template = self.FIELD_TEMPLATES.get(field_name, {})
        
        # Preprocess text for outcome fields
        if "outcome_measures" in field_name:
            text = self._preprocess_text_for_outcomes(text, field_name)
        
        prompt = f"""{self.ROLE_CONTEXT}

{self.ANTI_HALLUCINATION_WARNING}

DOCUMENT-ONLY EXTRACTION TASK:
You are scanning a {doc_type} document for a specific field.

FIELD TO EXTRACT: {field_name}

FIELD DETAILS:
- Description: {template.get('description', f'Extract {field_name}')}
"""
        
        # Add field-specific context
        field_context = get_field_context(field_name)
        if field_context:
            prompt += f"\n=== WHAT THIS FIELD MEANS ===\n{field_context}\n"
        
        extraction_hints = get_extraction_hints(field_name)
        if extraction_hints:
            prompt += "\n=== HOW TO FIND IT ===\n"
            for hint in extraction_hints:
                prompt += f"• {hint}\n"
        
        # Extra warning for NCT number
        if field_name == "nct_number":
            prompt += """

⛔ EXTREME CAUTION FOR NCT NUMBER ⛔
This is the #1 field where hallucination occurs. Common mistakes:
- Using NCT numbers from similar studies in training data
- "Remembering" the NCT number from the filename or context
- Generating a plausible NCT number when none exists
- Using NCT00514324 or other memorized numbers

CORRECT APPROACH:
1. Search for the pattern "NCT" followed by 8 digits IN THE PROVIDED TEXT
2. You must be able to quote the exact line where you found it
3. If not found in the text below, return NOT_FOUND
4. Common locations: cover page, header, footer, first few pages
5. The document might be about study NCT12345678 but if the text shows NCT87654321, extract NCT87654321

VERBATIM REQUIREMENT: You must show the exact text snippet where you found the NCT number, like:
"Found in text: 'ClinicalTrials.gov Identifier: NCT02454972'"
If you cannot provide this quote, return NOT_FOUND
"""
        
        if 'instructions' in template:
            prompt += f"\nDETAILED INSTRUCTIONS:\n{template['instructions']}\n"
        
        if 'valid_values' in template:
            prompt += f"- Valid Values: {', '.join(template['valid_values'])}\n"
        
        if 'examples' in template:
            prompt += f"- Examples:\n"
            for example in template['examples']:
                prompt += f"  • {example}\n"
        
        if 'format' in template:
            prompt += f"- Required Format: {template['format']}\n"
        
        if 'hints' in template:
            prompt += f"- Where to Look:\n"
            for hint in template['hints']:
                prompt += f"  • {hint}\n"
        
        if 'patterns' in template:
            prompt += f"- Patterns: {'; '.join(template['patterns'])}\n"
            
        if 'avoid' in template:
            prompt += f"- Avoid: {'; '.join(template['avoid'])}\n"
        
        prompt += f"""

EXTRACTION PROTOCOL:
1. SCAN the document text below for "{field_name}"
2. COPY exactly what you find (no modifications)
3. If the field is absent or unclear, return NOT_FOUND
4. DO NOT use any external knowledge or memory
5. For outcome measures, combine the outcome name with its timeframe
6. If multiple values exist, include all (separated by semicolons)

VERBATIM EXTRACTION REQUIREMENT:
For your extraction to be valid, you must:
1. Locate the exact text in the document
2. Be able to quote the surrounding context
3. Never complete partial information
4. Never use similar values from memory

DOCUMENT TEXT (THIS IS YOUR ONLY SOURCE):
{text}

REQUIRED RESPONSE FORMAT:
{field_name}: [exact value from document or NOT_FOUND]

For NCT numbers, also include where you found it:
Source quote: [exact line from document containing the NCT number]

FINAL CHECK: Before responding, ask yourself:
- Can I point to the EXACT location in the text above where I found this?
- Am I 100% certain this came from the document, not my memory?
- Would I extract the same value if this was a completely different clinical trial?
If any answer is NO, return NOT_FOUND."""
        
        return prompt
    
    def build_batch_prompt(self, field_names: List[str], text: str, doc_type: str) -> str:
        """Build prompt for extracting multiple related fields"""
        # Check if we're extracting outcomes
        has_outcomes = any("outcome_measures" in fn for fn in field_names)
        
        if has_outcomes:
            # Preprocess text to focus on outcome sections
            for fn in field_names:
                if "outcome_measures" in fn:
                    text = self._preprocess_text_for_outcomes(text, fn)
                    break  # Only need to do this once
        
        prompt = f"""You are extracting clinical trial data from a {doc_type} document.

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
        
        prompt += f"""

EXTRACTION RULES:
1. Extract each field exactly as it appears
2. For outcome measures, include timeframes in brackets
3. Use NOT_FOUND if a field is not present
4. Follow the specified format for each field
5. Separate multiple values with semicolons

DOCUMENT TEXT:
{text}

RESPONSE FORMAT (one per line):
field_name: value
"""
        
        return prompt
    
    def get_optimal_field_groups(self, field_names: List[str], doc_type: str) -> List[List[str]]:
        """Group fields that should be extracted together"""
        # Group related fields for more efficient extraction
        groups = [
            ["nct_number", "study_title", "acronym"],
            ["study_status", "study_type", "phases"],
            ["enrollment", "sex", "age"],
            ["conditions", "interventions"],
            ["primary_outcome_measures"],  # Extract primary outcomes separately
            ["secondary_outcome_measures"],  # Extract secondary outcomes separately
            ["other_outcome_measures"],  # Extract other outcomes separately
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
                    if value.upper() in ["NOT_FOUND", "NOT FOUND", "NONE", "N/A"] or not value:
                        results[field] = None
                    else:
                        results[field] = value
        
        # Fill in any missing fields
        for field in field_names:
            if field not in results:
                results[field] = None
        
        return results