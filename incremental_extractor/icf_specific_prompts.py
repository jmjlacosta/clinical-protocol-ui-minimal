"""ICF-specific prompt improvements for challenging fields"""

ICF_FIELD_IMPROVEMENTS = {
    "conditions": {
        "prompt": """Look for specific medical conditions in sections like:
- "What is the purpose of this study?"
- "Why are we doing this research?"
- "What condition is being studied?"

Extract the SPECIFIC diseases/conditions mentioned, not generic text.
Examples: "pneumonia", "urinary tract infections", "skin infections"
NOT: "Medical conditions or diseases being studied"
""",
        "search_patterns": [
            "children with (.*?) who",
            "patients with (.*?) will",
            "study of (.*?) in",
            "treatment for (.*?)",
            "(pneumonia|infection|disease|disorder|syndrome|cancer)"
        ]
    },
    
    "age": {
        "prompt": """In ICF documents, age eligibility is often in:
- "Who can participate?" section
- "Eligibility" or "Can my child participate?"
- May be described as "children", "adults", "18 years or older"

Convert descriptions to age ranges:
- "children" → "0 Years to 17 Years"  
- "adults" → "18 Years and older"
- "children and adults" → "All ages"
- "6 months to 18 years" → "6 Months to 18 Years"
""",
        "search_patterns": [
            "age(d)? (\\d+)",
            "(\\d+) (years|months) (old|to|and)",
            "children|pediatric|adult|infant",
            "participate if .{0,50}(age|years old)"
        ]
    },
    
    "enrollment": {
        "prompt": """For enrollment/sample size in ICF, look for:
- "How many people will be in this study?"
- "About [number] participants"
- "We plan to enroll [number]"
- "Up to [number] subjects"

Extract ONLY the number, not text.
If it says "about 5,600" → extract "5600"
If it says "1000-1200" → extract "1100" (midpoint)
""",
        "validation": "Must be a number between 1 and 100000"
    },
    
    "interventions": {
        "prompt": """In ICF documents, interventions are described in:
- "What will happen in this study?"
- "Study procedures"
- "What are we testing?"
- "Study treatment" or "Study drug"

Format as: "Type: Description"
Types: Drug, Device, Behavioral, Procedure, Other

Example extractions:
- "antibiotic guidelines" → "Behavioral: Antibiotic prescribing guidelines"
- "new medication called X" → "Drug: X"
- "education program" → "Behavioral: Education program"
""",
        "search_sections": [
            "what will happen",
            "study procedures", 
            "what are we testing",
            "study involves"
        ]
    },
    
    "primary_outcome_measures": {
        "prompt": """In ICF documents, primary outcomes are often described as:
- "What are we trying to find out?"
- "Main goal of the study"
- "What will we measure?"
- "Primary purpose"

Look for SPECIFIC measurements with timeframes.
Format: "Outcome description [Time Frame: X days/weeks/months]"

If timeframe not specified, use [Time Frame: End of study]
""",
        "examples": [
            "Antibiotic prescription appropriateness [Time Frame: 30 days post-discharge]",
            "Treatment failure rate [Time Frame: 7 days]"
        ]
    },
    
    "brief_summary": {
        "prompt": """Create a concise summary (2-3 sentences) that captures:
1. What condition/problem is being studied
2. What intervention is being tested  
3. What the study hopes to achieve

Avoid copying verbatim - synthesize the key information.

Good example: "This study tests whether discharge antibiotic guidelines can reduce inappropriate prescriptions in children. Hospitals will implement new procedures and measure prescription appropriateness. The goal is to improve antibiotic use and reduce treatment failures."
"""
    }
}

def get_enhanced_icf_prompt(field_name: str, base_prompt: str) -> str:
    """Enhance prompts specifically for ICF documents"""
    if field_name in ICF_FIELD_IMPROVEMENTS:
        improvements = ICF_FIELD_IMPROVEMENTS[field_name]
        enhanced = base_prompt + "\n\nICF-SPECIFIC GUIDANCE:\n"
        enhanced += improvements.get("prompt", "")
        
        if "examples" in improvements:
            enhanced += f"\n\nExamples:\n"
            for ex in improvements["examples"]:
                enhanced += f"- {ex}\n"
                
        if "search_patterns" in improvements:
            enhanced += f"\n\nLook for patterns like: {', '.join(improvements['search_patterns'][:3])}"
            
        return enhanced
    return base_prompt