"""
Field descriptions to provide context for better extraction accuracy.
"""
from typing import List

FIELD_DESCRIPTIONS = {
    'nct_number': {
        'description': 'The unique NCT identifier from ClinicalTrials.gov',
        'context': """The NCT number is a unique identifier that starts with "NCT" followed by 8 digits.
It may appear in the document or in the filename.
Example: NCT12345678""",
        'extraction_hints': [
            'Look for "NCT" followed by 8 digits',
            'Check the filename if not found in document',
            'Do not confuse with protocol numbers (e.g., PM1183-B-005-14)'
        ]
    },
    
    'enrollment': {
        'description': 'The total number of participants to be enrolled in the study',
        'context': """Enrollment refers to the NUMBER of patients/participants, not the duration.
Look for the total number of people who will participate in the study.
This is typically a number like "500 patients" or "100-200 participants".""",
        'extraction_hints': [
            'Extract the NUMBER of patients, not study duration',
            'Look for "X patients will be enrolled" or "planned enrollment of X"',
            'May be expressed as a range (e.g., "135-350 patients")',
            'Common keywords: patients, participants, subjects, evaluable'
        ]
    },
    
    'brief_title': {
        'description': 'A short descriptive title of the clinical study',
        'context': """The brief title is a concise name for the study.
It's usually shorter than the official title and summarizes the key elements.""",
        'extraction_hints': [
            'Often appears near the beginning of the document',
            'Shorter than the official/full title',
            'May include drug name and condition'
        ]
    },
    
    'official_title': {
        'description': 'The full scientific title of the clinical study',
        'context': """The official title is the complete, formal title of the study.
It typically includes study phase, design, drug name, and condition.""",
        'extraction_hints': [
            'Usually longer and more detailed than brief title',
            'Often includes "Phase", "Randomized", "Controlled", etc.',
            'May span multiple lines'
        ]
    },
    
    'primary_outcome_measures': {
        'description': 'The main outcomes used to determine if the treatment works',
        'context': """Primary outcomes are the most important measures of treatment effect.
They are the main results that the study is designed to evaluate.
Often includes a time frame for measurement.""",
        'extraction_hints': [
            'Look for "primary outcome", "primary endpoint", "primary efficacy"',
            'Usually includes measurement method and time frame',
            'May be a single outcome or a short list',
            'Format often includes "[Time Frame: X weeks/months]"'
        ]
    },
    
    'secondary_outcome_measures': {
        'description': 'Additional outcomes that provide supporting information',
        'context': """Secondary outcomes are additional measures that support the primary analysis.
They provide extra information about treatment effects but are not the main focus.""",
        'extraction_hints': [
            'Look for "secondary outcome", "secondary endpoint"',
            'Usually a list of multiple measures',
            'Each should include what is measured and when',
            'Often follows the primary outcomes section'
        ]
    },
    
    'sponsor': {
        'description': 'The organization responsible for the clinical trial',
        'context': """The sponsor is the organization that oversees and funds the study.
This is typically a pharmaceutical company, university, or research organization.""",
        'extraction_hints': [
            'Look for "Sponsor:", "Lead Sponsor:", "Sponsored by"',
            'Usually a company or institution name',
            'May include both sponsor and collaborators'
        ]
    },
    
    'conditions': {
        'description': 'The diseases, disorders, or conditions being studied',
        'context': """Conditions are the medical problems that the study is trying to treat or understand.
May be one or multiple conditions.""",
        'extraction_hints': [
            'Look for disease names, medical conditions',
            'May be listed as "Indication:", "Disease:", "Condition:"',
            'Can be a single condition or a list'
        ]
    },
    
    'interventions': {
        'description': 'The treatments or procedures being tested',
        'context': """Interventions are what participants receive during the study.
This includes drugs, devices, procedures, or other treatments being tested.""",
        'extraction_hints': [
            'Look for drug names, doses, procedures',
            'May include both study drug and comparators (placebo)',
            'Often includes route of administration (oral, IV, etc.)'
        ]
    },
    
    'inclusion_criteria': {
        'description': 'Requirements that participants must meet to join the study',
        'context': """Inclusion criteria define who can participate in the study.
These are the characteristics participants must have to be eligible.""",
        'extraction_hints': [
            'Look for "Inclusion Criteria:", "Eligible if:", "Must have:"',
            'Usually a numbered or bulleted list',
            'Includes age ranges, disease requirements, etc.'
        ]
    },
    
    'exclusion_criteria': {
        'description': 'Conditions that prevent participation in the study',
        'context': """Exclusion criteria define who cannot participate in the study.
These are characteristics or conditions that make someone ineligible.""",
        'extraction_hints': [
            'Look for "Exclusion Criteria:", "Not eligible if:", "Must not have:"',
            'Usually a numbered or bulleted list',
            'Includes safety concerns, conflicting medications, etc.'
        ]
    }
}

def get_field_context(field_name: str) -> str:
    """Get the context description for a field."""
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    return field_info.get('context', '')

def get_extraction_hints(field_name: str) -> List[str]:
    """Get extraction hints for a field."""
    field_info = FIELD_DESCRIPTIONS.get(field_name, {})
    return field_info.get('extraction_hints', [])