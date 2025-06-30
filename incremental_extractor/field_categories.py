"""Categorize fields by how they should be extracted and validated"""

# Fields that should exist verbatim in the document
VERBATIM_FIELDS = {
    "nct_number",          # Exact NCT number
    "sponsor",             # Company names
    "collaborators",       # Company names
    "start_date",          # Specific dates
    "completion_date",     # Specific dates
    "enrollment",          # Exact number
    "phases",              # "Phase I", "Phase II", etc.
    "locations",           # Site names/countries
}

# Fields that are summaries/interpretations created by the LLM
SUMMARY_FIELDS = {
    "brief_summary",       # LLM creates a summary
    "study_design",        # LLM summarizes the design
    "study_results",       # Summary of results
    "interventions",       # Formatted list from scattered info
    "primary_outcome_measures",    # Combined from objectives/endpoints
    "secondary_outcome_measures",  # Combined from objectives/endpoints
}

# Fields that may be inferred from context
INFERRED_FIELDS = {
    "study_status",        # Might say "ongoing" or infer from dates
    "study_type",          # Inferred from design
    "sex",                 # May say "both" or infer from eligibility
    "age",                 # Extracted from eligibility criteria
    "funder_type",         # Inferred from sponsor type
}

# Fields that only exist in registry, not protocols
REGISTRY_ONLY_FIELDS = {
    "study_url",           # ClinicalTrials.gov URL
    "first_posted",        # Registry posting date
    "results_first_posted",# Registry results date
    "last_update_posted",  # Registry update date
    "study_documents",     # Links to documents
}

# Fields that might use different terminology
TERMINOLOGY_VARIANTS = {
    "conditions": ["indication", "disease", "disorder", "cancer type", "tumor type"],
    "interventions": ["study drug", "treatment", "therapy", "investigational product"],
    "primary_outcome_measures": ["primary endpoint", "primary objective", "primary efficacy"],
    "secondary_outcome_measures": ["secondary endpoint", "secondary objective"],
}