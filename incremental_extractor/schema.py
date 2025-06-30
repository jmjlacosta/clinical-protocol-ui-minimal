"""Schema definitions for the incremental extraction system"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ExtractionStatus(str, Enum):
    """Status of field extraction"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class FieldExtraction(BaseModel):
    """Individual field extraction result"""
    field_name: str
    value: Optional[Any] = None
    status: ExtractionStatus = ExtractionStatus.PENDING
    error_message: Optional[str] = None
    extraction_time: Optional[datetime] = None
    confidence: Optional[float] = None
    source_text: Optional[str] = None  # Text snippet used for extraction

class ExtractionCheckpoint(BaseModel):
    """Checkpoint for resuming extraction"""
    nct_number: str
    pdf_path: str
    pdf_type: str  # Protocol, SAP, ICF
    total_fields: int
    completed_fields: int = 0
    failed_fields: int = 0
    skipped_fields: int = 0
    start_time: datetime
    last_update: datetime
    fields: Dict[str, FieldExtraction] = Field(default_factory=dict)
    pdf_text_hash: Optional[str] = None  # To detect if PDF changed
    
    @property
    def is_complete(self) -> bool:
        """Check if all fields have been processed"""
        return self.completed_fields + self.failed_fields + self.skipped_fields >= self.total_fields
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total_fields == 0:
            return 0.0
        return (self.completed_fields + self.failed_fields + self.skipped_fields) / self.total_fields * 100

class ComparisonResult(BaseModel):
    """Result of comparing extracted data with CT.gov data"""
    field_name: str
    extracted_value: Optional[Any] = None
    ctgov_value: Optional[Any] = None
    match: bool = False
    similarity_score: Optional[float] = None
    notes: Optional[str] = None

class StudyComparison(BaseModel):
    """Complete comparison for a study"""
    nct_number: str
    pdf_type: str
    pdf_path: str
    comparison_time: datetime
    total_fields: int
    matching_fields: int = 0
    mismatched_fields: int = 0
    missing_in_extraction: int = 0
    missing_in_ctgov: int = 0
    field_comparisons: List[ComparisonResult] = Field(default_factory=list)
    
    @property
    def match_percentage(self) -> float:
        """Get matching percentage"""
        if self.total_fields == 0:
            return 0.0
        return self.matching_fields / self.total_fields * 100

# CT.gov field mappings - based on the CSV headers
CTGOV_FIELD_MAPPING = {
    "nct_number": "NCT Number",
    "study_title": "Study Title",
    "study_url": "Study URL",
    "acronym": "Acronym",
    "study_status": "Study Status",
    "brief_summary": "Brief Summary",
    "study_results": "Study Results",
    "conditions": "Conditions",
    "interventions": "Interventions",
    "primary_outcome_measures": "Primary Outcome Measures",
    "secondary_outcome_measures": "Secondary Outcome Measures",
    "other_outcome_measures": "Other Outcome Measures",
    "sponsor": "Sponsor",
    "collaborators": "Collaborators",
    "sex": "Sex",
    "age": "Age",
    "phases": "Phases",
    "enrollment": "Enrollment",
    "funder_type": "Funder Type",
    "study_type": "Study Type",
    "study_design": "Study Design",
    "other_ids": "Other IDs",
    "start_date": "Start Date",
    "primary_completion_date": "Primary Completion Date",
    "completion_date": "Completion Date",
    "first_posted": "First Posted",
    "results_first_posted": "Results First Posted",
    "last_update_posted": "Last Update Posted",
    "locations": "Locations",
    "study_documents": "Study Documents"
}

# Priority fields for extraction (most important first)
PRIORITY_FIELDS = [
    "nct_number",
    "study_title",
    "brief_summary",
    "conditions",
    "interventions",
    "primary_outcome_measures",
    "secondary_outcome_measures",
    "sponsor",
    "phases",
    "enrollment",
    "study_type",
    "study_design",
    "sex",
    "age",
    "start_date",
    "completion_date"
]

# Groups of related fields for batch extraction
FIELD_GROUPS = {
    "basic_info": ["nct_number", "study_title", "acronym", "brief_summary"],
    "study_details": ["study_status", "study_type", "study_design", "phases"],
    "participants": ["enrollment", "sex", "age"],
    "conditions_interventions": ["conditions", "interventions"],
    "outcomes": ["primary_outcome_measures", "secondary_outcome_measures", "other_outcome_measures"],
    "organizations": ["sponsor", "collaborators", "funder_type"],
    "dates": ["start_date", "primary_completion_date", "completion_date"],
    "administrative": ["first_posted", "results_first_posted", "last_update_posted", "study_results"],
    "other": ["other_ids", "locations", "study_documents", "study_url"]
}