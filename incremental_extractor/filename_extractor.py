"""Extract metadata from filenames to prevent hallucinations and provide reliable baseline data"""
import re
import logging
from typing import Dict, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class FilenameExtractor:
    """Extract clinical trial metadata from filenames"""
    
    def __init__(self):
        # Regex patterns for various metadata
        self.patterns = {
            'nct_number': re.compile(r'NCT\d{8}', re.IGNORECASE),
            'document_type': re.compile(r'(_Prot_|Protocol|_SAP_|SAP|_ICF_|ICF)', re.IGNORECASE),
            'version': re.compile(r'[_v](\d+)|_(\d{3})'),
            'study_code': re.compile(r'([A-Z]+\d+(?:-[A-Z0-9]+-\d+|-\d+)*)', re.IGNORECASE),
            'date': re.compile(r'(\d{4}[-_]\d{2}[-_]\d{2}|\d{8})'),
        }
        
        # Document type mappings
        self.doc_type_map = {
            'prot': 'Protocol',
            'protocol': 'Protocol',
            'sap': 'SAP',
            'icf': 'ICF',
            'informed_consent': 'ICF',
            'consent': 'ICF'
        }
    
    def extract_all(self, filename: str) -> Dict[str, Optional[str]]:
        """Extract all available metadata from filename"""
        results = {}
        
        # Clean filename (remove path, keep only basename)
        basename = Path(filename).name
        
        # Extract NCT number (highest priority)
        nct_match = self.patterns['nct_number'].search(basename)
        if nct_match:
            nct_number = nct_match.group(0).upper()
            results['nct_number'] = nct_number
            logger.info(f"Extracted NCT number from filename: {nct_number}")
        
        # Extract document type
        doc_type = self.extract_document_type(basename)
        if doc_type:
            results['document_type'] = doc_type
            logger.info(f"Extracted document type from filename: {doc_type}")
        
        # Extract version
        version_match = self.patterns['version'].search(basename)
        if version_match:
            version = version_match.group(1) or version_match.group(2)
            results['version'] = version
            logger.info(f"Extracted version from filename: {version}")
        
        # Extract study codes (could be other_ids)
        study_codes = self.extract_study_codes(basename)
        if study_codes:
            results['study_codes'] = study_codes
            logger.info(f"Extracted study codes from filename: {study_codes}")
        
        # Extract dates
        date_match = self.patterns['date'].search(basename)
        if date_match:
            date_str = date_match.group(1).replace('_', '-')
            results['file_date'] = date_str
            logger.info(f"Extracted date from filename: {date_str}")
        
        return results
    
    def extract_nct_number(self, filename: str) -> Optional[str]:
        """Extract NCT number from filename - most reliable source"""
        basename = Path(filename).name
        match = self.patterns['nct_number'].search(basename)
        if match:
            return match.group(0).upper()
        return None
    
    def extract_document_type(self, filename: str) -> Optional[str]:
        """Extract document type from filename"""
        basename = Path(filename).name.lower()
        
        # Check against known patterns
        for pattern, doc_type in self.doc_type_map.items():
            if pattern in basename:
                return doc_type
        
        # Check regex pattern
        match = self.patterns['document_type'].search(basename)
        if match:
            matched_text = match.group(1).strip('_').lower()
            return self.doc_type_map.get(matched_text, matched_text.title())
        
        return None
    
    def extract_study_codes(self, filename: str) -> List[str]:
        """Extract study codes/identifiers from filename"""
        basename = Path(filename).name
        
        # Remove NCT number first to avoid matching it
        nct_removed = self.patterns['nct_number'].sub('', basename)
        
        # Find all potential study codes
        matches = self.patterns['study_code'].findall(nct_removed)
        
        # Filter out common false positives
        codes = []
        for match in matches:
            # Skip if it's just a file extension or common word
            if match.lower() not in ['pdf', 'doc', 'docx', 'txt', 'prot', 'sap', 'icf']:
                codes.append(match)
        
        return codes
    
    def create_extraction_hints(self, filename: str) -> Dict[str, str]:
        """Create hints for the LLM based on filename extraction"""
        hints = {}
        metadata = self.extract_all(filename)
        
        if 'nct_number' in metadata:
            hints['nct_number'] = f"The filename indicates this document is for trial {metadata['nct_number']}. Verify this matches the NCT number in the document."
        
        if 'document_type' in metadata:
            hints['document_type'] = f"This appears to be a {metadata['document_type']} document based on the filename."
        
        if 'study_codes' in metadata:
            hints['other_ids'] = f"The filename contains these potential study identifiers: {', '.join(metadata['study_codes'])}"
        
        return hints
    
    def validate_extraction(self, field_name: str, extracted_value: str, filename: str) -> Tuple[bool, Optional[str]]:
        """Validate extracted values against filename metadata"""
        metadata = self.extract_all(filename)
        
        if field_name == 'nct_number' and 'nct_number' in metadata:
            filename_nct = metadata['nct_number']
            
            # If extracted value doesn't match filename, prefer filename
            if extracted_value and extracted_value != filename_nct:
                logger.warning(f"NCT mismatch: Document extracted '{extracted_value}' but filename has '{filename_nct}'")
                return False, filename_nct
            
            # If no extraction but filename has NCT, use filename
            if not extracted_value:
                logger.info(f"Using NCT from filename: {filename_nct}")
                return True, filename_nct
        
        return True, extracted_value