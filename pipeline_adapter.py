"""
Pipeline adapter for integrating extraction pipelines with Streamlit UI
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import tempfile
import subprocess
from dataclasses import dataclass, field
import csv

from agents.orchestrator.main import extract_info as extract_info_legacy
from incremental_extractor.extractor import IncrementalExtractor
from unified_extractor_enhanced import EnhancedUnifiedExtractor

logger = logging.getLogger(__name__)

@dataclass
class ExtractionJob:
    """Represents an extraction job"""
    job_id: str
    nct_number: str
    pdf_files: List[Tuple[str, bytes, str]]  # (filename, content, doc_type)
    pipeline_type: str  # 'legacy' or 'enhanced'
    status: str = 'pending'  # pending, running, completed, failed
    progress: float = 0.0
    current_step: str = ''
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class PipelineAdapter:
    """Unified interface for extraction pipelines"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Check for API key
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        
    def extract_nct_number(self, filename: str) -> Optional[str]:
        """Extract NCT number from filename"""
        # Common patterns: NCT12345678_Protocol.pdf, NCT12345678_SAP_v1.pdf
        import re
        match = re.search(r'(NCT\d{8})', filename)
        return match.group(1) if match else None
    
    def determine_doc_type(self, filename: str) -> str:
        """Determine document type from filename"""
        filename_lower = filename.lower()
        if '_prot_' in filename_lower or '_protocol' in filename_lower or 'protocol' in filename_lower:
            return 'Protocol'
        elif '_sap_' in filename_lower or 'sap' in filename_lower:
            return 'SAP'
        elif '_icf_' in filename_lower or 'icf' in filename_lower:
            return 'ICF'
        else:
            return 'Unknown'
    
    def run_legacy_extraction(self, job: ExtractionJob, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run legacy extraction pipeline"""
        results = {}
        
        for idx, (filename, pdf_bytes, doc_type) in enumerate(job.pdf_files):
            if progress_callback:
                progress = (idx / len(job.pdf_files)) * 100
                progress_callback(progress, f"Processing {filename} with legacy pipeline...")
            
            try:
                # Legacy pipeline expects single PDF
                extraction = extract_info_legacy(pdf_bytes)
                
                # Add metadata
                extraction['_metadata'] = {
                    'filename': filename,
                    'doc_type': doc_type,
                    'nct_number': job.nct_number,
                    'extraction_time': datetime.now().isoformat()
                }
                
                results[filename] = {
                    'status': 'success',
                    'extraction': extraction
                }
                
            except Exception as e:
                logger.error(f"Legacy extraction failed for {filename}: {e}")
                results[filename] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def run_enhanced_extraction(self, job: ExtractionJob, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run enhanced extraction pipeline with CT.gov comparison"""
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Save PDFs to temp directory
        temp_dir = Path(tempfile.mkdtemp())
        pdf_paths = []
        
        try:
            # Save PDFs temporarily
            for filename, pdf_bytes, doc_type in job.pdf_files:
                pdf_path = temp_dir / filename
                pdf_path.write_bytes(pdf_bytes)
                pdf_paths.append((str(pdf_path), doc_type))
            
            if progress_callback:
                progress_callback(10, "PDFs saved, initializing enhanced extractor...")
            
            # Initialize enhanced extractor
            extractor = EnhancedUnifiedExtractor()
            
            # Run extraction for each document
            all_extractions = {}
            total_docs = len(pdf_paths)
            
            for idx, (pdf_path, doc_type) in enumerate(pdf_paths):
                if progress_callback:
                    progress = 10 + (idx / total_docs) * 60  # 10-70% for extraction
                    progress_callback(progress, f"Extracting from {doc_type}...")
                
                # Run extraction directly using IncrementalExtractor
                try:
                    incremental_extractor = IncrementalExtractor(api_key=self.api_key)
                    
                    # Look for CT.gov CSV in examples directory
                    ctgov_csv_path = None
                    examples_dir = Path("examples")
                    if examples_dir.exists():
                        csv_files = list(examples_dir.glob(f"{job.nct_number}_ct_*.csv"))
                        if csv_files:
                            ctgov_csv_path = str(csv_files[0])
                    
                    # Extract from PDF using temp path directly
                    checkpoint = incremental_extractor.extract_from_pdf(
                        pdf_path=pdf_path,  # Use temp file directly
                        nct_number=job.nct_number,
                        pdf_type=doc_type,
                        resume=False,  # Always start fresh
                        compare_immediately=True,
                        ctgov_csv_path=ctgov_csv_path
                    )
                    
                    # Convert checkpoint to dict for compatibility
                    extraction_dict = {
                        'nct_number': checkpoint.nct_number,
                        'pdf_type': checkpoint.pdf_type,
                        'pdf_path': pdf_path,  # Add the missing pdf_path
                        'fields': {
                            field_name: {
                                'value': field.value,
                                'status': field.status.value,
                                'extraction_time': field.extraction_time.isoformat() if field.extraction_time else None
                            }
                            for field_name, field in checkpoint.fields.items()
                        },
                        'completed_fields': checkpoint.completed_fields,
                        'total_fields': checkpoint.total_fields,
                        'progress_percentage': checkpoint.progress_percentage
                    }
                    
                    all_extractions[doc_type] = extraction_dict
                    
                except Exception as e:
                    logger.error(f"Extraction failed for {doc_type}: {e}")
            
            if progress_callback:
                progress_callback(70, "Merging extractions...")
            
            # Merge extractions
            if all_extractions:
                unified_data = extractor.merge_extractions(job.nct_number, all_extractions)
                
                # Check for CT.gov data
                ctgov_csv = None
                csv_files = list(Path("examples").glob(f"{job.nct_number}_ct_*.csv"))
                if csv_files:
                    ctgov_csv = csv_files[0]
                
                if progress_callback:
                    progress_callback(85, "Finalizing extraction...")
                
                # Load CT.gov data for comparison if available
                ctgov_data = {}
                if ctgov_csv:
                    with open(ctgov_csv, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        ctgov_data = next(reader, {})
                
                return {
                    'status': 'success',
                    'unified_data': unified_data,
                    'ctgov_data': ctgov_data,
                    'report_path': None  # Not saving to disk
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'No extractions succeeded'
                }
                
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def process_job(self, job: ExtractionJob, progress_callback: Optional[Callable] = None) -> ExtractionJob:
        """Process an extraction job"""
        job.status = 'running'
        job.start_time = datetime.now()
        
        try:
            if job.pipeline_type == 'legacy':
                job.results = self.run_legacy_extraction(job, progress_callback)
            elif job.pipeline_type == 'enhanced':
                job.results = self.run_enhanced_extraction(job, progress_callback)
            else:
                raise ValueError(f"Unknown pipeline type: {job.pipeline_type}")
            
            job.status = 'completed'
            if progress_callback:
                progress_callback(100, "Extraction complete!")
                
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            logger.error(f"Job {job.job_id} failed: {e}")
            if progress_callback:
                progress_callback(0, f"Extraction failed: {e}")
        
        job.end_time = datetime.now()
        return job
    
    def get_comparison_stats(self, unified_data: Dict[str, Any], ctgov_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison statistics"""
        stats = {
            'total_fields': 0,
            'extracted_fields': 0,
            'matched_fields': 0,
            'mismatched_fields': 0,
            'unique_extractions': 0,
            'extraction_rate': 0.0,
            'match_rate': 0.0,
            'by_document': {}
        }
        
        if 'statistics' in unified_data:
            stats.update(unified_data['statistics'])
        
        # Count matches/mismatches if we have CT.gov data
        if ctgov_data and 'fields' in unified_data:
            field_mapping = {
                'nct_number': 'NCT Number',
                'study_title': 'Study Title',
                'conditions': 'Conditions',
                'interventions': 'Interventions',
                'sponsor': 'Sponsor',
                # Add more mappings as needed
            }
            
            for field_name, field_data in unified_data['fields'].items():
                if field_data.get('value'):
                    ctgov_field = field_mapping.get(field_name, field_name)
                    ctgov_value = ctgov_data.get(ctgov_field, '')
                    
                    if ctgov_value:
                        # Simple comparison - could be enhanced
                        if str(field_data['value']).lower().strip() == ctgov_value.lower().strip():
                            stats['matched_fields'] += 1
                        else:
                            stats['mismatched_fields'] += 1
                    else:
                        stats['unique_extractions'] += 1
            
            total_compared = stats['matched_fields'] + stats['mismatched_fields']
            if total_compared > 0:
                stats['match_rate'] = (stats['matched_fields'] / total_compared) * 100
        
        return stats