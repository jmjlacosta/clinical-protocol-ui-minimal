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
from incremental_extractor.checkpoint_manager import CheckpointManager
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
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(str(self.results_dir / "extractions"))
        
        # Check for API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        
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
            
            # Create extraction directory for this NCT
            nct_dir = self.results_dir / "extractions" / job.nct_number
            nct_dir.mkdir(parents=True, exist_ok=True)
            
            # Run extraction for each document
            all_extractions = {}
            total_docs = len(pdf_paths)
            
            for idx, (pdf_path, doc_type) in enumerate(pdf_paths):
                if progress_callback:
                    progress = 10 + (idx / total_docs) * 60  # 10-70% for extraction
                    progress_callback(progress, f"Extracting from {doc_type}...")
                
                # Save PDF to examples directory (expected by extractor)
                examples_dir = Path("examples")
                examples_dir.mkdir(exist_ok=True)
                
                # Copy PDF with proper naming
                dest_filename = f"{job.nct_number}_{doc_type}_{Path(pdf_path).name}"
                dest_path = examples_dir / dest_filename
                
                import shutil
                shutil.copy2(pdf_path, dest_path)
                
                # Run extraction using subprocess
                cmd = ["python3", "run_incremental_extraction.py", 
                       "--nct", job.nct_number, 
                       "--pdf-type", doc_type]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Load checkpoint
                    checkpoint_file = Path("checkpoints") / f"{job.nct_number}_{doc_type.lower()}_checkpoint.json"
                    if checkpoint_file.exists():
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                            all_extractions[doc_type] = checkpoint
                            
                            # Copy checkpoint to results directory
                            dest_checkpoint = nct_dir / "checkpoints" / checkpoint_file.name
                            dest_checkpoint.parent.mkdir(exist_ok=True)
                            shutil.copy2(checkpoint_file, dest_checkpoint)
                else:
                    logger.error(f"Extraction failed for {doc_type}: {result.stderr}")
            
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
                    progress_callback(85, "Generating comparison report...")
                
                # Generate report (saves directly to results directory)
                extractor.generate_enhanced_report(job.nct_number, unified_data)
                
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
                    'report_path': str(nct_dir / "unified" / f"{job.nct_number}_unified_report.md")
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