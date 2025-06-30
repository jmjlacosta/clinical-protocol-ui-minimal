"""Main extraction engine - simplified version without checkpoints"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .pdf_extractor import extract_text_from_pdf
from .schema import (
    ExtractionCheckpoint, ExtractionStatus, FieldExtraction,
    ComparisonResult, StudyComparison, CTGOV_FIELD_MAPPING, PRIORITY_FIELDS
)
from .enhanced_prompt_builder_v2 import EnhancedPromptBuilderV2 as EnhancedPromptBuilder
from .intelligent_comparator import IntelligentComparator
from .smart_outcome_extractor import SmartOutcomeExtractor
from .smart_validator import SmartValidator
from .filename_extractor import FilenameExtractor
from .chunked_extractor import ChunkedExtractor
from .intelligent_chunker import IntelligentChunker
from .chunk_mapper import ChunkMapper

logger = logging.getLogger(__name__)

class IncrementalExtractor:
    """Main extraction engine - simplified without checkpoints"""
    
    def __init__(self, checkpoint_dir: str = None, api_key: Optional[str] = None):
        # Ignore checkpoint_dir parameter for compatibility
        self.prompt_builder = EnhancedPromptBuilder()
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in API_KEY or OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # Initialize intelligent comparator
        self.comparator = IntelligentComparator(self.api_key)
        
        # Initialize smart outcome extractor
        self.smart_outcome_extractor = SmartOutcomeExtractor(self.api_key)
        
        # Initialize smart validator
        self.validator = SmartValidator()
        
        # Initialize filename extractor
        self.filename_extractor = FilenameExtractor()
        
        # Initialize chunked extractor for large documents
        self.chunked_extractor = ChunkedExtractor()
        
        # Initialize intelligent chunking
        self.intelligent_chunker = IntelligentChunker(
            chunk_size=8000,
            overlap_size=1000
        )
        self.chunk_mapper = ChunkMapper(model="gpt-3.5-turbo")
        logger.info("Intelligent chunking initialized")
    
    def extract_from_pdf(self, pdf_path: str, nct_number: str, pdf_type: str,
                        resume: bool = True, compare_immediately: bool = True,
                        ctgov_csv_path: Optional[str] = None) -> ExtractionCheckpoint:
        """
        Extract fields from PDF (without checkpoint support).
        
        Args:
            pdf_path: Path to the PDF file
            nct_number: NCT number for the study
            pdf_type: Type of PDF (Protocol, SAP, ICF)
            resume: Ignored (kept for compatibility)
            compare_immediately: Whether to compare with CT.gov data after each extraction
            ctgov_csv_path: Path to CT.gov CSV for immediate comparison
            
        Returns:
            ExtractionCheckpoint with results
        """
        logger.info(f"Starting extraction for {nct_number} from {pdf_type} PDF")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Create extraction result container
        checkpoint = ExtractionCheckpoint(
            nct_number=nct_number,
            pdf_path=pdf_path,
            pdf_type=pdf_type,
            total_fields=len(CTGOV_FIELD_MAPPING),
            start_time=datetime.now(),
            last_update=datetime.now(),
            pdf_text_hash=""  # Not used anymore
        )
        
        # Initialize all fields as pending
        for field_name in CTGOV_FIELD_MAPPING.keys():
            checkpoint.fields[field_name] = FieldExtraction(
                field_name=field_name,
                status=ExtractionStatus.PENDING
            )
        
        # Extract metadata from filename
        filename_metadata = self.filename_extractor.extract_all(pdf_path)
        
        # Pre-populate NCT number from filename if available
        if 'nct_number' in filename_metadata:
            filename_nct = filename_metadata['nct_number']
            logger.info(f"Pre-populating NCT number from filename: {filename_nct}")
            
            checkpoint.fields['nct_number'] = FieldExtraction(
                field_name='nct_number',
                value=filename_nct,
                status=ExtractionStatus.COMPLETED,
                extraction_time=datetime.now(),
                confidence=1.0,
                source_text=f"Extracted from filename: {os.path.basename(pdf_path)}"
            )
            checkpoint.completed_fields += 1
        
        # Load CT.gov data if comparing immediately
        ctgov_data = None
        if compare_immediately and ctgov_csv_path:
            ctgov_data = self._load_ctgov_data(ctgov_csv_path)
        
        # Get fields to extract (prioritized)
        pending_fields = self._get_pending_fields(checkpoint)
        
        # Use intelligent chunking
        logger.info("Using intelligent chunking for extraction")
        document_chunks = self.intelligent_chunker.chunk_document(pdf_text)
        
        # Analyze chunks to find field locations
        logger.info(f"Analyzing {len(document_chunks)} chunks for field mapping...")
        chunk_mappings = self.chunk_mapper.analyze_chunks(document_chunks)
        
        # Create extraction plan
        extraction_plan = self.chunk_mapper.create_extraction_plan(chunk_mappings)
        logger.info(f"Extraction plan created for {len(extraction_plan)} fields")
        
        # Group fields for efficient extraction
        field_groups = self.prompt_builder.get_optimal_field_groups(pending_fields, pdf_type)
        
        logger.info(f"Extracting {len(pending_fields)} pending fields in {len(field_groups)} groups")
        
        # Extract fields group by group
        for group_idx, field_group in enumerate(field_groups):
            logger.info(f"Processing group {group_idx + 1}/{len(field_groups)}: {field_group}")
            
            try:
                # Extract fields
                if len(field_group) == 1:
                    # Single field extraction
                    field_name = field_group[0]
                    
                    # Try intelligent chunking first if available
                    value = None
                    if extraction_plan and field_name in extraction_plan:
                        chunk_id = extraction_plan[field_name]
                        chunk_text = document_chunks[chunk_id].text
                        logger.info(f"Trying chunk {chunk_id} for {field_name} (intelligent mapping)")
                        value = self._extract_single_field(field_name, chunk_text, pdf_type, pdf_path)
                        
                        # If not found in chunk, fall back to full document
                        if not value or value.lower() in ["not found", "not_found"]:
                            logger.info(f"{field_name} not found in chunk {chunk_id}, falling back to full document")
                            value = self._extract_single_field(field_name, pdf_text, pdf_type, pdf_path)
                    else:
                        # No chunk mapping, use full document
                        logger.info(f"No chunk mapping for {field_name}, using full document")
                        value = self._extract_single_field(field_name, pdf_text, pdf_type, pdf_path)
                    
                    # Validate against filename for NCT number
                    if field_name == 'nct_number' and value:
                        is_valid, corrected_value = self.filename_extractor.validate_extraction(
                            field_name, value, pdf_path
                        )
                        if not is_valid:
                            logger.warning(f"NCT validation failed, using filename value: {corrected_value}")
                            value = corrected_value
                    
                    # Update field status
                    checkpoint.fields[field_name] = FieldExtraction(
                        field_name=field_name,
                        value=value,
                        status=ExtractionStatus.COMPLETED if value and value.lower() not in ["not found", "not_found"] else ExtractionStatus.FAILED,
                        extraction_time=datetime.now()
                    )
                    
                    if checkpoint.fields[field_name].status == ExtractionStatus.COMPLETED:
                        checkpoint.completed_fields += 1
                    
                    # Compare immediately if requested
                    if compare_immediately and ctgov_data and value:
                        self._compare_field(field_name, value, ctgov_data)
                
                else:
                    # Batch extraction
                    results = self._extract_batch_fields(field_group, pdf_text, pdf_type)
                    
                    for field_name, value in results.items():
                        checkpoint.fields[field_name] = FieldExtraction(
                            field_name=field_name,
                            value=value,
                            status=ExtractionStatus.COMPLETED if value and value.lower() not in ["not found", "not_found"] else ExtractionStatus.FAILED,
                            extraction_time=datetime.now()
                        )
                        
                        if checkpoint.fields[field_name].status == ExtractionStatus.COMPLETED:
                            checkpoint.completed_fields += 1
                        
                        # Compare immediately if requested
                        if compare_immediately and ctgov_data and value:
                            self._compare_field(field_name, value, ctgov_data)
                
                # Log progress
                checkpoint.last_update = datetime.now()
                logger.info(f"Progress: {checkpoint.progress_percentage:.1f}% "
                           f"({checkpoint.completed_fields}/{checkpoint.total_fields} completed)")
                
            except Exception as e:
                logger.error(f"Error extracting fields {field_group}: {e}")
                
                # Mark fields as failed
                for field_name in field_group:
                    checkpoint.fields[field_name] = FieldExtraction(
                        field_name=field_name,
                        status=ExtractionStatus.FAILED,
                        error_message=str(e),
                        extraction_time=datetime.now()
                    )
        
        logger.info(f"Extraction complete. {checkpoint.completed_fields} fields extracted successfully.")
        return checkpoint
    
    def _get_pending_fields(self, checkpoint: ExtractionCheckpoint) -> List[str]:
        """Get list of pending fields in priority order"""
        pending = []
        
        # First add priority fields that are pending
        for field in PRIORITY_FIELDS:
            if field in checkpoint.fields and checkpoint.fields[field].status == ExtractionStatus.PENDING:
                pending.append(field)
        
        # Then add remaining fields
        for field_name, field in checkpoint.fields.items():
            if field.status == ExtractionStatus.PENDING and field_name not in pending:
                pending.append(field_name)
        
        return pending
    
    def _extract_single_field(self, field_name: str, text: str, doc_type: str, pdf_path: str = None) -> Optional[str]:
        """Extract a single field using LLM"""
        
        # Use specialized extraction for outcome measures
        if field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
            outcome_type = "primary" if "primary" in field_name else "secondary"
            logger.info(f"Using smart extraction for {field_name}")
            outcome_text = text if len(text) < 300000 else text[:250000]
            outcomes = self.smart_outcome_extractor.extract_outcomes(outcome_text, outcome_type)
            
            if outcomes:
                return "; ".join(outcomes)
            else:
                logger.info(f"No {outcome_type} outcomes found")
                return None
        
        # For other fields, use standard extraction
        max_chars = 48000
        
        if field_name == "nct_number":
            truncated_text = text[:max_chars]
        elif field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
            start = min(20000, len(text) // 4)
            truncated_text = text[start:start + max_chars]
        else:
            truncated_text = text[:max_chars]
        
        # Add filename hints if available
        filename_hints = ""
        if pdf_path and field_name == "nct_number":
            filename_nct = self.filename_extractor.extract_nct_number(pdf_path)
            if filename_nct:
                filename_hints = f"\n\nIMPORTANT: The filename contains '{filename_nct}'. Verify if this matches the NCT number in the document text. If you cannot find any NCT number in the text, return NOT_FOUND."
            
        prompt = self.prompt_builder.build_single_field_prompt(field_name, truncated_text + filename_hints, doc_type)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a clinical research data extractor. Extract only the requested information from clinical trial documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            if "NOT_FOUND" in result.upper() or "NOT FOUND" in result.upper():
                return None
            
            # Extract value after field name
            if ":" in result:
                lines = result.split('\n')
                value = None
                
                for line in lines:
                    if line.strip().startswith(field_name + ":"):
                        value = line.split(":", 1)[1].strip()
                        break
                
                if value and value.upper() not in ["NOT_FOUND", "NOT FOUND", "NONE", "N/A"]:
                    # Use simple validation
                    is_valid, error_msg = self.validator.validate_extraction(
                        field_name, value, text, result
                    )
                    
                    if not is_valid:
                        logger.error(f"Validation failed for {field_name}: {error_msg}")
                        return None
                    
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"LLM extraction error for {field_name}: {e}")
            return None
    
    def _extract_batch_fields(self, field_names: List[str], text: str, doc_type: str) -> Dict[str, Optional[str]]:
        """Extract multiple fields in one LLM call"""
        max_chars = 48000
        truncated_text = text[:max_chars]
            
        prompt = self.prompt_builder.build_batch_prompt(field_names, truncated_text, doc_type)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a clinical research data extractor. Extract only the requested information from clinical trial documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            parsed_results = self.prompt_builder.parse_extraction_response(result, field_names)
            
            # Validate all extracted fields
            validation_results = self.validator.batch_validate(parsed_results, text)
            
            for field_name, (is_valid, error_msg) in validation_results.items():
                if not is_valid and parsed_results.get(field_name):
                    logger.error(f"Batch extraction: {field_name} validation failed - {error_msg}")
                    parsed_results[field_name] = None
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"LLM batch extraction error: {e}")
            return {field: None for field in field_names}
    
    def _load_ctgov_data(self, csv_path: str) -> Dict[str, str]:
        """Load CT.gov data from CSV"""
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return row
            return {}
        except Exception as e:
            logger.error(f"Error loading CT.gov CSV: {e}")
            return {}
    
    def _compare_field(self, field_name: str, extracted_value: str, ctgov_data: Dict[str, str]) -> None:
        """Compare extracted field with CT.gov data"""
        ctgov_field = CTGOV_FIELD_MAPPING.get(field_name)
        if not ctgov_field or ctgov_field not in ctgov_data:
            return
        
        ctgov_value = ctgov_data[ctgov_field]
        
        # Use intelligent comparison
        match, confidence, explanation = self.comparator.compare_fields(
            field_name, extracted_value, ctgov_value
        )
        
        if match:
            logger.info(f"{field_name}: MATCH ({confidence:.0%}) - {explanation}")
        else:
            logger.warning(f"{field_name}: MISMATCH ({confidence:.0%}) - {explanation}\n  Extracted: {extracted_value[:100]}...\n  CT.gov: {ctgov_value[:100]}...")
    
    def compare_with_ctgov(self, checkpoint: ExtractionCheckpoint, ctgov_csv_path: str) -> StudyComparison:
        """Compare extracted data with CT.gov data"""
        logger.info(f"Comparing extraction with CT.gov data for {checkpoint.nct_number}")
        
        # Load CT.gov data
        ctgov_data = self._load_ctgov_data(ctgov_csv_path)
        
        comparison = StudyComparison(
            nct_number=checkpoint.nct_number,
            pdf_type=checkpoint.pdf_type,
            pdf_path=checkpoint.pdf_path,
            comparison_time=datetime.now(),
            total_fields=len(CTGOV_FIELD_MAPPING)
        )
        
        # Compare each field
        for field_name, ctgov_field in CTGOV_FIELD_MAPPING.items():
            extracted_field = checkpoint.fields.get(field_name)
            extracted_value = extracted_field.value if extracted_field else None
            ctgov_value = ctgov_data.get(ctgov_field)
            
            if not extracted_value:
                comparison.missing_in_extraction += 1
            if not ctgov_value:
                comparison.missing_in_ctgov += 1
            
            # Determine match
            match = False
            similarity_score = 0.0
            
            if extracted_value and ctgov_value:
                # Simple exact match for now
                match = extracted_value.lower().strip() == ctgov_value.lower().strip()
                similarity_score = 100.0 if match else 0.0
                
                if match:
                    comparison.matching_fields += 1
                else:
                    comparison.mismatched_fields += 1
            
            comparison.field_comparisons.append(
                ComparisonResult(
                    field_name=field_name,
                    extracted_value=extracted_value,
                    ctgov_value=ctgov_value,
                    match=match,
                    similarity_score=similarity_score
                )
            )
        
        logger.info(f"Comparison complete: {comparison.match_percentage:.1f}% match "
                   f"({comparison.matching_fields}/{comparison.total_fields} fields)")
        
        return comparison
    
    def resume_extraction(self, nct_number: str, pdf_type: str) -> Optional[ExtractionCheckpoint]:
        """Compatibility method - always starts fresh"""
        logger.warning("Resume functionality disabled - starting fresh extraction")
        return None