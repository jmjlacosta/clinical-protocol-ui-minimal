"""Main extraction engine with incremental processing and LLM integration"""
import os
import logging
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

from .pdf_extractor import extract_text_from_pdf
from .schema import (
    ExtractionCheckpoint, ExtractionStatus, FieldExtraction,
    ComparisonResult, StudyComparison, CTGOV_FIELD_MAPPING, PRIORITY_FIELDS
)
from .checkpoint_manager import CheckpointManager
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
    """Main extraction engine with checkpoint support"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", api_key: Optional[str] = None):
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.prompt_builder = EnhancedPromptBuilder()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Initialize OpenAI
        try:
            import openai
            openai.api_key = self.api_key
            self.openai = openai
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
            chunk_size=50000,
            overlap_size=1000
        )
        self.chunk_mapper = ChunkMapper(model="gpt-3.5-turbo-0125")
        logger.info("Intelligent chunking initialized")
    
    def extract_from_pdf(self, pdf_path: str, nct_number: str, pdf_type: str,
                        resume: bool = True, compare_immediately: bool = True,
                        ctgov_csv_path: Optional[str] = None) -> ExtractionCheckpoint:
        """
        Extract fields from PDF with checkpoint support.
        
        Args:
            pdf_path: Path to the PDF file
            nct_number: NCT number for the study
            pdf_type: Type of PDF (Protocol, SAP, ICF)
            resume: Whether to resume from checkpoint if available
            compare_immediately: Whether to compare with CT.gov data after each extraction
            ctgov_csv_path: Path to CT.gov CSV for immediate comparison
            
        Returns:
            ExtractionCheckpoint with results
        """
        logger.info(f"Starting extraction for {nct_number} from {pdf_type} PDF")
        
        # Load checkpoint if resuming
        checkpoint = None
        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(nct_number, pdf_type)
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(pdf_path)
        text_hash = CheckpointManager.compute_text_hash(pdf_text)
        
        # Check if PDF changed since last checkpoint
        if checkpoint and checkpoint.pdf_text_hash != text_hash:
            logger.warning("PDF content has changed since last checkpoint. Starting fresh.")
            checkpoint = None
        
        # Create new checkpoint if needed
        if not checkpoint:
            checkpoint = ExtractionCheckpoint(
                nct_number=nct_number,
                pdf_path=pdf_path,
                pdf_type=pdf_type,
                total_fields=len(CTGOV_FIELD_MAPPING),
                start_time=datetime.now(),
                last_update=datetime.now(),
                pdf_text_hash=text_hash
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
            
            # Add study codes as potential other_ids hint
            if 'study_codes' in filename_metadata:
                logger.info(f"Found potential study identifiers in filename: {filename_metadata['study_codes']}")
            
            self.checkpoint_manager.save_checkpoint(checkpoint)
        
        # Load CT.gov data if comparing immediately
        ctgov_data = None
        if compare_immediately and ctgov_csv_path:
            ctgov_data = self._load_ctgov_data(ctgov_csv_path)
        
        # Get fields to extract (prioritized)
        pending_fields = self._get_pending_fields(checkpoint)
        
        # Use intelligent chunking
        logger.info("Using intelligent chunking for extraction")
        # Create chunks
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
                # Mark fields as in progress
                for field_name in field_group:
                    self.checkpoint_manager.update_field_status(
                        checkpoint, field_name, ExtractionStatus.IN_PROGRESS
                    )
                
                # Extract fields
                if len(field_group) == 1:
                    # Single field extraction
                    field_name = field_group[0]
                    
                    # Use intelligent chunking
                    if extraction_plan and field_name in extraction_plan:
                        chunk_id = extraction_plan[field_name]
                        chunk_text = document_chunks[chunk_id].text
                        logger.info(f"Using chunk {chunk_id} for {field_name} (intelligent mapping)")
                        value = self._extract_single_field(field_name, chunk_text, pdf_type, pdf_path)
                    else:
                        value = self._extract_single_field(field_name, pdf_text, pdf_type, pdf_path)
                    
                    # Validate against filename for NCT number
                    if field_name == 'nct_number' and value:
                        is_valid, corrected_value = self.filename_extractor.validate_extraction(
                            field_name, value, pdf_path
                        )
                        if not is_valid:
                            logger.warning(f"NCT validation failed, using filename value: {corrected_value}")
                            value = corrected_value
                    
                    self.checkpoint_manager.update_field_status(
                        checkpoint, field_name, 
                        ExtractionStatus.COMPLETED if value and value.lower() not in ["not found", "not_found"] else ExtractionStatus.FAILED,
                        value=value
                    )
                    
                    # Compare immediately if requested
                    if compare_immediately and ctgov_data and value:
                        self._compare_field(field_name, value, ctgov_data)
                
                else:
                    # Batch extraction
                    # For batch, use intelligent chunking if all fields in the group map to the same chunk
                    batch_chunk_text = None
                    if extraction_plan:
                        chunk_ids = [extraction_plan.get(field) for field in field_group if field in extraction_plan]
                        if chunk_ids and all(cid == chunk_ids[0] for cid in chunk_ids):
                            # All fields map to the same chunk
                            chunk_id = chunk_ids[0]
                            batch_chunk_text = document_chunks[chunk_id].text
                            logger.info(f"Using chunk {chunk_id} for batch extraction of {field_group}")
                    
                    if batch_chunk_text:
                        results = self._extract_batch_fields(field_group, batch_chunk_text, pdf_type)
                    else:
                        results = self._extract_batch_fields(field_group, pdf_text, pdf_type)
                    
                    for field_name, value in results.items():
                        self.checkpoint_manager.update_field_status(
                            checkpoint, field_name,
                            ExtractionStatus.COMPLETED if value and value.lower() not in ["not found", "not_found"] else ExtractionStatus.FAILED,
                            value=value
                        )
                        
                        # Compare immediately if requested
                        if compare_immediately and ctgov_data and value:
                            self._compare_field(field_name, value, ctgov_data)
                
                # Log progress
                logger.info(f"Progress: {checkpoint.progress_percentage:.1f}% "
                           f"({checkpoint.completed_fields}/{checkpoint.total_fields} completed)")
                
            except Exception as e:
                logger.error(f"Error extracting fields {field_group}: {e}")
                
                # Mark fields as failed
                for field_name in field_group:
                    self.checkpoint_manager.update_field_status(
                        checkpoint, field_name, ExtractionStatus.FAILED,
                        error_message=str(e)
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
        """Extract a single field using LLM - with chunking for large documents"""
        
        # First try the original extraction method
        result = self._extract_single_field_from_chunk(field_name, text, doc_type, pdf_path)
        
        # If not found and document is large, try chunked extraction
        if not result and len(text) > 60000:  # Only chunk for large docs
            logger.info(f"Field {field_name} not found in first chunk, trying chunked extraction")
            
            def extract_func(chunk_text):
                return self._extract_single_field_from_chunk(field_name, chunk_text, doc_type, pdf_path, is_chunk=True)
            
            result = self.chunked_extractor.extract_with_chunks(text, field_name, extract_func)
            
        return result
    
    def _extract_single_field_from_chunk(self, field_name: str, text: str, doc_type: str, pdf_path: str = None, is_chunk: bool = False) -> Optional[str]:
        """Extract a single field using LLM"""
        # Use specialized extraction for outcome measures
        if field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
            outcome_type = "primary" if "primary" in field_name else "secondary"
            
            # Use smart extraction that mimics the original approach
            logger.info(f"Using smart extraction for {field_name}")
            # For outcomes, use more text since they often appear later in documents
            # Use full text if under 300k, otherwise use 250k
            outcome_text = text if len(text) < 300000 else text[:250000]
            outcomes = self.smart_outcome_extractor.extract_outcomes(outcome_text, outcome_type)
            
            if outcomes:
                # Format as semicolon-separated list
                return "; ".join(outcomes)
            else:
                logger.info(f"No {outcome_type} outcomes found")
                return None
        
        # For chunked extraction, use the full chunk
        if is_chunk:
            truncated_text = text
        else:
            # Calculate safe context size for GPT-3.5-turbo (16k tokens)
            # Roughly 4 chars per token, with overhead for prompts
            # Safe limit is about 12k tokens for content = 48k chars
            max_chars = 48000
            
            if field_name == "nct_number":
                # NCT numbers appear early, so less context needed
                truncated_text = text[:max_chars]
            elif field_name in ["primary_outcome_measures", "secondary_outcome_measures"]:
                # Outcomes often appear later in the document
                # Try to get middle section where outcomes typically are
                start = min(20000, len(text) // 4)
                truncated_text = text[start:start + max_chars]
            else:
                # For other fields, use beginning of document
                truncated_text = text[:max_chars]
        
        # Add filename hints if available
        filename_hints = ""
        if pdf_path and field_name == "nct_number":
            filename_nct = self.filename_extractor.extract_nct_number(pdf_path)
            if filename_nct:
                filename_hints = f"\n\nIMPORTANT: The filename contains '{filename_nct}'. Verify if this matches the NCT number in the document text. If you cannot find any NCT number in the text, return NOT_FOUND."
            
        prompt = self.prompt_builder.build_single_field_prompt(field_name, truncated_text + filename_hints, doc_type)
        
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using compatible model for v0.28.0
                messages=[
                    {"role": "system", "content": "You are a clinical research data extractor. Extract only the requested information from clinical trial documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response['choices'][0]['message']['content'].strip()
            
            # Parse the response
            if "NOT_FOUND" in result.upper() or "NOT FOUND" in result.upper():
                return None
            
            # Extract value after field name
            if ":" in result:
                # Handle multi-line responses (e.g., with source quotes)
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
        # Calculate safe context size for GPT-3.5-turbo (16k tokens)
        # Safe limit is about 12k tokens for content = 48k chars
        max_chars = 48000
        
        # For batch extraction, use the beginning of document
        # Most key fields appear in the first parts
        truncated_text = text[:max_chars]
            
        prompt = self.prompt_builder.build_batch_prompt(field_names, truncated_text, doc_type)
        
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a clinical research data extractor. Extract only the requested information from clinical trial documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response['choices'][0]['message']['content'].strip()
            
            # Parse the response
            parsed_results = self.prompt_builder.parse_extraction_response(result, field_names)
            
            # Validate all extracted fields using simple validator
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
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Return first row (should only be one row per NCT)
                    return row
            return {}
        except Exception as e:
            logger.error(f"Error loading CT.gov CSV: {e}")
            return {}
    
    def _compare_field(self, field_name: str, extracted_value: str, ctgov_data: Dict[str, str]) -> None:
        """Compare extracted field with CT.gov data using intelligent comparison"""
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
        """
        Compare extracted data with CT.gov data.
        
        Args:
            checkpoint: Completed extraction checkpoint
            ctgov_csv_path: Path to CT.gov CSV file
            
        Returns:
            StudyComparison with detailed results
        """
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
        """Resume a previous extraction from checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint(nct_number, pdf_type)
        
        if not checkpoint:
            logger.error(f"No checkpoint found for {nct_number} ({pdf_type})")
            return None
        
        if checkpoint.is_complete:
            logger.info(f"Extraction already complete for {nct_number} ({pdf_type})")
            return checkpoint
        
        logger.info(f"Resuming extraction for {nct_number} ({pdf_type}) from {checkpoint.progress_percentage:.1f}%")
        
        return self.extract_from_pdf(
            checkpoint.pdf_path,
            nct_number,
            pdf_type,
            resume=True
        )
