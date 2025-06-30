"""Checkpoint management for incremental extraction"""
import json
import os
import hashlib
from datetime import datetime
from typing import Optional, Dict
import logging

from .schema import ExtractionCheckpoint, FieldExtraction, ExtractionStatus

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages checkpoints for resumable extraction"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self, nct_number: str, pdf_type: str) -> str:
        """Get checkpoint file path for a study"""
        filename = f"{nct_number}_{pdf_type.lower()}_checkpoint.json"
        return os.path.join(self.checkpoint_dir, filename)
    
    def save_checkpoint(self, checkpoint: ExtractionCheckpoint) -> None:
        """Save checkpoint to disk"""
        checkpoint.last_update = datetime.now()
        checkpoint_path = self.get_checkpoint_path(checkpoint.nct_number, checkpoint.pdf_type)
        
        try:
            # Convert to dict with datetime serialization
            checkpoint_dict = checkpoint.model_dump()
            
            # Custom JSON encoder for datetime
            def datetime_handler(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_dict, f, indent=2, default=datetime_handler)
            
            logger.info(f"Saved checkpoint for {checkpoint.nct_number} ({checkpoint.pdf_type}) - "
                       f"{checkpoint.completed_fields}/{checkpoint.total_fields} fields completed")
                       
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, nct_number: str, pdf_type: str) -> Optional[ExtractionCheckpoint]:
        """Load checkpoint from disk"""
        checkpoint_path = self.get_checkpoint_path(nct_number, pdf_type)
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_dict = json.load(f)
            
            # Parse datetime strings back to datetime objects
            for date_field in ['start_time', 'last_update']:
                if date_field in checkpoint_dict and checkpoint_dict[date_field]:
                    checkpoint_dict[date_field] = datetime.fromisoformat(checkpoint_dict[date_field])
            
            # Parse datetime in field extractions
            for field_name, field_data in checkpoint_dict.get('fields', {}).items():
                if 'extraction_time' in field_data and field_data['extraction_time']:
                    field_data['extraction_time'] = datetime.fromisoformat(field_data['extraction_time'])
            
            checkpoint = ExtractionCheckpoint(**checkpoint_dict)
            logger.info(f"Loaded checkpoint for {nct_number} ({pdf_type}) - "
                       f"{checkpoint.completed_fields}/{checkpoint.total_fields} fields completed")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def delete_checkpoint(self, nct_number: str, pdf_type: str) -> None:
        """Delete checkpoint file"""
        checkpoint_path = self.get_checkpoint_path(nct_number, pdf_type)
        
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint for {nct_number} ({pdf_type})")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint: {e}")
    
    def update_field_status(self, checkpoint: ExtractionCheckpoint, field_name: str, 
                          status: ExtractionStatus, value: Optional[str] = None,
                          error_message: Optional[str] = None,
                          confidence: Optional[float] = None,
                          source_text: Optional[str] = None) -> None:
        """Update the status of a field in the checkpoint"""
        
        if field_name not in checkpoint.fields:
            checkpoint.fields[field_name] = FieldExtraction(field_name=field_name)
        
        field = checkpoint.fields[field_name]
        old_status = field.status
        
        field.status = status
        field.extraction_time = datetime.now()
        
        if value is not None:
            field.value = value
        if error_message is not None:
            field.error_message = error_message
        if confidence is not None:
            field.confidence = confidence
        if source_text is not None:
            field.source_text = source_text
        
        # Update counters
        if old_status in [ExtractionStatus.PENDING, ExtractionStatus.IN_PROGRESS]:
            if status == ExtractionStatus.COMPLETED:
                checkpoint.completed_fields += 1
            elif status == ExtractionStatus.FAILED:
                checkpoint.failed_fields += 1
            elif status == ExtractionStatus.SKIPPED:
                checkpoint.skipped_fields += 1
        
        # Save checkpoint after each field update
        self.save_checkpoint(checkpoint)
    
    def get_next_pending_field(self, checkpoint: ExtractionCheckpoint) -> Optional[str]:
        """Get the next field that needs to be extracted"""
        for field_name, field in checkpoint.fields.items():
            if field.status == ExtractionStatus.PENDING:
                return field_name
        return None
    
    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute hash of text for change detection"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def list_checkpoints(self) -> Dict[str, Dict]:
        """List all available checkpoints"""
        checkpoints = {}
        
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('_checkpoint.json'):
                parts = filename.replace('_checkpoint.json', '').split('_')
                if len(parts) >= 2:
                    nct_number = parts[0]
                    pdf_type = '_'.join(parts[1:]).upper()
                    
                    checkpoint = self.load_checkpoint(nct_number, pdf_type)
                    if checkpoint:
                        checkpoints[f"{nct_number}_{pdf_type}"] = {
                            'nct_number': nct_number,
                            'pdf_type': pdf_type,
                            'progress': checkpoint.progress_percentage,
                            'completed_fields': checkpoint.completed_fields,
                            'total_fields': checkpoint.total_fields,
                            'last_update': checkpoint.last_update
                        }
        
        return checkpoints