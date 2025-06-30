"""Main entry point for the incremental extraction system"""
import os
import sys
import logging
import argparse
from typing import List, Tuple, Optional
from pathlib import Path

from .extractor import IncrementalExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def identify_study_files(examples_dir: str) -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Identify and map study files in the examples directory.
    
    Returns:
        List of tuples: (nct_number, pdf_type, pdf_path, ctgov_csv_path)
    """
    study_files = []
    examples_path = Path(examples_dir)
    
    # Find all PDF files
    pdf_files = list(examples_path.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        filename = pdf_file.name
        
        # Extract NCT number (assumes format: NCT12345678_type_xxx.pdf)
        if filename.startswith("NCT"):
            parts = filename.split("_")
            if len(parts) >= 2:
                nct_number = parts[0]
                
                # Determine PDF type
                pdf_type = None
                if "Prot" in filename:
                    pdf_type = "Protocol"
                elif "SAP" in filename:
                    pdf_type = "SAP"
                elif "ICF" in filename:
                    pdf_type = "ICF"
                
                if pdf_type:
                    # Find corresponding CT.gov CSV
                    csv_pattern = f"{nct_number}_ct_*.csv"
                    csv_files = list(examples_path.glob(csv_pattern))
                    ctgov_csv = csv_files[0] if csv_files else None
                    
                    study_files.append((
                        nct_number,
                        pdf_type,
                        str(pdf_file),
                        str(ctgov_csv) if ctgov_csv else None
                    ))
    
    # Sort by NCT number and type
    study_files.sort(key=lambda x: (x[0], x[1]))
    
    return study_files

def extract_single_study(extractor: IncrementalExtractor, nct_number: str, 
                        pdf_type: str, pdf_path: str, ctgov_csv_path: Optional[str],
                        compare_immediately: bool = True) -> None:
    """Extract data from a single study PDF"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {nct_number} - {pdf_type}")
    logger.info(f"PDF: {pdf_path}")
    if ctgov_csv_path:
        logger.info(f"CT.gov CSV: {ctgov_csv_path}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Extract with checkpointing
        checkpoint = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            nct_number=nct_number,
            pdf_type=pdf_type,
            resume=True,
            compare_immediately=compare_immediately and ctgov_csv_path is not None,
            ctgov_csv_path=ctgov_csv_path
        )
        
        # If not comparing immediately, do final comparison
        if not compare_immediately and ctgov_csv_path and checkpoint.is_complete:
            logger.info("\nPerforming final comparison with CT.gov data...")
            comparison = extractor.compare_with_ctgov(checkpoint, ctgov_csv_path)
            
            # Print summary
            logger.info(f"\nComparison Summary:")
            logger.info(f"  Total fields: {comparison.total_fields}")
            logger.info(f"  Matching: {comparison.matching_fields} ({comparison.match_percentage:.1f}%)")
            logger.info(f"  Mismatched: {comparison.mismatched_fields}")
            logger.info(f"  Missing in extraction: {comparison.missing_in_extraction}")
            logger.info(f"  Missing in CT.gov: {comparison.missing_in_ctgov}")
        
        logger.info(f"\n✓ Completed {nct_number} - {pdf_type}")
        
    except Exception as e:
        logger.error(f"\n✗ Failed to process {nct_number} - {pdf_type}: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Incremental PDF extraction system for clinical trials")
    parser.add_argument("--examples-dir", default="examples", help="Directory containing example files")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoint files")
    parser.add_argument("--nct", help="Process specific NCT number only")
    parser.add_argument("--pdf-type", choices=["Protocol", "SAP", "ICF"], help="Process specific PDF type only")
    parser.add_argument("--no-compare", action="store_true", help="Skip immediate comparison with CT.gov")
    parser.add_argument("--list-checkpoints", action="store_true", help="List all available checkpoints")
    parser.add_argument("--resume", help="Resume specific extraction (format: NCT12345678_Protocol)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    # Handle list checkpoints
    if args.list_checkpoints:
        checkpoints = checkpoint_manager.list_checkpoints()
        if checkpoints:
            logger.info("\nAvailable checkpoints:")
            for key, info in checkpoints.items():
                logger.info(f"  {key}: {info['progress']:.1f}% complete "
                           f"({info['completed_fields']}/{info['total_fields']} fields)")
        else:
            logger.info("\nNo checkpoints found.")
        return
    
    # Initialize extractor
    try:
        extractor = IncrementalExtractor(
            checkpoint_dir=args.checkpoint_dir,
            api_key=args.api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        sys.exit(1)
    
    # Handle resume
    if args.resume:
        parts = args.resume.split("_")
        if len(parts) >= 2:
            nct_number = parts[0]
            pdf_type = "_".join(parts[1:])
            
            logger.info(f"Resuming extraction for {nct_number} ({pdf_type})...")
            checkpoint = extractor.resume_extraction(nct_number, pdf_type)
            
            if checkpoint:
                logger.info(f"✓ Extraction complete: {checkpoint.completed_fields}/{checkpoint.total_fields} fields")
        else:
            logger.error("Invalid resume format. Use: NCT12345678_Protocol")
        return
    
    # Find study files
    study_files = identify_study_files(args.examples_dir)
    
    if not study_files:
        logger.error(f"No study files found in {args.examples_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(study_files)} study PDFs")
    
    # Filter if specific NCT or type requested
    if args.nct:
        study_files = [s for s in study_files if s[0] == args.nct]
    if args.pdf_type:
        study_files = [s for s in study_files if s[1] == args.pdf_type]
    
    if not study_files:
        logger.error("No matching study files found after filtering")
        sys.exit(1)
    
    # Process each study
    logger.info(f"\nProcessing {len(study_files)} study PDFs...")
    
    for nct_number, pdf_type, pdf_path, ctgov_csv in study_files:
        try:
            extract_single_study(
                extractor=extractor,
                nct_number=nct_number,
                pdf_type=pdf_type,
                pdf_path=pdf_path,
                ctgov_csv_path=ctgov_csv,
                compare_immediately=not args.no_compare
            )
        except Exception as e:
            logger.error(f"Failed to process {nct_number} - {pdf_type}: {e}")
            # Continue with next file
            continue
    
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*60)
    
    # Summary of checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    if checkpoints:
        logger.info("\nCheckpoint Summary:")
        complete = 0
        incomplete = 0
        
        for key, info in checkpoints.items():
            if info['progress'] >= 100:
                complete += 1
            else:
                incomplete += 1
        
        logger.info(f"  Complete: {complete}")
        logger.info(f"  Incomplete: {incomplete}")
        logger.info(f"  Total: {len(checkpoints)}")

if __name__ == "__main__":
    main()