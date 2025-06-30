#!/usr/bin/env python3
"""
Test script for intelligent chunking system.
Demonstrates Phase 1 implementation without requiring actual PDFs.
"""

import os
import sys
import logging
from incremental_extractor.intelligent_chunker import IntelligentChunker
from incremental_extractor.chunk_mapper import ChunkMapper
from incremental_extractor.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_document():
    """Create a sample clinical trial document for testing"""
    return """
NCT12345678 - A Phase 3 Randomized Controlled Trial

PROTOCOL TITLE: A Randomized, Double-Blind, Placebo-Controlled Study to Evaluate 
the Efficacy and Safety of Drug X in Patients with Condition Y

BRIEF SUMMARY:
This is a multicenter, randomized, double-blind, placebo-controlled phase 3 study 
designed to evaluate the efficacy and safety of Drug X compared to placebo in 
adult patients with moderate to severe Condition Y. The study will enroll 
approximately 500 patients across 50 sites globally.

STUDY DESIGN:
This is a parallel-group study with a 2:1 randomization ratio (Drug X:Placebo).
The study consists of:
- Screening period (up to 4 weeks)
- Treatment period (24 weeks)
- Follow-up period (4 weeks)

Patients will be stratified by baseline severity and geographic region.

INCLUSION CRITERIA:
1. Adults aged 18-75 years
2. Diagnosed with Condition Y for at least 6 months
3. Baseline severity score ≥ 20
4. Willing and able to provide informed consent

EXCLUSION CRITERIA:
1. Pregnant or breastfeeding women
2. History of hypersensitivity to Drug X
3. Significant liver or kidney impairment
4. Use of prohibited medications within 30 days

PRIMARY OUTCOME MEASURES:
The primary efficacy endpoint is the change from baseline in the Condition Y 
Severity Score (CYSS) at Week 24. The CYSS is a validated 100-point scale where 
higher scores indicate greater severity. A reduction of ≥10 points is considered 
clinically meaningful.

SECONDARY OUTCOME MEASURES:
1. Proportion of patients achieving CYSS reduction ≥20 points at Week 24
2. Change from baseline in Quality of Life Questionnaire (QLQ) score at Week 24
3. Time to first symptom improvement (defined as ≥5 point reduction in CYSS)
4. Patient Global Impression of Change (PGIC) at Week 24
5. Physician Global Assessment (PGA) at Week 24

SAFETY ASSESSMENTS:
Safety will be assessed through:
- Adverse event monitoring
- Laboratory tests (hematology, chemistry, urinalysis)
- Vital signs and physical examinations
- Electrocardiograms

STATISTICAL ANALYSIS:
The primary analysis will be performed on the Intent-to-Treat (ITT) population 
using ANCOVA with baseline CYSS as a covariate. A hierarchical testing procedure 
will be used to control for multiplicity.

Sample size calculation: With 333 patients on Drug X and 167 on placebo, the 
study has 90% power to detect a 5-point difference in CYSS change, assuming a 
standard deviation of 12 points and a two-sided alpha of 0.05.

STUDY SPONSOR: PharmaCorp International
PRINCIPAL INVESTIGATOR: Dr. Jane Smith, MD, PhD
STUDY START DATE: January 2024
ESTIMATED COMPLETION DATE: December 2025
"""

def test_intelligent_chunking():
    """Test the intelligent chunking system"""
    logger.info("=== Testing Intelligent Chunking System ===")
    
    # Create sample document
    document = create_sample_document()
    logger.info(f"Sample document length: {len(document)} characters")
    
    # Test 1: Document Chunking
    logger.info("\n1. Testing Document Chunking")
    chunker = IntelligentChunker(chunk_size=1000, overlap_size=100)
    chunks = chunker.chunk_document(document)
    
    logger.info(f"Created {len(chunks)} chunks")
    for chunk in chunks[:3]:  # Show first 3 chunks
        logger.info(f"  Chunk {chunk.chunk_id}: {chunk.start_char}-{chunk.end_char} "
                   f"({len(chunk.text)} chars)")
        logger.info(f"    Preview: {chunk.text[:50]}...")
    
    # Show chunk summary
    summary = chunker.get_chunk_summary(chunks)
    logger.info(f"\nChunk Summary: {summary}")
    
    # Test 2: Chunk Mapping (if OpenAI available)
    logger.info("\n2. Testing Chunk Mapping")
    
    # Check if we can use chunk mapper
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found, testing chunk analysis...")
        
        # Set a small test chunk
        test_chunk = chunks[6] if len(chunks) > 6 else chunks[0]  # Primary outcomes chunk
        logger.info(f"Analyzing chunk {test_chunk.chunk_id}:")
        logger.info(f"Content preview: {test_chunk.text[:200]}...")
        
        mapper = ChunkMapper()
        mapping = mapper.analyze_chunk(test_chunk)
        
        logger.info(f"Identified fields: {mapping.identified_fields}")
        logger.info(f"Confidence scores: {mapping.confidence_scores}")
        logger.info(f"Relevant sections: {mapping.relevant_sections}")
        
        # Test extraction plan
        logger.info("\n3. Testing Extraction Plan")
        all_mappings = [mapping]  # In real usage, we'd analyze all chunks
        extraction_plan = mapper.create_extraction_plan(all_mappings)
        logger.info(f"Extraction plan: {extraction_plan}")
        
    else:
        logger.warning("No OpenAI API key found. Skipping chunk analysis test.")
        logger.info("Set OPENAI_API_KEY environment variable to test chunk analysis.")
    
    # Test 3: Configuration
    logger.info("\n4. Testing Configuration")
    logger.info(f"Current config: {config.get_chunk_config()}")
    
    # Test enabling/disabling
    config.enable_intelligent_chunking()
    logger.info(f"After enabling: use_intelligent_chunking = {config.use_intelligent_chunking}")
    
    config.disable_intelligent_chunking()
    logger.info(f"After disabling: use_intelligent_chunking = {config.use_intelligent_chunking}")
    
    logger.info("\n=== Test Complete ===")

if __name__ == "__main__":
    test_intelligent_chunking()