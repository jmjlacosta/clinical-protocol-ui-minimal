#!/usr/bin/env python3
"""
Run the incremental extraction system

Usage:
    # Extract all PDFs in examples directory
    python run_incremental_extraction.py
    
    # Extract specific NCT number
    python run_incremental_extraction.py --nct NCT02454972
    
    # Extract only Protocol PDFs
    python run_incremental_extraction.py --pdf-type Protocol
    
    # Skip immediate comparison (compare at the end)
    python run_incremental_extraction.py --no-compare
    
    # List available checkpoints
    python run_incremental_extraction.py --list-checkpoints
    
    # Resume a specific extraction
    python run_incremental_extraction.py --resume NCT02454972_Protocol
"""

import sys
from incremental_extractor.main import main

if __name__ == "__main__":
    main()