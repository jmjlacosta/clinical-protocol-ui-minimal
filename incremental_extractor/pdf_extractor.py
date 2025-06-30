"""Fast PDF text extraction using PyPDF2 with fallback to pdfplumber"""
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF2 (fast) with fallback to pdfplumber (more accurate).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try PyPDF2 first (faster)
    try:
        import PyPDF2
        logger.info(f"Extracting text from {pdf_path} using PyPDF2...")
        
        text_parts = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        full_text = "\n\n".join(text_parts)
        
        if full_text.strip():
            logger.info(f"Successfully extracted {len(full_text)} characters using PyPDF2")
            return full_text
        else:
            logger.warning("PyPDF2 extracted empty text, trying pdfplumber...")
            
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}, trying pdfplumber...")
    
    # Fallback to pdfplumber
    try:
        import pdfplumber
        logger.info(f"Extracting text from {pdf_path} using pdfplumber...")
        
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
        
        full_text = "\n\n".join(text_parts)
        
        if full_text.strip():
            logger.info(f"Successfully extracted {len(full_text)} characters using pdfplumber")
            return full_text
        else:
            raise ValueError("No text could be extracted from the PDF")
            
    except Exception as e:
        logger.error(f"All PDF extraction methods failed: {e}")
        raise