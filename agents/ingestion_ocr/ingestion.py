import os
import logging
import re
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info("Extracting text from PDF using docling DocumentConverter...")

    try:
        try:
            from docling.document_converter import DocumentConverter
            logger.info(f"Converting PDF with DocumentConverter: {pdf_path}")
            converter = DocumentConverter()
            result = converter.convert(pdf_path)

            # Export to markdown
            text = result.document.export_to_markdown()
            logger.info("Successfully converted PDF to markdown")
            return text
        except Exception as converter_error:
            logger.error(f"DocumentConverter failed: {converter_error}")
            raise

    except Exception as e:
        logger.error(f"All docling methods failed: {e}")


def chunk_text(text: str) -> List[str]:
    """Break text into manageable chunks while preserving section context"""
    if not isinstance(text, str):
        logger.error(f"chunk_text received non-string input: {type(text)}")
        text = str(text)

    section_pattern = r'(#{1,6}\s+.+?(?:\n|$))'

    try:
        sections = re.split(section_pattern, text, flags=re.MULTILINE)
    except Exception as e:
        logger.error(f"Error splitting text into sections: {e}")
        # Fallback to simple chunking
        return [text[i:i+64000] for i in range(0, len(text), 64000)]

    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_size = 64000

    for i in range(0, len(sections)):
        section = sections[i]
        if not section.strip():
            continue

        section_length = len(section)

        if current_length + section_length > max_chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [section]
            current_length = section_length
        else:
            current_chunk.append(section)
            current_length += section_length

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    if not chunks:
        logger.warning("Chunking produced no chunks, adding the whole text as one chunk")
        if len(text) > max_chunk_size:
            return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        else:
            chunks = [text]

    logger.info(f"Split document into {len(chunks)} chunks")
    return chunks
