"""
Simple ingestion — identical to the original extractor_core implementation.
* Page_map: in development
* No OCR fallback
"""

from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from libs.schema.ingestion import IngestionResult

logger = logging.getLogger(__name__)


def run(pdf_bytes: bytes) -> dict:
    tmp = Path("/tmp/ingest.pdf")
    tmp.write_bytes(pdf_bytes)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp))
        text = result.document.export_to_markdown()
        confidence = 0.95 if len(text.strip()) > 200 else 0.0
    except Exception as e:
        logger.error("Docling failed: %s", e, exc_info=True)
        text, confidence = "", 0.0

    # NOTE: page_map is now always an empty list
    res = IngestionResult(
        text=text,
        page_map=[],      # keep schema stable; just empty
        engine="docling",
        confidence=confidence,
    )
    return {"ingest": res}
