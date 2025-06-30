from pathlib import Path
import csv
import io
import json
from agents.ingestion_ocr.extractor_core import (
    extract_clinical_info,
    process_pdf_to_xml,
)

from agents.ingestion_ocr.runner import run as run_ingest
from agents.chunker_indexer.runner import run as run_chunker
from agents.outcome_extractor.runner import run as run_outcomes


# # ── NEW: return the structured dict ───────────────────────────────────────────
# def extract_info(pdf_bytes: bytes) -> dict:
#     """Parse PDF → return clinical_info dict (no XML)."""
#     tmp = Path("/tmp/in_pdf.pdf")
#     tmp.write_bytes(pdf_bytes)

#     pdf_text = extract_text_from_pdf(str(tmp))
#     chunks = chunk_text(pdf_text)
#     return extract_clinical_info(chunks)

def extract_info(pdf_bytes: bytes) -> dict:
    """
    PDF bytes → Ingestion → Chunker → Outcome extractor (+ legacy extract_clinical_info).
    Returns a dict ready for Streamlit display.
    """
    # 1. Ingestion
    ingest = run_ingest(pdf_bytes)["ingest"]
    # Path("/tmp/ingest_debug.txt").write_text(ingest.text[:5000], "utf-8")
    # print("🛠  first 5 000 chars dumped to /tmp/ingest_debug.txt")
    # 2. Chunking
    chunk_res = run_chunker(ingest)["chunk_res"]               # ChunkerResult
    chunk_texts = [c.text for c in chunk_res.chunks]

    # 3. Legacy code
    info_dict = extract_clinical_info(chunk_texts)

    # 4. Outcomes 
    outcome_res = run_outcomes(chunk_res.chunks)
    info_dict["primary_outcomes"]   = outcome_res.get("primary_outcomes", [])
    info_dict["secondary_outcomes"] = outcome_res.get("secondary_outcomes", [])
    info_dict["outcomes_evidence"]  = outcome_res.get("evidence", [])
    info_dict["outcomes_confidence"] = outcome_res.get("confidence")

    # 5. Attach provenance / summary meta
    info_dict["ingestion_meta"] = ingest.model_dump()
    info_dict["chunk_meta"]     = chunk_res.model_dump(exclude={"chunks"})

    return info_dict

# existing helper that returns XML
def run_pipeline(pdf_bytes: bytes) -> str:
    tmp = Path("/tmp/in_pdf.pdf")
    tmp.write_bytes(pdf_bytes)
    return process_pdf_to_xml(str(tmp))

def pdf_bytes_to_xml(pdf_bytes: bytes) -> str:
    """Convenience wrapper around ``process_pdf_to_xml`` for byte streams."""
    tmp = Path("/tmp/in_pdf_for_xml.pdf")
    tmp.write_bytes(pdf_bytes)
    return process_pdf_to_xml(str(tmp))


def info_to_csv(info: dict) -> str:
    """Serialize extracted info dict to a simple two-column CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["key", "value"])
    for key, value in info.items():
        writer.writerow([key, json.dumps(value)])
    return buffer.getvalue()
