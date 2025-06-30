"""
Chunker + (optional) index builder.
Keeps page provenance so downstream extractors can point back to the PDF.
"""

from __future__ import annotations
import re
from typing import List

# Try to import tiktoken, but make it optional
try:
    import tiktoken
    _tok_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    def _tok_count(txt: str) -> int:
        return len(_tok_enc.encode(txt))
except ImportError:
    # Fallback: estimate tokens as chars/4 (rough approximation)
    def _tok_count(txt: str) -> int:
        return len(txt) // 4

from libs.schema.ingestion import IngestionResult
from libs.schema.chunker import Chunk, ChunkerResult

# ── Helpers ───────────────────────────────────────────────────────────────────
SECTION_PATTERN = r'(#{1,6}\s+.+?(?:\n|$))'


# ── Public API ────────────────────────────────────────────────────────────────
def run(ingest: IngestionResult, max_chars: int = 64_000) -> dict:
    """
    Parameters
    ----------
    ingest : IngestionResult
        .text      → the full markdown-ish document
        .page_map  → list[{"page_no", "start", "end"}]  char offsets in the *same* text
    max_chars : int
        Upper bound of characters per chunk (≈ 16 k tokens)

    Returns
    -------
    ChunkerResult  (pydantic model)
    """
    text, page_map = ingest.text, ingest.page_map

    # 1 ▸ Split the doc   (same logic as the old chunk_text)
    try:
        sections = re.split(SECTION_PATTERN, text, flags=re.MULTILINE)
    except Exception:
        sections = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: List[Chunk] = []
    buf, buf_len = [], 0        # staging buffer for current chunk
    char_cursor = 0             # absolute position in full text

    def _flush():
        nonlocal char_cursor, buf, buf_len
        chunk_text = "\n".join(buf)
        # ▸ Which PDF pages overlap this char span?
        pages = [
            p["page_no"]
            for p in page_map
            if p["start"] < char_cursor + buf_len and p["end"] > char_cursor
        ] or [None]

        chunks.append(
            Chunk(
                chunk_id=f"c{len(chunks)}",
                text=chunk_text,
                start_page=pages[0],
                end_page=pages[-1],
                token_count=_tok_count(chunk_text),
            )
        )
        char_cursor += buf_len
        buf, buf_len = [], 0

    # iterate through pre-sections
    for part in sections:
        if not part.strip():
            continue
        if buf_len + len(part) > max_chars and buf:
            _flush()
        buf.append(part)
        buf_len += len(part)
    if buf:
        _flush()

    # 2 ▸ Return model
    res = ChunkerResult(
        chunks=chunks,
        num_chunks=len(chunks),
        total_tokens=sum(c.token_count for c in chunks),
        embedding_model=None,       # fill once you add embeddings
    )
    return {"chunk_res": res, "chunks": res.chunks}
