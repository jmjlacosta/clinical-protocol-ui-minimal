# agents/outcome_extractor/runner.py
from typing import List, Dict
from libs.schema.chunker import Chunk
import openai
import json, logging

import os

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
if not api_key:
    raise ValueError("Neither OPENAI_API_KEY nor API_KEY environment variable is set")

openai.api_key = api_key

# openai.api_key = ""

log = logging.getLogger(__name__)

def run(chunks: List[Chunk]) -> Dict[str, object]:
    candidate_chunks = [c for c in chunks if "outcome" in c.text.lower()]

    prompt = (
        "Extract primary and secondary outcome measures from the text.\n"
        "Return JSON with keys primary_outcomes and secondary_outcomes, "
        "each an array of objects {outcome_measure, outcome_time_frame, outcome_description}."
    )
    joined = "\n\n".join(c.text for c in candidate_chunks)[:30_000]

    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=2048,
        messages=[{"role":"user","content": prompt + "\n\n" + joined}]
    )
    try:
        data = json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        log.error("Outcome JSON parse error: %s", e)
        data = {"primary_outcomes": [], "secondary_outcomes": []}

    return {
        **data,
        "confidence": 0.85,                # stub until you compute it
        "evidence": [
            {"chunk_id": c.chunk_id, "start_page": c.start_page, "end_page": c.end_page}
            for c in candidate_chunks
        ],
    }
