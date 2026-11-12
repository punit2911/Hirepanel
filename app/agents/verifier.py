"""Evidence Verifier: no quote, no credit.

Every score must point at text that really exists in the resume. This is the layer that makes
LLM hallucinations and keyword stuffing expensive instead of free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..doc import Doc
from ..schemas import Requirement
from ..skills import skills_in
from ..util import truncate_words
from .scorers import RawRow


@dataclass
class Grounded:
    score: float
    evidence: str = ""
    section: str = ""
    verified: bool = True
    capped: str = ""
    hallucinated: bool = False
    raw: float = 0.0
    note: str = ""


def _mentions(text: str, term: str) -> bool:
    if term in skills_in(text):
        return True
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text.lower()))


def ground(doc: Doc, req: Requirement, raw: RawRow) -> Grounded:
    s = max(0.0, min(1.0, raw.score))
    if s <= 0:
        return Grounded(0.0, raw=raw.score, note=raw.note)
    quote = (raw.quote or "").strip()
    if not quote:
        return Grounded(min(s, 0.3), "", "", False, "no_quote", False, raw.score, raw.note)
    found, offset, _conf = doc.locate(quote)
    if not found:
        return Grounded(min(s, 0.2), "", "", False, "quote_not_found", True, raw.score, raw.note)
    ln = doc.line_at(offset)
    section = ln.section if ln else doc.section_at(offset)
    window = quote + " " + (ln.text if ln else "")
    terms = list(dict.fromkeys(req.keywords + [t for g in req.groups for t in g]))
    capped, cap = "", 1.0
    if ln is not None and ln.copied:
        capped, cap = "copied_from_job_description", 0.2
    elif terms and not any(_mentions(window, t) for t in terms):
        capped, cap = "quote_off_topic", 0.4
    elif ln is not None and (ln.listlike or ln.section == "skills"):
        limit = 0.25 if doc.stuffed else 0.5
        if s > limit:
            capped, cap = "listed_only", limit
    return Grounded(min(s, cap), truncate_words(quote, 220), section, True, capped, False, raw.score, raw.note)
