"""DocIndex: one pass over a resume giving every line its section, skills and quality signals."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sections import Section, heading_name, section_at, split_sections
from .skills import find_skills
from .util import norm, normalize_with_map, truncate_words

_BULLET = re.compile(r"^\s*(?:[-*•–·▪●◦]+|\d+[.)])\s*")
_VERB_START = re.compile(
    r"^\W*(?:[A-Za-z]+ed|built|led|ran|wrote|drove|won|made|grew|cut|set|spearheaded|"
    r"own|owned|manage|develop|design|create|implement|deliver)\b", re.I)
_QUANT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|x\b|k\b|m\b|ms\b|hours?|users|customers|requests|million|billion)|[$₹€£]\s?\d|\b\d{2,}\b")


@dataclass
class Line:
    i: int
    start: int
    end: int
    text: str
    section: str
    skills: list[str] = field(default_factory=list)
    listlike: bool = False
    quant: bool = False
    heading: bool = False
    copied: bool = False  # mostly lifted from the job description: never counts as evidence

    @property
    def clean(self) -> str:
        return _BULLET.sub("", self.text).strip()


@dataclass
class Doc:
    text: str
    lines: list[Line]
    sections: list[Section]
    stuffed: bool = False  # skills section looks like a keyword dump: listed skills earn very little

    def mark_copied(self, jd_text: str, n: int = 4, threshold: float = 0.6) -> int:
        if not jd_text:
            return 0
        jt = norm(jd_text).split()
        jd = {tuple(jt[i : i + n]) for i in range(max(0, len(jt) - n + 1))}
        count = 0
        # short bullets ("Strong SQL and Excel") are too short for phrase overlap, so match whole JD lines exactly
        jd_lines = {" ".join(norm(x).split()) for x in jd_text.splitlines()}
        jd_lines = {x for x in jd_lines if len(x.split()) >= 3}
        for ln in self.lines:
            toks = norm(ln.text).split()
            if 3 <= len(toks) < n + 2:
                if " ".join(toks) in jd_lines:
                    ln.copied = True
                    count += 1
                continue
            if len(toks) < n + 2:
                continue
            sh = {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
            if sh and len(sh & jd) / len(sh) >= threshold:
                ln.copied = True
                count += 1
        return count

    def section_at(self, offset: int) -> str:
        return section_at(self.sections, offset)

    def line_at(self, offset: int) -> Line | None:
        for ln in self.lines:
            if ln.start <= offset <= ln.end:
                return ln
        return None

    def in_section(self, *names: str) -> list[Line]:
        return [ln for ln in self.lines if ln.section in names and not ln.heading and ln.text.strip()]

    def locate(self, quote: str) -> tuple[bool, int, float]:
        """Find a (possibly slightly altered) quote. Returns (found, start_offset, confidence)."""
        nq, _ = normalize_with_map(quote)
        if len(nq) < 6:
            return False, -1, 0.0
        norm_text, idx = normalize_with_map(self.text)
        pos = norm_text.find(nq)
        if pos != -1:
            return True, idx[pos], 1.0
        # fuzzy: token overlap against windows of 1-3 consecutive lines
        q_tokens = [t for t in nq.split() if len(t) >= 3]
        if len(q_tokens) < 3:
            return False, -1, 0.0
        q_set = set(q_tokens)
        best = (0.0, -1)
        content = [ln for ln in self.lines if ln.text.strip()]
        for w in (1, 2, 3):
            for k in range(0, max(0, len(content) - w + 1)):
                win = content[k : k + w]
                toks = set(normalize_with_map(" ".join(x.text for x in win))[0].split())
                score = len(q_set & toks) / len(q_set)
                if score > best[0]:
                    best = (score, win[0].start)
        if best[0] >= 0.85:
            return True, best[1], round(best[0], 2)
        return False, -1, round(best[0], 2)

    def rank(self, ln: Line) -> float:
        """How convincing is this line as evidence? (higher is better)"""
        base = {"experience": 2.0, "projects": 2.0, "education": 1.5, "certifications": 1.2,
                "summary": 1.0, "skills": 0.5, "header": 0.5, "other": 0.8}.get(ln.section, 1.0)
        if ln.listlike:
            return 0.5
        if ln.quant and base >= 2.0:
            base += 0.5
        return base

    def quote_for(self, ln: Line, max_chars: int = 220) -> str:
        return truncate_words(ln.clean, max_chars)


def build_doc(text: str) -> Doc:
    sections = split_sections(text)
    lines: list[Line] = []
    pos = 0
    for i, raw in enumerate(text.split("\n")):
        start, end = pos, pos + len(raw)
        pos = end + 1
        stripped = raw.strip()
        hits = find_skills(raw)
        skills = list(dict.fromkeys(h.canon for h in hits))
        sep = sum(raw.count(c) for c in ",|;•/")
        has_digit = bool(re.search(r"\d", raw))
        listlike = (
            len(skills) >= 3 and sep >= 2 and not has_digit and not _VERB_START.match(_BULLET.sub("", raw))
            and len(raw.split()) <= 3 * len(skills) + 4
        )
        lines.append(Line(
            i=i, start=start, end=end, text=raw, section=section_at(sections, start),
            skills=skills, listlike=listlike, quant=bool(_QUANT.search(raw)),
            heading=bool(stripped) and heading_name(stripped) is not None,
        ))
    return Doc(text=text, lines=lines, sections=sections)
