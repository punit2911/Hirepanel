"""Resume section detection (works on plain text, headings optional)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_HEADINGS: dict[str, str] = {
    "summary": r"summary|professional summary|profile|objective|about(?: me)?|career objective|career summary",
    "experience": r"experience|work experience|professional experience|employment(?: history)?|work history|career history|internships?",
    "projects": r"projects?|personal projects|academic projects|key projects|selected projects|open[- ]source",
    "education": r"education|academics?|academic background|qualifications?",
    "skills": r"skills|technical skills|core competencies|key skills|technologies|tech stack|skills summary|tools",
    "certifications": r"certifications?|licenses?(?: (?:&|and) certifications?)?|courses|training",
    "other": r"achievements?|awards|publications|languages|interests|hobbies|references|volunteer(?:ing)?|extracurriculars?",
}
_COMPILED = {k: re.compile(rf"(?:{v})", re.I) for k, v in _HEADINGS.items()}
_DECOR = re.compile(r"^[\s#*=_\-–—:|•·]+|[\s#*=_\-–—:|•·]+$")


@dataclass
class Section:
    name: str
    start: int  # offset of the heading line (0 for the header block)


def heading_name(line: str) -> str | None:
    core = _DECOR.sub("", line)
    if not core or len(core.split()) > 5:
        return None
    for name, rx in _COMPILED.items():
        if rx.fullmatch(core):
            return name
    return None


def split_sections(text: str) -> list[Section]:
    sections = [Section("header", 0)]
    pos = 0
    for raw in text.split("\n"):
        name = heading_name(raw.strip())
        if name:
            sections.append(Section(name, pos))
        pos += len(raw) + 1
    return sections


def section_at(sections: list[Section], offset: int) -> str:
    current = sections[0].name
    for s in sections:
        if s.start <= offset:
            current = s.name
        else:
            break
    return current
