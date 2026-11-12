"""Blind-screening redactor. Runs locally, *before* any text is sent to an LLM.

Removes direct identifiers (name, email, phone, links, address) and common bias proxies
(pronouns, titles, date of birth / age / marital status lines, graduation years).
The contact details are kept locally so notifications can still reach the candidate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sections import split_sections, section_at

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|(?:linkedin\.com|github\.com|gitlab\.com|bitbucket\.org|kaggle\.com|medium\.com|twitter\.com|x\.com)/\S*",
    re.I)
PHONE_RE = re.compile(r"(?<![\w/.])(\+?\(?\d[\d\s().\-]{7,}\d)(?![\w/])")
SENSITIVE_FIELD = re.compile(
    r"^[ \t]*(?:date of birth|dob|d\.o\.b\.?|age|marital status|gender|sex|nationality|religion|caste|"
    r"father'?s name|mother'?s name|passport(?: no\.?| number)?|visa status|place of birth|"
    r"(?:permanent |current |home )?address|residence|location)[ \t]*[:\-–][^\n]*$",
    re.I | re.M)
TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx)\.?\s+")
INSTITUTION_RE = re.compile(r"[^\n,|]*\b(?:University|Institute|College|School of|IIT|NIT)\b[^\n,|]*", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_PRONOUNS = [
    (re.compile(r"\b(he|she)\b", re.I), "they"),
    (re.compile(r"\b(his|hers?)\b", re.I), "their"),
    (re.compile(r"\bhim\b", re.I), "them"),
    (re.compile(r"\b(himself|herself)\b", re.I), "themself"),
]
_NOT_NAME_WORDS = {
    "resume", "curriculum", "vitae", "cv", "engineer", "developer", "scientist", "manager", "analyst",
    "summary", "profile", "objective", "experience", "education", "skills", "contact", "address", "phone",
    "email", "intern", "consultant", "architect", "designer", "lead", "senior", "junior", "data", "machine",
    "learning", "software", "full", "stack", "backend", "frontend", "specialist", "executive", "student",
    "graduate", "fresher", "professional", "technical", "applied", "research", "researcher", "officer",
}
_NAME_TOKEN = re.compile(r"^[A-Z][A-Za-z'’.\-]+$")
_COMMON = {"the", "and", "for", "with", "van", "der", "von", "del", "bin", "bint", "ben", "mac", "will", "mark", "rose", "grace", "hope", "art", "may", "bill", "pat", "sue", "ray"}


@dataclass
class Redaction:
    text: str
    contact: dict = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def guess_name(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:6]
    for line in lines:
        clean = re.split(r"[|•·]", line)[0].strip()
        if EMAIL_RE.search(clean) or re.search(r"\d", clean):
            continue
        toks = clean.replace(",", " ").split()
        if not 2 <= len(toks) <= 4:
            continue
        if any(t.lower().strip(".") in _NOT_NAME_WORDS for t in toks):
            continue
        if all(_NAME_TOKEN.match(t) or t.isupper() for t in toks):
            return " ".join(toks)
    return None


def _find_phone(text: str) -> str | None:
    for m in PHONE_RE.finditer(text):
        s = m.group(1)
        digits = re.sub(r"\D", "", s)
        if not 9 <= len(digits) <= 15:
            continue
        if re.fullmatch(r"\s*(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\s*", s):
            continue
        return s.strip()
    return None


def _keep_case(repl: str):
    def f(m: re.Match) -> str:
        return repl.capitalize() if m.group(0)[0].isupper() else repl
    return f


def _redact_education_years(text: str) -> tuple[str, int]:
    sections = split_sections(text)
    out, pos, n = [], 0, 0
    for raw in text.split("\n"):
        if section_at(sections, pos) == "education":
            new, k = YEAR_RE.subn("[YEAR]", raw)
            n += k
            out.append(new)
        else:
            out.append(raw)
        pos += len(raw) + 1
    return "\n".join(out), n


def extract_contact(text: str, name: str | None = None, email: str | None = None) -> dict:
    m = EMAIL_RE.search(text)
    return {
        "name": (name or guess_name(text) or "").strip(),
        "email": (email or (m.group(0) if m else "")).strip(),
        "phone": _find_phone(text) or "",
    }


def redact(text: str, *, name: str | None = None, email: str | None = None,
           redact_institutions: bool = False) -> Redaction:
    contact = extract_contact(text, name, email)
    counts: dict[str, int] = {}

    def bump(key: str, n: int) -> None:
        if n:
            counts[key] = counts.get(key, 0) + n

    out, n = SENSITIVE_FIELD.subn("", text)
    bump("sensitive_fields", n)
    out, n = EMAIL_RE.subn("[EMAIL]", out)
    bump("emails", n)
    out, n = URL_RE.subn("[LINK]", out)
    bump("links", n)
    phones = 0
    for m in list(PHONE_RE.finditer(out))[::-1]:
        digits = re.sub(r"\D", "", m.group(1))
        if 9 <= len(digits) <= 15 and not re.fullmatch(r"\s*(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\s*", m.group(1)):
            out = out[: m.start(1)] + "[PHONE]" + out[m.end(1):]
            phones += 1
    bump("phones", phones)

    person = contact["name"]
    if person:
        out, n = re.subn(re.escape(person), "[CANDIDATE]", out, flags=re.I)
        bump("names", n)
        for tok in {t.strip(".,") for t in person.split()}:
            if len(tok) >= 3 and tok.lower() not in _COMMON:
                out, n = re.subn(rf"\b{re.escape(tok)}\b", "[CANDIDATE]", out)
                bump("names", n)
    local = contact["email"].split("@")[0] if contact["email"] else ""
    if len(local) >= 5:
        out, n = re.subn(re.escape(local), "[CANDIDATE]", out, flags=re.I)
        bump("names", n)

    out, n = TITLE_RE.subn("", out)
    bump("titles", n)
    pron = 0
    for rx, repl in _PRONOUNS:
        out, k = rx.subn(_keep_case(repl), out)
        pron += k
    bump("pronouns", pron)
    out, n = _redact_education_years(out)
    bump("education_years", n)
    if redact_institutions:
        out, n = INSTITUTION_RE.subn("[INSTITUTION]", out)
        bump("institutions", n)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return Redaction(text=out, contact=contact, counts=counts)


def leaks(redacted: str, contact: dict) -> list[str]:
    """Return identifiers that survived redaction (used as a fail-closed safety check)."""
    found = []
    if contact.get("email") and contact["email"].lower() in redacted.lower():
        found.append("email")
    if contact.get("phone"):
        d = re.sub(r"\D", "", contact["phone"])
        if d and d in re.sub(r"\D", "", redacted):
            found.append("phone")
    for tok in (contact.get("name") or "").split():
        tok = tok.strip(".,")
        if len(tok) >= 3 and tok.lower() not in _COMMON and re.search(rf"\b{re.escape(tok)}\b", redacted):
            found.append(f"name:{tok}")
    return found
