"""Parser agent: resume text -> structured profile. The LLM extracts; code computes durations."""
from __future__ import annotations

import re
from datetime import date

from ..dates import RANGE_RE, months_of, parse_range, years_from_ranges
from ..doc import Doc, build_doc
from ..orchestrator import Span
from ..schemas import CandidateProfile, Education, Experience
from ..skills import canonical, skills_in
from .base import UNTRUSTED, Agent, wrap_resume

SYSTEM = f"""You are the Parser agent on a hiring panel. Extract structured data from the resume.
{UNTRUSTED}
Rules:
- Only extract what is explicitly written. Never guess, infer or invent anything.
- Never record or infer age, gender, ethnicity, religion, nationality, marital status or health.
- Copy dates exactly as written (e.g. "Jan 2020", "2019", "Present"). Do not calculate durations.
- "skills": individual skills/tools/technologies mentioned anywhere, short names.
- "highlights": up to 5 short bullet points per job, close to the original wording."""

DEGREE_RE = re.compile(
    r"(?<![A-Za-z])(?:B\.?E|B\.?Tech|B\.?Sc|B\.?A|B\.?Com|BS|BCA|M\.?E|M\.?Tech|M\.?Sc|M\.?A|MS|MCA|MBA|M\.?Com|Ph\.?D)\.?(?![A-Za-z])"
    r"|\b(?:Bachelor|Master|Doctor|Diploma|Associate)\b[^\n,|]*")
INSTITUTION_WORD = re.compile(r"University|Institute|College|School|Academy|\[INSTITUTION\]|IIT|NIT", re.I)
_SEPS = [" | ", " at ", " @ ", ", ", " - ", " – ", " — "]


def _split_title_company(s: str) -> tuple[str, str]:
    s = re.sub(r"[()\[\]]", " ", s).strip(" -–—|,•·\t")
    s = re.sub(r"\s{2,}", " ", s)
    for sep in _SEPS:
        if sep in s:
            a, b = s.split(sep, 1)
            return a.strip(), b.strip(" -–—|,")
    return s, ""


def parse_heuristic(doc: Doc) -> CandidateProfile:
    exp_lines = [ln for ln in doc.lines if ln.section == "experience" and not ln.heading]
    if not exp_lines:
        exp_lines = [ln for ln in doc.lines
                     if ln.section not in ("education", "skills", "certifications") and not ln.heading]
    entries: list[Experience] = []
    for k, ln in enumerate(exp_lines):
        m = RANGE_RE.search(ln.text)
        if not m or DEGREE_RE.search(ln.text):
            continue
        rest = (ln.text[: m.start()] + " " + ln.text[m.end():]).strip()
        if len(re.sub(r"[\W_]+", "", rest)) < 3:  # only dates on this line: title is on the line above
            prev = next((p for p in reversed(exp_lines[:k]) if p.text.strip()), None)
            rest = prev.clean if prev else ""
        title, company = _split_title_company(rest)
        highlights: list[str] = []
        for nxt in exp_lines[k + 1 :]:
            if RANGE_RE.search(nxt.text) and not nxt.text.lstrip().startswith(("-", "•", "*")):
                break
            if nxt.text.strip() and len(highlights) < 8:
                highlights.append(nxt.clean)
        entries.append(Experience(title=title, company=company, start=m.group("s"), end=m.group("e"),
                                  highlights=highlights))
    edu: list[Education] = []
    edu_lines = [ln for ln in doc.lines if ln.section == "education" and not ln.heading and ln.text.strip()]
    for k, ln in enumerate(edu_lines):
        dm = DEGREE_RE.search(ln.text)
        if not dm:
            continue
        near = " ".join(x.text for x in edu_lines[max(0, k - 1) : k + 2])
        inst = next((x.clean for x in edu_lines[max(0, k - 1) : k + 2] if INSTITUTION_WORD.search(x.text)), "")
        edu.append(Education(degree=re.sub(r"\[YEAR\]|\b(?:19|20)\d{2}\b", "", ln.clean).strip(" -–—|,"),
                             field="", institution=inst if inst else ("" if not INSTITUTION_WORD.search(near) else near[:80])))
    certs = [ln.clean for ln in doc.lines if ln.section == "certifications" and not ln.heading and ln.text.strip()][:8]
    headline = next((ln.clean for ln in doc.lines if ln.section == "summary" and not ln.heading and ln.text.strip()), "")
    return CandidateProfile(headline=headline[:160], skills=skills_in(doc.text), experience=entries,
                            education=edu, certifications=certs)


def finalize_profile(profile: CandidateProfile, doc: Doc, today: date | None = None) -> CandidateProfile:
    """Deterministic post-processing shared by LLM and heuristic paths."""
    today = today or date.today()
    ranges, issues = [], []
    for e in profile.experience:
        rng = parse_range(e.start, e.end, today)
        if rng is None:
            if e.start or e.end:
                issues.append(f"could not read dates for '{e.title or 'a role'}'")
            e.months = None
            continue
        if rng[0] > today.year * 12 + today.month - 1:
            issues.append(f"'{e.title or 'a role'}' starts in the future")
            continue
        e.months = months_of(rng)
        ranges.append(rng)
    profile.total_years = years_from_ranges(ranges)
    if profile.total_years > 45:
        issues.append("implausible total experience")
        profile.total_years = 0.0
    merged: list[str] = []
    for s in list(profile.skills) + skills_in(doc.text):
        c = canonical(s) or s.strip().lower()
        if c and c not in merged:
            merged.append(c)
    profile.skills = merged
    profile.date_issues = issues
    return profile


class ParserAgent(Agent):
    name = "parser"

    async def run(self, text: str, span: Span, today: date | None = None) -> CandidateProfile:
        doc = build_doc(text)

        async def via_llm() -> CandidateProfile:
            return await self.ask(span, system=SYSTEM, user="Parse this resume.\n\n" + wrap_resume(text),
                                  model_cls=CandidateProfile, temperature=0.0)

        profile = await self.with_fallback(span, via_llm, lambda: parse_heuristic(doc))
        profile.total_years, profile.date_issues = 0.0, []
        for e in profile.experience:
            e.months = None
        return finalize_profile(profile, doc, today)
