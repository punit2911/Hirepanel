"""JD analyzer: job description -> structured requirements, plus an inclusive-language audit."""
from __future__ import annotations

import re

from ..doc import build_doc  # noqa: F401  (kept for symmetry / future use)
from ..orchestrator import Span
from ..schemas import Flag, JobSpec, Requirement
from ..skills import canonical, expand_group, find_skills
from .base import Agent

SYSTEM = """You are the Job Analyzer agent on a hiring panel. Turn a job description into scoreable requirements.
Rules:
- 5 to 12 requirements. Each must be checkable from a resume (skills, tools, experience, domain knowledge).
- kind: "must" for required qualifications, "nice" for preferred/bonus ones. Weight: must 0.6-1.0, nice 0.2-0.5.
- "keywords": the concrete skills/tools in that requirement, lowercase short names.
- "groups": AND of ORs. Each inner list is a set of interchangeable skills (any one is enough).
  Example: "PyTorch or TensorFlow, and Docker" -> [["pytorch","tensorflow"],["docker"]].
- Put years of experience in "min_years" (a number); do NOT make a requirement for it.
- Never require or prefer age, gender, nationality, accent, marital status or similar attributes."""

_STOP = set("""with and the for from that this have has will your our you are able ability strong good great
experience experienced knowledge understanding familiarity proficiency proficient hands years year plus work
working team teams such including etc using use used within across into more than least minimum required
preferred bonus nice must should would could role position candidate candidates skills skill background
excellent solid deep related relevant similar other about they their them able""".split())

_MUST_HDR = re.compile(r"requirements?|qualifications?|must[- ]haves?|what you(?:'|’)?ll need|what we(?:'|’)?re looking for|required|you have|about you", re.I)
_NICE_HDR = re.compile(r"nice[- ]to[- ]haves?|preferred|bonus|good to have|plus|desirable|extra credit", re.I)
_RESP_HDR = re.compile(r"responsibilit|what you(?:'|’)?ll do|the role|about the|about us|benefits|perks|we offer|how to apply", re.I)
_NICE_INLINE = re.compile(r"\b(nice to have|preferred|bonus|a plus|is a plus|desirable|good to have)\b", re.I)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•–·▪●◦]+|\d+[.)])\s*")
_CONNECT = re.compile(r"^[\s,;&/]*(?:and|or|and/or|&|/|,)?[\s,;]*$", re.I)

JD_LEXICON: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"\b(rock\s?stars?|ninjas?|gurus?|wizards?)\b", re.I), "coded_jargon",
     "Informal jargon like this tends to put off some applicants.", "Say what the person will actually do, e.g. 'skilled engineer'."),
    (re.compile(r"\b(young|youthful|energetic|digital natives?|recent graduates?|fresh graduates?)\b", re.I), "age_proxy",
     "Wording that signals an age preference.", "Describe the skills needed instead of a life stage."),
    (re.compile(r"\bnative[- ]english\b|\bnative speakers?\b", re.I), "language_origin",
     "Native-speaker requirements exclude fluent non-native speakers.", "Ask for 'fluent English' or a proficiency level."),
    (re.compile(r"\b(he/she|his/her|he or she)\b|\bhe\b|\bshe\b", re.I), "gendered_pronoun",
     "Gendered pronouns for the role holder.", "Use 'they' or 'you'."),
    (re.compile(r"\b(manpower|chairman|salesman|workmanship|man-hours)\b", re.I), "gendered_term",
     "Gendered job terminology.", "Use 'workforce', 'chair', 'salesperson'."),
    (re.compile(r"\b(aggressive|dominant|killer instinct|competitive environment|ruthless)\b", re.I), "masculine_coded",
     "Masculine-coded language reduces applications from some groups.", "Use 'ambitious', 'collaborative', 'driven'."),
    (re.compile(r"\bculture fit\b|\bcultural fit\b", re.I), "culture_fit",
     "'Culture fit' invites subjective, biased judgements.", "Name the specific values or working style you need."),
]


def audit_jd(description: str, spec: JobSpec) -> list[Flag]:
    flags: list[Flag] = []
    for rx, kind, text, fix in JD_LEXICON:
        m = rx.search(description)
        if m:
            flags.append(Flag(type=kind, severity="medium", text=f"“{m.group(0)}”: {text}", suggestion=fix))
    musts = [r for r in spec.requirements if r.kind == "must"]
    if len(musts) > 10:
        flags.append(Flag(type="long_must_have_list", severity="low",
                          text=f"{len(musts)} must-haves. Long lists discourage qualified applicants.",
                          suggestion="Keep must-haves to the 5-7 that truly block someone from doing the job."))
    if spec.min_years >= 8 and not re.search(r"lead|principal|staff|director|head|architect|manager", spec.title, re.I):
        flags.append(Flag(type="experience_inflation", severity="low",
                          text=f"{spec.min_years:g}+ years required for a non-leadership title.",
                          suggestion="Check that the years requirement reflects real need."))
    return flags


def _seniority(title: str) -> str:
    t = title.lower()
    for key, label in (("intern", "intern"), ("junior", "junior"), ("associate", "junior"), ("principal", "principal"),
                       ("staff", "staff"), ("lead", "lead"), ("senior", "senior"), ("sr.", "senior"),
                       ("manager", "manager"), ("director", "director"), ("head", "director")):
        if key in t:
            return label
    return "mid"


def _groups_for_line(line: str) -> list[list[str]]:
    """'Python and PyTorch or TensorFlow' -> [[python], [pytorch, tensorflow]] (AND of ORs)."""
    hits = find_skills(line)
    groups: list[list[str]] = []
    prev_end = None
    for h in hits:
        if prev_end is not None:
            conn = line[prev_end : h.start]
            if len(conn) <= 14 and _CONNECT.match(conn):
                if re.search(r"\bor\b|/", conn, re.I) and groups:
                    if h.canon not in groups[-1]:
                        groups[-1].append(h.canon)
                    prev_end = h.end
                    continue
                groups.append([h.canon])
                prev_end = h.end
                continue
        groups.append([h.canon])
        prev_end = h.end
    seen, out = set(), []
    for g in groups:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def _tokens(line: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-+]{3,}", line.lower())
    out = []
    for w in words:
        if w not in _STOP and w not in out:
            out.append(w)
    return out[:4]


def analyze_heuristic(title: str, description: str) -> JobSpec:
    section = "must"
    reqs: list[Requirement] = []
    min_years = 0.0
    seen_keys: set[tuple] = set()

    def add(text: str, kind: str, groups: list[list[str]], category: str) -> None:
        key = (kind, tuple(sorted(tuple(sorted(g)) for g in groups)), text.lower()[:40] if not groups else "")
        if key in seen_keys:
            return
        seen_keys.add(key)
        reqs.append(Requirement(text=text, kind=kind, category=category, weight=1.0 if kind == "must" else 0.4,
                                keywords=[c for g in groups for c in g], groups=groups))

    first_line = True
    for raw in description.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(_BULLET.match(line))
        body = _BULLET.sub("", line).strip()
        if first_line:
            first_line = False
            if not is_bullet and len(body.split()) <= 10 and not body.endswith("."):
                continue  # the job title line is not a requirement
        if body.lower() == title.strip().lower():
            continue
        if not is_bullet and len(body.split()) <= 7 and not body.endswith("."):
            hdr = body.rstrip(":")
            if _NICE_HDR.search(hdr):
                section = "nice"
                continue
            if _RESP_HDR.search(hdr):
                section = "ignore"
                continue
            if _MUST_HDR.search(hdr):
                section = "must"
                continue
        if section == "ignore":
            hits = find_skills(body)
            if is_bullet and hits:
                add(body, "nice", _groups_for_line(body), "skill")
            continue
        if not is_bullet and len(body.split()) > 7 and section == "must" and not _MUST_HDR.search(body):
            hits = find_skills(body)
            if not hits:
                continue
        kind = "nice" if (section == "nice" or _NICE_INLINE.search(body)) else "must"
        ym = _YEARS.search(body)
        if ym and kind == "must":
            min_years = max(min_years, float(ym.group(1)))
        groups = _groups_for_line(body)
        if ym and not groups:
            continue  # pure "N+ years" line: handled by the computed requirement
        if groups:
            add(body, kind, groups, "skill")
        else:
            toks = _tokens(body)
            if toks:
                add(body, kind, [[t] for t in toks], "domain")
    musts = [r for r in reqs if r.kind == "must"][:10]
    nices = [r for r in reqs if r.kind == "nice"][:5]
    return JobSpec(title=title, seniority=_seniority(title), min_years=min_years, requirements=musts + nices)


def normalize_spec(spec: JobSpec, title: str = "") -> JobSpec:
    """Make any spec (LLM or heuristic) safe and consistent for the scorers."""
    reqs: list[Requirement] = []
    for r in spec.requirements:
        if r.computed or re.search(r"\d+\s*\+?\s*years?", r.text, re.I) and r.category == "experience":
            continue
        if not r.text.strip():
            continue
        r.kind = r.kind if r.kind in ("must", "nice") else "must"
        if not (0.05 <= r.weight <= 1.0):
            r.weight = 1.0 if r.kind == "must" else 0.4
        r.keywords = [canonical(k) or k.strip().lower() for k in r.keywords if k.strip()]
        r.groups = [[canonical(k) or k.strip().lower() for k in g if k.strip()] for g in r.groups if g]
        r.groups = [expand_group(g) for g in r.groups if g]
        if not r.groups and r.keywords:
            r.groups = [[k] for k in r.keywords]
        if r.groups and not r.keywords:
            r.keywords = [k for g in r.groups for k in g]
        reqs.append(r)
    reqs = reqs[:15]
    for i, r in enumerate(reqs, 1):
        r.id = f"R{i}"
    if spec.min_years > 0:
        reqs.insert(0, Requirement(id="R_years", text=f"At least {spec.min_years:g} years of professional experience",
                                   kind="must", category="experience", weight=0.8, computed=True))
    spec.requirements = reqs
    spec.title = spec.title or title
    spec.seniority = spec.seniority or _seniority(spec.title)
    return spec


class JDAgent(Agent):
    name = "jd_analyzer"

    async def run(self, title: str, description: str, span: Span) -> JobSpec:
        async def via_llm() -> JobSpec:
            out = await self.ask(span, system=SYSTEM,
                                 user=f"Job title: {title}\n\nJob description:\n{description[:12000]}",
                                 model_cls=JobSpec, temperature=0.0)
            if not out.requirements:
                raise_empty()
            return out

        def raise_empty():
            from ..llm import LLMUnavailable
            raise LLMUnavailable("job analyzer returned no requirements")

        spec = await self.with_fallback(span, via_llm, lambda: analyze_heuristic(title, description))
        spec = normalize_spec(spec, title)
        spec.jd_flags = audit_jd(description, spec)
        return spec
