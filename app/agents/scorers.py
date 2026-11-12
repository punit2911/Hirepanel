"""Two independent scorers with deliberately different lenses.

* evidence   - strict auditor: only proven, hands-on use earns full marks.
* trajectory - potential assessor: also credits adjacent skills and growth.

Disagreement between the two is a feature: it is what the Adjudicator resolves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..doc import Doc, Line
from ..orchestrator import Span
from ..schemas import JobSpec, Requirement, ScorerOut, ScorerSummary
from ..skills import related_to
from .base import UNTRUSTED, Agent, wrap_resume

LENS_PROMPTS = {
    "evidence": (
        "You are the Evidence Auditor on a hiring panel. Score each requirement from 0 to 1 STRICTLY by proof "
        "of hands-on use in the resume.\n"
        "1.0 = specific, concrete work clearly shown. 0.7 = relevant but thin or vague. "
        "A skill that only appears in a skills list can never score above 0.5. 0 = no evidence."
    ),
    "trajectory": (
        "You are the Potential Assessor on a hiring panel. Score each requirement from 0 to 1 by how well the "
        "candidate could meet it. Credit directly proven work fully (1.0), and also credit adjacent or "
        "transferable experience (e.g. TensorFlow work for a PyTorch requirement is roughly 0.5-0.7) and clear "
        "growth in scope. 0 = nothing relevant at all."
    ),
}

COMMON_RULES = f"""{UNTRUSTED}
Rules:
- Judge only evidence in the resume. Never use or infer name, gender, age, nationality, religion, marital status or school prestige.
- "evidence_quote" MUST be copied verbatim from the resume (max 25 words). If there is no evidence, use score 0 and an empty quote.
- Return one entry for EVERY requirement id given, with a one-sentence "note".
- "strengths" and "gaps": up to 4 short phrases each. "rationale": two sentences at most."""


@dataclass
class RawRow:
    req_id: str
    score: float
    quote: str = ""
    note: str = ""
    source: str = "heuristic"  # llm | heuristic


def _term_in_line(ln: Line, term: str) -> bool:
    if term in ln.skills:
        return True
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", ln.text.lower()))


def _line_score(ln: Line, lens: str, headless: bool, stuffed: bool = False) -> float:
    sec = ln.section
    if headless and sec == "header" and not ln.listlike:
        sec = "experience"
    if ln.listlike or sec in ("skills", "header", "other"):
        if stuffed:
            return 0.25
        return 0.4 if lens == "evidence" else 0.5
    if sec in ("experience", "projects"):
        return 1.0 if (ln.quant or lens == "trajectory") else 0.85
    if sec in ("education", "certifications"):
        return 0.7 if lens == "evidence" else 0.8
    if sec == "summary":
        return 0.6 if lens == "evidence" else 0.7
    return 0.6


def heuristic_row(doc: Doc, req: Requirement, lens: str) -> RawRow:
    groups = req.groups or [[k] for k in req.keywords]
    if not groups:
        return RawRow(req.id, 0.0, "", "no checkable keywords in this requirement")
    headless = len(doc.sections) == 1
    cands = [ln for ln in doc.lines if ln.text.strip() and not ln.heading and not ln.copied]
    scores: list[float] = []
    best: tuple[float, float, Line | None] = (0.0, 0.0, None)
    matched = 0
    for g in groups:
        gs, gl, grank = 0.0, None, -1.0
        for ln in cands:
            if any(_term_in_line(ln, t) for t in g):
                s, r = _line_score(ln, lens, headless, doc.stuffed), doc.rank(ln)
                if (s, r) > (gs, grank):
                    gs, gl, grank = s, ln, r
        if gs == 0 and lens == "trajectory":
            rel: set[str] = set()
            for t in g:
                rel |= related_to(t)
            for ln in cands:
                if rel & set(ln.skills):
                    s = 0.5 if ln.section in ("experience", "projects") and not ln.listlike else 0.3
                    r = doc.rank(ln)
                    if (s, r) > (gs, grank):
                        gs, gl, grank = s, ln, r
        scores.append(gs)
        if gs > 0:
            matched += 1
        if (gs, grank) > (best[0], best[1]):
            best = (gs, grank, gl)
    score = round(sum(scores) / len(scores), 3)
    ln = best[2]
    note = f"{matched}/{len(groups)} skill group(s) found"
    return RawRow(req.id, score, doc.quote_for(ln) if (ln and score > 0) else "", note)


def summarize_rows(lens: str, spec: JobSpec, rows: dict[str, RawRow], mode: str, provider: str = "") -> ScorerSummary:
    scoreable = [r for r in spec.requirements if not r.computed]
    wsum = sum(r.weight for r in scoreable) or 1.0
    score = 100 * sum(r.weight * rows[r.id].score for r in scoreable if r.id in rows) / wsum
    strengths = [r.text for r in scoreable if rows[r.id].score >= 0.8][:4]
    gaps = [r.text for r in scoreable if rows[r.id].score < 0.4 and r.kind == "must"][:4]
    proven = sum(1 for r in scoreable if rows[r.id].score >= 0.7)
    rationale = f"Direct evidence found for {proven} of {len(scoreable)} requirements."
    return ScorerSummary(lens=lens, mode=mode, provider=provider, score=round(score, 1), rationale=rationale,
                         strengths=[s[:140] for s in strengths], gaps=[g[:140] for g in gaps])


class ScorerAgent(Agent):
    def __init__(self, settings, llm, lens: str):
        super().__init__(settings, llm)
        self.lens = lens
        self.name = f"scorer_{'a' if lens == 'evidence' else 'b'}"

    def _prefer(self) -> str | None:
        s = self.settings
        pref = s.scorer_a_provider if self.lens == "evidence" else s.scorer_b_provider
        if pref:
            return pref
        names = self.llm.names()
        if len(names) >= 2:
            return names[0] if self.lens == "evidence" else names[1]
        return names[0] if names else None

    def _heuristic(self, doc: Doc, spec: JobSpec) -> dict[str, RawRow]:
        return {r.id: heuristic_row(doc, r, self.lens) for r in spec.requirements if not r.computed}

    async def run(self, doc: Doc, text: str, spec: JobSpec, span: Span) -> tuple[dict[str, RawRow], ScorerSummary]:
        scoreable = [r for r in spec.requirements if not r.computed]

        async def via_llm() -> tuple[dict[str, RawRow], ScorerSummary]:
            req_lines = "\n".join(f"- id={r.id} [{r.kind}] {r.text}" for r in scoreable)
            user = f"Requirements:\n{req_lines}\n\nScore every requirement id.\n\n{wrap_resume(text[:9000])}"
            out = await self.ask(span, system=LENS_PROMPTS[self.lens] + "\n" + COMMON_RULES, user=user,
                                 model_cls=ScorerOut, temperature=0.1, prefer=self._prefer())
            by_id = {rs.requirement_id: rs for rs in out.requirement_scores}
            rows: dict[str, RawRow] = {}
            filled = 0
            for r in scoreable:
                rs = by_id.get(r.id)
                if rs is None:
                    rows[r.id] = heuristic_row(doc, r, self.lens)
                    filled += 1
                else:
                    rows[r.id] = RawRow(r.id, rs.score, rs.evidence_quote, rs.note, "llm")
            if filled:
                span.status = "degraded"
                span.note(f"model skipped {filled} requirement(s); offline rules filled them in")
            summ = summarize_rows(self.lens, spec, rows, "llm", span.provider)
            if out.rationale:
                summ.rationale = out.rationale[:400]
            if out.strengths:
                summ.strengths = [s[:140] for s in out.strengths[:4]]
            if out.gaps:
                summ.gaps = [g[:140] for g in out.gaps[:4]]
            return rows, summ

        def via_rules() -> tuple[dict[str, RawRow], ScorerSummary]:
            rows = self._heuristic(doc, spec)
            return rows, summarize_rows(self.lens, spec, rows, "heuristic")

        return await self.with_fallback(span, via_llm, via_rules)
