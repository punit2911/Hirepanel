"""Adjudicator: merges the two scorers, and calls a tie-break only where they really disagree."""
from __future__ import annotations

from ..doc import Doc
from ..orchestrator import Span
from ..schemas import AdjOut, AdjSummary, FinalRow, JobSpec
from ..util import clamp
from .base import UNTRUSTED, Agent, wrap_resume
from .scorers import RawRow
from .verifier import Grounded, ground

SYSTEM = f"""You are the Adjudicator on a hiring panel. Two assessors disagree about some requirements.
For each disputed requirement decide the fair score from 0 to 1 using ONLY the resume text.
{UNTRUSTED}
Rules:
- Prefer the assessor whose evidence quote is real, specific and relevant. Ignore claims without proof.
- "evidence_quote" must be copied verbatim from the resume. If nothing supports the requirement, score 0 with an empty quote.
- Never use name, gender, age, nationality, religion or school prestige.
- Return one resolution per disputed requirement_id, with a one-sentence "note"."""


def _pick_evidence(a: Grounded, b: Grounded) -> Grounded:
    cands = [g for g in (a, b) if g.evidence]
    if not cands:
        return a
    return max(cands, key=lambda g: g.score)


class AdjudicatorAgent(Agent):
    name = "adjudicator"

    async def run(self, *, doc: Doc, text: str, spec: JobSpec, ga: dict[str, Grounded], gb: dict[str, Grounded],
                  raw_a: dict[str, RawRow], raw_b: dict[str, RawRow], years: float,
                  span: Span) -> tuple[list[FinalRow], AdjSummary]:
        delta = self.settings.disagreement_delta
        rows: dict[str, FinalRow] = {}
        disputed: list[str] = []
        for req in spec.requirements:
            if req.computed:
                need = spec.min_years or 1.0
                rows[req.id] = FinalRow(
                    requirement_id=req.id, text=req.text, kind=req.kind, weight=req.weight,
                    final=round(clamp(years / need), 2), section="experience", resolved_by="computed",
                    evidence=f"Computed from the dates in the resume: {years:g} years")
                continue
            a, b = ga[req.id], gb[req.id]
            best = _pick_evidence(a, b)
            caps = ";".join(dict.fromkeys(c for c in (a.capped, b.capped) if c))
            row = FinalRow(
                requirement_id=req.id, text=req.text, kind=req.kind, weight=req.weight,
                raw_a=round(raw_a[req.id].score, 2), raw_b=round(raw_b[req.id].score, 2),
                score_a=round(a.score, 2), score_b=round(b.score, 2),
                evidence=best.evidence, section=best.section, verified=a.verified and b.verified,
                capped=caps, note=best.note or "")
            if abs(a.score - b.score) <= delta:
                row.final = round((a.score + b.score) / 2, 2)
                row.resolved_by = "average"
            else:
                row.disputed = True
                disputed.append(req.id)
            rows[req.id] = row

        summary = AdjSummary(disputed=disputed)
        resolved: dict[str, Grounded] = {}
        if disputed and self.llm.available:
            try:
                resolved = await self._ask(doc, text, spec, disputed, ga, gb, raw_a, raw_b, span)
                summary.used_llm = True
            except Exception as e:  # LLMUnavailable or anything unexpected: never block the pipeline
                span.status = "degraded"
                span.note(f"tie-break model unavailable, used the conservative score instead: {type(e).__name__}")
        if disputed and not summary.used_llm:
            span.mode = "heuristic"
        for rid in disputed:
            row, a, b = rows[rid], ga[rid], gb[rid]
            g = resolved.get(rid)
            if g is not None:
                row.final = round(g.score, 2)
                row.resolved_by = "adjudicator"
                row.evidence = g.evidence or row.evidence
                row.section = g.section or row.section
                row.verified = row.verified and g.verified
                if g.capped:
                    row.capped = ";".join(dict.fromkeys([c for c in row.capped.split(";") if c] + [g.capped]))
                row.note = g.note or row.note
                summary.notes.append(f"{rid}: tie-break settled at {row.final}")
            else:
                low = a if a.score <= b.score else b
                row.final = round(min(a.score, b.score), 2)
                row.resolved_by = "conservative"
                row.evidence = low.evidence or row.evidence
                row.section = low.section or row.section
                summary.notes.append(f"{rid}: used the lower score ({row.final}) because no tie-break was available")
        span.note(f"{len(disputed)} disputed requirement(s)")
        return [rows[r.id] for r in spec.requirements], summary

    async def _ask(self, doc, text, spec, disputed, ga, gb, raw_a, raw_b, span) -> dict[str, Grounded]:
        by_id = {r.id: r for r in spec.requirements}
        blocks = []
        for rid in disputed:
            req = by_id[rid]
            blocks.append(
                f"Requirement {rid} [{req.kind}]: {req.text}\n"
                f"  Assessor A scored {ga[rid].score:.2f}; quote: \"{raw_a[rid].quote}\"; note: {raw_a[rid].note}\n"
                f"  Assessor B scored {gb[rid].score:.2f}; quote: \"{raw_b[rid].quote}\"; note: {raw_b[rid].note}")
        user = "Disputed requirements:\n\n" + "\n\n".join(blocks) + "\n\n" + wrap_resume(text[:9000])
        out = await self.ask(span, system=SYSTEM, user=user, model_cls=AdjOut, temperature=0.0)
        resolved: dict[str, Grounded] = {}
        for res in out.resolutions:
            if res.requirement_id not in disputed:
                continue
            g = ground(doc, by_id[res.requirement_id], RawRow(res.requirement_id, res.score, res.evidence_quote,
                                                               res.note, "llm"))
            g.note = res.note
            resolved[res.requirement_id] = g
        return resolved
