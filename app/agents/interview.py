"""Interview Kit agent: probing questions aimed at the gaps the panel actually found."""
from __future__ import annotations

from ..orchestrator import Span
from ..schemas import FinalRow, InterviewQ, KitOut
from .base import Agent

SYSTEM = """You are the Interview Kit agent on a hiring panel. Write 4 structured interview questions for the recruiter.
Aim 3 at the gaps or weakly-evidenced requirements and 1 at the strongest claimed skill (to verify depth).
Each question: "question", "targets" (the requirement), "good_signals", "red_flags".
Questions must be job-related, behavioural or technical, and never touch age, family, nationality, religion, health or similar."""


def template_kit(rows: list[FinalRow]) -> list[InterviewQ]:
    weak = sorted([r for r in rows if r.resolved_by != "computed" and r.final < 0.7],
                  key=lambda r: (r.kind != "must", r.final, -r.weight))[:3]
    strong = sorted([r for r in rows if r.resolved_by != "computed" and r.final >= 0.8], key=lambda r: -r.weight)[:1]
    qs = [InterviewQ(
        question=f"Walk us through your hands-on experience with “{r.text[:110]}”. What was your own contribution, and what would you do differently now?",
        targets=r.text[:110],
        good_signals="Specific example, clear personal role, measurable result, honest reflection.",
        red_flags="Vague or team-only answers, cannot explain trade-offs.") for r in weak]
    qs += [InterviewQ(
        question=f"Your resume suggests strong experience with “{r.text[:90]}”. Walk us through the hardest problem you solved there.",
        targets=r.text[:110],
        good_signals="Explains design choices, failure modes and how success was measured.",
        red_flags="Cannot go beyond what the resume says.") for r in strong]
    return qs


class InterviewAgent(Agent):
    name = "interview_kit"

    async def run(self, rows: list[FinalRow], title: str, span: Span) -> list[InterviewQ]:
        async def via_llm() -> list[InterviewQ]:
            lines = "\n".join(f"- {r.text[:120]} (score {r.final:.2f})" for r in rows if r.resolved_by != "computed")
            out = await self.ask(span, system=SYSTEM, user=f"Role: {title}\nRequirement scores:\n{lines}",
                                 model_cls=KitOut, temperature=0.3)
            if not out.questions:
                from ..llm import LLMUnavailable
                raise LLMUnavailable("empty interview kit")
            return out.questions[:5]

        return await self.with_fallback(span, via_llm, lambda: template_kit(rows))
