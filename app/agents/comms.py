"""Comms agent: recruiter summary + both candidate email drafts. Nothing is ever sent automatically."""
from __future__ import annotations

from ..config import Settings
from ..orchestrator import Span
from ..schemas import CommsOut, Email
from .base import Agent

SYSTEM = """You are the Communications agent on a hiring panel. Write in plain, warm, professional English.
Produce three things:
1. "recruiter_summary": 3 to 4 sentences for the recruiter: fit, strongest evidence, biggest gaps, recommended next step.
2. "shortlist_email": a friendly note inviting the candidate to the next step (a short call). Do not promise anything.
3. "reject_email": a kind, respectful rejection. Do NOT mention scores, rankings, AI, or detailed reasons.
Rules:
- Begin both emails with exactly "Hi {{name}}," and keep that placeholder as written.
- Sign off with the recruiter name and company given. Keep each email under 120 words.
- Never mention or imply gender, age, nationality, religion, family status or school prestige.
- Use only the facts given. Never invent details."""


def templates(settings: Settings, title: str) -> tuple[Email, Email]:
    co, rec = settings.company_name, settings.recruiter_name
    short = Email(
        subject=f"Next steps for the {title} role",
        body=(f"Hi {{{{name}}}},\n\nThank you for applying for the {title} role at {co}. Your background stood out to us "
              f"and we would like to set up a short call to talk about the role and your experience.\n\n"
              f"Please reply with a few times that suit you over the next week.\n\nBest regards,\n{rec}\n{co}"))
    rej = Email(
        subject=f"Your application for {title}",
        body=(f"Hi {{{{name}}}},\n\nThank you for taking the time to apply for the {title} role at {co}. "
              f"After careful consideration we have decided to move forward with other candidates whose experience "
              f"more closely matches what we need right now.\n\nWe appreciate your interest and wish you every success "
              f"in your search.\n\nBest regards,\n{rec}\n{co}"))
    return short, rej


def recruiter_template(title: str, score: float, confidence: float, action: str,
                       strengths: list[str], gaps: list[str], flags: list[str]) -> str:
    verb = {"shortlist": "Recommend moving to a call.", "human_review": "Needs a person to review before any decision.",
            "reject": "Recommend a polite rejection."}[action]
    parts = [f"Scored {score:g}/100 for {title} with {confidence:.0%} panel confidence. {verb}"]
    if strengths:
        parts.append("Strongest evidence: " + "; ".join(strengths[:3]) + ".")
    if gaps:
        parts.append("Gaps to probe: " + "; ".join(gaps[:3]) + ".")
    if flags:
        parts.append("Notes: " + "; ".join(flags[:3]) + ".")
    return " ".join(parts)


class CommsAgent(Agent):
    name = "comms"

    async def run(self, *, title: str, score: float, confidence: float, action: str, strengths: list[str],
                  gaps: list[str], flags: list[str], span: Span) -> tuple[str, dict[str, Email]]:
        short, rej = templates(self.settings, title)
        summary = recruiter_template(title, score, confidence, action, strengths, gaps, flags)

        async def via_llm() -> tuple[str, dict[str, Email]]:
            facts = (f"Role: {title}\nCompany: {self.settings.company_name}\nRecruiter name: {self.settings.recruiter_name}\n"
                     f"Score: {score:g}/100 (confidence {confidence:.0%})\nPanel recommendation: {action}\n"
                     f"Strengths: {strengths}\nGaps: {gaps}\nNotes: {flags}")
            out = await self.ask(span, system=SYSTEM, user=facts, model_cls=CommsOut, temperature=0.4)
            ok = lambda e: "{{name}}" in e.body and e.subject.strip()  # noqa: E731
            return (out.recruiter_summary.strip() or summary,
                    {"shortlist": out.shortlist_email if ok(out.shortlist_email) else short,
                     "reject": out.reject_email if ok(out.reject_email) else rej})

        return await self.with_fallback(span, via_llm, lambda: (summary, {"shortlist": short, "reject": rej}))
