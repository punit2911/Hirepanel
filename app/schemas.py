"""Typed contracts between agents. Every LLM reply is validated against one of these."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _to01(v):
    if v is None or v == "":
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"not a number: {v!r}") from None
    if f > 1.0:  # model answered on a 0-100 scale
        f = f / 100.0 if f <= 100.0 else 1.0
    return max(0.0, min(1.0, f))


Score01 = Annotated[float, BeforeValidator(_to01)]


class Flag(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"] = "medium"
    text: str = ""
    suggestion: str = ""


# ---------------------------------------------------------------- candidate profile
class Experience(BaseModel):
    title: str = ""
    company: str = ""
    start: str | None = Field(default=None, description="exactly as written, e.g. 'Jan 2020'")
    end: str | None = Field(default=None, description="exactly as written, or 'Present'")
    months: int | None = None  # computed by code, never by the model
    highlights: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree: str = ""
    field: str = ""
    institution: str = ""


class CandidateProfile(BaseModel):
    headline: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_years: float = 0.0  # computed by code from dates
    date_issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- job spec
class Requirement(BaseModel):
    id: str = ""
    text: str
    kind: Literal["must", "nice"] = "must"
    category: Literal["skill", "experience", "education", "domain", "soft", "other"] = "skill"
    weight: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    groups: list[list[str]] = Field(
        default_factory=list,
        description="AND of ORs: every inner list is an any-of group of equivalent skills",
    )
    computed: bool = False  # scored by code (e.g. years of experience), not by agents


class JobSpec(BaseModel):
    title: str = ""
    seniority: str = ""
    min_years: float = 0.0
    requirements: list[Requirement] = Field(default_factory=list)
    jd_flags: list[Flag] = Field(default_factory=list)


# ---------------------------------------------------------------- agent outputs (LLM-facing)
class ReqScore(BaseModel):
    requirement_id: str
    score: Score01 = Field(default=0.0, description="0 to 1")
    evidence_quote: str = Field(default="", description="verbatim text copied from the resume")
    note: str = ""


class ScorerOut(BaseModel):
    requirement_scores: list[ReqScore] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    rationale: str = ""


class AdjOut(BaseModel):
    resolutions: list[ReqScore] = Field(default_factory=list)


class BiasOut(BaseModel):
    flags: list[Flag] = Field(default_factory=list)


class Email(BaseModel):
    subject: str = ""
    body: str = ""


class CommsOut(BaseModel):
    recruiter_summary: str = ""
    shortlist_email: Email = Field(default_factory=Email)
    reject_email: Email = Field(default_factory=Email)


class InterviewQ(BaseModel):
    question: str
    targets: str = ""
    good_signals: str = ""
    red_flags: str = ""


class KitOut(BaseModel):
    questions: list[InterviewQ] = Field(default_factory=list)


# ---------------------------------------------------------------- pipeline result
class ShieldReport(BaseModel):
    risk: float = 0.0
    level: Literal["low", "medium", "high"] = "low"
    flags: list[Flag] = Field(default_factory=list)
    removed_lines: int = 0


class BiasReport(BaseModel):
    passed: bool = True
    flags: list[Flag] = Field(default_factory=list)
    scrubbed: bool = False


class Decision(BaseModel):
    action: Literal["shortlist", "reject", "human_review"]
    reasons: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class FinalRow(BaseModel):
    requirement_id: str
    text: str
    kind: str
    weight: float
    raw_a: float | None = None
    raw_b: float | None = None
    score_a: float | None = None  # after evidence grounding
    score_b: float | None = None
    final: float = 0.0
    evidence: str = ""
    section: str = ""
    verified: bool = True
    capped: str = ""
    disputed: bool = False
    resolved_by: Literal["average", "adjudicator", "computed", "conservative"] = "average"
    note: str = ""


class ScorerSummary(BaseModel):
    lens: str
    mode: str = ""
    provider: str = ""
    score: float = 0.0
    rationale: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class AdjSummary(BaseModel):
    disputed: list[str] = Field(default_factory=list)
    used_llm: bool = False
    notes: list[str] = Field(default_factory=list)


class PanelResult(BaseModel):
    version: str = "1"
    score: float
    confidence: float
    decision: Decision
    rows: list[FinalRow]
    gate_failures: list[str] = Field(default_factory=list)
    shield: ShieldReport
    redaction: dict[str, int] = Field(default_factory=dict)
    profile: CandidateProfile
    scorer_a: ScorerSummary
    scorer_b: ScorerSummary
    adjudication: AdjSummary
    bias: BiasReport
    recruiter_summary: str = ""
    drafts: dict[str, Email] = Field(default_factory=dict)  # keyed by "shortlist" / "reject"
    interview_kit: list[InterviewQ] = Field(default_factory=list)
    hallucinated_quotes: int = 0
    mode: str = "heuristic"  # llm | heuristic | mixed
    trace: list[dict] = Field(default_factory=list)
    contact: dict = Field(default_factory=dict)  # kept locally; stripped from stored result
