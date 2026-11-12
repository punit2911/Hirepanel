"""Decision policy. Deterministic, explainable, auditable: no LLM is allowed to make this call."""
from __future__ import annotations

from ..config import Settings
from ..schemas import CandidateProfile, Decision, FinalRow, ShieldReport
from ..util import clamp


def weighted_score(rows: list[FinalRow]) -> float:
    w = sum(r.weight for r in rows)
    if w <= 0:
        return 0.0
    return round(100 * sum(r.weight * r.final for r in rows) / w, 1)


def gate_failures(rows: list[FinalRow]) -> list[str]:
    out = []
    for r in rows:
        if r.kind != "must":
            continue
        if r.resolved_by == "computed":
            if r.final < 0.5:
                out.append("Experience is below the required minimum")
        elif r.final < 0.2:
            out.append(f"No evidence for must-have: {r.text[:100]}")
    return out


def compute_confidence(rows: list[FinalRow], hallucinated: int, shield: ShieldReport,
                       profile: CandidateProfile, mode: str, text_len: int) -> float:
    scored = [r for r in rows if r.resolved_by != "computed"]
    n = max(1, len(scored))
    agree = sum(1 for r in scored if not r.disputed) / n
    verified = sum(1 for r in scored if r.verified) / n
    conf = 0.45 + 0.25 * agree + 0.15 * verified + (0.1 if mode in ("llm", "mixed") else 0.0)
    conf -= 0.25 * shield.risk
    conf -= min(0.2, 0.05 * hallucinated)
    conf -= 0.05 * min(2, len(profile.date_issues))
    if not profile.experience and text_len > 500:
        conf -= 0.15
    return round(clamp(conf), 2)


def decide(settings: Settings, *, score: float, confidence: float, gates: list[str],
           shield: ShieldReport) -> Decision:
    thr, band = settings.shortlist_threshold, settings.review_band
    reasons: list[str] = []
    problems: list[str] = []

    if shield.level == "high":
        problems.append("Possible manipulation detected in the resume. A person should read it.")
    if confidence < settings.min_confidence:
        problems.append(f"Panel confidence is low ({confidence:.0%}).")

    if problems:
        action = "human_review"
        reasons.append(f"Score {score:g} with {confidence:.0%} confidence")
    elif score >= thr and not gates:
        action = "shortlist"
        reasons.append(f"Score {score:g} meets the {thr:g} threshold with no missing must-haves")
    elif score >= thr and gates:
        action = "human_review"
        reasons.append(f"Score {score:g} meets the threshold but a must-have looks missing")
        problems.extend(gates)
    elif score >= thr - band:
        action = "human_review"
        reasons.append(f"Score {score:g} is within {band:g} points of the {thr:g} threshold")
    else:
        action = "reject"
        reasons.append(f"Score {score:g} is below the {thr:g} threshold")
        if gates:
            reasons.append(gates[0])
    return Decision(action=action, reasons=reasons, problems=problems)
