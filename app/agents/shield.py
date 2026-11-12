"""Shield agent: defends the panel against hostile resumes (prompt injection, hidden text, stuffing).

Deterministic on purpose: a security layer should not depend on an LLM's mood.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ..doc import build_doc
from ..orchestrator import Span
from ..schemas import Flag, ShieldReport
from ..util import norm

INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")

_P = lambda pat: re.compile(pat, re.I)  # noqa: E731
PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (_P(r"\b(ignore|disregard|forget|override)\b.{0,25}\b(previous|prior|above|earlier|all|any|your|the)\b.{0,25}\b(instructions?|prompts?|messages?|rules?|guidelines?|context)\b"),
     0.9, "instruction_override"),
    (_P(r"\b(rate|score|rank|grade|evaluate|assess|mark)\b.{0,20}\b(this|the)\b.{0,12}\b(candidate|applicant|resume|cv)\b.{0,25}\b(\d+|10/10|highest|perfect|top|excellent|maximum|max)\b"),
     0.9, "score_manipulation"),
    (_P(r"\b(recommend|shortlist|select|hire|advance|prioriti[sz]e)\b.{0,12}\b(this|the)\b.{0,12}\b(candidate|applicant|resume)\b"),
     0.8, "decision_manipulation"),
    (_P(r"\b(override|bypass|disable|skip)\b.{0,15}\b(the\s+)?(score|scoring|filter|screening|rules?|guardrails?|checks?|safety)\b"),
     0.8, "guardrail_bypass"),
    (_P(r"^\s*(system|assistant|developer)\s*:"), 0.7, "role_spoofing"),
    (_P(r"</?\s*(resume|system|instructions?|candidate|prompt)\s*>"), 0.7, "delimiter_spoofing"),
    (_P(r"\b(score|rating|match)\s*[:=]\s*(100|10/10|1\.0|perfect)\b"), 0.8, "score_injection"),
    (_P(r"\b(do not|don't|never)\b.{0,12}\b(reject|flag|mention|reveal|disclose|penali[sz]e)\b"), 0.6, "suppression_request"),
    (_P(r"\bas an ai( language)? model\b"), 0.5, "ai_voice"),
    (_P(r"\byou are now\b|\bnew instructions?\b|\bnew task\b"), 0.5, "role_reassignment"),
]


def _level(risk: float) -> str:
    return "high" if risk >= 0.6 else "medium" if risk >= 0.3 else "low"


def _sev(p: float) -> str:
    return "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"


def _shingles(text: str, n: int = 6) -> set[tuple[str, ...]]:
    toks = norm(text).split()
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


class ShieldAgent:
    name = "shield"

    def run(self, raw: str, jd_text: str, span: Span) -> tuple[ShieldReport, str]:
        span.mode = "deterministic"
        flags: list[Flag] = []
        probs: dict[str, float] = {}

        text = unicodedata.normalize("NFKC", raw)
        n_inv = len(INVISIBLE.findall(text))
        text = INVISIBLE.sub("", text)
        if n_inv >= 3:
            flags.append(Flag(type="hidden_characters", severity="medium",
                              text=f"{n_inv} invisible characters were removed"))
            probs["hidden_characters"] = 0.45

        lines = text.split("\n")
        remove = [False] * len(lines)

        def hits(candidate: str) -> list[tuple[float, str]]:
            return [(p, label) for rx, p, label in PATTERNS if candidate.strip() and rx.search(candidate)]

        def record(candidate: str, found: list[tuple[float, str]], idxs: list[int]) -> None:
            for p, label in found:
                if label not in probs:
                    flags.append(Flag(type=label, severity=_sev(p), text=candidate.strip()[:140],
                                      suggestion="Line removed before scoring" if p >= 0.5 else ""))
                    probs[label] = p
                if p >= 0.5:
                    for j in idxs:
                        remove[j] = True

        single = [hits(ln) for ln in lines]
        for i, found in enumerate(single):
            if found:
                record(lines[i], found, [i])  # only the offending line is removed
        for i in range(len(lines) - 1):  # an instruction split across two lines
            if not single[i] and not single[i + 1]:
                both = lines[i] + " " + lines[i + 1]
                found = hits(both)
                if found:
                    record(both, found, [i, i + 1])
        removed = sum(remove)
        if removed:
            lines = ["[removed: suspected prompt injection]" if r else ln for ln, r in zip(lines, remove)]
        text = "\n".join(lines)

        doc = build_doc(text)
        skill_set = {s for ln in doc.in_section("skills") for s in ln.skills}
        if len(skill_set) >= 35:
            flags.append(Flag(type="skills_dump", severity="low",
                              text=f"{len(skill_set)} distinct skills in the skills section"))
            probs["skills_dump"] = 0.25
        words = [w for w in norm(text).split() if len(w) >= 4]
        if len(words) >= 80:
            word, count = Counter(words).most_common(1)[0]
            if count >= 12 and count / len(words) >= 0.06:
                flags.append(Flag(type="keyword_repetition", severity="low",
                                  text=f"'{word}' appears {count} times"))
                probs["keyword_repetition"] = 0.3

        if jd_text:
            r, j = _shingles(text), _shingles(jd_text)
            if len(r) >= 40:
                overlap = len(r & j) / len(r)
                if overlap >= 0.12:
                    p = 0.8 if overlap >= 0.3 else 0.5
                    flags.append(Flag(type="jd_copy", severity=_sev(p),
                                      text=f"{overlap:.0%} of the resume's phrases are copied from the job description"))
                    probs["jd_copy"] = p

        keep = 1.0
        for p in probs.values():
            keep *= (1 - p)
        risk = round(1 - keep, 2)
        span.note(f"risk={risk} flags={[f.type for f in flags]}")
        return ShieldReport(risk=risk, level=_level(risk), flags=flags, removed_lines=removed), text
