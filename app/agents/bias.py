"""Bias Auditor: the last gate. Scans everything a human or candidate will read and scrubs it.

Deterministic lexicon always runs. An optional LLM review (BIAS_LLM=true) can add flags on top.
"""
from __future__ import annotations

import re

from ..orchestrator import Span
from ..schemas import BiasOut, BiasReport, Flag
from .base import Agent

_I = re.I
HIGH: list[tuple[str, re.Pattern]] = [
    ("gender", re.compile(r"\b(male|female|woman|women|girl|boy|gentleman|lady|husband|wife|mother|father|pregnan\w*|maternity|paternity)\b", _I)),
    ("age", re.compile(r"\b(young|youthful|elderly|older|millennial|gen z|boomer|overqualified|digital native|\d{2}[- ]years?[- ]old|aged\s+\d+)\b", _I)),
    ("family_status", re.compile(r"\b(married|unmarried|divorced|newlywed|children|kids|childcare|family planning)\b", _I)),
    ("religion", re.compile(r"\b(christian|muslim|hindu|jewish|sikh|buddhist|church|mosque|synagogue)\b", _I)),
    ("health", re.compile(r"\b(disabled|disability|wheelchair|illness|health condition|medical leave|mental health)\b", _I)),
    ("ethnicity", re.compile(r"\b(ethnic\w*|racial|race|caste|immigrant|accent)\b", _I)),
]
MEDIUM: list[tuple[str, re.Pattern]] = [
    ("nationality", re.compile(r"\b(indian|american|chinese|african|asian|european|foreign)\b(?!\s+(?:express|airlines|bank|university))", _I)),
    ("subjective_fit", re.compile(r"\b(culture fit|cultural fit|gut feel\w*|likeable|likable|bubbly|personality fit)\b", _I)),
    ("prestige", re.compile(r"\b(ivy league|top[- ]tier (?:university|school)|prestigious (?:university|school|college))\b", _I)),
]
PLACEHOLDER = re.compile(r"\[(?:CANDIDATE|EMAIL|PHONE|LINK|INSTITUTION|YEAR)\]")
PRONOUNS = [
    (re.compile(r"\b(he|she)\b", _I), "they"),
    (re.compile(r"\b(his|hers?)\b", _I), "their"),
    (re.compile(r"\bhim\b", _I), "them"),
    (re.compile(r"\b(himself|herself)\b", _I), "themself"),
]
SENT = re.compile(r"(?<=[.!?])\s+")


def _case(repl: str):
    return lambda m: repl.capitalize() if m.group(0)[0].isupper() else repl


def scrub_text(text: str) -> tuple[str, list[Flag]]:
    flags: list[Flag] = []
    out = PLACEHOLDER.sub("the candidate", text)
    n = 0
    for rx, repl in PRONOUNS:
        out, k = rx.subn(_case(repl), out)
        n += k
    if n:
        flags.append(Flag(type="gendered_pronoun", severity="medium", text=f"{n} gendered pronoun(s) neutralised"))
    kept = []
    for sent in SENT.split(out):
        hit = next(((lbl, rx.search(sent)) for lbl, rx in HIGH if rx.search(sent)), None)
        if hit:
            flags.append(Flag(type=hit[0], severity="high", text=sent.strip()[:140],
                              suggestion="Sentence removed automatically"))
        else:
            kept.append(sent)
    out = " ".join(kept)
    for lbl, rx in MEDIUM:
        m = rx.search(out)
        if m:
            flags.append(Flag(type=lbl, severity="medium", text=f"“{m.group(0)}” appears in the output"))
    return out, flags


SYSTEM = """You are the Bias Auditor on a hiring panel. Review the text written about a candidate.
Report flags for anything that references or implies protected characteristics (gender, age, race, ethnicity,
nationality, religion, disability, marital or family status), school prestige, or subjective 'fit'.
Return an empty list if the text is clean. Do not rewrite the text."""


class BiasAgent(Agent):
    name = "bias_auditor"

    async def run(self, fields: dict[str, str], span: Span) -> tuple[BiasReport, dict[str, str]]:
        span.mode = "deterministic"
        cleaned: dict[str, str] = {}
        flags: list[Flag] = []
        scrubbed = False
        for key, value in fields.items():
            out, fl = scrub_text(value or "")
            if out != (value or ""):
                scrubbed = True
            cleaned[key] = out
            flags.extend(Flag(type=f.type, severity=f.severity, text=f"[{key}] {f.text}", suggestion=f.suggestion)
                         for f in fl)
        if self.settings.bias_llm and self.llm.available:
            try:
                joined = "\n\n".join(f"[{k}]\n{v}" for k, v in cleaned.items())
                out = await self.ask(span, system=SYSTEM, user=joined[:6000], model_cls=BiasOut, temperature=0.0)
                flags.extend(out.flags)
            except Exception as e:
                span.note(f"LLM bias review skipped: {type(e).__name__}")
        high_left = any(f.severity == "high" and f.suggestion != "Sentence removed automatically" for f in flags)
        span.note(f"{len(flags)} flag(s), scrubbed={scrubbed}")
        return BiasReport(passed=not high_left, flags=flags, scrubbed=scrubbed), cleaned
