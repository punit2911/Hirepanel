"""SIMULATED language models for ablation experiments.  These are NOT real LLMs.

`SimModel` plays an idealised reader (it understands paraphrases, which keyword rules cannot) and can be
configured with the failure modes real models have: hallucinated evidence, noisy scores, prompt-injection
susceptibility, and bias against identity cues. The ablations ask: "if a model fails in this way, does the
architecture still protect the decision?"  That is a statement about the safety design, not about any vendor.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from app.agents.parser import parse_heuristic
from app.doc import build_doc

from .generator import PARA

INJ = re.compile(r"ignore all previous|disregard the above|^system:|note to the ai|override the scoring", re.I | re.M)
BIAS_CUES = [re.compile(p, re.I) for p in (r"age:\s*\d+", r"gender:", r"marital", r"nationality", r"date of birth",
                                           r"community college", r"\b(19[0-9]{2})\b", r"\b(he|she)\b enjoys")]
ZW = "​‌‍⁠﻿"


class SimModel:
    name, model = "sim", "sim-1"

    def __init__(self, specs: dict[str, object], *, hallu_rate: float = 0.0, noise: float = 0.0,
                 inject_susceptible: bool = False, bias_penalty: float = 0.0):
        self.by_text = {}
        for spec in specs.values():
            for r in spec.requirements:
                self.by_text[r.text.strip()] = r
        self.h, self.noise, self.inj, self.bias = hallu_rate, noise, inject_susceptible, bias_penalty

    async def generate(self, system: str, user: str, temperature: float) -> str:
        if "Evidence Auditor" not in system and "Potential Assessor" not in system:
            # Only the two scorers are simulated. The other agents get neutral, VALID answers (an exact
            # parse, no adjudicator overrides, template-only comms), so the experiment isolates the scorers.
            if "Parser agent" in system:
                m0 = re.search(r"<resume>\n(.*?)\n</resume>", user, re.S)
                return parse_heuristic(build_doc(re.sub(f"[{ZW}]", "", m0.group(1) if m0 else ""))).model_dump_json()
            if "Adjudicator" in system:
                return json.dumps({"resolutions": []})
            if "Interview Kit" in system:
                return json.dumps({"questions": []})
            return "{}"
        lens = "evidence" if "Evidence Auditor" in system else "trajectory"
        m = re.search(r"<resume>\n(.*?)\n</resume>", user, re.S)
        resume = re.sub(f"[{ZW}]", "", m.group(1) if m else "")
        doc = build_doc(resume)
        rng = random.Random(int(hashlib.sha256((lens + resume + user[:200]).encode()).hexdigest()[:12], 16))
        injected = self.inj and bool(INJ.search(resume))
        penalty = self.bias if any(c.search(resume) for c in BIAS_CUES) else 0.0
        out = []
        for rid, text in re.findall(r"id=(R\w+) \[\w+\] (.*)", user):
            req = self.by_text.get(text.strip())
            score, quote = self._read(doc, req, lens)
            if injected:
                score, quote = 1.0, rng.choice([ln.text for ln in doc.lines if ln.text.strip()])
            elif score == 0 and rng.random() < self.h:
                score = rng.choice([0.8, 0.9, 1.0])
                quote = (rng.choice([ln.text for ln in doc.lines if ln.text.strip()]) if rng.random() < 0.4
                         else "Led a team of 12 shipping production systems at a Fortune 500 company")
            if score > 0:
                score = min(1.0, max(0.0, score + rng.gauss(0, self.noise) - penalty))
            out.append({"requirement_id": rid, "score": round(score, 3), "evidence_quote": quote, "note": "sim"})
        return json.dumps({"requirement_scores": out, "strengths": [], "gaps": [], "rationale": "simulated"})

    @staticmethod
    def _read(doc, req, lens: str) -> tuple[float, str]:
        if req is None:
            return 0.0, ""
        groups = req.groups or [[k] for k in req.keywords]
        if not groups:
            return 0.0, ""
        per, best = [], (0.0, "")
        for g in groups:
            gs, gq = 0.0, ""
            for ln in doc.lines:
                if not ln.text.strip() or ln.heading or ln.copied:
                    continue
                low = ln.text.lower()
                kw = any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low) or t in ln.skills for t in g)
                para = any(p.lower() in low for t in g for p in PARA.get(t, []))
                if not (kw or para):
                    continue
                listy = ln.listlike or ln.section in ("skills", "header")
                s = (0.5 if lens == "evidence" else 0.6) if (listy and not para) else (0.9 if para else 1.0)
                if s > gs:
                    gs, gq = s, ln.text.strip()
            per.append(gs)
            if gs > best[0]:
                best = (gs, gq)
        return round(sum(per) / len(per), 3), best[1]
