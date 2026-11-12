"""The HirePanel pipeline: a DAG of specialist agents.

shield -> redact -> [parse | scorer A | scorer B] -> verify -> adjudicate -> decide
       -> [comms | interview kit] -> bias audit -> final result
"""
from __future__ import annotations

import re
from datetime import date

from .agents.adjudicator import AdjudicatorAgent
from .agents.bias import BiasAgent
from .agents.comms import CommsAgent
from .agents.decision import compute_confidence, decide, gate_failures, weighted_score
from .agents.interview import InterviewAgent
from .agents.jd import JDAgent
from .agents.parser import ParserAgent
from .agents.scorers import ScorerAgent
from .agents.shield import ShieldAgent
from .agents.verifier import Grounded, ground
from .config import Settings
from .doc import build_doc
from .llm import LLMRouter
from .orchestrator import DAG, Span, Trace
from .pii import extract_contact, leaks, redact
from .schemas import Decision, JobSpec, PanelResult, ShieldReport
from .util import truncate_words

LLM_NODE_TIMEOUT = 240.0


class Panel:
    def __init__(self, settings: Settings, llm: LLMRouter):
        self.s, self.llm = settings, llm
        self.jd = JDAgent(settings, llm)
        self.shield = ShieldAgent()
        self.parser = ParserAgent(settings, llm)
        self.scorer_a = ScorerAgent(settings, llm, "evidence")
        self.scorer_b = ScorerAgent(settings, llm, "trajectory")
        self.adjudicator = AdjudicatorAgent(settings, llm)
        self.comms = CommsAgent(settings, llm)
        self.interview = InterviewAgent(settings, llm)
        self.bias = BiasAgent(settings, llm)

    # ------------------------------------------------------------ job analysis
    async def analyze_job(self, title: str, description: str) -> tuple[JobSpec, list[dict]]:
        trace = Trace()
        span = trace.new_span("jd_analyzer")
        spec = await self.jd.run(title, description, span)
        return spec, trace.to_list()

    # ------------------------------------------------------------ candidate run
    async def run(self, *, spec: JobSpec, jd_text: str, text: str, name_hint: str = "", email_hint: str = "",
                  today: date | None = None) -> PanelResult:
        s = self.s
        trace = Trace()
        dag = DAG(trace)
        # slow local models (Ollama on a CPU) need a longer leash than hosted APIs
        LLM_NODE_TIMEOUT = max(240.0, s.llm_timeout_s * (s.llm_max_retries + 1))  # noqa: N806

        async def n_shield(_d, sp: Span):
            if not s.shield_enabled:
                sp.status, sp.mode = "skipped", "deterministic"
                sp.note("shield disabled by SHIELD_ENABLED=false (experiment switch)")
                return ShieldReport(), text
            return self.shield.run(text, jd_text, sp)

        async def n_redact(d, sp: Span):
            report, cleaned = d["shield"]
            sp.mode = "deterministic"
            if not s.blind_mode:
                sp.note("blind mode is OFF: identifiers are not redacted")
                return cleaned, extract_contact(cleaned, name_hint, email_hint), {}
            red = redact(cleaned, name=name_hint or None, email=email_hint or None,
                         redact_institutions=s.redact_institutions)
            out = red.text
            left = leaks(out, red.contact)
            if left:  # fail closed: never let an identifier reach a model
                for item in left:
                    if item.startswith("name:"):
                        out = re.sub(rf"\b{re.escape(item[5:])}\b", "[CANDIDATE]", out)
                    elif item == "email":
                        out = out.replace(red.contact["email"], "[EMAIL]")
                sp.note(f"second-pass redaction removed: {left}")
            sp.note("redacted " + ", ".join(f"{k}={v}" for k, v in red.counts.items()) if red.counts else "nothing to redact")
            return out, red.contact, red.counts

        async def n_parse(d, sp: Span):
            return await self.parser.run(d["redact"][0], sp, today)

        def mkdoc(d):
            doc = build_doc(d["redact"][0])
            doc.mark_copied(jd_text)
            doc.stuffed = any(f.type == "skills_dump" for f in d["shield"][0].flags)
            return doc

        mode_sel = s.scorer_mode

        async def n_score_a(d, sp: Span):
            if mode_sel == "b":  # single-scorer experiment: reuse B's opinion
                sp.mode = "deterministic"
                sp.note("single-scorer mode: reusing scorer B")
                return d["score_b"]
            return await self.scorer_a.run(mkdoc(d), d["redact"][0], spec, sp)

        async def n_score_b(d, sp: Span):
            if mode_sel == "a":
                sp.mode = "deterministic"
                sp.note("single-scorer mode: reusing scorer A")
                return d["score_a"]
            return await self.scorer_b.run(mkdoc(d), d["redact"][0], spec, sp)

        async def n_verify(d, sp: Span):
            sp.mode = "deterministic"
            doc = mkdoc(d)
            raw_a, _ = d["score_a"]
            raw_b, _ = d["score_b"]
            ga, gb, hallu = {}, {}, 0
            for r in spec.requirements:
                if r.computed:
                    continue
                checked_a, checked_b = ground(doc, r, raw_a[r.id]), ground(doc, r, raw_b[r.id])
                hallu += int(checked_a.hallucinated) + int(checked_b.hallucinated)
                if s.verifier_enabled:
                    ga[r.id], gb[r.id] = checked_a, checked_b
                else:  # experiment switch: trust the scorers blindly (still counts what slipped through)
                    ga[r.id] = Grounded(raw_a[r.id].score, raw_a[r.id].quote, "", True, "", False, raw_a[r.id].score, raw_a[r.id].note)
                    gb[r.id] = Grounded(raw_b[r.id].score, raw_b[r.id].quote, "", True, "", False, raw_b[r.id].score, raw_b[r.id].note)
            if not s.verifier_enabled:
                sp.note("verifier disabled by VERIFIER_ENABLED=false (experiment switch)")
            capped = sum(1 for g in list(ga.values()) + list(gb.values()) if g.capped)
            sp.note(f"hallucinated quotes={hallu}, capped scores={capped}")
            if hallu:
                sp.status = "degraded"
            return ga, gb, hallu

        async def n_adjudicate(d, sp: Span):
            clean = d["redact"][0]
            ga, gb, _ = d["verify"]
            raw_a, _ = d["score_a"]
            raw_b, _ = d["score_b"]
            return await self.adjudicator.run(doc=mkdoc(d), text=clean, spec=spec, ga=ga, gb=gb,
                                              raw_a=raw_a, raw_b=raw_b, years=d["parse"].total_years, span=sp)

        async def n_decide(d, sp: Span):
            sp.mode = "deterministic"
            rows, _adj = d["adjudicate"]
            report, _ = d["shield"]
            profile = d["parse"]
            hallu = d["verify"][2]
            score = weighted_score(rows)
            gates = gate_failures(rows)
            mode = trace.overall_mode()
            conf = compute_confidence(rows, hallu, report, profile, mode, len(text))
            decision = decide(s, score=score, confidence=conf, gates=gates, shield=report)
            sp.note(f"score={score} confidence={conf} action={decision.action}")
            return score, conf, gates, decision

        def _strengths_gaps(rows):
            ranked = sorted([r for r in rows if r.resolved_by != "computed"], key=lambda r: -r.final)
            strengths = [r.text[:110] for r in ranked if r.final >= 0.7][:4]
            gaps = [r.text[:110] for r in sorted(ranked, key=lambda r: r.final) if r.final < 0.5 and r.kind == "must"][:4]
            return strengths, gaps

        async def n_comms(d, sp: Span):
            rows, _ = d["adjudicate"]
            score, conf, gates, decision = d["decide"]
            strengths, gaps = _strengths_gaps(rows)
            flags = [f.type.replace("_", " ") for f in d["shield"][0].flags[:2]]
            return await self.comms.run(title=spec.title or "the role", score=score, confidence=conf,
                                        action=decision.action, strengths=strengths, gaps=gaps, flags=flags, span=sp)

        async def n_interview(d, sp: Span):
            rows, _ = d["adjudicate"]
            if d["decide"][3].action == "reject":
                sp.status, sp.mode = "skipped", "deterministic"
                return []
            return await self.interview.run(rows, spec.title or "the role", sp)

        async def n_bias(d, sp: Span):
            _, adj = d["adjudicate"]
            summary, drafts = d["comms"]
            fields = {"recruiter_summary": summary,
                      "scorer_a_rationale": d["score_a"][1].rationale,
                      "scorer_b_rationale": d["score_b"][1].rationale,
                      "shortlist_email": drafts["shortlist"].body, "reject_email": drafts["reject"].body}
            for i, q in enumerate(d["interview_kit"] or []):
                fields[f"question_{i + 1}"] = q.question
            report, cleaned = await self.bias.run(fields, sp)
            return report, cleaned

        dag.add("shield", n_shield)
        dag.add("redact", n_redact, ("shield",))
        dag.add("parse", n_parse, ("redact",), timeout=LLM_NODE_TIMEOUT)
        dag.add("score_a", n_score_a, ("redact", "shield") + (("score_b",) if mode_sel == "b" else ()),
                timeout=LLM_NODE_TIMEOUT)
        dag.add("score_b", n_score_b, ("redact", "shield") + (("score_a",) if mode_sel == "a" else ()),
                timeout=LLM_NODE_TIMEOUT)
        dag.add("verify", n_verify, ("redact", "shield", "score_a", "score_b"))
        dag.add("adjudicate", n_adjudicate, ("redact", "shield", "verify", "score_a", "score_b", "parse"),
                timeout=LLM_NODE_TIMEOUT)
        dag.add("decide", n_decide, ("adjudicate", "shield", "parse", "verify"))
        dag.add("comms", n_comms, ("adjudicate", "decide", "shield"), timeout=LLM_NODE_TIMEOUT)
        dag.add("interview_kit", n_interview, ("adjudicate", "decide"), optional=True, timeout=LLM_NODE_TIMEOUT)
        dag.add("bias_audit", n_bias, ("adjudicate", "comms", "interview_kit", "score_a", "score_b"))
        res = await dag.run()
        return self._assemble(spec, res, trace)

    def _assemble(self, spec: JobSpec, res: dict, trace: Trace) -> PanelResult:
        clean_text, contact, counts = res["redact"]
        shield, _ = res["shield"]
        profile = res["parse"]
        rows, adj = res["adjudicate"]
        _, summ_a = res["score_a"]
        _, summ_b = res["score_b"]
        score, conf, gates, decision = res["decide"]
        hallu = res["verify"][2]
        bias, cleaned = res["bias_audit"]
        summary, drafts = res["comms"]
        kit = res["interview_kit"] or []

        summary = cleaned.get("recruiter_summary", summary)
        drafts["shortlist"].body = cleaned.get("shortlist_email", drafts["shortlist"].body)
        drafts["reject"].body = cleaned.get("reject_email", drafts["reject"].body)
        summ_a.rationale = cleaned.get("scorer_a_rationale", summ_a.rationale)
        summ_b.rationale = cleaned.get("scorer_b_rationale", summ_b.rationale)
        for i, q in enumerate(kit):
            q.question = cleaned.get(f"question_{i + 1}", q.question)

        high = [f for f in bias.flags if f.severity == "high"]
        if high and decision.action != "human_review":
            decision = Decision(action="human_review", reasons=decision.reasons,
                                problems=decision.problems + ["The AI wrote something about a protected characteristic. It was removed. A person should review."])
        return PanelResult(
            score=score, confidence=conf, decision=decision, rows=rows, gate_failures=gates, shield=shield,
            redaction=counts, profile=profile, scorer_a=summ_a, scorer_b=summ_b, adjudication=adj, bias=bias,
            recruiter_summary=truncate_words(summary, 900), drafts=drafts, interview_kit=kit,
            hallucinated_quotes=hallu, mode=trace.overall_mode(), trace=trace.to_list(), contact=contact)
