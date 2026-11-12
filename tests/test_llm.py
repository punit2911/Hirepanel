"""LLM layer + the language-model path of the pipeline, with fake providers (no network)."""
import asyncio
import json
import re

import httpx
import pytest

from app.llm import LLMRouter, LLMUnavailable
from app.llm.limiter import RateLimiter
from app.llm.providers import GeminiProvider
from app.pipeline import Panel
from app.schemas import Email
from evals.dataset import JD, JOB_TITLE, UNRELATED
from tests.conftest import TODAY

from pydantic import BaseModel


class Pong(BaseModel):
    answer: str


def gemini_ok(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def groq_ok(text):
    return {"choices": [{"message": {"content": text}}]}


def make_router(settings, with_settings, handler):
    s = with_settings(settings, gemini_api_key="g", groq_api_key="k", gemini_model="gemini-test", groq_model="groq-test",
                      llm_max_retries=3, gemini_rpm=0, gemini_rpd=0, groq_rpm=0, groq_rpd=0, groq_tpm=0)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = LLMRouter(s, client)
    r._backoff = staticmethod(lambda attempt: 0.0)  # do not actually sleep in tests
    return r


def run(coro):
    return asyncio.run(coro)


def test_model_picker_prefers_stable_flash():
    names = ["gemini-2.5-pro", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-image",
             "gemini-2.5-flash-lite", "gemma-3-27b", "gemini-embedding-001"]
    assert GeminiProvider.pick_model(names) == "gemini-2.5-flash"


def test_json_mode_rejection_falls_back_to_plain_mode(settings, with_settings):
    seen = []

    def handler(req):
        cfg = json.loads(req.content)["generationConfig"]
        seen.append("responseMimeType" in cfg)
        if "responseMimeType" in cfg:
            return httpx.Response(400, json={"error": {"message": "JSON mode is not enabled for this model"}})
        return httpx.Response(200, json=gemini_ok('{"answer": "plain"}'))
    r = make_router(settings, with_settings, handler)
    out, _ = run(r.complete_json(system="s", user="u", model_cls=Pong))
    out2, _ = run(r.complete_json(system="s", user="u2", model_cls=Pong))
    assert out.answer == out2.answer == "plain"
    assert seen == [True, False, False]  # learned once, did not retry JSON mode again


def test_retired_model_falls_back_to_next_best(settings, with_settings):
    s = with_settings(settings, gemini_api_key="g", groq_api_key="", gemini_model="auto", llm_providers=("gemini",))
    tried = []

    def handler(req):
        url = str(req.url)
        if url.endswith("/models?pageSize=200"):
            return httpx.Response(200, json={"models": [
                {"name": "models/gemini-9-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]})
        tried.append(url.split("/models/")[1].split(":")[0])
        if "gemini-9-flash" in url:
            return httpx.Response(404, json={"error": {"message": "models/gemini-9-flash is not found"}})
        return httpx.Response(200, json=gemini_ok('{"answer": "ok"}'))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = LLMRouter(s, client)
    out, info = run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert out.answer == "ok" and tried == ["gemini-9-flash", "gemini-2.5-flash"] and info.model == "gemini-2.5-flash"


def test_router_collects_latency_and_validity_stats(settings, with_settings):
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(200, json=gemini_ok("garbage" if n["c"] == 1 else '{"answer": "x"}'))
    r = make_router(settings, with_settings, handler)
    run(r.complete_json(system="s", user="u", model_cls=Pong))
    st = r.stats_summary()["gemini"]
    assert st["calls"] == 2 and st["ok"] == 1 and st["invalid_json"] == 1 and st["json_valid_rate"] == 0.5
    assert st["latency_p50_ms"] is not None


def test_success_and_json_fence_tolerance(settings, with_settings):
    def handler(req):
        return httpx.Response(200, json=gemini_ok('```json\n{"answer": "pong"}\n```'))
    r = make_router(settings, with_settings, handler)
    out, info = run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert out.answer == "pong" and info.provider == "gemini"


def test_rate_limit_then_fallback_to_second_provider(settings, with_settings):
    calls = {"gemini": 0, "groq": 0}

    def handler(req):
        if "generativelanguage" in str(req.url):
            calls["gemini"] += 1
            return httpx.Response(429, json={"error": "quota"})
        calls["groq"] += 1
        return httpx.Response(200, json=groq_ok('{"answer": "from groq"}'))
    r = make_router(settings, with_settings, handler)
    out, info = run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert out.answer == "from groq" and info.provider == "groq" and calls["gemini"] == 3


def test_invalid_json_is_retried_with_feedback(settings, with_settings):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body["contents"][0]["parts"][0]["text"])
        return httpx.Response(200, json=gemini_ok("not json at all" if len(seen) == 1 else '{"answer": "ok"}'))
    r = make_router(settings, with_settings, handler)
    out, info = run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert out.answer == "ok" and info.attempts == 2 and "rejected" in seen[1]


def test_all_providers_down_raises_and_circuit_opens(settings, with_settings):
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(500)
    r = make_router(settings, with_settings, handler)
    with pytest.raises(LLMUnavailable):
        run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert n["c"] == 6  # 3 attempts on each of 2 providers


def test_bad_key_is_permanent_and_pauses_provider(settings, with_settings):
    def handler(req):
        return httpx.Response(401, json={"error": "bad key"})
    r = make_router(settings, with_settings, handler)
    with pytest.raises(LLMUnavailable):
        run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert r._is_open("gemini") and r._is_open("groq")


def test_daily_budget_is_enforced_client_side():
    async def go():
        lim = RateLimiter(rpd=2)
        await lim.acquire()
        await lim.acquire()
        with pytest.raises(Exception):
            await lim.acquire()
    run(go())


def test_cache_avoids_second_call(settings, with_settings):
    n = {"c": 0}

    def handler(req):
        n["c"] += 1
        return httpx.Response(200, json=gemini_ok('{"answer": "x"}'))
    s = with_settings(settings, llm_cache=True)
    r = make_router(s, with_settings, handler)
    run(r.complete_json(system="s", user="u", model_cls=Pong))
    run(r.complete_json(system="s", user="u", model_cls=Pong))
    assert n["c"] == 1


# ------------------------------------------------------------------ pipeline with a lying model
class LyingModel:
    """Pretends every requirement is perfectly met, with invented quotes; also writes biased text."""
    name, model = "fake", "fake-1"

    def __init__(self):
        self.systems = []

    async def generate(self, system, user, temperature):
        self.systems.append(system)
        if "Parser agent" in system:
            return json.dumps({"skills": ["python"], "experience": [
                {"title": "Coordinator", "company": "BrightAds", "start": "Mar 2018", "end": "Present", "highlights": []}]})
        if "Evidence Auditor" in system or "Potential Assessor" in system:
            ids = re.findall(r"id=(R\w+)", user)
            return json.dumps({"requirement_scores": [
                {"requirement_id": i, "score": 1.0, "evidence_quote": "Led a 200 person ML org at Google Brain", "note": "great"} for i in ids],
                "strengths": ["Amazing"], "gaps": [], "rationale": "She is a wonderful fit and her husband supports her career."})
        if "Adjudicator" in system:
            return json.dumps({"resolutions": []})
        if "Communications agent" in system:
            return json.dumps({
                "recruiter_summary": "Great candidate. She is married with two kids. Strong fit overall.",
                "shortlist_email": {"subject": "Next steps", "body": "Hi {{name}}, let's talk."},
                "reject_email": {"subject": "Update", "body": "No placeholder here"}})
        if "Interview Kit" in system:
            return json.dumps({"questions": [{"question": "Tell us about a hard project.", "targets": "x", "good_signals": "y", "red_flags": "z"}]})
        return "{}"


def lying_panel(settings):
    llm = LLMRouter(settings)
    fake = LyingModel()
    llm.providers["fake"], llm.limiters["fake"] = fake, RateLimiter()
    return Panel(settings, llm), fake


def test_hallucinating_model_cannot_lift_an_unqualified_candidate(settings, offline_panel):
    spec, _ = run(offline_panel.analyze_job(JOB_TITLE, JD))
    panel, fake = lying_panel(settings)
    res = run(panel.run(spec=spec, jd_text=JD, text=UNRELATED, today=TODAY))
    assert res.mode in ("llm", "mixed")
    assert res.hallucinated_quotes >= 2 * len([r for r in spec.requirements if not r.computed])
    assert res.score < 35 and res.decision.action != "shortlist"
    assert all(not r.evidence for r in res.rows if r.resolved_by != "computed")


def test_ai_written_bias_is_scrubbed_and_escalated(settings, offline_panel):
    spec, _ = run(offline_panel.analyze_job(JOB_TITLE, JD))
    panel, _ = lying_panel(settings)
    res = run(panel.run(spec=spec, jd_text=JD, text=UNRELATED, today=TODAY))
    assert "married" not in res.recruiter_summary.lower() and "kids" not in res.recruiter_summary.lower()
    assert "husband" not in res.scorer_a.rationale.lower() and " she " not in (" " + res.scorer_a.rationale.lower() + " ")
    assert res.bias.scrubbed and any(f.severity == "high" for f in res.bias.flags)
    assert res.decision.action == "human_review"
    assert "{{name}}" in res.drafts["shortlist"].body
    assert "{{name}}" in res.drafts["reject"].body  # invalid model draft was replaced by the safe template


def test_model_never_sees_identifiers(settings, offline_panel):
    spec, _ = run(offline_panel.analyze_job(JOB_TITLE, JD))
    seen = []

    class Spy(LyingModel):
        async def generate(self, system, user, temperature):
            seen.append(user)
            return await super().generate(system, user, temperature)

    llm = LLMRouter(settings)
    llm.providers["fake"], llm.limiters["fake"] = Spy(), RateLimiter()
    text = ("Maria Gonzalez\nmaria.g@corp.com | +1 (555) 010-7788\nExperience\nCoordinator, BrightAds | Mar 2018 - Present\n"
            "- Ran campaigns. She managed the budget.\nEducation\nB.A. Communications, 2014\n")
    res = run(Panel(settings, llm).run(spec=spec, jd_text=JD, text=text, today=TODAY))
    blob = "\n".join(seen)
    assert seen and "Maria" not in blob and "Gonzalez" not in blob and "maria.g" not in blob and "555" not in blob
    assert res.contact["name"] == "Maria Gonzalez"  # but we still know who to contact, locally


def test_outage_degrades_to_offline_rules_instead_of_failing(settings, with_settings, offline_panel):
    spec, _ = run(offline_panel.analyze_job(JOB_TITLE, JD))

    def handler(req):
        return httpx.Response(503)
    r = make_router(settings, with_settings, handler)
    from evals.dataset import STRONG_STANDARD
    res = run(Panel(settings, r).run(spec=spec, jd_text=JD, text=STRONG_STANDARD, today=TODAY))
    assert res.decision.action == "shortlist" and res.mode == "heuristic"
    assert any(t["status"] == "degraded" for t in res.trace)
