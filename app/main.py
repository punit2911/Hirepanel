"""FastAPI application: REST API, background queue, review workflow and the single-page UI."""
from __future__ import annotations

import asyncio
import csv
import hmac
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .agents.jd import _groups_for_line, normalize_spec
from .config import Settings, get_settings
from .extract import ExtractError, extract_text
from .llm import LLMRouter
from .notify import Notifier
from .orchestrator import PipelineError
from .pipeline import Panel
from .schemas import Email, JobSpec, Requirement
from .security import RateLimit, SecurityHeaders, check_key, is_public_read, mask_name
from .store import Store
from .util import now_iso

log = logging.getLogger("hirepanel")
UI_FILE = Path(__file__).parent / "ui" / "index.html"
MAX_FILES = 25


# ----------------------------------------------------------------------------- request models
class JobIn(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    description: str = Field(min_length=30, max_length=20000)


class ReqIn(BaseModel):
    text: str = Field(min_length=2, max_length=300)
    kind: Literal["must", "nice"] = "must"
    keywords: list[str] = Field(default_factory=list, max_length=12)
    groups: list[list[str]] = Field(default_factory=list, max_length=12)


class SpecIn(BaseModel):
    min_years: float = Field(default=0, ge=0, le=40)
    requirements: list[ReqIn] = Field(min_length=1, max_length=15)


class TextIn(BaseModel):
    text: str = Field(min_length=80, max_length=40000)
    name: str = Field(default="", max_length=120)
    email: str = Field(default="", max_length=200)


class ReviewIn(BaseModel):
    action: Literal["approve_shortlist", "approve_reject", "hold"]
    note: str = Field(default="", max_length=2000)


class WebhookReview(ReviewIn):
    candidate_id: str = Field(min_length=4, max_length=32)


# ----------------------------------------------------------------------------- runtime
class Runtime:
    def __init__(self, settings: Settings, llm: LLMRouter | None = None, notifier: Notifier | None = None):
        self.settings = settings
        self.store = Store(settings.database_url)
        self.llm = llm or LLMRouter(settings)
        self.panel = Panel(settings, self.llm)
        self.notifier = notifier or Notifier(settings)
        self.sem = asyncio.Semaphore(settings.max_concurrent_runs)
        self.tasks: set[asyncio.Task] = set()
        self.loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    def spawn(self, coro):
        """Start a background coroutine from the event loop OR from a threadpool (sync endpoints)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    # ------------------------------------------------------------ processing
    async def process(self, cid: str) -> None:
        async with self.sem:
            cand = self.store.get_candidate(cid)
            if not cand or cand["status"] not in ("queued", "processing"):
                return
            job = self.store.get_job(cand["job_id"])
            if not job:
                self.store.update_candidate(cid, status="failed", error="The job for this candidate was deleted")
                return
            self.store.update_candidate(cid, status="processing", error="")
            t0 = time.perf_counter()
            try:
                spec = JobSpec.model_validate(job["spec"])
                res = await asyncio.wait_for(
                    self.panel.run(spec=spec, jd_text=job["description"], text=cand["raw_text"],
                                   name_hint=cand.get("name_hint", "")),
                    self.settings.run_timeout_s)
            except asyncio.TimeoutError:
                self._fail(cid, "Timed out. The free AI quota may be exhausted. Try again in a few minutes.")
                return
            except PipelineError as e:
                log.exception("pipeline failed for %s", cid)
                self._fail(cid, f"The {e.node} step failed. See server logs.")
                return
            except Exception as e:  # noqa: BLE001
                log.exception("unexpected failure for %s", cid)
                self._fail(cid, f"Unexpected error: {type(e).__name__}")
                return
            ms = int((time.perf_counter() - t0) * 1000)
            contact = res.contact
            data = res.model_dump(exclude={"contact"})
            self.store.update_candidate(
                cid, status="done", decision=res.decision.action, score=res.score, confidence=res.confidence,
                result=data, contact=contact, run_ms=ms, error="")
            self.store.audit("panel", "scored", cid,
                             f"score={res.score} confidence={res.confidence} decision={res.decision.action} mode={res.mode}")
        await self._notify_decided(cid, job)

    def _fail(self, cid: str, msg: str) -> None:
        self.store.update_candidate(cid, status="failed", error=msg)
        self.store.audit("panel", "failed", cid, msg)

    async def _notify_decided(self, cid: str, job: dict) -> None:
        if not self.notifier.enabled:
            return
        c = self.store.get_candidate(cid)
        if not c or not c.get("result"):
            return
        r = c["result"]
        strengths = [x["text"] for x in sorted(r["rows"], key=lambda x: -x["final"])[:3] if x["final"] >= 0.7]
        payload = {
            "candidate_id": cid, "job_id": job["id"], "job_title": job["title"],
            "candidate_name": (c.get("contact") or {}).get("name", ""),
            "score": r["score"], "confidence": r["confidence"], "decision": r["decision"]["action"],
            "reasons": r["decision"]["reasons"], "problems": r["decision"]["problems"],
            "summary": r["recruiter_summary"], "top_strengths": strengths, "gaps": r["gate_failures"],
            "review_url": f"{self.settings.public_base_url}/#/candidate/{cid}",
        }
        status = await self.notifier.send("candidate_decided", payload)
        notified = dict(c.get("notified") or {})
        notified.update({k: {"status": v, "at": now_iso()} for k, v in status.items()})
        self.store.update_candidate(cid, notified=notified)
        self.store.audit("system", "notified", cid, str(status))

    def enqueue(self, cid: str) -> None:
        self.spawn(self.process(cid))

    # ------------------------------------------------------------ review
    async def review(self, cid: str, body: ReviewIn, actor: str) -> dict:
        c = self.store.get_candidate(cid)
        if not c:
            raise HTTPException(404, "Candidate not found")
        if c["status"] != "done":
            raise HTTPException(409, "This candidate has not finished scoring yet")
        job = self.store.get_job(c["job_id"]) or {"title": "the role"}
        self.store.update_candidate(cid, human_action=body.action, human_note=body.note, reviewed_at=now_iso())
        self.store.audit(actor, f"review:{body.action}", cid, body.note)
        out: dict = {"ok": True, "action": body.action, "email": None, "email_status": None}
        if body.action == "hold":
            return out
        kind = "shortlist" if body.action == "approve_shortlist" else "reject"
        draft = Email.model_validate(((c["result"] or {}).get("drafts") or {}).get(kind) or {})
        contact = c.get("contact") or {}
        first = (contact.get("name") or "").split(" ")[0] or "there"
        filled = {"subject": draft.subject, "body": draft.body.replace("{{name}}", first),
                  "to": contact.get("email", "")}
        out["email"] = filled
        if self.settings.n8n_webhook_url and filled["to"]:
            status = await self.notifier.send("send_candidate_email", {
                "candidate_id": cid, "job_title": job["title"], "kind": kind, **filled})
            out["email_status"] = status.get("n8n")
            self.store.audit("system", "email_handoff", cid, f"{kind}: {status}")
        return out

    # ------------------------------------------------------------ spec edit
    def build_spec(self, job: dict, body: SpecIn) -> JobSpec:
        old = JobSpec.model_validate(job["spec"])
        reqs = []
        for r in body.requirements:
            groups = (r.groups if r.groups else [[k] for k in r.keywords] if r.keywords
                      else _groups_for_line(r.text))
            reqs.append(Requirement(text=r.text, kind=r.kind, keywords=[k for g in groups for k in g], groups=groups,
                                    category="skill" if groups else "domain"))
        spec = JobSpec(title=old.title, seniority=old.seniority, min_years=body.min_years, requirements=reqs,
                       jd_flags=old.jd_flags)
        return normalize_spec(spec, old.title)

    async def retention_loop(self) -> None:
        while True:
            try:
                n = self.store.purge_older_than(self.settings.retention_days)
                if n:
                    self.store.audit("system", "retention_purge", None, f"deleted {n} candidate(s)")
            except Exception:  # noqa: BLE001
                log.exception("retention purge failed")
            await asyncio.sleep(6 * 3600)


# ----------------------------------------------------------------------------- app factory
def _csv_safe(v) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def create_app(settings: Settings | None = None, *, llm: LLMRouter | None = None,
               notifier: Notifier | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = Runtime(settings, llm, notifier)
        app.state.rt = rt
        for cid in rt.store.pending_ids():
            rt.store.update_candidate(cid, status="queued")
            rt.enqueue(cid)
        if settings.retention_days > 0:
            rt.spawn(rt.retention_loop())
        log.info("HirePanel %s ready (env=%s, llm=%s)", __version__, settings.env, rt.llm.names() or "offline mode")
        yield
        for t in list(rt.tasks):
            t.cancel()
        await rt.llm.aclose()
        await rt.notifier.aclose()

    app = FastAPI(title="HirePanel", version=__version__, lifespan=lifespan,
                  docs_url=None if settings.is_production else "/docs", redoc_url=None,
                  openapi_url=None if settings.is_production else "/openapi.json")
    app.add_middleware(SecurityHeaders)
    app.add_middleware(RateLimit, per_minute=settings.rate_limit_per_min)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["x-api-key", "content-type", "authorization"])

    def rt_of(request: Request) -> Runtime:
        return request.app.state.rt

    def auth(request: Request) -> None:
        # DEMO_MODE: strangers may READ (names masked); anything else, or any request carrying a key, is checked normally.
        if (settings.demo_mode and settings.api_key and is_public_read(request)
                and not request.headers.get("x-api-key") and not request.headers.get("authorization")):
            request.state.public = True
            return
        request.state.public = False
        check_key(request, settings.api_key)

    def _mask(c: dict) -> dict:
        c["name"] = mask_name(c.get("name", ""))
        if "contact" in c:
            c["contact"] = {"name": c["name"]}
        return c

    api = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    # ---------------------------------------------------------------- jobs
    @api.post("/jobs")
    async def create_job(body: JobIn, request: Request):
        rt = rt_of(request)
        spec, trace = await rt.panel.analyze_job(body.title.strip(), body.description)
        job = rt.store.create_job(body.title.strip(), body.description, spec.model_dump())
        rt.store.audit("recruiter", "job_created", None, job["id"])
        return {**job, "trace": trace}

    @api.get("/jobs")
    def list_jobs(request: Request):
        return rt_of(request).store.list_jobs()

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        job = rt_of(request).store.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @api.put("/jobs/{job_id}/spec")
    def update_spec(job_id: str, body: SpecIn, request: Request):
        rt = rt_of(request)
        job = rt.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        spec = rt.build_spec(job, body)
        rt.store.audit("recruiter", "spec_edited", None, job_id)
        return rt.store.update_job_spec(job_id, spec.model_dump())

    @api.post("/jobs/{job_id}/rescore")
    def rescore(job_id: str, request: Request):
        rt = rt_of(request)
        if not rt.store.get_job(job_id):
            raise HTTPException(404, "Job not found")
        ids = [c["id"] for c in rt.store.list_candidates(job_id) if c["status"] in ("done", "failed")]
        for cid in ids:
            rt.store.update_candidate(cid, status="queued", error="", human_action=None)
            rt.enqueue(cid)
        return {"requeued": len(ids)}

    # ---------------------------------------------------------------- candidates
    @api.post("/jobs/{job_id}/candidates")
    async def upload(job_id: str, request: Request, files: list[UploadFile] = File(...)):
        rt = rt_of(request)
        if not rt.store.get_job(job_id):
            raise HTTPException(404, "Job not found")
        if len(files) > MAX_FILES:
            raise HTTPException(413, f"Upload at most {MAX_FILES} files at a time")
        accepted, rejected = [], []
        for f in files:
            name = (f.filename or "resume")[:200]
            data = await f.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                rejected.append({"filename": name, "error": f"larger than {settings.max_upload_mb:g} MB"})
                continue
            try:
                text = await asyncio.to_thread(extract_text, name, data)
            except ExtractError as e:
                rejected.append({"filename": name, "error": str(e)})
                continue
            c = rt.store.create_candidate(job_id, name, text)
            rt.store.audit("recruiter", "uploaded", c["id"], name)
            rt.enqueue(c["id"])
            accepted.append(c)
        return {"accepted": accepted, "rejected": rejected}

    @api.post("/jobs/{job_id}/candidates/text")
    def add_text(job_id: str, body: TextIn, request: Request):
        rt = rt_of(request)
        if not rt.store.get_job(job_id):
            raise HTTPException(404, "Job not found")
        c = rt.store.create_candidate(job_id, "pasted-resume.txt", body.text.strip(), body.name.strip())
        rt.store.audit("recruiter", "uploaded", c["id"], "pasted text")
        rt.enqueue(c["id"])
        return c

    @api.get("/jobs/{job_id}/candidates")
    def list_candidates(job_id: str, request: Request):
        rows = rt_of(request).store.list_candidates(job_id)
        return [_mask(dict(c)) for c in rows] if getattr(request.state, "public", False) else rows

    @api.get("/jobs/{job_id}/export.csv")
    def export_csv(job_id: str, request: Request):
        rows = rt_of(request).store.list_candidates(job_id)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "name", "file", "status", "score", "confidence", "ai_decision", "human_action"])
        for c in rows:
            w.writerow([_csv_safe(x) for x in (c["id"], c["name"], c["filename"], c["status"], c["score"],
                                               c["confidence"], c["decision"], c["human_action"])])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="candidates-{job_id}.csv"'})

    @api.get("/candidates/{cid}")
    def get_candidate(cid: str, request: Request):
        rt = rt_of(request)
        c = rt.store.get_candidate(cid)
        if not c:
            raise HTTPException(404, "Candidate not found")
        c.pop("raw_text", None)
        c.pop("name_hint", None)
        c["audit"] = rt.store.audit_for(cid)
        return _mask(c) if getattr(request.state, "public", False) else c

    @api.post("/candidates/{cid}/review")
    async def review(cid: str, body: ReviewIn, request: Request):
        return await rt_of(request).review(cid, body, "recruiter")

    @api.post("/candidates/{cid}/retry")
    def retry(cid: str, request: Request):
        rt = rt_of(request)
        c = rt.store.get_candidate(cid, full=False)
        if not c:
            raise HTTPException(404, "Candidate not found")
        rt.store.update_candidate(cid, status="queued", error="")
        rt.enqueue(cid)
        return {"ok": True}

    @api.delete("/candidates/{cid}")
    def delete_candidate(cid: str, request: Request):
        rt = rt_of(request)
        if not rt.store.delete_candidate(cid):
            raise HTTPException(404, "Candidate not found")
        rt.store.audit("recruiter", "deleted", cid, "candidate data erased on request")
        return {"ok": True}

    # ---------------------------------------------------------------- insight
    @api.get("/status")
    def status(request: Request):
        rt = rt_of(request)
        return {
            "version": __version__, "env": settings.env,
            "providers": [{"name": n, "model": getattr(p, "model", ""),
                           "calls_left_today": rt.llm.limiters[n].remaining_today()}
                          for n, p in rt.llm.providers.items()],
            "offline_mode": not rt.llm.available,
            "blind_mode": settings.blind_mode,
            "threshold": settings.shortlist_threshold, "review_band": settings.review_band,
            "notifications": {"n8n": bool(settings.n8n_webhook_url), "discord": bool(settings.discord_webhook_url)},
            "queue": len(rt.tasks),
        }

    @api.get("/metrics")
    def metrics(request: Request):
        rows = rt_of(request).store.decided_rows()
        n = len(rows)
        by: dict[str, int] = {}
        for r in rows:
            by[r["decision"] or "none"] = by.get(r["decision"] or "none", 0) + 1
        reviewed = [r for r in rows if r["human_action"] and r["human_action"] != "hold"]
        agree = sum(1 for r in reviewed if (r["human_action"] == "approve_shortlist") == (r["decision"] == "shortlist")
                    and not (r["decision"] == "human_review"))
        scored = [r["score"] for r in rows if r["score"] is not None]
        buckets = [0] * 10
        for s in scored:
            buckets[min(9, int(s // 10))] += 1
        return {"scored": n, "by_decision": by, "avg_score": round(sum(scored) / len(scored), 1) if scored else None,
                "reviewed": len(reviewed), "ai_human_agreement": round(agree / len(reviewed), 2) if reviewed else None,
                "score_histogram": buckets}

    app.include_router(api)

    # ---------------------------------------------------------------- unauthenticated
    @app.get("/healthz")
    def healthz():
        return {"ok": True, "version": __version__}

    @app.get("/readyz")
    def readyz(request: Request):
        try:
            rt_of(request).store.ping()
        except Exception:  # noqa: BLE001
            return JSONResponse({"ok": False, "detail": "database unreachable"}, status_code=503)
        return {"ok": True}

    @app.get("/api/public")
    def public():
        return {"auth_required": bool(settings.api_key), "demo": bool(settings.demo_mode and settings.api_key),
                "version": __version__, "name": "HirePanel"}

    @app.post("/api/webhooks/n8n")
    async def n8n_inbound(body: WebhookReview, request: Request):
        got = request.headers.get("x-webhook-secret", "")
        if not settings.webhook_secret or not hmac.compare_digest(got.encode(), settings.webhook_secret.encode()):
            raise HTTPException(401, "Bad webhook secret")
        return await rt_of(request).review(body.candidate_id, ReviewIn(action=body.action, note=body.note), "n8n")

    @app.get("/", include_in_schema=False)
    def index():
        if not UI_FILE.exists():
            return JSONResponse({"name": "HirePanel", "docs": "/docs"})
        return FileResponse(UI_FILE, media_type="text/html")

    return app
