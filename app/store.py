"""SQLite (default) or Postgres storage via SQLAlchemy. All methods are synchronous and short."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import JSON, Float, Integer, String, Text, create_engine, delete, event, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .util import now_iso


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text)
    spec: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))


class Candidate(Base):
    __tablename__ = "candidates"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    filename: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    name_hint: Mapped[str] = mapped_column(String(200), default="")
    contact: Mapped[dict] = mapped_column(JSON, default=dict)
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    human_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    human_note: Mapped[str] = mapped_column(Text, default="")
    reviewed_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    notified: Mapped[dict] = mapped_column(JSON, default=dict)
    run_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))


class Audit(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[str] = mapped_column(String(40))
    actor: Mapped[str] = mapped_column(String(60))
    action: Mapped[str] = mapped_column(String(60))
    candidate_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    detail: Mapped[str] = mapped_column(Text, default="")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _cand_dict(c: Candidate, *, full: bool) -> dict[str, Any]:
    d = {
        "id": c.id, "job_id": c.job_id, "filename": c.filename, "status": c.status,
        "decision": c.decision, "score": c.score, "confidence": c.confidence,
        "name": (c.contact or {}).get("name", ""), "human_action": c.human_action,
        "human_note": c.human_note, "reviewed_at": c.reviewed_at, "error": c.error,
        "created_at": c.created_at, "run_ms": c.run_ms, "notified": c.notified or {},
    }
    if full:
        d.update(contact=c.contact or {}, result=c.result, raw_text=c.raw_text, name_hint=c.name_hint)
    else:
        res = c.result or {}
        d["shield_risk"] = (res.get("shield") or {}).get("risk")
        d["bias_passed"] = (res.get("bias") or {}).get("passed")
    return d


class Store:
    def __init__(self, url: str):
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        is_sqlite = url.startswith("sqlite")
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        if is_sqlite:
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=5000")
                cur.close()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def ping(self) -> bool:
        with self.Session() as s:
            s.execute(select(1))
        return True

    # ---------------------------------------------------------------- jobs
    def create_job(self, title: str, description: str, spec: dict) -> dict:
        job = Job(id=new_id(), title=title, description=description, spec=spec, created_at=now_iso())
        with self.Session.begin() as s:
            s.add(job)
        return self._job_dict(job)

    @staticmethod
    def _job_dict(j: Job) -> dict:
        return {"id": j.id, "title": j.title, "description": j.description, "spec": j.spec,
                "created_at": j.created_at}

    def get_job(self, job_id: str) -> dict | None:
        with self.Session() as s:
            j = s.get(Job, job_id)
            return self._job_dict(j) if j else None

    def list_jobs(self) -> list[dict]:
        with self.Session() as s:
            rows = s.scalars(select(Job).order_by(Job.created_at.desc())).all()
            counts = dict(s.execute(select(Candidate.job_id, func.count()).group_by(Candidate.job_id)).all())
            return [{**self._job_dict(j), "candidates": counts.get(j.id, 0)} for j in rows]

    def update_job_spec(self, job_id: str, spec: dict) -> dict | None:
        with self.Session.begin() as s:
            j = s.get(Job, job_id)
            if not j:
                return None
            j.spec = spec
            return self._job_dict(j)

    # ---------------------------------------------------------------- candidates
    def create_candidate(self, job_id: str, filename: str, raw_text: str, name_hint: str = "") -> dict:
        now = now_iso()
        c = Candidate(id=new_id(), job_id=job_id, filename=filename, raw_text=raw_text,
                      name_hint=name_hint, contact={}, status="queued", created_at=now, updated_at=now)
        with self.Session.begin() as s:
            s.add(c)
        return _cand_dict(c, full=False)

    def get_candidate(self, cid: str, *, full: bool = True) -> dict | None:
        with self.Session() as s:
            c = s.get(Candidate, cid)
            return _cand_dict(c, full=full) if c else None

    def update_candidate(self, cid: str, **fields: Any) -> None:
        with self.Session.begin() as s:
            c = s.get(Candidate, cid)
            if not c:
                return
            for k, v in fields.items():
                setattr(c, k, v)
            c.updated_at = now_iso()

    def list_candidates(self, job_id: str) -> list[dict]:
        with self.Session() as s:
            rows = s.scalars(select(Candidate).where(Candidate.job_id == job_id)).all()
            out = [_cand_dict(c, full=False) for c in rows]
        out.sort(key=lambda d: (d["score"] is None, -(d["score"] or 0)))
        return out

    def delete_candidate(self, cid: str) -> bool:
        with self.Session.begin() as s:
            c = s.get(Candidate, cid)
            if not c:
                return False
            s.delete(c)
            return True

    def pending_ids(self) -> list[str]:
        with self.Session() as s:
            return list(s.scalars(select(Candidate.id).where(Candidate.status.in_(["queued", "processing"]))))

    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self.Session.begin() as s:
            res = s.execute(delete(Candidate).where(Candidate.created_at < cutoff))
            return res.rowcount or 0

    # ---------------------------------------------------------------- audit + metrics
    def audit(self, actor: str, action: str, candidate_id: str | None = None, detail: str = "") -> None:
        with self.Session.begin() as s:
            s.add(Audit(ts=now_iso(), actor=actor, action=action, candidate_id=candidate_id, detail=detail[:2000]))

    def audit_for(self, cid: str) -> list[dict]:
        with self.Session() as s:
            rows = s.scalars(select(Audit).where(Audit.candidate_id == cid).order_by(Audit.id)).all()
            return [{"ts": a.ts, "actor": a.actor, "action": a.action, "detail": a.detail} for a in rows]

    def decided_rows(self) -> list[dict]:
        with self.Session() as s:
            rows = s.scalars(select(Candidate).where(Candidate.status == "done")).all()
            return [{"id": c.id, "job_id": c.job_id, "score": c.score, "decision": c.decision,
                     "human_action": c.human_action, "created_at": c.created_at,
                     "name": (c.contact or {}).get("name", ""), "confidence": c.confidence} for c in rows]
