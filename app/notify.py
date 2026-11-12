"""Outbound notifications. n8n is the main channel; Discord is a zero-setup fallback.

Every n8n call is signed with HMAC-SHA256 so the workflow can prove the call came from this app.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time

import httpx

from .config import Settings

log = logging.getLogger("hirepanel.notify")


def sign(secret: str, body: bytes, ts: str) -> str:
    return hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()


def verify_signature(secret: str, body: bytes, ts: str, signature: str, tolerance_s: int = 300) -> bool:
    try:
        if abs(time.time() - float(ts)) > tolerance_s:
            return False
    except ValueError:
        return False
    return hmac.compare_digest(sign(secret, body, ts), signature or "")


class Notifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.s = settings
        self._client = client or httpx.AsyncClient(timeout=15)
        self._own = client is None

    @property
    def enabled(self) -> bool:
        return bool(self.s.n8n_webhook_url or self.s.discord_webhook_url)

    async def aclose(self) -> None:
        if self._own:
            await self._client.aclose()

    async def send(self, event: str, payload: dict) -> dict:
        """Returns {channel: 'ok' | error text}. Never raises."""
        out: dict[str, str] = {}
        if self.s.n8n_webhook_url:
            out["n8n"] = await self._n8n(event, payload)
        if self.s.discord_webhook_url and event == "candidate_decided":
            out["discord"] = await self._discord(payload)
        return out

    async def _n8n(self, event: str, payload: dict) -> str:
        body = json.dumps({"event": event, **payload}, separators=(",", ":")).encode()
        ts = str(int(time.time()))
        headers = {"content-type": "application/json", "x-hirepanel-timestamp": ts}
        if self.s.webhook_secret:
            headers["x-hirepanel-signature"] = sign(self.s.webhook_secret, body, ts)
        err = "unknown error"
        for attempt in range(3):
            try:
                r = await self._client.post(self.s.n8n_webhook_url, content=body, headers=headers)
                if r.status_code < 300:
                    return "ok"
                err = f"HTTP {r.status_code}"
                if r.status_code < 500 and r.status_code != 429:
                    break
            except httpx.HTTPError as e:
                err = type(e).__name__
            await asyncio.sleep(2 ** attempt)
        log.warning("n8n notification failed: %s", err)
        return err

    async def _discord(self, p: dict) -> str:
        icon = {"shortlist": "🟢", "human_review": "🟡", "reject": "🔴"}.get(p.get("decision", ""), "⚪")
        text = (f"{icon} **{p.get('candidate_name') or 'Candidate'}** for **{p.get('job_title', '')}**: "
                f"{p.get('score')}/100 ({p.get('decision', '').replace('_', ' ')})\n"
                f"{(p.get('summary') or '')[:600]}\n{p.get('review_url', '')}")
        try:
            r = await self._client.post(self.s.discord_webhook_url, json={"content": text[:1900]})
            return "ok" if r.status_code < 300 else f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            return type(e).__name__
