"""Auth, rate limiting and security headers."""
from __future__ import annotations

import hmac
import re
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")


# Pages a stranger may READ when DEMO_MODE is on. No exports, no writes, no webhooks.
_PUBLIC_READ = re.compile(r"^/api/(jobs|jobs/[^/]+|jobs/[^/]+/candidates|candidates/[^/]+|status|metrics)$")


def is_public_read(request: Request) -> bool:
    return request.method == "GET" and bool(_PUBLIC_READ.match(request.url.path))


def mask_name(name: str) -> str:
    """'Maria Gonzalez' -> 'M. G.' so a public demo never shows a full name."""
    parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
    return " ".join(p[0].upper() + "." for p in parts[:3]) or "Candidate"


def check_key(request: Request, expected: str) -> None:
    if not expected:
        return  # development without a key
    supplied = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong API key")


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        if request.url.path.startswith("/api"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp


class RateLimit(BaseHTTPMiddleware):
    """Sliding-window limit per client IP. Good enough for one instance; use a gateway for more."""

    def __init__(self, app, per_minute: int):
        super().__init__(app)
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request, call_next):
        if self.per_minute <= 0 or not request.url.path.startswith("/api"):
            return await call_next(request)
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "?")
        now = time.monotonic()
        q = self.hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return JSONResponse({"detail": "Too many requests. Try again in a minute."}, status_code=429,
                                headers={"Retry-After": "30"})
        q.append(now)
        if len(self.hits) > 5000:  # bound memory
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                self.hits.pop(k, None)
        return await call_next(request)
