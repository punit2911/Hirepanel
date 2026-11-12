"""Client-side rate limiting so free tiers are never hammered into 429s."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from .errors import DailyQuotaExhausted


class RateLimiter:
    def __init__(self, rpm: int = 0, rpd: int = 0, tpm: int = 0):
        self.rpm, self.rpd, self.tpm = rpm, rpd, tpm
        self._calls: list[tuple[float, int]] = []
        self._day = datetime.now(timezone.utc).date()
        self._day_count = 0
        self._lock = asyncio.Lock()

    def remaining_today(self) -> int | None:
        return None if not self.rpd else max(0, self.rpd - self._day_count)

    async def acquire(self, est_tokens: int = 0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                today = datetime.now(timezone.utc).date()
                if today != self._day:
                    self._day, self._day_count = today, 0
                if self.rpd and self._day_count >= self.rpd:
                    raise DailyQuotaExhausted(f"daily budget of {self.rpd} calls used")
                self._calls = [c for c in self._calls if now - c[0] < 60]
                wait = 0.0
                if self.rpm and len(self._calls) >= self.rpm:
                    wait = max(wait, self._calls[0][0] + 60 - now)
                if self.tpm and self._calls and sum(t for _, t in self._calls) + est_tokens > self.tpm:
                    wait = max(wait, self._calls[0][0] + 60 - now)
                if wait <= 0:
                    self._calls.append((now, est_tokens))
                    self._day_count += 1
                    return
            await asyncio.sleep(min(wait, 5.0) + 0.05)
