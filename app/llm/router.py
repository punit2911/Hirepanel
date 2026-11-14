"""LLMRouter: provider fallback, retries with backoff, circuit breaker, schema validation, cache."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..util import extract_json, schema_hint, sha256
from .errors import (DailyQuotaExhausted, LLMUnavailable, PermanentError, ProviderError, RateLimited,
                     TransientError)
from .limiter import RateLimiter
from .providers import GROQ_BASE, GeminiProvider, OpenAICompatProvider, Provider

log = logging.getLogger("hirepanel.llm")
T = TypeVar("T", bound=BaseModel)


@dataclass
class CallInfo:
    provider: str
    model: str
    attempts: int = 1
    cached: bool = False
    latency_ms: int = 0


@dataclass
class ProviderStats:
    """What we measured about one provider during this process's lifetime."""
    calls: int = 0           # HTTP generations that returned text
    ok: int = 0              # answers that parsed and validated
    invalid_json: int = 0    # answers that did not match the required shape
    errors: int = 0          # rate limits, timeouts, 5xx, bad keys
    failed: int = 0          # whole requests that gave up on this provider
    chars_in: int = 0
    chars_out: int = 0
    latencies_ms: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)

        def pct(p: float) -> int | None:
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
        return {"calls": self.calls, "ok": self.ok, "invalid_json": self.invalid_json, "errors": self.errors,
                "failed": self.failed, "json_valid_rate": round(self.ok / self.calls, 3) if self.calls else None,
                "latency_p50_ms": pct(0.5), "latency_p95_ms": pct(0.95),
                "approx_tokens_in": self.chars_in // 4, "approx_tokens_out": self.chars_out // 4}


class LLMRouter:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.llm_timeout_s)
        self._own_client = client is None
        self.providers: dict[str, Provider] = {}
        self.limiters: dict[str, RateLimiter] = {}
        self._open_until: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        self.stats: dict[str, ProviderStats] = {}
        self._cache_dir: Path | None = None
        if settings.llm_cache:
            self._cache_dir = settings.data_dir / "llm_cache"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._build()

    # ---------------------------------------------------------------- setup
    def _build(self) -> None:
        s, c = self.settings, self._client
        for name in s.llm_providers:
            if name == "gemini" and s.gemini_api_key:
                self.providers[name] = GeminiProvider(s.gemini_api_key, s.gemini_model, c)
                self.limiters[name] = RateLimiter(rpm=s.gemini_rpm, rpd=s.gemini_rpd)
            elif name == "groq" and s.groq_api_key:
                self.providers[name] = OpenAICompatProvider(
                    "groq", GROQ_BASE, s.groq_api_key, s.groq_model, c, json_mode=True,
                    prefer=("gpt-oss-120b", "llama-3.3-70b", "gpt-oss-20b", "llama-3.1-8b", "qwen"))
                self.limiters[name] = RateLimiter(rpm=s.groq_rpm, rpd=s.groq_rpd, tpm=s.groq_tpm)
            elif name == "openrouter" and s.openrouter_api_key and s.openrouter_model:
                self.providers[name] = OpenAICompatProvider(
                    "openrouter", "https://openrouter.ai/api/v1", s.openrouter_api_key,
                    s.openrouter_model, c, json_mode=False)
                self.limiters[name] = RateLimiter(rpm=s.openrouter_rpm, rpd=s.openrouter_rpd)
            elif name == "ollama" and s.ollama_model:
                self.providers[name] = OpenAICompatProvider(
                    "ollama", s.ollama_base_url, "", s.ollama_model, c, json_mode=False)
                self.limiters[name] = RateLimiter()

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def names(self) -> list[str]:
        return list(self.providers)

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    # ---------------------------------------------------------------- helpers
    def _order(self, prefer: str | None) -> list[str]:
        names = list(self.providers)
        if prefer and prefer in names:
            names.remove(prefer)
            names.insert(0, prefer)
        return names

    def _is_open(self, name: str) -> bool:
        return time.monotonic() < self._open_until.get(name, 0.0)

    def _trip(self, name: str, seconds: float) -> None:
        self._open_until[name] = time.monotonic() + seconds
        log.warning("provider %s paused for %.0fs", name, seconds)

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(20.0, (2 ** attempt) + random.random())

    def _cache_path(self, key: str) -> Path | None:
        return self._cache_dir / f"{key}.json" if self._cache_dir else None

    # ---------------------------------------------------------------- main entry
    async def complete_json(self, *, system: str, user: str, model_cls: type[T], temperature: float = 0.1,
                            prefer: str | None = None, task: str = "") -> tuple[T, CallInfo]:
        if not self.providers:
            raise LLMUnavailable("no LLM provider configured")
        full_system = (
            system.strip()
            + "\n\nReply with ONE valid JSON object and nothing else (no markdown fences). "
            + "Shape:\n" + schema_hint(model_cls)
        )
        key = sha256(json.dumps([full_system, user, temperature, model_cls.__name__]))
        cp = self._cache_path(key)
        if cp and cp.exists():
            try:
                return model_cls.model_validate_json(cp.read_text(encoding="utf-8")), CallInfo("cache", "cache", 0, True, 0)
            except (ValidationError, ValueError, OSError):
                pass

        last: Exception | None = None
        for name in self._order(prefer):
            if self._is_open(name):
                continue
            try:
                obj, info = await self._try_provider(name, full_system, user, model_cls, temperature, task)
                if cp:
                    try:
                        cp.write_text(obj.model_dump_json(), encoding="utf-8")
                    except OSError:
                        pass
                return obj, info
            except LLMUnavailable as e:
                last = e
        raise LLMUnavailable(f"all providers failed ({last})" if last else "all providers are paused")

    async def _try_provider(self, name: str, system: str, user: str, model_cls: type[T],
                            temperature: float, task: str) -> tuple[T, CallInfo]:
        provider, limiter = self.providers[name], self.limiters[name]
        st = self.stats.setdefault(name, ProviderStats())
        user_cur, bad_outputs, t0 = user, 0, time.perf_counter()
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                await limiter.acquire(est_tokens=(len(system) + len(user_cur)) // 4 + 800)
                g0 = time.perf_counter()
                text = await provider.generate(system, user_cur, temperature)
            except DailyQuotaExhausted as e:
                st.failed += 1
                self._trip(name, 900)
                raise LLMUnavailable(f"{name}: {e}") from e
            except RateLimited as e:
                st.errors += 1
                if e.retry_after and e.retry_after > 30:
                    st.failed += 1
                    self._trip(name, e.retry_after)
                    raise LLMUnavailable(f"{name}: quota reset in {e.retry_after:.0f}s") from e
                await asyncio.sleep(e.retry_after or self._backoff(attempt))
                continue
            except TransientError as e:
                st.errors += 1
                log.info("%s transient error on %s (attempt %d): %s", task, name, attempt, e)
                await asyncio.sleep(self._backoff(attempt))
                continue
            except PermanentError as e:
                st.errors += 1
                st.failed += 1
                self._trip(name, 300)
                raise LLMUnavailable(f"{name}: {e}") from e
            except ProviderError as e:  # pragma: no cover
                st.failed += 1
                raise LLMUnavailable(f"{name}: {e}") from e
            st.calls += 1
            st.latencies_ms.append(int((time.perf_counter() - g0) * 1000))
            st.chars_in += len(system) + len(user_cur)
            st.chars_out += len(text)
            try:
                obj = model_cls.model_validate(extract_json(text))
            except (ValueError, ValidationError) as e:
                st.invalid_json += 1
                bad_outputs += 1
                if bad_outputs >= 2:
                    break
                user_cur = (user + "\n\nYour previous reply was rejected: " + str(e)[:300]
                            + "\nReply again with ONLY one valid JSON object in the required shape.")
                continue
            st.ok += 1
            self._failures[name] = 0
            used = (getattr(provider, "_ranked", None) or [getattr(provider, "model", "")])[0]
            return obj, CallInfo(name, used, attempt, False, int((time.perf_counter() - t0) * 1000))
        st.failed += 1
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= 3:
            self._trip(name, 60)
        raise LLMUnavailable(f"{name} exhausted retries")

    def stats_summary(self) -> dict[str, dict]:
        return {n: s.summary() for n, s in self.stats.items()}
