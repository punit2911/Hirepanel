"""Thin REST clients for free LLM APIs. No SDKs: just httpx, so the Docker image stays small.

Both clients are defensive about things providers change without warning:
* a model that is retired or not enabled for your key  -> try the next best model
* JSON mode rejected for a particular model            -> retry once without JSON mode
"""
from __future__ import annotations

import re
from typing import Protocol

import httpx

from .errors import PermanentError, RateLimited, TransientError

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"
MAX_MODEL_FALLBACKS = 3


class Provider(Protocol):
    name: str
    model: str

    async def generate(self, system: str, user: str, temperature: float) -> str: ...


def _retry_after(r: httpx.Response) -> float | None:
    v = r.headers.get("retry-after")
    try:
        return float(v) if v else None
    except ValueError:
        return None


async def _request(client: httpx.AsyncClient, method: str, url: str, **kw) -> dict:
    try:
        r = await client.request(method, url, **kw)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise TransientError(f"network error: {type(e).__name__}") from e
    if r.status_code == 429:
        raise RateLimited("429 from provider", _retry_after(r))
    if r.status_code >= 500:
        raise TransientError(f"provider returned {r.status_code}")
    if r.status_code >= 400:
        raise PermanentError(f"{r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise TransientError("provider returned non-JSON body") from e


def model_missing(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in ("404", "not found", "not supported", "does not exist", "decommission",
                                "model_not_found", "no longer available", "is not available"))


def json_mode_unsupported(msg: str) -> bool:
    m = msg.lower()
    return m.startswith("400") and any(k in m for k in ("json", "mime", "response_format", "response format"))


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, client: httpx.AsyncClient):
        self.api_key, self.model, self._client = api_key, model or "auto", client
        self._ranked: list[str] | None = None
        self._json_mode = True

    @property
    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key, "content-type": "application/json"}

    @staticmethod
    def rank_models(names: list[str]) -> list[str]:
        bad = ("image", "tts", "audio", "embedding", "live", "robotics", "computer", "veo", "imagen",
               "native", "learnlm", "aqa", "gemma", "vision", "thinking-exp")
        scored: list[tuple[float, str]] = []
        for n in names:
            low = n.lower()
            if not low.startswith("gemini") or any(b in low for b in bad):
                continue
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", low)
            score = (float(m.group(1)) if m else 0.0) * 10
            if "flash" in low and "lite" not in low:
                score += 30
            elif "flash-lite" in low:
                score += 20
            elif "pro" in low:
                score += 5
            if not any(t in low for t in ("preview", "exp")):
                score += 8
            scored.append((score, n))
        return [n for _, n in sorted(scored, key=lambda x: -x[0])]

    @classmethod
    def pick_model(cls, names: list[str]) -> str | None:
        ranked = cls.rank_models(names)
        return ranked[0] if ranked else None

    async def models(self) -> list[str]:
        if self.model != "auto":
            return [self.model]
        if self._ranked:
            return self._ranked
        ranked: list[str] = []
        try:
            data = await _request(self._client, "GET", f"{GEMINI_BASE}/models",
                                  headers=self._headers, params={"pageSize": 200})
            names = [m["name"].split("/", 1)[-1] for m in data.get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            ranked = self.rank_models(names)
        except (TransientError, RateLimited):
            pass
        self._ranked = ranked or ["gemini-2.5-flash"]
        return self._ranked

    async def resolve_model(self) -> str:
        return (await self.models())[0]

    async def _call(self, model: str, system: str, user: str, temperature: float) -> str:
        cfg: dict = {"temperature": temperature, "maxOutputTokens": 8192}
        if self._json_mode:
            cfg["responseMimeType"] = "application/json"
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": cfg}
        data = await _request(self._client, "POST", f"{GEMINI_BASE}/models/{model}:generateContent",
                              headers=self._headers, json=payload)
        cands = data.get("candidates") or []
        if not cands:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            raise TransientError(f"gemini returned no candidates (blockReason={block})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise TransientError(f"gemini returned empty text (finishReason={cands[0].get('finishReason')})")
        return text

    async def generate(self, system: str, user: str, temperature: float) -> str:
        models = (await self.models())[:MAX_MODEL_FALLBACKS]
        last: PermanentError | None = None
        for i, model in enumerate(models):
            try:
                try:
                    text = await self._call(model, system, user, temperature)
                except PermanentError as e:
                    if self._json_mode and json_mode_unsupported(str(e)):
                        self._json_mode = False
                        text = await self._call(model, system, user, temperature)
                    else:
                        raise
                if self.model == "auto" and i:  # remember the model that worked
                    self._ranked = models[i:] + [m for m in (self._ranked or []) if m not in models[i:]]
                return text
            except PermanentError as e:
                last = e
                if self.model == "auto" and model_missing(str(e)) and i + 1 < len(models):
                    continue
                raise
        raise last or PermanentError("no gemini model available for this key")


class OpenAICompatProvider:
    """Groq, OpenRouter and Ollama all speak the OpenAI chat-completions dialect."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, client: httpx.AsyncClient,
                 json_mode: bool = True, prefer: tuple[str, ...] = ()):
        self.name, self.base_url, self.api_key = name, base_url.rstrip("/"), api_key
        self.model, self._client, self.json_mode, self.prefer = model or "auto", client, json_mode, prefer
        self._ranked: list[str] | None = None

    @property
    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    async def models(self) -> list[str]:
        if self.model != "auto":
            return [self.model]
        if self._ranked:
            return self._ranked
        data = await _request(self._client, "GET", f"{self.base_url}/models", headers=self._headers)
        ids = [m.get("id", "") for m in data.get("data", [])]
        skip = ("whisper", "guard", "tts", "compound", "embed", "orpheus", "distil", "safeguard")
        usable = [i for i in ids if i and not any(s in i.lower() for s in skip)]
        ranked: list[str] = []
        for pref in self.prefer:
            ranked += [i for i in usable if pref in i.lower() and i not in ranked]
        ranked += [i for i in usable if i not in ranked]
        if not ranked:
            raise PermanentError("no usable chat model found for this key")
        self._ranked = ranked
        return ranked

    async def resolve_model(self) -> str:
        return (await self.models())[0]

    async def _call(self, model: str, system: str, user: str, temperature: float) -> str:
        payload: dict = {"model": model, "temperature": temperature,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await _request(self._client, "POST", f"{self.base_url}/chat/completions",
                              headers=self._headers, json=payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise TransientError("unexpected response shape") from e
        if not text.strip():
            raise TransientError("empty completion")
        return text

    async def generate(self, system: str, user: str, temperature: float) -> str:
        models = (await self.models())[:MAX_MODEL_FALLBACKS]
        last: PermanentError | None = None
        for i, model in enumerate(models):
            try:
                try:
                    text = await self._call(model, system, user, temperature)
                except PermanentError as e:
                    if self.json_mode and json_mode_unsupported(str(e)):
                        self.json_mode = False
                        text = await self._call(model, system, user, temperature)
                    else:
                        raise
                if self.model == "auto" and i:
                    self._ranked = models[i:] + [m for m in (self._ranked or []) if m not in models[i:]]
                return text
            except PermanentError as e:
                last = e
                if self.model == "auto" and model_missing(str(e)) and i + 1 < len(models):
                    continue
                raise
        raise last or PermanentError("no model available for this key")
