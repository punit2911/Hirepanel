"""Small shared helpers."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a model reply (tolerates code fences / chatter)."""
    t = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(t[start : end + 1])


def schema_hint(model: type[BaseModel]) -> str:
    """Compact, readable description of a pydantic model's JSON shape for prompts."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def render(node: dict) -> str:
        if "$ref" in node:
            return render(defs[node["$ref"].split("/")[-1]])
        if "anyOf" in node:
            options = [n for n in node["anyOf"] if n.get("type") != "null"]
            return render(options[0]) if options else "null"
        if "enum" in node:
            return " | ".join(json.dumps(e) for e in node["enum"])
        t = node.get("type")
        if t == "object":
            props = node.get("properties", {})
            return "{" + ", ".join(f'"{k}": {render(v)}' for k, v in props.items()) + "}"
        if t == "array":
            return "[" + render(node.get("items", {})) + ", ...]"
        desc = node.get("description")
        if t == "string":
            return f'"<string: {desc}>"' if desc else '"<string>"'
        if t in ("number", "integer"):
            return f"<{t}: {desc}>" if desc else f"<{t}>"
        if t == "boolean":
            return "<true|false>"
        return "<any>"

    return render(schema)


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase alphanumerics only, whitespace collapsed; map[i] = original index of char i."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            idx.append(i)
            prev_space = True
    return "".join(out).rstrip(), idx


def norm(text: str) -> str:
    return normalize_with_map(text)[0]


def truncate_words(s: str, max_chars: int) -> str:
    """Truncate to a prefix on a word boundary (result is always a prefix of s)."""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > max_chars * 0.6 else cut).rstrip()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
