#!/usr/bin/env python3
"""Creates your .env with generated secrets. Safe to re-run: it never overwrites keys you already set."""
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV, EXAMPLE = ROOT / ".env", ROOT / ".env.example"


def read(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).split("  #")[0].strip()
    return out


def set_key(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{key}=.*$", re.M)
    return pattern.sub(f"{key}={value}", text, count=1) if pattern.search(text) else text + f"\n{key}={value}\n"


def ask(label: str, hint: str) -> str:
    if not sys.stdin.isatty():
        return ""
    return input(f"{label}\n  {hint}\n  Paste it here (or press Enter to skip): ").strip()


def main() -> None:
    text = ENV.read_text() if ENV.exists() else EXAMPLE.read_text()
    current = read(ENV) if ENV.exists() else {}
    if not current.get("API_KEY"):
        key = secrets.token_urlsafe(24)
        text = set_key(text, "API_KEY", key)
        print(f"\nYour app password (API_KEY) is:\n\n    {key}\n\nSave it. You type it into the website the first time you open it.\n")
    if not current.get("WEBHOOK_SECRET"):
        text = set_key(text, "WEBHOOK_SECRET", secrets.token_urlsafe(24))
    if not current.get("GEMINI_API_KEY"):
        v = ask("Gemini key (free, no card):", "Get it at https://aistudio.google.com/apikey")
        if v:
            text = set_key(text, "GEMINI_API_KEY", v)
    if not current.get("GROQ_API_KEY"):
        v = ask("Groq key (free, no card):", "Get it at https://console.groq.com/keys")
        if v:
            text = set_key(text, "GROQ_API_KEY", v)
    ENV.write_text(text)
    print(f"Saved {ENV}")
    final = read(ENV)
    if not (final.get("GEMINI_API_KEY") or final.get("GROQ_API_KEY")):
        print("No AI key set: HirePanel will run in offline mode (built-in rules). Add a key to .env any time.")


if __name__ == "__main__":
    main()
