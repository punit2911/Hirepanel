#!/usr/bin/env python3
"""Tests the AI keys in your .env with one tiny call each. Run this first if anything looks wrong.

    python scripts/check_keys.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import LLMRouter  # noqa: E402


async def main() -> int:
    s = Settings.from_env()
    router = LLMRouter(s)
    if not router.providers:
        print("No AI keys found in .env. That is fine: HirePanel will use its built-in offline rules.")
        print("To switch on real models, add GEMINI_API_KEY and/or GROQ_API_KEY (free, see README Step 1).")
        await router.aclose()
        return 0
    failed = 0
    for name, p in router.providers.items():
        try:
            await p.generate("Reply with JSON only.", 'Return exactly {"ok": true}', 0.0)
            model = await p.resolve_model()
            print(f"OK    {name}: working, using model '{model}'")
        except Exception as e:  # noqa: BLE001
            failed += 1
            hint = ""
            msg = str(e)
            if "401" in msg or "403" in msg or "API key" in msg:
                hint = "  -> the key looks wrong or not enabled. Re-copy it into .env"
            elif "429" in msg or "rate" in msg.lower():
                hint = "  -> free quota is used up right now. Wait a minute and retry"
            elif "network" in msg.lower():
                hint = "  -> cannot reach the internet from here"
            print(f"FAIL  {name}: {type(e).__name__}: {msg[:200]}{hint}")
    await router.aclose()
    if failed:
        print("\nProviders that fail are skipped automatically. HirePanel falls back to the next one, then to offline rules.")
    return 1 if failed == len(router.providers) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
