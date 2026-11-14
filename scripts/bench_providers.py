#!/usr/bin/env python3
"""Benchmark REAL models on the labelled resumes.  One report per provider, written to docs/bench/.

    python scripts/bench_providers.py --providers ollama --n 6 --attacks 2
    python scripts/bench_providers.py --providers gemini,groq,ollama

It measures what matters for choosing a model: accuracy vs the hidden labels, how often the model invents
quotes, how often its JSON is valid, and how slow it is. Keep --n small on free tiers (each resume = ~8 calls).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import LLMRouter  # noqa: E402
from evals.benchmark import base_settings, evaluate  # noqa: E402


def explain(err: str) -> str:
    low = err.lower()
    if "connect" in low or "refused" in low:
        return "Cannot reach the model server. For Ollama: open the Ollama app (or run `ollama serve`), then try again."
    if "404" in low or "not found" in low:
        return "The model name was not found. Run `ollama list` and make OLLAMA_MODEL in .env match it exactly."
    if "timeout" in low or "timed out" in low:
        return ("The model took too long. Close other programs, stop the HirePanel app (it may be using the model), "
                "run `ollama run <model> \"hi\"` once to load it, then try again. Or use a smaller model.")
    return "See the error above. `python scripts/check_keys.py` shows the same problem in plain words."


async def preflight(router: LLMRouter, name: str) -> str | None:
    """One tiny real call first. If the model cannot answer this, a benchmark would only measure the fallback rules."""
    try:
        await asyncio.wait_for(
            router.providers[name].generate("Reply with JSON only.", 'Return exactly {"ok": true}', 0.0), 300)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:300] or 'no message'}"


async def one(name: str, n: int, attacks: int) -> dict:
    s = base_settings(True, llm_providers=(name,), run_timeout_s=900, llm_timeout_s=300)
    router = LLMRouter(s)
    if not router.available:
        return {"provider": name, "error": "not configured (missing key or model name in .env)"}
    print("  checking that the model answers (the first call loads it, so it can take a minute)...", flush=True)
    bad = await preflight(router, name)
    if bad:
        await router.aclose()
        return {"provider": name, "error": f"The model did not answer: {bad}\n{explain(bad)}"}
    out = await evaluate({"llm_providers": (name,), "llm_timeout_s": 300}, live=True, n_per_family=n,
                         attacks_per_family=attacks, fairness=False, llm=router)
    rows = out.pop("rows")
    stats = out.pop("provider_stats", {})
    st = stats.get(name, {})
    if not st.get("calls"):
        return {"provider": name, "error": "No model call succeeded during the run, so every score came from the offline "
                "fallback rules. These numbers say nothing about the model. Check the model is running and try again."}
    return {"provider": name, "summary": out, "llm_stats": st, "cases": len(rows)}


async def baseline(n: int, attacks: int) -> dict:
    """Offline rules on the SAME resumes (the generator is deterministic), for an apples-to-apples row."""
    out = await evaluate({}, live=False, n_per_family=n, attacks_per_family=attacks, fairness=False)
    out.pop("rows")
    out.pop("provider_stats", None)
    return out


def md(r: dict) -> str:
    if "error" in r:
        return f"# {r['provider']}\n\n{r['error']}\n"
    s, t, b = r["summary"], r["llm_stats"], r.get("baseline") or {}
    return f"""# Provider benchmark: {r['provider']}

{r['cases']} labelled resumes (synthetic, hidden ground truth). Small samples: treat differences of a few points as noise.
The right-hand column is the built-in offline rules on exactly the same resumes.

| Metric | {r['provider']} (full pipeline) | Offline rules |
|---|---|---|
| AUROC strong vs rest | {s['auroc_strong_vs_rest']} | {b.get('auroc_strong_vs_rest', '-')} |
| Decision acceptable | {s['decision_not_wrong']} | {b.get('decision_not_wrong', '-')} |
| Weak candidates shortlisted | {s['false_shortlist_weak']} | {b.get('false_shortlist_weak', '-')} |
| Attacks shortlisted | {s['attack_false_shortlist']} | {b.get('attack_false_shortlist', '-')} |
| Invented quotes caught by verifier | {s['hallucinated_quotes_total']} | - |
| Valid-JSON rate | {t.get('json_valid_rate')} | - |
| Latency p50 / p95 (ms) | {t.get('latency_p50_ms')} / {t.get('latency_p95_ms')} | - |
| Model calls | {t.get('calls')} | - |
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="gemini,groq,ollama")
    ap.add_argument("--n", type=int, default=4, help="clean resumes per role (3 roles)")
    ap.add_argument("--attacks", type=int, default=2, help="attack resumes per role")
    ap.add_argument("--tag", default="", help="label for the output files, e.g. llama3b -> docs/bench/ollama-llama3b.md")
    a = ap.parse_args()
    Path("docs/bench").mkdir(parents=True, exist_ok=True)
    for name in [p.strip() for p in a.providers.split(",") if p.strip()]:
        print(f"== {name} ...", flush=True)
        r = await one(name, a.n, a.attacks)
        if "error" in r:
            print(f"\nSKIPPED {name}: {r['error']}\nNothing was written to docs/bench for this provider.\n")
            continue
        r["baseline"] = await baseline(a.n, a.attacks)
        label = f"{name}-{a.tag}" if a.tag else name
        if a.tag:
            r["provider"] = label
        Path(f"docs/bench/{label}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        Path(f"docs/bench/{label}.md").write_text(md(r), encoding="utf-8")
        print(md(r))


if __name__ == "__main__":
    asyncio.run(main())