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


async def one(name: str, n: int, attacks: int) -> dict:
    s = base_settings(True, llm_providers=(name,), run_timeout_s=900, llm_timeout_s=300)
    router = LLMRouter(s)
    if not router.available:
        return {"provider": name, "error": "not configured (missing key or model name in .env)"}
    out = await evaluate({"llm_providers": (name,), "llm_timeout_s": 300}, live=True, n_per_family=n,
                         attacks_per_family=attacks, fairness=False, llm=router)
    rows = out.pop("rows")
    stats = out.pop("provider_stats", {})
    return {"provider": name, "summary": out, "llm_stats": stats.get(name, {}), "cases": len(rows)}


def md(r: dict) -> str:
    if "error" in r:
        return f"# {r['provider']}\n\n{r['error']}\n"
    s, t = r["summary"], r["llm_stats"]
    return f"""# Provider benchmark: {r['provider']}

{r['cases']} labelled resumes (synthetic, hidden ground truth). Small samples: treat differences of a few points as noise.

| Metric | Value |
|---|---|
| AUROC strong vs rest | {s['auroc_strong_vs_rest']} |
| Decision acceptable | {s['decision_not_wrong']} |
| Weak candidates shortlisted | {s['false_shortlist_weak']} |
| Attacks shortlisted | {s['attack_false_shortlist']} |
| Invented quotes caught by verifier | {s['hallucinated_quotes_total']} |
| Valid-JSON rate | {t.get('json_valid_rate')} |
| Latency p50 / p95 (ms) | {t.get('latency_p50_ms')} / {t.get('latency_p95_ms')} |
| Model calls | {t.get('calls')} |
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="gemini,groq,ollama")
    ap.add_argument("--n", type=int, default=4, help="clean resumes per role (3 roles)")
    ap.add_argument("--attacks", type=int, default=2, help="attack resumes per role")
    a = ap.parse_args()
    Path("docs/bench").mkdir(parents=True, exist_ok=True)
    for name in [p.strip() for p in a.providers.split(",") if p.strip()]:
        print(f"== {name} ...", flush=True)
        r = await one(name, a.n, a.attacks)
        Path(f"docs/bench/{name}.json").write_text(json.dumps(r, indent=2))
        Path(f"docs/bench/{name}.md").write_text(md(r))
        print(md(r))


if __name__ == "__main__":
    asyncio.run(main())
