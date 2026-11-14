"""Large labelled benchmark.  `python -m evals.benchmark [--live] [--n 40]`
Runs every generated resume through the real pipeline and reports headline metrics.
Library use: `await evaluate(settings_overrides)` returns a dict (used by the ablation script)."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

from app.config import Settings
from app.llm import LLMRouter
from app.pipeline import Panel

from .generator import FAMILIES, fairness_groups, generate
from .metrics import auroc, rate, wilson

TODAY = date(2026, 11, 11)
OK = {"strong": {"shortlist", "human_review"}, "medium": {"human_review"}, "weak": {"reject", "human_review"}}


def base_settings(live: bool, **over) -> Settings:
    s = Settings.from_env()
    if not live:
        over.setdefault("llm_providers", ())
    over.setdefault("llm_cache", False)
    return dataclasses.replace(s, **over)


async def evaluate(over: dict | None = None, *, live: bool = False, n_per_family: int = 40,
                   attacks_per_family: int = 10, fairness: bool = True, llm: LLMRouter | None = None) -> dict:
    s = base_settings(live, **(over or {}))
    router = llm or LLMRouter(s)
    panel = Panel(s, router)
    specs = {}
    for fam, f in FAMILIES.items():
        specs[fam], _ = await panel.analyze_job(f["title"], f["jd"])
    rows = []
    for c in generate(n_per_family=n_per_family, attacks_per_family=attacks_per_family):
        r = await panel.run(spec=specs[c.family], jd_text=FAMILIES[c.family]["jd"], text=c.text, today=TODAY)
        rows.append({"id": c.id, "family": c.family, "tier": c.tier, "attack": c.attack, "paraphrase": c.n_paraphrased > 0,
                     "score": r.score, "conf": r.confidence, "action": r.decision.action,
                     "hallu": r.hallucinated_quotes, "flags": sorted({f.type for f in r.shield.flags})})
    out = summarize(rows)
    if fairness:
        out["fairness"] = await _fairness(panel, specs)
    out["rows"] = rows
    out["provider_stats"] = router.stats_summary()
    return out


def summarize(rows: list[dict]) -> dict:
    clean = [r for r in rows if not r["attack"]]
    atk = [r for r in rows if r["attack"]]
    strong = [r["score"] for r in clean if r["tier"] == "strong"]
    rest = [r["score"] for r in clean if r["tier"] != "strong"]
    weak = [r["score"] for r in clean if r["tier"] == "weak"]
    mid = [r["score"] for r in clean if r["tier"] == "medium"]
    ok = sum(r["action"] in OK[r["tier"]] for r in clean)
    auto = [r for r in clean if r["action"] != "human_review"]
    auto_ok = sum((r["action"] == "shortlist") == (r["tier"] == "strong") for r in auto)
    unq = [r for r in clean if r["tier"] == "weak"]
    para = [r for r in clean if r["paraphrase"] and r["tier"] == "strong"]
    shortlisted_attacks = sum(r["action"] == "shortlist" for r in atk)
    return {
        "n_clean": len(clean), "n_attacks": len(atk),
        "auroc_strong_vs_rest": auroc(strong, rest),
        "auroc_medium_vs_weak": auroc(mid, weak),
        "decision_not_wrong": rate(ok, len(clean)), "decision_not_wrong_ci": wilson(ok, len(clean)),
        "false_shortlist_weak": rate(sum(r["action"] == "shortlist" for r in unq), len(unq)),
        "strong_recall_shortlist_or_review": rate(sum(r["action"] != "reject" for r in clean if r["tier"] == "strong"),
                                                  sum(1 for r in clean if r["tier"] == "strong")),
        "strong_shortlisted": rate(sum(r["action"] == "shortlist" for r in clean if r["tier"] == "strong"),
                                   sum(1 for r in clean if r["tier"] == "strong")),
        "paraphrase_strong_rejected": rate(sum(r["action"] == "reject" for r in para), len(para)),
        "auto_decision_coverage": rate(len(auto), len(clean)),
        "auto_decision_precision": rate(auto_ok, len(auto)),
        "attack_false_shortlist": rate(shortlisted_attacks, len(atk)),
        "attack_false_shortlist_count": shortlisted_attacks,
        "attack_flagged": rate(sum(bool(r["flags"]) for r in atk), len(atk)),
        "weak_mean_score": round(sum(weak) / len(weak), 1) if weak else None,
        "strong_mean_score": round(sum(strong) / len(strong), 1) if strong else None,
        "hallucinated_quotes_total": sum(r["hallu"] for r in rows),
    }


async def _fairness(panel: Panel, specs: dict) -> dict:
    spreads, flips = [], 0
    groups = fairness_groups()
    for g in groups:
        scores, acts = [], set()
        for t in g["texts"]:
            r = await panel.run(spec=specs[g["family"]], jd_text=FAMILIES[g["family"]]["jd"], text=t, today=TODAY,
                                name_hint="", email_hint="")
            scores.append(r.score)
            acts.add(r.decision.action)
        spreads.append(max(scores) - min(scores))
        flips += int(len(acts) > 1)
    return {"groups": len(groups), "max_score_spread": round(max(spreads), 2),
            "mean_score_spread": round(sum(spreads) / len(spreads), 2), "decision_flips": flips}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    out = asyncio.run(evaluate(live=a.live, n_per_family=a.n))
    rows = out.pop("rows")
    out.pop("provider_stats", None)
    Path("evals").mkdir(exist_ok=True)
    Path("evals/benchmark.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
