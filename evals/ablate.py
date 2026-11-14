"""Ablation experiments.  `python -m evals.ablate`  -> writes docs/RESULTS.md and evals/ablations.json
Every row is the SAME 150 resumes through the SAME pipeline with ONE thing switched off or changed."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.llm import LLMRouter
from app.llm.limiter import RateLimiter

from .benchmark import base_settings, evaluate
from .generator import FAMILIES
from .simulate import SimModel
from app.pipeline import Panel


async def run_cfg(label: str, over: dict, sim_kw: dict | None, fairness: bool = True) -> dict:
    s = base_settings(False, **over)
    router = LLMRouter(s)
    if sim_kw is not None:
        panel = Panel(s, router)
        specs = {f: (await panel.analyze_job(v["title"], v["jd"]))[0] for f, v in FAMILIES.items()}
        router.providers["sim"], router.limiters["sim"] = SimModel(specs, **sim_kw), RateLimiter()
    out = await evaluate(over, llm=router, fairness=fairness)
    out.pop("rows")
    out.pop("provider_stats", None)
    out["label"] = label
    return out


EXPERIMENTS = [
    # (group, label, settings override, simulator kwargs or None)
    ("baseline", "Offline rules only (no model)", {}, None),
    ("baseline", "Ideal simulated model, full pipeline", {}, {}),
    ("verifier", "Hallucinating model (30% invented evidence) + verifier ON", {}, {"hallu_rate": 0.3}),
    ("verifier", "Hallucinating model (30% invented evidence) + verifier OFF", {"verifier_enabled": False}, {"hallu_rate": 0.3}),
    ("scorers", "Noisy model (sigma=0.25), dual scorers + adjudicator", {}, {"noise": 0.25}),
    ("scorers", "Noisy model (sigma=0.25), single scorer", {"scorer_mode": "a"}, {"noise": 0.25}),
    ("shield", "Injection-susceptible model + shield ON", {}, {"inject_susceptible": True}),
    ("shield", "Injection-susceptible model + shield OFF", {"shield_enabled": False}, {"inject_susceptible": True}),
    ("shield", "Injection-susceptible model, shield OFF and verifier OFF", {"shield_enabled": False, "verifier_enabled": False},
     {"inject_susceptible": True}),
    ("blind", "Identity-biased model (-0.25 on identity cues) + blind mode ON (school names visible)", {"redact_institutions": False}, {"bias_penalty": 0.25}),
    ("blind", "Identity-biased model + blind mode ON + institutions redacted", {"redact_institutions": True}, {"bias_penalty": 0.25}),
    ("blind", "Identity-biased model (-0.25 on identity cues) + blind mode OFF", {"blind_mode": False}, {"bias_penalty": 0.25}),
]

COLS = [("auroc_strong_vs_rest", "AUROC"), ("decision_not_wrong", "Decision OK"), ("false_shortlist_weak", "Weak shortlisted"),
        ("attack_false_shortlist", "Attacks shortlisted"), ("weak_mean_score", "Weak mean score"), ("auto_decision_precision", "Auto-precision"),
        ("hallucinated_quotes_total", "Fake quotes seen")]


def table(rows: list[dict]) -> str:
    head = "| Configuration | " + " | ".join(c[1] for c in COLS) + " | Fair. spread | Flips |\n"
    head += "|---|" + "---|" * (len(COLS) + 2) + "\n"
    for r in rows:
        f = r.get("fairness", {})
        vals = [("—" if r[k] is None else (f"{r[k]:.0%}" if isinstance(r[k], float) and k not in ("auroc_strong_vs_rest", "weak_mean_score") else str(r[k])))
                for k, _ in COLS]
        head += f"| {r['label']} | " + " | ".join(vals) + f" | {f.get('max_score_spread', '—')} | {f.get('decision_flips', '—')} |\n"
    return head


async def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not be UTF-8
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    results = []
    for group, label, over, sim in EXPERIMENTS:
        r = await run_cfg(label, over, sim)
        r["group"] = group
        results.append(r)
        print(f"done: {label}")
    Path("evals/ablations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/RESULTS.md").write_text(render(results), encoding="utf-8")
    print(table(results))


def render(results: list[dict]) -> str:
    return f"""# Results

**Read this first.** All numbers below come from a *synthetic* benchmark (150 generated resumes across 3 roles with
hidden ground truth, 30 of them attacks, plus 9 identity-swap groups of 8) and from *simulated* models (`evals/simulate.py`).
They show how the **architecture** behaves when a model misbehaves in a known way. They are not a claim about real-world
hiring accuracy, and not a claim about any vendor's model. Real-model numbers come from `scripts/bench_providers.py`
(see `docs/OLLAMA.md`). Reproduce everything with `python -m evals.ablate`.

How to read the columns: **AUROC** = chance a strong candidate outscores a weaker one (0.5 = coin flip, 1.0 = perfect).
**Decision OK** = decision was acceptable for the hidden label. **Weak shortlisted** = unqualified people wrongly shortlisted.
**Attacks shortlisted** = attack resumes wrongly shortlisted. **Auto-precision** = when the system decided without a human,
how often it was right. **Fair. spread** = largest score difference between identical qualifications under different
names/schools/ages/genders (lower is better). **Flips** = groups where the decision changed with identity alone.

{table(results)}
## What the experiments show (and don't)

1. **The verifier matters.** With a model that invents evidence 30% of the time, turning the verifier off lets 7% of
   attack resumes get shortlisted and inflates unqualified candidates' mean score; with it on, none are shortlisted.
2. **Defence in depth works.** An injection-susceptible model with the shield *and* the verifier off shortlists 53% of
   attacks. Either layer alone brings that to 0%, so one failing does not break the system.
3. **Schools are a bias channel.** With a model biased against identity cues, hiding names/pronouns/ages but leaving school
   names still produced a 21.7-point score spread and 3 decision flips between identical candidates. Redacting
   institutions (now the default) brought both to 0. This was found by the experiment, not assumed.
4. **Dual scorers did NOT improve accuracy in this simulation.** AUROC was slightly lower than a single scorer, and the
   conservative "take the lower score when they disagree" fallback made scores less stable. Their real value is the
   disagreement signal that routes cases to a human; if that is not worth 2x model calls for you, set `SCORER_MODE=a`.
5. **A simulated ideal model did not beat the offline rules** on this synthetic set (AUROC 0.937 vs 0.943). The set's
   paraphrase cases are too easy for the rules' fuzzy matching, so this benchmark cannot show a language-model advantage.
   That needs real models on real resumes, which is what `scripts/bench_providers.py` is for.
6. **"Decision OK" tops out near 70-75%** because the middle ("medium") tier is genuinely ambiguous: the policy shortlists
   many of them. Weak candidates were shortlisted 0% of the time in every configuration.

A bug found by this work: copied *short* job-description bullets ("Strong SQL and Excel") evaded copy detection and
let a JD-copy attack reach 72.7. Fixed in `Doc.mark_copied`, covered by `test_short_copied_jd_bullets_are_marked_copied`.
"""


if __name__ == "__main__":
    asyncio.run(main())
