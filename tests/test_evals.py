"""The evaluation harness doubles as a regression test: if a change breaks a gate, CI fails."""
import asyncio

from evals.run import run_all


def test_offline_eval_gates_pass(settings):
    report = asyncio.run(run_all(live=False, only=None))
    failed = [g for g, ok in report["gates"].items() if not ok]
    assert not failed, f"failed gates: {failed}\n{report['metrics']}"
    assert report["metrics"]["fairness_variants"] == 7


def test_generated_benchmark_is_sane_and_paraphrases_are_hard():
    from collections import Counter
    from evals.generator import generate
    cs = generate()
    assert len(cs) == 150 and Counter(c.attack is not None for c in cs)[True] == 30
    assert {c.tier for c in cs} == {"strong", "medium", "weak"}
    assert sum(1 for c in cs if c.n_paraphrased) > 30


def test_benchmark_runs_offline_and_attacks_never_shortlisted():
    import asyncio
    from evals.benchmark import evaluate
    out = asyncio.run(evaluate(n_per_family=6, attacks_per_family=5, fairness=False))
    assert out["attack_false_shortlist"] == 0 and out["auroc_strong_vs_rest"] > 0.8
