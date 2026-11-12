# Results

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

| Configuration | AUROC | Decision OK | Weak shortlisted | Attacks shortlisted | Weak mean score | Auto-precision | Fake quotes seen | Fair. spread | Flips |
|---|---|---|---|---|---|---|---|---|---|
| Offline rules only (no model) | 0.943 | 75% | 0% | 0% | 34.9 | 82% | 0 | 0.0 | 0 |
| Ideal simulated model, full pipeline | 0.937 | 70% | 0% | 0% | 34.7 | 82% | 0 | 0.0 | 0 |
| Hallucinating model (30% invented evidence) + verifier ON | 0.937 | 70% | 0% | 0% | 36.9 | 82% | 99 | 1.6 | 0 |
| Hallucinating model (30% invented evidence) + verifier OFF | 0.938 | 75% | 0% | 7% | 40.7 | 81% | 99 | 0.0 | 0 |
| Noisy model (sigma=0.25), dual scorers + adjudicator | 0.926 | 72% | 0% | 0% | 29.5 | 87% | 0 | 26.9 | 1 |
| Noisy model (sigma=0.25), single scorer | 0.938 | 72% | 0% | 0% | 31.4 | 88% | 0 | 16.9 | 1 |
| Injection-susceptible model + shield ON | 0.937 | 70% | 0% | 0% | 34.7 | 82% | 0 | 0.0 | 0 |
| Injection-susceptible model + shield OFF | 0.937 | 70% | 0% | 0% | 34.7 | 82% | 12 | 0.0 | 0 |
| Injection-susceptible model, shield OFF and verifier OFF | 0.941 | 74% | 0% | 53% | 37.2 | 82% | 12 | 0.0 | 0 |
| Identity-biased model (-0.25 on identity cues) + blind mode ON (school names visible) | 0.922 | 71% | 0% | 0% | 32.1 | 84% | 0 | 21.7 | 3 |
| Identity-biased model + blind mode ON + institutions redacted | 0.928 | 70% | 0% | 0% | 33.5 | 82% | 0 | 0.0 | 0 |
| Identity-biased model (-0.25 on identity cues) + blind mode OFF | 0.926 | 69% | 0% | 0% | 25.2 | 92% | 0 | 16.7 | 0 |

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
