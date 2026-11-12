# Interview Q&A for HirePanel (answer in your own words)

**1. Give me the 30-second pitch.**
A multi-agent hiring assistant where models propose scores but code decides. Every score needs a verbatim quote the verifier can find in the resume, identifiers are redacted before any model sees the text, and a human approves every outcome. I measured each safety layer with ablations.

**2. Why multiple agents instead of one prompt?**
Different jobs need different guarantees: parsing and scoring are model tasks, verification and the decision are deterministic. Splitting them lets me test each, run independent steps in parallel (a DAG), and degrade gracefully when one fails.

**3. What does the verifier do and why does it matter?**
It checks every quote exists in the resume, is on-topic, is not copied from the job description, and is not only in a skills list. Ablation: a model inventing evidence 30% of the time shortlisted 7% of attacks with the verifier off and 0% with it on.

**4. How do you handle prompt injection?**
Two layers: a shield that detects and removes instruction-like text, hidden characters and stuffing, and the verifier that makes injected claims worthless. With both off, 53% of attacks got shortlisted in simulation; either one alone gave 0%.

**5. How did you test fairness?**
Counterfactual identity swaps: same qualifications, different name, school, age, gender, marital status. Metric = score spread and decision flips. The experiment found that redacting names but not schools left a 21.7-point spread against a biased model, so institution redaction became the default (spread 0).

**6. Your dual scorers did not improve accuracy. Why keep them?**
Honest answer: in simulation they did not raise AUROC. Their value is the disagreement signal that routes uncertain cases to a human. It is configurable (`SCORER_MODE`) so the cost is a choice.

**7. How reliable are your numbers?**
Synthetic data with hidden labels I generated, so they compare configurations, not real-world accuracy. I report confidence intervals and list limitations in `docs/LIMITATIONS.md`. Real-model numbers come from `scripts/bench_providers.py`.

**8. Tell me about a bug the evaluation found.**
Short copied job-description bullets ("Strong SQL and Excel") escaped copy detection and a JD-copy attack scored 72.7. Fixed with whole-line matching and a regression test.

**9. How do you handle rate limits on free APIs?**
Client-side RPM/RPD/TPM limiters, retries with backoff, a circuit breaker per provider, fallback across providers, and a final deterministic fallback.

**10. What if the model returns invalid JSON?**
Validate against a Pydantic schema, retry once with the error fed back, learn if a provider rejects JSON mode, then fall back to the next provider or the rules. Stats track JSON validity per provider.

**11. Why Ollama?**
Privacy and cost: resumes never leave the machine. Trade-off: slower and weaker; the benchmark script quantifies that.

**12. How is it deployed?**
Docker, docker-compose with n8n, render.yaml, CI on GitHub Actions. Signed (HMAC) webhooks to n8n; API-key auth; retention and deletion endpoints.

**13. What would you do with more time?**
Evaluate on a real labelled resume set with consent, calibrate the confidence score, add OCR, and run the provider benchmark on larger samples.

**14. Is this legal to use for hiring?**
It is a decision-support tool with human approval, not a compliance product. Laws like NYC LL144 and the EU AI Act add audit/notice duties. I list this as a limitation.

**15. What is the biggest weakness?**
No real-world labelled validation yet. I say so up front and built the tooling to close that gap.
