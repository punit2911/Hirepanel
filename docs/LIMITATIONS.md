# Limitations (read before you trust or present this)

- **Benchmark is synthetic.** 150 generated resumes with hidden labels, written by the same author as the system. Real resumes are messier. Treat metrics as comparisons between configurations, not real-world accuracy.
- **Simulated models are not real models.** Ablations show how the architecture reacts to a defined failure, nothing more.
- **Real-model results must be measured by you** (`scripts/bench_providers.py`). Live Gemini/Groq/Ollama paths were hardened and unit-tested with fakes, but were not run against the real services during development.
- **Docker image build, docker-compose and the n8n workflow were not executed** in the development sandbox.
- **Offline rules are a skill-keyword model** (taxonomy in `app/skills.py`); unseen skills are invisible to them.
- **Not a legal compliance product.** Hiring-AI laws (e.g. NYC Local Law 144, EU AI Act) require audits and notices beyond this. A human approves every outcome by design.
- **PDF/DOCX extraction** is best-effort; scanned images are not OCR'd.
- **Fairness checks cover the identity cues we thought of** (name, pronouns, age, school, etc.). Proxies we did not model could remain.
- **Dual scorers did not improve accuracy** in simulation (docs/RESULTS.md); their value is the disagreement signal.
