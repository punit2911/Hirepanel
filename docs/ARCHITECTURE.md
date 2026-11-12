# Architecture

```
resume + job description
  -> shield (injection, hidden chars, keyword stuffing, JD copy)
  -> redact (names, contact, pronouns, ages, school names) : no model ever sees identifiers
  -> in parallel: parser | scorer A (strict evidence) | scorer B (potential)
  -> verifier   "no quote, no credit": every quote must exist in the resume and be on-topic
  -> adjudicator (only when A and B disagree by >= 0.25)
  -> decision policy (deterministic: shortlist / human_review / reject, with gates + confidence)
  -> in parallel: comms drafts | interview kit
  -> bias audit (scrubs protected-characteristic language from AI output)
  -> recruiter approves  -> signed webhook -> n8n -> email / Discord
```

## Why it is built this way
- **LLMs propose, code decides.** Scores come from models but the verifier and the decision policy are plain code, so a model cannot talk its way to a shortlist.
- **Two lenses, different providers.** A strict and a generous scorer; disagreement is information and is escalated, not averaged away.
- **Fail safe.** Any model failure degrades to deterministic rules; the circuit breaker stops hammering a down provider; nothing is sent without human approval.
- **Privacy by construction.** Redaction runs before any model call and fails closed if an identifier remains.
- **Measured, not claimed.** `evals/` holds the benchmark, `evals/ablate.py` the ablations, `scripts/bench_providers.py` real-model measurements.

## Where things live
`app/pipeline.py` DAG wiring | `app/agents/` the agents | `app/llm/` provider router (retries, rate limits, circuit breaker, stats) | `app/main.py` API | `evals/` benchmark + ablations | `n8n/` workflow.
