# Run HirePanel on your own computer's AI (Ollama), from zero

Ollama runs an AI model on your own laptop. It is free, needs no internet after the download, and no resume ever leaves your machine. That last point is a real selling point for a hiring tool.

## 1. Install
1. Go to **https://ollama.com/download** and install it for your system (Windows / Mac / Linux).
2. Open a terminal (Windows: search "PowerShell"). Type `ollama --version`. If it prints a version, it worked.

## 2. Pick a model that fits your RAM
Check your RAM first (Windows: Task Manager > Performance > Memory).

| Your RAM | Try this | Notes |
|---|---|---|
| 8 GB | `llama3.2:3b` | small and fast, weaker at following JSON rules |
| 16 GB | `qwen2.5:7b` | good all-rounder, our recommended start |
| 32 GB+ | a 14B model (see ollama.com/library) | slower, better |

Download it (one time, a few GB): `ollama pull qwen2.5:7b`
Then check the exact name: `ollama list` (copy the name shown, e.g. `qwen2.5:7b`).
Quick test: `ollama run qwen2.5:7b "say hi"` (type `/bye` to leave).

Model names change over time. If a pull fails, search the model on https://ollama.com/library and use the name it shows.

## 3. Tell HirePanel to use it
Open the `.env` file in the project folder and set:
```
LLM_PROVIDERS=ollama
OLLAMA_MODEL=qwen2.5:7b
LLM_TIMEOUT_S=300
RUN_TIMEOUT_S=900
```
(A laptop is slower than Google's servers, so we give it more time.) Keep Ollama running in the background.

Check it works: `python scripts/check_keys.py` should print `OK ollama`.
Then start the app as usual (README) and analyse a resume. The result page will say the mode is `llm`.

## 4. Measure it (this is what makes it a strong project)
```
python scripts/bench_providers.py --providers ollama --n 4 --attacks 2
```
It writes `docs/bench/ollama.md` with accuracy, fake-quote rate, JSON-validity rate and latency. Run it for each model you try (3B vs 7B) and for Gemini/Groq if you have keys, then put the comparison table in your README. Small models usually invent more quotes and break JSON more often. The verifier catching them is exactly the story to tell.

## Troubleshooting
- `connection refused`: Ollama is not running. Open the Ollama app, or run `ollama serve`.
- Very slow: use a smaller model, or close other programs. First call is slow while the model loads.
- `model not found`: the name in `.env` must match `ollama list` exactly.
- Lots of "invalid JSON": the model is too small. Try a bigger one.
