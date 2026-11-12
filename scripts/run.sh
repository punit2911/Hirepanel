#!/usr/bin/env bash
# Mac/Linux one-command start:  bash scripts/run.sh
set -e
cd "$(dirname "$0")/.."
command -v python3 >/dev/null || { echo "Install Python 3.10+ first: https://www.python.org/downloads/"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
python scripts/setup_env.py
echo
echo "Starting HirePanel at http://localhost:8000   (Ctrl+C to stop)"
exec uvicorn app.asgi:app --host 127.0.0.1 --port 8000
