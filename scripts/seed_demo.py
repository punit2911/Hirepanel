#!/usr/bin/env python3
"""Loads a demo job plus 9 fake candidates into a RUNNING HirePanel so you can click around.

    python scripts/seed_demo.py                       # local, no password
    python scripts/seed_demo.py http://localhost:8000 YOUR_API_KEY
"""
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.dataset import CASES, JD, JOB_TITLE  # noqa: E402


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    key = sys.argv[2] if len(sys.argv) > 2 else ""
    c = httpx.Client(base_url=base + "/api", headers={"x-api-key": key} if key else {}, timeout=60)
    job = c.post("/jobs", json={"title": JOB_TITLE, "description": JD}).raise_for_status().json()
    for case in CASES:
        c.post(f"/jobs/{job['id']}/candidates", files=[("files", (case["id"] + ".txt", case["text"].encode(), "text/plain"))]).raise_for_status()
    print("Scoring 9 demo resumes (free AI limits can make this take a few minutes)...")
    for _ in range(300):
        rows = c.get(f"/jobs/{job['id']}/candidates").json()
        if all(r["status"] in ("done", "failed") for r in rows):
            break
        time.sleep(2)
    for r in rows:
        print(f"  {r['filename']:<34} {r['status']:<7} score={r['score']} decision={r['decision']}")
    print(f"\nOpen {base}/#/job/{job['id']}")


if __name__ == "__main__":
    main()
