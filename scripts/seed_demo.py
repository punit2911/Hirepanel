#!/usr/bin/env python3
"""Loads a demo job plus 9 fake candidates into a RUNNING HirePanel so you can click around.

    python scripts/seed_demo.py                       # local, no password
    python scripts/seed_demo.py http://localhost:8000 YOUR_API_KEY

Safe to run again: it reuses the demo job and only uploads resumes that are missing.
Patient on free hosts: it waits for the server to wake up and retries slow requests.
"""
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.dataset import CASES, JD, JOB_TITLE  # noqa: E402


def wake(base: str) -> None:
    """Free hosts sleep. Keep knocking until /healthz answers (up to about 5 minutes)."""
    print("Waking the server (the first request on a free host can take a minute or two)...")
    for i in range(30):
        try:
            if httpx.get(base + "/healthz", timeout=30).status_code == 200:
                print("Server is awake.")
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    sys.exit("The server did not wake up in 5 minutes. Open the site in your browser, wait, then run this again.")


def retry(fn, what: str, tries: int = 4):
    for n in range(1, tries + 1):
        try:
            return fn()
        except httpx.TimeoutException:
            print(f"  {what}: slow, trying again ({n}/{tries})...")
            time.sleep(5)
    sys.exit(f"{what} kept timing out. Wait a minute and run this script again; it will pick up where it left off.")


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    key = sys.argv[2] if len(sys.argv) > 2 else ""
    wake(base)
    c = httpx.Client(base_url=base + "/api", headers={"x-api-key": key} if key else {}, timeout=httpx.Timeout(300, connect=30))

    def get_jobs():
        r = c.get("/jobs")
        if r.status_code == 401:
            sys.exit("The server refused the key (401). Copy API_KEY from Render > Environment exactly, with no spaces.")
        return r.raise_for_status().json()

    job = next((j for j in retry(get_jobs, "Listing jobs") if j["title"] == JOB_TITLE), None)
    if job:
        print(f"Reusing the existing demo job {job['id']}.")
    else:
        print("Creating the demo job (the AI reads the job description, this can take a minute)...")
        job = retry(lambda: c.post("/jobs", json={"title": JOB_TITLE, "description": JD}).raise_for_status().json(), "Creating the job")
        # A slow request may have succeeded on the server even though we timed out; never end up with two.
    have = {r["filename"] for r in retry(lambda: c.get(f"/jobs/{job['id']}/candidates").raise_for_status().json(), "Listing candidates")}
    for case in CASES:
        name = case["id"] + ".txt"
        if name in have:
            continue
        print(f"  uploading {name}")
        retry(lambda: c.post(f"/jobs/{job['id']}/candidates",
                             files=[("files", (name, case["text"].encode(), "text/plain"))]).raise_for_status(), f"Uploading {name}")
    print("Scoring 9 demo resumes (free AI limits can make this take a few minutes)...")
    rows = []
    for _ in range(400):
        rows = retry(lambda: c.get(f"/jobs/{job['id']}/candidates").raise_for_status().json(), "Checking progress")
        if all(r["status"] in ("done", "failed") for r in rows):
            break
        time.sleep(3)
    for r in rows:
        print(f"  {r['filename']:<34} {r['status']:<7} score={r['score']} decision={r['decision']}")
    print(f"\nOpen {base}/#/job/{job['id']}")


if __name__ == "__main__":
    main()