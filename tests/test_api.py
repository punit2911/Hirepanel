"""HTTP API: auth, uploads, queue, review flow, webhooks, hardening."""
import io
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.notify import sign, verify_signature
from evals.dataset import JD, STRONG_STANDARD

KEY = "test-key-1234567890abcd"
H = {"x-api-key": KEY}


@pytest.fixture()
def client(settings, with_settings):
    s = with_settings(settings, api_key=KEY, webhook_secret="whsec-test-1234567890")
    with TestClient(create_app(s)) as c:
        yield c


def wait_done(client, cid, timeout=15):
    t = time.time()
    while time.time() - t < timeout:
        d = client.get(f"/api/candidates/{cid}", headers=H).json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(0.1)
    raise AssertionError("candidate did not finish")


def new_job(client):
    return client.post("/api/jobs", json={"title": "Senior ML Engineer", "description": JD}, headers=H).json()


def test_auth_required_but_health_is_open(client):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/api/jobs", headers={"authorization": f"Bearer {KEY}"}).status_code == 200
    assert client.get("/healthz").json()["ok"] and client.get("/readyz").json()["ok"]
    assert client.get("/api/public").json()["auth_required"] is True


def test_full_flow_upload_score_review_erase(client):
    job = new_job(client)
    assert any(r["id"] == "R_years" for r in job["spec"]["requirements"])
    up = client.post(f"/api/jobs/{job['id']}/candidates", headers=H, files=[
        ("files", ("priya.txt", STRONG_STANDARD.encode(), "text/plain")),
        ("files", ("virus.exe", b"MZ" + b"0" * 200, "application/octet-stream")),
        ("files", ("empty.txt", b"", "text/plain"))]).json()
    assert len(up["accepted"]) == 1 and len(up["rejected"]) == 2
    cid = up["accepted"][0]["id"]
    d = wait_done(client, cid)
    assert d["status"] == "done" and d["decision"] == "shortlist" and d["score"] > 70
    assert "raw_text" not in d and d["contact"]["email"] == "priya.sharma@example.com"
    assert "contact" not in d["result"]  # PII lives in its own column, not inside the result blob

    r = client.post(f"/api/candidates/{cid}/review", headers=H, json={"action": "approve_shortlist", "note": "ok"}).json()
    assert r["email"]["to"] == "priya.sharma@example.com" and r["email"]["body"].startswith("Hi Priya")
    assert r["email_status"] is None  # no n8n configured: draft is handed back, nothing is sent
    m = client.get("/api/metrics", headers=H).json()
    assert m["scored"] == 1 and m["reviewed"] == 1
    audit = [a["action"] for a in client.get(f"/api/candidates/{cid}", headers=H).json()["audit"]]
    assert audit[:2] == ["uploaded", "scored"] and "review:approve_shortlist" in audit

    assert client.delete(f"/api/candidates/{cid}", headers=H).status_code == 200
    assert client.get(f"/api/candidates/{cid}", headers=H).status_code == 404


def test_pdf_and_docx_extraction(client):
    import docx
    from reportlab.pdfgen import canvas  # noqa: F401
    pytest.skip("reportlab not required") if False else None
    job = new_job(client)
    d = docx.Document()
    for line in STRONG_STANDARD.splitlines():
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    up = client.post(f"/api/jobs/{job['id']}/candidates", headers=H,
                     files=[("files", ("cv.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))]).json()
    assert len(up["accepted"]) == 1
    assert wait_done(client, up["accepted"][0]["id"])["decision"] == "shortlist"
    fake = client.post(f"/api/jobs/{job['id']}/candidates", headers=H, files=[("files", ("cv.pdf", b"not a pdf", "application/pdf"))]).json()
    assert fake["rejected"] and "not a real PDF" in fake["rejected"][0]["error"]


def test_upload_size_limit(client, settings, with_settings):
    s = with_settings(settings, api_key=KEY, max_upload_mb=0.001)
    with TestClient(create_app(s)) as c:
        job = c.post("/api/jobs", json={"title": "Dev", "description": JD}, headers=H).json()
        r = c.post(f"/api/jobs/{job['id']}/candidates", headers=H, files=[("files", ("big.txt", b"a" * 5000, "text/plain"))]).json()
        assert r["rejected"] and "larger than" in r["rejected"][0]["error"]


def test_requirement_editing_and_rescore(client):
    job = new_job(client)
    cid = client.post(f"/api/jobs/{job['id']}/candidates/text", headers=H, json={"text": STRONG_STANDARD}).json()["id"]
    before = wait_done(client, cid)["score"]
    spec = client.put(f"/api/jobs/{job['id']}/spec", headers=H, json={
        "min_years": 0, "requirements": [{"text": "Experience with Rust", "kind": "must", "keywords": ["rust"]}]}).json()["spec"]
    assert [r["id"] for r in spec["requirements"]] == ["R1"]
    assert client.post(f"/api/jobs/{job['id']}/rescore", headers=H).json()["requeued"] == 1
    time.sleep(0.3)
    after = wait_done(client, cid)["score"]
    assert after < before and after < 30


def test_webhook_review_requires_secret(client):
    job = new_job(client)
    cid = client.post(f"/api/jobs/{job['id']}/candidates/text", headers=H, json={"text": STRONG_STANDARD}).json()["id"]
    wait_done(client, cid)
    body = {"candidate_id": cid, "action": "hold", "note": "from n8n"}
    assert client.post("/api/webhooks/n8n", json=body).status_code == 401
    assert client.post("/api/webhooks/n8n", json=body, headers={"x-webhook-secret": "nope"}).status_code == 401
    ok = client.post("/api/webhooks/n8n", json=body, headers={"x-webhook-secret": "whsec-test-1234567890"})
    assert ok.status_code == 200
    assert any(a["actor"] == "n8n" for a in client.get(f"/api/candidates/{cid}", headers=H).json()["audit"])


def test_csv_export_neutralises_formula_injection(client):
    job = new_job(client)
    cid = client.post(f"/api/jobs/{job['id']}/candidates/text", headers=H,
                      json={"text": STRONG_STANDARD, "name": "=HYPERLINK(\"http://evil\")"}).json()["id"]
    wait_done(client, cid)
    csv_text = client.get(f"/api/jobs/{job['id']}/export.csv", headers=H).text
    assert "'=HYPERLINK" in csv_text and ",=HYPERLINK" not in csv_text


def test_input_validation(client):
    assert client.post("/api/jobs", json={"title": "x", "description": "short"}, headers=H).status_code == 422
    job = new_job(client)
    assert client.post(f"/api/candidates/nope/review", json={"action": "hold"}, headers=H).status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/candidates/text", json={"text": "too short"}, headers=H).status_code == 422
    assert client.post("/api/candidates/x/review", json={"action": "delete_everything"}, headers=H).status_code == 422


def test_rate_limit_and_security_headers(settings, with_settings):
    s = with_settings(settings, rate_limit_per_min=5)
    with TestClient(create_app(s)) as c:
        codes = [c.get("/api/public").status_code for _ in range(8)]
        assert 429 in codes
        r = c.get("/healthz")
        assert r.headers["x-content-type-options"] == "nosniff" and "frame-ancestors" in r.headers["content-security-policy"]


def test_production_refuses_to_boot_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        Settings.from_env()


def test_ui_is_served_and_docs_hidden_in_production(settings, with_settings):
    s = with_settings(settings, env="production", api_key=KEY, webhook_secret="x" * 20)
    with TestClient(create_app(s)) as c:
        assert "HirePanel" in c.get("/").text
        assert c.get("/docs").status_code == 404


def test_webhook_signature_roundtrip():
    body, ts = b'{"a":1}', str(int(time.time()))
    sig = sign("secret", body, ts)
    assert verify_signature("secret", body, ts, sig)
    assert not verify_signature("secret", b'{"a":2}', ts, sig)
    assert not verify_signature("secret", body, str(int(time.time()) - 4000), sign("secret", body, str(int(time.time()) - 4000)))


@pytest.fixture()
def demo_client(settings, with_settings):
    s = with_settings(settings, api_key=KEY, webhook_secret="whsec-test-1234567890", demo_mode=True)
    with TestClient(create_app(s)) as c:
        yield c


def test_demo_mode_is_read_only_masked_and_owner_keeps_full_access(demo_client):
    c = demo_client
    job = c.post("/api/jobs", json={"title": "Senior ML Engineer", "description": JD}, headers=H).json()
    cid = c.post(f"/api/jobs/{job['id']}/candidates", headers=H,
                 files=[("files", ("priya.txt", STRONG_STANDARD.encode(), "text/plain"))]).json()["accepted"][0]["id"]
    wait_done(c, cid)
    assert c.get("/api/public").json()["demo"] is True
    # strangers can read
    assert c.get("/api/jobs").status_code == 200 and c.get("/api/status").status_code == 200
    rows = c.get(f"/api/jobs/{job['id']}/candidates").json()
    assert rows and rows[0]["name"] == "P. S."
    d = c.get(f"/api/candidates/{cid}").json()
    assert d["contact"] == {"name": "P. S."} and "priya.sharma@example.com" not in str(d)
    # strangers cannot write, export, review, delete, or use a wrong key
    assert c.post("/api/jobs", json={"title": "x" * 5, "description": JD}).status_code == 401
    assert c.get(f"/api/jobs/{job['id']}/export.csv").status_code == 401
    assert c.post(f"/api/candidates/{cid}/review", json={"action": "approve_shortlist"}).status_code == 401
    assert c.delete(f"/api/candidates/{cid}").status_code == 401
    assert c.get("/api/jobs", headers={"x-api-key": "wrong"}).status_code == 401
    # the owner still sees everything unmasked
    assert c.get(f"/api/candidates/{cid}", headers=H).json()["contact"]["email"] == "priya.sharma@example.com"


def test_demo_mode_off_by_default_and_needs_a_key_to_exist(client, settings, with_settings):
    assert client.get("/api/public").json()["demo"] is False
    s = with_settings(settings, api_key="", demo_mode=True)  # no key set means nothing to bypass
    with TestClient(create_app(s)) as c:
        assert c.get("/api/public").json()["demo"] is False
