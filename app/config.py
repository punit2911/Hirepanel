"""All configuration comes from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        raise RuntimeError(f"Config error: {name} must be a number, got {v!r}") from None


def _db_url(raw: str, data_dir: Path) -> str:
    if not raw:
        return f"sqlite:///{data_dir / 'hirepanel.db'}"
    # Render / Heroku / Neon style URLs
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        raw = "postgresql+psycopg2://" + raw[len("postgresql://"):]
    return raw


@dataclass(frozen=True)
class Settings:
    # --- app ---
    env: str
    api_key: str
    webhook_secret: str
    public_base_url: str
    database_url: str
    data_dir: Path
    cors_origins: tuple[str, ...]
    max_upload_mb: float
    rate_limit_per_min: int
    retention_days: int
    max_concurrent_runs: int
    run_timeout_s: float
    # --- llm ---
    llm_providers: tuple[str, ...]
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    openrouter_api_key: str
    openrouter_model: str
    ollama_base_url: str
    ollama_model: str
    gemini_rpm: int
    gemini_rpd: int
    groq_rpm: int
    groq_rpd: int
    groq_tpm: int
    openrouter_rpm: int
    openrouter_rpd: int
    scorer_a_provider: str
    scorer_b_provider: str
    llm_cache: bool
    llm_timeout_s: float
    llm_max_retries: int
    bias_llm: bool
    # --- policy ---
    shortlist_threshold: float
    review_band: float
    min_confidence: float
    disagreement_delta: float
    blind_mode: bool
    redact_institutions: bool
    # --- research switches (leave at defaults in production; used by the ablation experiments) ---
    verifier_enabled: bool
    scorer_mode: str  # dual | a | b
    shield_enabled: bool
    # --- notifications ---
    n8n_webhook_url: str
    discord_webhook_url: str
    company_name: str
    recruiter_name: str

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @staticmethod
    def from_env() -> "Settings":
        env = _str("ENV", "development").lower()
        data_dir = Path(_str("DATA_DIR", "./data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        s = Settings(
            env=env,
            api_key=_str("API_KEY"),
            webhook_secret=_str("WEBHOOK_SECRET"),
            public_base_url=_str("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
            database_url=_db_url(_str("DATABASE_URL"), data_dir),
            data_dir=data_dir,
            cors_origins=tuple(o.strip() for o in _str("CORS_ORIGINS").split(",") if o.strip()),
            max_upload_mb=_num("MAX_UPLOAD_MB", 5),
            rate_limit_per_min=int(_num("RATE_LIMIT_PER_MIN", 120)),
            retention_days=int(_num("RETENTION_DAYS", 0)),
            max_concurrent_runs=max(1, int(_num("MAX_CONCURRENT_RUNS", 2))),
            run_timeout_s=_num("RUN_TIMEOUT_S", 300),
            llm_providers=tuple(
                p.strip().lower()
                for p in _str("LLM_PROVIDERS", "gemini,groq,openrouter,ollama").split(",")
                if p.strip()
            ),
            gemini_api_key=_str("GEMINI_API_KEY"),
            gemini_model=_str("GEMINI_MODEL", "auto"),
            groq_api_key=_str("GROQ_API_KEY"),
            groq_model=_str("GROQ_MODEL", "auto"),
            openrouter_api_key=_str("OPENROUTER_API_KEY"),
            openrouter_model=_str("OPENROUTER_MODEL"),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            ollama_model=_str("OLLAMA_MODEL"),
            gemini_rpm=int(_num("GEMINI_RPM", 10)),
            gemini_rpd=int(_num("GEMINI_RPD", 200)),
            groq_rpm=int(_num("GROQ_RPM", 25)),
            groq_rpd=int(_num("GROQ_RPD", 900)),
            groq_tpm=int(_num("GROQ_TPM", 7000)),
            openrouter_rpm=int(_num("OPENROUTER_RPM", 15)),
            openrouter_rpd=int(_num("OPENROUTER_RPD", 150)),
            scorer_a_provider=_str("SCORER_A_PROVIDER").lower(),
            scorer_b_provider=_str("SCORER_B_PROVIDER").lower(),
            llm_cache=_bool("LLM_CACHE", False),
            llm_timeout_s=_num("LLM_TIMEOUT_S", 90),
            llm_max_retries=max(1, int(_num("LLM_MAX_RETRIES", 3))),
            bias_llm=_bool("BIAS_LLM", False),
            shortlist_threshold=_num("SHORTLIST_THRESHOLD", 70),
            review_band=_num("REVIEW_BAND", 10),
            min_confidence=_num("MIN_CONFIDENCE", 0.55),
            disagreement_delta=_num("DISAGREEMENT_DELTA", 0.25),
            blind_mode=_bool("BLIND_MODE", True),
            redact_institutions=_bool("REDACT_INSTITUTIONS", True),
            verifier_enabled=_bool("VERIFIER_ENABLED", True),
            scorer_mode=_str("SCORER_MODE", "dual").lower() if _str("SCORER_MODE", "dual").lower() in ("dual", "a", "b") else "dual",
            shield_enabled=_bool("SHIELD_ENABLED", True),
            n8n_webhook_url=_str("N8N_WEBHOOK_URL"),
            discord_webhook_url=_str("DISCORD_WEBHOOK_URL"),
            company_name=_str("COMPANY_NAME", "our team"),
            recruiter_name=_str("RECRUITER_NAME", "The Recruiting Team"),
        )
        if s.is_production:
            missing = [k for k, v in (("API_KEY", s.api_key), ("WEBHOOK_SECRET", s.webhook_secret)) if not v]
            if missing:
                raise RuntimeError(
                    "ENV=production requires these to be set: " + ", ".join(missing)
                    + ". Run `python scripts/setup_env.py` to generate them."
                )
            if len(s.api_key) < 16:
                raise RuntimeError("API_KEY must be at least 16 characters in production.")
        return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
