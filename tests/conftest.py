import dataclasses
import re
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.llm import LLMRouter  # noqa: E402
from app.pipeline import Panel  # noqa: E402

TODAY = date(2026, 11, 11)


# Every environment variable the app reads. Tests must never depend on the developer's own .env
# (for example LLM_PROVIDERS=ollama), so the fixture clears all of them.
_CONFIG_VARS = sorted(set(re.findall(r'_(?:str|num|bool)\("([A-Z0-9_]+)"',
                                     (Path(__file__).resolve().parents[1] / "app" / "config.py").read_text())))


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    for k in _CONFIG_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return Settings.from_env()


@pytest.fixture()
def offline_panel(settings):
    return Panel(settings, LLMRouter(settings))


@pytest.fixture()
def with_settings():
    def make(base: Settings, **kw) -> Settings:
        return dataclasses.replace(base, **kw)
    return make