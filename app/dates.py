"""Date-range parsing. The LLM extracts date strings; code does the arithmetic."""
from __future__ import annotations

import re
from datetime import date

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_YEAR = r"(?:19|20)\d{2}"
_DATE = rf"(?:{_MONTH}\.?,?\s+{_YEAR}|\d{{1,2}}/{_YEAR}|{_YEAR})"
_PRESENT = r"(?:present|current|now|till date|to date|ongoing)"
RANGE_RE = re.compile(
    rf"(?P<s>{_DATE})\s*(?:-|–|—|to|until|till)\s*(?P<e>{_DATE}|{_PRESENT})", re.I)
_POINT_RE = re.compile(rf"^(?:(?P<mon>{_MONTH})\.?,?\s+(?P<y1>{_YEAR})|(?P<m2>\d{{1,2}})/(?P<y2>{_YEAR})|(?P<y3>{_YEAR}))$", re.I)


def _idx(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def parse_point(s: str, *, end: bool, today: date) -> int | None:
    """Return a month index (year*12 + month-1) or None."""
    s = s.strip()
    if re.fullmatch(_PRESENT, s, re.I):
        return _idx(today.year, today.month)
    m = _POINT_RE.match(s)
    if not m:
        return None
    if m.group("mon"):
        return _idx(int(m.group("y1")), _MONTHS[m.group("mon")[:3].lower()])
    if m.group("m2"):
        mo = int(m.group("m2"))
        return _idx(int(m.group("y2")), mo) if 1 <= mo <= 12 else None
    y = int(m.group("y3"))
    return _idx(y, 12 if end else 1)


def parse_range(start: str | None, end: str | None, today: date | None = None) -> tuple[int, int] | None:
    today = today or date.today()
    if not start:
        return None
    a = parse_point(start, end=False, today=today)
    b = parse_point(end or "present", end=True, today=today)
    if a is None or b is None or b < a:
        return None
    return a, b


def find_ranges(text: str, today: date | None = None) -> list[tuple[str, str, tuple[int, int] | None]]:
    today = today or date.today()
    out = []
    for m in RANGE_RE.finditer(text):
        s, e = m.group("s"), m.group("e")
        out.append((s, e, parse_range(s, e, today)))
    return out


def months_of(rng: tuple[int, int]) -> int:
    return rng[1] - rng[0] + 1


def years_from_ranges(ranges: list[tuple[int, int]]) -> float:
    """Union of month intervals (overlapping jobs are not double counted)."""
    if not ranges:
        return 0.0
    merged: list[list[int]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return round(sum(b - a + 1 for a, b in merged) / 12.0, 1)
