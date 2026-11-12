"""Small, dependency-free metrics."""
from __future__ import annotations


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(random positive scores higher than random negative); ties count half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a proportion, so small samples are not over-claimed."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return round((c - m) / d, 3), round((c + m) / d, 3)
