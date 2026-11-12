"""A tiny async DAG runner: independent agents run in parallel, every run leaves a trace."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Span:
    name: str
    status: str = "ok"  # ok | degraded | failed | skipped
    mode: str = "deterministic"  # deterministic | llm | heuristic
    provider: str = ""
    model: str = ""
    started_ms: int = 0
    duration_ms: int = 0
    attempts: int = 0
    cached: bool = False
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "name", "status", "mode", "provider", "model", "started_ms", "duration_ms",
            "attempts", "cached", "notes", "error")}


class Trace:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[Span] = []

    def now_ms(self) -> int:
        return int((time.perf_counter() - self.t0) * 1000)

    def new_span(self, name: str) -> Span:
        sp = Span(name=name, started_ms=self.now_ms())
        self.spans.append(sp)
        return sp

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.spans]

    def overall_mode(self) -> str:
        modes = {s.mode for s in self.spans if s.mode in ("llm", "heuristic")}
        if modes == {"llm"}:
            return "llm"
        if modes == {"heuristic"}:
            return "heuristic"
        return "mixed" if modes else "heuristic"


class PipelineError(Exception):
    def __init__(self, node: str, original: BaseException):
        super().__init__(f"agent '{node}' failed: {type(original).__name__}: {original}")
        self.node, self.original = node, original


NodeFn = Callable[[dict[str, Any], Span], Awaitable[Any]]


@dataclass
class Node:
    name: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    optional: bool = False
    timeout: float = 120.0


class DAG:
    def __init__(self, trace: Trace):
        self.trace = trace
        self.nodes: dict[str, Node] = {}

    def add(self, name: str, fn: NodeFn, deps: tuple[str, ...] = (), *, optional: bool = False,
            timeout: float = 120.0) -> None:
        if name in self.nodes:
            raise ValueError(f"duplicate node {name}")
        self.nodes[name] = Node(name, fn, tuple(deps), optional, timeout)

    def _toposort(self) -> list[str]:
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(n: str) -> None:
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                raise ValueError(f"cycle at {n}")
            state[n] = 1
            for d in self.nodes[n].deps:
                if d not in self.nodes:
                    raise ValueError(f"{n} depends on unknown node {d}")
                visit(d)
            state[n] = 2
            order.append(n)

        for n in self.nodes:
            visit(n)
        return order

    async def run(self) -> dict[str, Any]:
        futures: dict[str, asyncio.Future] = {}

        async def runner(node: Node) -> Any:
            try:
                dep_vals = {d: await futures[d] for d in node.deps}
            except Exception:
                sp = self.trace.new_span(node.name)
                sp.status, sp.mode = "skipped", "deterministic"
                raise
            span = self.trace.new_span(node.name)
            t0 = time.perf_counter()
            try:
                return await asyncio.wait_for(node.fn(dep_vals, span), node.timeout)
            except Exception as e:
                span.status = "failed"
                span.error = f"{type(e).__name__}: {e}"[:300]
                if node.optional:
                    return None
                raise
            finally:
                span.duration_ms = int((time.perf_counter() - t0) * 1000)

        order = self._toposort()
        for name in order:
            futures[name] = asyncio.ensure_future(runner(self.nodes[name]))
        outcomes = await asyncio.gather(*futures.values(), return_exceptions=True)
        results = dict(zip(futures.keys(), outcomes))
        for name in order:
            sp = next((s for s in self.trace.spans if s.name == name), None)
            if isinstance(results[name], BaseException) and sp and sp.status == "failed":
                raise PipelineError(name, results[name])
        for name, val in results.items():
            if isinstance(val, BaseException):
                raise PipelineError(name, val)
        return results
