"""Tiny ordered parallel map for independent per-frame work."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_jobs(n_jobs: int | None) -> int:
    """1 = serial (default); -1 or 0 = all cores; otherwise capped at the core count."""
    cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs <= 0:
        return cores
    return min(int(n_jobs), cores)


def pmap(
    fn: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int | None = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[R]:
    """``[fn(x) for x in items]`` on a thread pool, results in input order.

    ``n_jobs=1`` runs a plain loop, so the serial path is exactly the old code.
    ``on_progress(done, total)`` is called after each item (from worker threads
    when parallel).
    """
    items = list(items)
    if on_progress is not None:
        total, done, lock, inner = len(items), [0], threading.Lock(), fn

        def fn(x: T) -> R:  # noqa: F811
            out = inner(x)
            with lock:  # report inside the lock so percentages never go backwards
                done[0] += 1
                on_progress(done[0], total)
            return out

    jobs = min(resolve_jobs(n_jobs), len(items))
    if jobs <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items))
