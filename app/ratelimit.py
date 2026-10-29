"""Tiny in-memory sliding-window limiter (single process; fine for local/small deployments)."""
import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def _prune(q: deque, window: float, now: float) -> None:
    while q and q[0] <= now - window:
        q.popleft()


def count(key: str, window: float) -> int:
    with _lock:
        q = _hits[key]
        _prune(q, window, time.monotonic())
        return len(q)


def hit(key: str, window: float) -> int:
    with _lock:
        q = _hits[key]
        now = time.monotonic()
        _prune(q, window, now)
        q.append(now)
        return len(q)


def clear(key: str) -> None:
    with _lock:
        _hits.pop(key, None)
