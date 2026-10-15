"""Tiny file cache for HTTP responses."""
from __future__ import annotations

import hashlib
import time

import requests

from ..config import CACHE_DIR

_session = requests.Session()
_session.headers["User-Agent"] = "MoneyBagTips/1.0 (+football research tool)"


def fetch(url: str, ttl: float | None, offline: bool = False) -> str | None:
    """Return the body of ``url``, cached on disk.

    ttl: seconds a cached copy stays fresh; None means forever.
    Returns None on 404. A stale cached copy is used if the network fails.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / hashlib.sha1(url.encode()).hexdigest()
    if path.exists():
        fresh = ttl is None or time.time() - path.stat().st_mtime < ttl
        if fresh or offline:
            return path.read_text(encoding="utf-8")
    elif offline:
        return None
    try:
        r = _session.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException:
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise
    text = r.content.decode("utf-8-sig", errors="replace")
    path.write_text(text, encoding="utf-8")
    return text
