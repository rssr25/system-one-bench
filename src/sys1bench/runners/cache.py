"""SQLite cache of raw responses keyed by (adapter, model id, tunables, request). Shipped with results so
every number can be re-derived offline."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

from ..schemas import DecisionResponse


class ResponseCache:
    def __init__(self, path: str | Path = "cache.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, adapter TEXT, model TEXT, "
                         "created REAL, payload TEXT)")
        self._db.commit()

    def get(self, key: str) -> DecisionResponse | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM responses WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return DecisionResponse.model_validate_json(row[0])
        except Exception:
            # legacy entries stored failed answers with NaN probabilities (serialised as null); repair and let the
            # runner rebuild them from `raw` via adapter.reparse
            d = json.loads(row[0])
            for a in d.get("answers", {}).values():
                pr = a.get("probs") or []
                if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in pr):
                    a["probs"] = []
                    a["error"] = a.get("error") or "legacy_nan"
            return DecisionResponse.model_validate(d)

    def put(self, key: str, adapter: str, model: str, resp: DecisionResponse) -> None:
        import time

        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
                             (key, adapter, model, time.time(), resp.model_dump_json()))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
