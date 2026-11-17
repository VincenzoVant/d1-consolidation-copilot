"""Remember a model's answers on disk, so the same question is never computed twice.

Laya takes about a fifth of a second per account. Training (the dev calibration),
the dev evaluation run and the notebook all ask it the same 3,500 questions; with
this cache the first run pays and the others are instant. The LLM method (Task
3.4) uses the same cache, where a repeated question would also cost money.

The key is a fingerprint of everything that decides the answer: the method's
version (model, revision, settings), the input and the question. Change any of
them and it's a different key, so a stale answer can't be returned.

It is one sqlite file (Python's built-in database), under .cache/, which git ignores.
"""

import hashlib
import json
import sqlite3
from pathlib import Path


def cache_key(*parts) -> str:
    """A 64-character fingerprint of the parts. Dict key order doesn't matter; list order does."""
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        except sqlite3.DatabaseError as e:
            db.close()
            raise sqlite3.DatabaseError(f"{self.path} is not a readable cache ({e}): delete it and run again") from None
        return db

    def get(self, key: str) -> dict | None:
        return self.get_many([key]).get(key)

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        """The cached answers among keys (missing keys are simply absent), in one query per 500 keys."""
        found: dict[str, dict] = {}
        if not keys:
            return found
        db = self._connect()
        try:
            for start in range(0, len(keys), 500):  # sqlite caps the number of ? in one statement
                part = keys[start : start + 500]
                rows = db.execute(f"SELECT key, value FROM answers WHERE key IN ({','.join('?' * len(part))})", part)
                found.update({k: json.loads(v) for k, v in rows})
        finally:
            db.close()
        return found

    def put(self, key: str, value: dict) -> None:
        self.put_many({key: value})

    def put_many(self, items: dict[str, dict]) -> None:
        """Store answers in one transaction; a key already there is replaced."""
        if not items:
            return
        db = self._connect()
        try:
            with db:
                db.executemany("INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)",
                               [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()])  # fmt: skip
        finally:
            db.close()

    def __len__(self) -> int:
        db = self._connect()
        try:
            return db.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        finally:
            db.close()
