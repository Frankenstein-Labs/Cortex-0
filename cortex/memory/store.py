"""Persistent multi-level memory backed by SQLite (FTS5 + hashed embeddings).

Levels
------
working      current turn scratchpad (short TTL)
short_term   recent conversation facts
long_term    durable facts worth keeping
episodic     raw conversation turns
semantic     consolidated summaries
project      decisions, architecture notes
preference   explicit user preferences

This is *external* memory.  It never updates model weights.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .consolidation import extract_preferences, extractive_summary
from .retrieval import (
    HashingEmbedder,
    cosine,
    dedupe_texts,
    recency_score,
)

KINDS = {"working", "short_term", "long_term", "episodic", "semantic", "project", "preference"}
KIND_HALF_LIFE_DAYS = {
    "working": 1 / 24,
    "short_term": 1,
    "long_term": 90,
    "episodic": 14,
    "semantic": 180,
    "project": 60,
    "preference": 365,
}
PREFERENCE_HINTS = re.compile(
    r"(je préfère|j'aime|rappelle-toi|souviens-toi|retiens que|i prefer|i like|remember that)",
    re.I,
)


@dataclass
class MemoryHit:
    id: int
    kind: str
    content: str
    summary: str
    importance: float
    created_at: float
    score: float = 0.0
    lexical: float = 0.0
    recency: float = 0.0
    semantic: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "summary": self.summary,
            "importance": round(self.importance, 3),
            "created_at": self.created_at,
            "score": round(self.score, 4),
            "lexical": round(self.lexical, 4),
            "recency": round(self.recency, 4),
            "semantic": round(self.semantic, 4),
        }


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    source TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_accessed REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, id);
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reasoning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    verified INTEGER NOT NULL,
    confidence REAL NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS initiatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    score REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    sent_at REAL
);
CREATE TABLE IF NOT EXISTS missions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    instruction TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    plan TEXT NOT NULL DEFAULT '[]',
    instructions TEXT NOT NULL DEFAULT '[]',
    checkpoints TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    priority REAL NOT NULL DEFAULT 0.5,
    progress REAL NOT NULL DEFAULT 0.0,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    priority INTEGER NOT NULL DEFAULT 0,
    run_at REAL NOT NULL,
    locked_by TEXT,
    locked_at REAL,
    error TEXT,
    result TEXT,
    idempotency_key TEXT UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, run_at);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, summary, content='memories', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
    VALUES ('delete', old.id, old.content, old.summary);
    INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary);
END;
"""


class MemoryStore:
    def __init__(
        self,
        path: str | Path = "cortex.sqlite",
        embedder: Optional[HashingEmbedder] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.path = str(path)
        self.embedder = embedder or HashingEmbedder()
        self.weights = weights or {
            "lexical": 0.45,
            "recency": 0.25,
            "importance": 0.15,
            "semantic": 0.15,
        }
        self._lock = threading.RLock()
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.fts = self._init_schema()

    # ------------------------------------------------------------- connection
    @contextmanager
    def _connect(self):
        if self._memory_conn is not None:
            with self._lock:
                try:
                    yield self._memory_conn
                    self._memory_conn.commit()
                except Exception:
                    self._memory_conn.rollback()
                    raise
            return
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def connection(self):
        """Public accessor used by the planning and task modules."""
        return self._connect()

    def _init_schema(self) -> bool:
        with self._connect() as conn:
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            try:
                conn.executescript(FTS_SCHEMA)
                return True
            except sqlite3.OperationalError:
                return False

    # ------------------------------------------------------------------ write
    def add(
        self,
        content: str,
        kind: str = "long_term",
        importance: Optional[float] = None,
        source: str = "",
        session_id: Optional[str] = None,
        meta: Optional[dict] = None,
        summary: Optional[str] = None,
    ) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind: {kind!r}")
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if importance is None:
            importance = self._estimate_importance(content, kind)
        importance = float(max(0.0, min(1.0, importance)))
        if summary is None:
            summary = self._make_summary(content)
        embedding = self.embedder.to_bytes(self.embedder.embed(content))
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO memories
                   (kind, content, summary, importance, source, session_id, meta,
                    created_at, last_accessed, access_count, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    kind,
                    content,
                    summary,
                    importance,
                    source,
                    session_id,
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                    now,
                    embedding,
                ),
            )
            return int(cursor.lastrowid)

    def add_event(
        self, session_id: str, role: str, content: str, meta: Optional[dict] = None
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events(session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(meta or {}, ensure_ascii=False), time.time()),
            )
            return int(cursor.lastrowid)

    def set_preference(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO preferences(key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, value, time.time()),
            )

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def preferences(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM preferences ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def kv_set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO kv(key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, value, time.time()),
            )

    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def delete(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cursor.rowcount > 0

    def record_reasoning(self, payload: dict) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO reasoning_runs(question, answer, verified, confidence, payload, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    payload.get("question", ""),
                    payload.get("answer", ""),
                    int(bool(payload.get("verified"))),
                    float(payload.get("confidence", 0.0)),
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------- read
    def get(self, memory_id: int) -> Optional[MemoryHit]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._to_hit(row) if row else None

    def search(
        self,
        query: str,
        limit: int = 8,
        kinds: Optional[Sequence[str]] = None,
        min_score: float = 0.0,
        now: Optional[float] = None,
    ) -> List[MemoryHit]:
        now = now or time.time()
        query = query.strip()
        query_embedding = self.embedder.embed(query)
        rows: List[sqlite3.Row] = []
        ranks: Dict[int, float] = {}

        kind_clause = ""
        kind_params: List[str] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            kind_clause = f" AND m.kind IN ({placeholders})"
            kind_params = list(kinds)

        if self.fts and query:
            tokens = re.findall(r"\w+", query, re.UNICODE)
            if tokens:
                fts_query = " OR ".join(f'"{token}"' for token in tokens)
                sql = (
                    "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts "
                    "JOIN memories m ON m.id = memories_fts.rowid "
                    f"WHERE memories_fts MATCH ?{kind_clause} ORDER BY rank LIMIT ?"
                )
                with self._connect() as conn:
                    rows = conn.execute(
                        sql, [fts_query, *kind_params, max(limit * 4, 20)]
                    ).fetchall()
                for row in rows:
                    ranks[int(row["id"])] = -float(row["rank"])

        if not rows:
            sql = f"SELECT m.*, NULL AS rank FROM memories m WHERE 1=1{kind_clause} ORDER BY importance DESC, created_at DESC LIMIT ?"
            with self._connect() as conn:
                rows = conn.execute(sql, [*kind_params, max(limit * 4, 20)]).fetchall()

        if not rows:
            return []

        lexical_values = list(ranks.values())
        lex_min = min(lexical_values) if lexical_values else 0.0
        lex_max = max(lexical_values) if lexical_values else 0.0
        hits: List[MemoryHit] = []
        for row in rows:
            hit = self._to_hit(row)
            memory_id = hit.id
            if ranks and lex_max > lex_min:
                hit.lexical = (ranks.get(memory_id, lex_min) - lex_min) / (lex_max - lex_min)
            elif ranks:
                hit.lexical = 1.0 if memory_id in ranks else 0.0
            age = max(0.0, now - hit.created_at)
            half_life = KIND_HALF_LIFE_DAYS.get(hit.kind, 30) * 86400
            hit.recency = recency_score(age, half_life)
            embedding = HashingEmbedder.from_bytes(row["embedding"] or b"", self.embedder.dim)
            hit.semantic = max(0.0, cosine(query_embedding, embedding))
            hit.score = (
                self.weights["lexical"] * hit.lexical
                + self.weights["recency"] * hit.recency
                + self.weights["importance"] * hit.importance
                + self.weights["semantic"] * hit.semantic
            )
            hits.append(hit)

        hits.sort(key=lambda item: item.score, reverse=True)
        hits = [hit for hit in hits if hit.score >= min_score][:limit]
        if hits:
            with self._connect() as conn:
                conn.executemany(
                    "UPDATE memories SET last_accessed=?, access_count=access_count+1 WHERE id=?",
                    [(now, hit.id) for hit in hits],
                )
        return hits

    def recent(self, kind: Optional[str] = None, limit: int = 10) -> List[MemoryHit]:
        sql = "SELECT * FROM memories"
        params: List[object] = []
        if kind:
            sql += " WHERE kind=?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._to_hit(row) for row in rows]

    def recent_events(self, session_id: str, limit: int = 20) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in reversed(rows)
        ]

    def sessions(self, limit: int = 20) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT session_id, COUNT(*) AS turns, MAX(created_at) AS last_at
                   FROM events GROUP BY session_id ORDER BY last_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reasoning_runs(self, limit: int = 50) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM reasoning_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        runs = []
        for row in rows:
            payload = json.loads(row["payload"] or "{}")
            payload.setdefault("created_at", row["created_at"])
            runs.append(payload)
        return runs

    def working_memory(
        self, session_id: str, limit: int = 12, ttl_minutes: int = 120
    ) -> List[dict]:
        cutoff = time.time() - ttl_minutes * 60
        events = [event for event in self.recent_events(session_id, limit=limit) if event["created_at"] >= cutoff]
        return events

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
            by_kind = {
                row["kind"]: row["c"]
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS c FROM memories GROUP BY kind"
                ).fetchall()
            }
            events = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
            reasoning = conn.execute("SELECT COUNT(*) AS c FROM reasoning_runs").fetchone()["c"]
            initiatives = conn.execute("SELECT COUNT(*) AS c FROM initiatives").fetchone()["c"]
        return {
            "memories": total,
            "by_kind": by_kind,
            "events": events,
            "reasoning_runs": reasoning,
            "initiatives": initiatives,
            "fts": self.fts,
        }

    # ------------------------------------------------------------- maintenance
    def consolidate(
        self,
        session_id: Optional[str] = None,
        min_events: int = 8,
        max_events: int = 200,
    ) -> dict:
        key = f"last_consolidation:{session_id or 'global'}"
        last = float(self.kv_get(key, "0") or 0)
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM events WHERE session_id=? AND created_at>? ORDER BY id LIMIT ?",
                    (session_id, last, max_events),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM events WHERE created_at>? ORDER BY id LIMIT ?",
                    (last, max_events),
                ).fetchall()
        if len(rows) < min_events:
            return {"consolidated": 0, "events": len(rows), "reason": "not enough new events"}

        texts = [row["content"] for row in rows if row["role"] == "user"]
        summary = extractive_summary(texts)
        created = 0
        if summary:
            self.add(
                content=summary,
                kind="semantic",
                importance=0.75,
                source="consolidation",
                session_id=session_id,
                meta={"method": "extractive", "events": len(rows)},
            )
            created += 1
        preferences = extract_preferences(texts)
        for preference in dedupe_texts(preferences):
            self.add(
                content=preference,
                kind="preference",
                importance=0.9,
                source="consolidation",
                session_id=session_id,
                meta={"method": "pattern"},
            )
            created += 1
        latest = max(row["created_at"] for row in rows)
        self.kv_set(key, str(latest))
        return {"consolidated": created, "events": len(rows), "summary": summary[:300]}

    def prune_working(self, ttl_minutes: int = 120) -> int:
        cutoff = time.time() - ttl_minutes * 60
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE kind='working' AND created_at<?", (cutoff,)
            )
            return cursor.rowcount

    # ---------------------------------------------------------------- helpers
    def _to_hit(self, row: sqlite3.Row) -> MemoryHit:
        return MemoryHit(
            id=int(row["id"]),
            kind=row["kind"],
            content=row["content"],
            summary=row["summary"],
            importance=float(row["importance"]),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _make_summary(content: str) -> str:
        first = re.split(r"(?<=[.!?])\s+", content.strip(), maxsplit=1)[0]
        return (first or content)[:240]

    @staticmethod
    def _estimate_importance(content: str, kind: str) -> float:
        score = 0.5
        if kind in {"preference", "project"}:
            score += 0.2
        if len(content) > 200:
            score += 0.1
        if PREFERENCE_HINTS.search(content):
            score += 0.2
        return max(0.0, min(1.0, score))
