"""Multi-tier memory for CORTEX-0.

Tiers are deliberately separated:

* working memory  - in-process, bounded by a token budget
* episodic memory - conversation turns and events (SQLite)
* semantic memory - facts and consolidated knowledge (SQLite)
* preference memory - user settings (SQLite)

This is *external* memory. It does not modify the model weights; anything that
should become model knowledge must go through a controlled fine-tuning run.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëîïôöùûüç']+", re.IGNORECASE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "i", "you", "he", "she",
    "we", "they", "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "en", "est",
    "sont", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "que", "qui", "à",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS and len(t) > 1]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used for context budgeting (chars / 4)."""
    return max(1, math.ceil(len(text) / 4))


@dataclass
class Episode:
    id: int
    session_id: str
    role: str
    content: str
    ts: float
    importance: float = 0.5
    tags: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
            "importance": self.importance,
            "tags": self.tags,
        }


@dataclass
class Fact:
    id: int
    subject: str
    content: str
    source: str
    confidence: float
    ts: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "ts": self.ts,
        }


class WorkingMemory:
    """Bounded, token-budgeted scratchpad for the current turn."""

    def __init__(self, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self._items: deque[tuple[str, int]] = deque()
        self._tokens = 0

    def add(self, text: str) -> None:
        cost = estimate_tokens(text)
        self._items.append((text, cost))
        self._tokens += cost
        while self._tokens > self.max_tokens and self._items:
            _, removed = self._items.popleft()
            self._tokens -= removed

    def render(self) -> str:
        return "\n".join(text for text, _ in self._items)

    def clear(self) -> None:
        self._items.clear()
        self._tokens = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def tokens(self) -> int:
        return self._tokens


class MemoryStore:
    """SQLite-backed episodic, semantic and preference memory."""

    def __init__(self, path: str | Path = "cortex_memory.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.working = WorkingMemory()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                tags TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id, ts);
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.7,
                ts REAL NOT NULL,
                UNIQUE(subject, content)
            );
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                goal TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                notes TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS initiatives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                score REAL NOT NULL,
                created_at REAL NOT NULL,
                delivered INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_initiatives_time ON initiatives(created_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ episodes
    def add_episode(
        self,
        session_id: str,
        role: str,
        content: str,
        importance: float = 0.5,
        tags: str = "",
        ts: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO episodes(session_id, role, content, ts, importance, tags) VALUES (?,?,?,?,?,?)",
            (session_id, role, content, ts if ts is not None else time.time(), importance, tags),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent_episodes(self, session_id: str | None = None, limit: int = 20) -> list[Episode]:
        if session_id is None:
            rows = self._conn.execute("SELECT * FROM episodes ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE session_id=? ORDER BY ts DESC LIMIT ?", (session_id, limit)
            ).fetchall()
        return [Episode(**dict(r)) for r in rows][::-1]

    def search_episodes(self, query: str, limit: int = 8, now: float | None = None) -> list[Episode]:
        """Rank episodes with TF-IDF-ish overlap plus a mild recency bonus."""
        q_terms = Counter(tokenize(query))
        if not q_terms:
            return self.recent_episodes(limit=limit)
        rows = [Episode(**dict(r)) for r in self._conn.execute("SELECT * FROM episodes").fetchall()]
        if not rows:
            return []
        doc_terms = [Counter(tokenize(r.content)) for r in rows]
        n_docs = len(rows)
        df = Counter()
        for counts in doc_terms:
            df.update(counts.keys())
        now = now if now is not None else time.time()
        scored: list[tuple[float, Episode]] = []
        for ep, counts in zip(rows, doc_terms):
            length = max(1, sum(counts.values()))
            score = 0.0
            for term, qf in q_terms.items():
                tf = counts.get(term, 0)
                if tf:
                    idf = math.log(1 + n_docs / (1 + df[term]))
                    score += (1 + math.log(tf)) * idf * (1 + 0.1 * qf) / math.sqrt(length)
            if score > 0:
                age_days = max(0.0, (now - ep.ts) / 86400)
                score *= 0.5 ** (age_days / 30)  # 30-day half-life
                score *= 0.5 + ep.importance
                scored.append((score, ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    # ---------------------------------------------------------------------- facts
    def add_fact(
        self, subject: str, content: str, source: str = "", confidence: float = 0.7, ts: float | None = None
    ) -> int:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO facts(subject, content, source, confidence, ts) VALUES (?,?,?,?,?)",
            (subject, content, source, confidence, ts if ts is not None else time.time()),
        )
        self._conn.commit()
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = self._conn.execute(
            "SELECT id FROM facts WHERE subject=? AND content=?", (subject, content)
        ).fetchone()
        return int(row["id"])

    def get_facts(self, subject: str | None = None, limit: int = 50) -> list[Fact]:
        if subject is None:
            rows = self._conn.execute("SELECT * FROM facts ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE subject=? ORDER BY ts DESC LIMIT ?", (subject, limit)
            ).fetchall()
        return [Fact(**dict(r)) for r in rows]

    # ---------------------------------------------------------------- preferences
    def set_preference(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO preferences(key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, time.time()),
        )
        self._conn.commit()

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def all_preferences(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # -------------------------------------------------------------------- projects
    def upsert_project(self, name: str, goal: str = "", status: str = "active", notes: str = "") -> None:
        self._conn.execute(
            "INSERT INTO projects(name, goal, status, notes, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET goal=excluded.goal, status=excluded.status, "
            "notes=excluded.notes, updated_at=excluded.updated_at",
            (name, goal, status, notes, time.time()),
        )
        self._conn.commit()

    def projects(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- initiatives
    def add_initiative(self, kind: str, content: str, score: float, session_id: str = "default", delivered: bool = True, ts: float | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO initiatives(session_id, kind, content, score, created_at, delivered) VALUES (?,?,?,?,?,?)",
            (session_id, kind, content, score, ts if ts is not None else time.time(), 1 if delivered else 0),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def recent_initiatives(self, limit: int = 20, session_id: str | None = None) -> list[dict]:
        if session_id:
            rows = self._conn.execute(
                "SELECT * FROM initiatives WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM initiatives ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows][::-1]

    def count_initiatives_since(self, since_ts: float, session_id: str | None = None) -> int:
        if session_id:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM initiatives WHERE created_at>=? AND session_id=? AND delivered=1",
                (since_ts, session_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM initiatives WHERE created_at>=? AND delivered=1", (since_ts,)
            ).fetchone()
        return int(row["c"])

    # ---------------------------------------------------------------- consolidation
    def summarize_session(self, session_id: str, max_items: int = 6) -> str:
        """Extractive summary: the most important, least redundant turns."""
        eps = self.recent_episodes(session_id, limit=200)
        if not eps:
            return ""
        scored: list[tuple[float, Episode]] = []
        seen_terms: Counter = Counter()
        for ep in eps:
            terms = set(tokenize(ep.content))
            if not terms:
                continue
            novelty = 1.0 - (len(terms & set(seen_terms)) / len(terms))
            score = ep.importance * (0.3 + novelty) * math.log1p(len(terms))
            scored.append((score, ep))
            seen_terms.update(terms)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = sorted(scored[:max_items], key=lambda pair: pair[1].ts)
        return "\n".join(f"{ep.role}: {ep.content[:240]}" for _, ep in chosen)

    def consolidate(self, min_mentions: int = 3) -> list[Fact]:
        """Promote repeated episodic statements into semantic facts.

        A statement is only consolidated when the same normalised sentence
        appears at least ``min_mentions`` times. No model is involved.
        """
        rows = self._conn.execute("SELECT role, content FROM episodes").fetchall()
        counts: Counter = Counter()
        for row in rows:
            for sentence in re.split(r"(?<=[.!?])\s+", row["content"]):
                norm = " ".join(tokenize(sentence))
                if len(norm.split()) >= 4:
                    counts[norm] += 1
        promoted: list[Fact] = []
        for norm, count in counts.items():
            if count >= min_mentions:
                fact_id = self.add_fact(
                    subject="consolidated", content=norm, source="episodic-consolidation", confidence=min(0.9, 0.4 + 0.1 * count)
                )
                promoted.append(
                    Fact(id=fact_id, subject="consolidated", content=norm, source="episodic-consolidation",
                         confidence=min(0.9, 0.4 + 0.1 * count), ts=time.time())
                )
        return promoted

    def stats(self) -> dict:
        episodes = self._conn.execute("SELECT COUNT(*) AS c FROM episodes").fetchone()["c"]
        facts = self._conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
        prefs = self._conn.execute("SELECT COUNT(*) AS c FROM preferences").fetchone()["c"]
        projects = self._conn.execute("SELECT COUNT(*) AS c FROM projects").fetchone()["c"]
        by_role = {
            r["role"]: r["c"]
            for r in self._conn.execute("SELECT role, COUNT(*) AS c FROM episodes GROUP BY role").fetchall()
        }
        return {
            "episodes": episodes,
            "facts": facts,
            "preferences": prefs,
            "projects": projects,
            "episodes_by_role": by_role,
            "working_memory_items": len(self.working),
            "working_memory_tokens": self.working.tokens,
            "initiatives": self._conn.execute("SELECT COUNT(*) AS c FROM initiatives").fetchone()["c"],
        }

    def close(self) -> None:
        self._conn.close()
