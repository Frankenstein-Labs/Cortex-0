"""Autonomous initiative engine.

The engine proposes messages only when they pass explicit usefulness gates:
relevance to a project, novelty against past messages, a minimum score, a
cooldown, a daily cap and quiet hours. It returns ``None`` when nothing is
worth saying, which is the common case by design.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from cortex.memory.store import MemoryStore, tokenize


@dataclass
class InitiativeConfig:
    enabled: bool = True
    min_interval_sec: float = 900.0
    max_per_day: int = 4
    quiet_hours_start: int = 22
    quiet_hours_end: int = 8
    min_score: float = 0.55
    min_novelty: float = 0.35
    lookback_messages: int = 30

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Candidate:
    kind: str
    text: str
    relevance: float = 0.0
    novelty: float = 0.0
    specificity: float = 0.0

    @property
    def score(self) -> float:
        return round(0.45 * self.relevance + 0.35 * self.novelty + 0.20 * self.specificity, 4)


@dataclass
class InitiativeMessage:
    kind: str
    text: str
    score: float
    created_at: float
    session_id: str = "default"
    delivered: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "text": self.text,
            "score": self.score,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "delivered": self.delivered,
        }


@dataclass
class Decision:
    message: InitiativeMessage | None
    reason: str
    considered: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "message": self.message.as_dict() if self.message else None,
            "reason": self.reason,
            "considered": self.considered,
        }


class Notifier:
    def send(self, message: InitiativeMessage) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    def send(self, message: InitiativeMessage) -> bool:
        print(f"[CORTEX initiative] {message.text}")
        return True


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.messages: list[InitiativeMessage] = []

    def send(self, message: InitiativeMessage) -> bool:
        self.messages.append(message)
        return True


class WebhookNotifier(Notifier):
    """Optional push notification. Disabled unless an explicit URL is provided."""

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("webhook URL must be http(s)")
        self.url = url
        self.timeout = timeout

    def send(self, message: InitiativeMessage) -> bool:
        import json
        import urllib.request

        payload = json.dumps(message.as_dict()).encode("utf-8")
        req = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


class InitiativeEngine:
    def __init__(
        self,
        memory: MemoryStore,
        tasks=None,
        config: InitiativeConfig | None = None,
        notifier: Notifier | None = None,
        model=None,
        tokenizer=None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.memory = memory
        self.tasks = tasks
        self.config = config or InitiativeConfig()
        self.notifier = notifier or ConsoleNotifier()
        self.model = model
        self.tokenizer = tokenizer
        self.clock = clock

    # --------------------------------------------------------------- utilities
    def _quiet(self, ts: float) -> bool:
        hour = datetime.fromtimestamp(ts).hour
        start, end = self.config.quiet_hours_start, self.config.quiet_hours_end
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _novelty(self, text: str, history: Sequence[dict]) -> float:
        terms = set(tokenize(text))
        if not terms:
            return 0.0
        if not history:
            return 1.0
        best_overlap = 0.0
        for item in history:
            other = set(tokenize(item["content"]))
            if not other:
                continue
            best_overlap = max(best_overlap, len(terms & other) / len(terms | other))
        return round(1.0 - best_overlap, 4)

    def _relevance(self, text: str, goals: list[str]) -> float:
        if not goals:
            return 0.5
        terms = set(tokenize(text))
        goal_terms: set[str] = set()
        for goal in goals:
            goal_terms |= set(tokenize(goal))
        if not terms or not goal_terms:
            return 0.0
        overlap = len(terms & goal_terms) / math.sqrt(len(terms) * len(goal_terms))
        return round(min(1.0, overlap * 2.0), 4)

    # -------------------------------------------------------------- candidates
    def _candidates(self, session_id: str) -> list[Candidate]:
        goals = [p["goal"] for p in self.memory.projects() if p.get("goal")]
        out: list[Candidate] = []

        if self.tasks is not None:
            for task in self.tasks.open_tasks()[:3]:
                nxt = task.next_step()
                if nxt:
                    text = (
                        f"Mission « {task.title} » : prochaine étape — {nxt.description}. "
                        f"Progression {task.progress:.0%}. On continue ?"
                    )
                    out.append(Candidate("task", text))

        questions = [
            ep.content.strip()
            for ep in self.memory.recent_episodes(session_id, limit=40)
            if ep.content.strip().endswith("?") and len(ep.content.strip()) > 20
        ]
        for question in questions[-2:]:
            out.append(Candidate("question", f"Une question reste ouverte : {question}"))

        projects = self.memory.projects()
        if projects:
            p = projects[0]
            out.append(
                Candidate(
                    "idea",
                    f"Idée pour « {p['name']} » : écrire un test qui mesure la prochaine amélioration "
                    "avant de coder la fonctionnalité.",
                )
            )

        facts = self.memory.get_facts(limit=5)
        if facts:
            fact = facts[0]
            out.append(Candidate("fact", f"À propos de « {fact.subject} » : {fact.content[:160]}"))

        if self.model is not None and self.tokenizer is not None:
            out.extend(self._model_ideas(session_id, goals))

        scored: list[Candidate] = []
        history = self.memory.recent_initiatives(self.config.lookback_messages, session_id=session_id)
        for cand in out:
            cand.relevance = self._relevance(cand.text, goals)
            cand.novelty = self._novelty(cand.text, history)
            words = tokenize(cand.text)
            cand.specificity = min(1.0, len(set(words)) / 12.0)
            scored.append(cand)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored

    def _model_ideas(self, session_id: str, goals: list[str]) -> list[Candidate]:
        from cortex.generation.sampler import GenerationConfig, generate_text

        goal = goals[0] if goals else "un petit modèle de langage autonome"
        prompt = (
            f"Projet : {goal}\n"
            "Propose une idée concrète et vérifiable pour faire progresser le projet en une phrase :"
        )
        ids = self.tokenizer.encode(
            self.tokenizer.chat_prompt([{"role": "user", "content": prompt}], add_generation_prompt=True)
        )
        try:
            text, _ = generate_text(
                self.model,
                self.tokenizer,
                ids,
                GenerationConfig(max_new_tokens=48, temperature=0.9, top_k=50, top_p=0.95, seed=None),
            )
        except Exception:
            return []
        text = " ".join(text.split()).strip()
        if len(text) < 25 or len(set(tokenize(text))) < 4:
            return []
        return [Candidate("model_idea", text)]

    # -------------------------------------------------------------------- tick
    def tick(self, session_id: str = "default", now: float | None = None) -> Decision:
        now = self.clock() if now is None else now
        cfg = self.config
        if not cfg.enabled:
            return Decision(None, "initiative disabled")
        if self._quiet(now):
            return Decision(None, "quiet hours")

        recent = self.memory.recent_initiatives(1, session_id=session_id)
        if recent and now - recent[-1]["created_at"] < cfg.min_interval_sec:
            return Decision(None, "cooldown")

        day_start = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        if self.memory.count_initiatives_since(day_start, session_id=session_id) >= cfg.max_per_day:
            return Decision(None, "daily cap reached")

        candidates = self._candidates(session_id)
        considered = [{"kind": c.kind, "score": c.score, "text": c.text} for c in candidates[:5]]
        if not candidates:
            return Decision(None, "no candidate", considered)

        best = candidates[0]
        if best.novelty < cfg.min_novelty:
            return Decision(None, f"not novel enough ({best.novelty:.2f})", considered)
        if best.score < cfg.min_score:
            return Decision(None, f"score below threshold ({best.score:.2f})", considered)

        message = InitiativeMessage(best.kind, best.text, best.score, now, session_id)
        message.delivered = bool(self.notifier.send(message))
        self.memory.add_initiative(
            best.kind, best.text, best.score, session_id=session_id, delivered=message.delivered, ts=now
        )
        return Decision(message, "delivered" if message.delivered else "delivery failed", considered)

    def history(self, session_id: str = "default", limit: int = 20) -> list[dict]:
        return self.memory.recent_initiatives(limit, session_id=session_id)

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled
        self.memory.set_preference("initiative.enabled", "1" if enabled else "0")
