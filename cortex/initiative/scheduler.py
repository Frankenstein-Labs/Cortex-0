"""Persistent scheduler that drives the initiative engine."""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

from ..memory.store import MemoryStore
from .engine import Initiative, InitiativeEngine


class InitiativeScheduler:
    def __init__(
        self,
        engine: InitiativeEngine,
        store: MemoryStore,
        interval_seconds: float = 60.0,
        notifier: Optional[Callable[[Initiative], None]] = None,
        session_id: Optional[str] = None,
        user_activity_provider: Optional[Callable[[], Optional[float]]] = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.engine = engine
        self.store = store
        self.interval_seconds = interval_seconds
        self.notifier = notifier
        self.session_id = session_id
        self.user_activity_provider = user_activity_provider
        self.clock = clock
        self.sleeper = sleeper
        self._stop = threading.Event()

    def run_once(self, now: Optional[float] = None) -> List[Initiative]:
        now = now or self.clock()
        last_tick = float(self.store.kv_get("scheduler:last_tick", "0") or 0)
        if now - last_tick < self.interval_seconds:
            return []
        self.store.kv_set("scheduler:last_tick", str(now))
        user_last_active = (
            self.user_activity_provider() if self.user_activity_provider else None
        )
        initiatives = self.engine.tick(
            session_id=self.session_id,
            now=now,
            user_last_active=user_last_active,
        )
        if self.notifier:
            for initiative in initiatives:
                self.notifier(initiative)
                self.engine.mark_sent(initiative.id, now=now)
        return initiatives

    def run_forever(
        self,
        stop_event: Optional[threading.Event] = None,
        max_iterations: Optional[int] = None,
    ) -> int:
        stop_event = stop_event or self._stop
        ticks = 0
        while not stop_event.is_set():
            self.run_once()
            ticks += 1
            if max_iterations is not None and ticks >= max_iterations:
                break
            self.sleeper(self.interval_seconds)
        return ticks

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="cortex-scheduler", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
