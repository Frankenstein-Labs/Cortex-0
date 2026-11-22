"""Proactivity policy: quiet hours, rate limits, idle gating.

The default policy is deliberately conservative.  CORTEX must earn the right
to interrupt; silence is the correct behaviour when there is nothing valuable
to say.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class InitiativePolicy:
    enabled: bool = True
    min_interval_seconds: int = 900
    daily_cap: int = 6
    quiet_hours: Optional[Tuple[int, int]] = (22, 7)
    idle_min_seconds: int = 300
    similarity_threshold: float = 0.82
    min_score: float = 0.55
    max_pending: int = 3
    per_kind_cooldown_seconds: int = 6 * 3600
    stalled_goal_seconds: int = 24 * 3600

    def in_quiet_hours(self, now: Optional[float] = None) -> bool:
        if not self.quiet_hours:
            return False
        now = now or time.time()
        start, end = self.quiet_hours
        if start == end:
            return False
        hour = time.localtime(now).tm_hour
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def check(
        self,
        now: float,
        last_sent_at: Optional[float],
        sent_today: int,
        pending: int,
        user_last_active: Optional[float],
    ) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "disabled"
        if self.in_quiet_hours(now):
            return False, "quiet_hours"
        if user_last_active is not None and now - user_last_active < self.idle_min_seconds:
            return False, "user_active"
        if pending >= self.max_pending:
            return False, "too_many_pending"
        if sent_today >= self.daily_cap:
            return False, "daily_cap"
        if last_sent_at is not None and now - last_sent_at < self.min_interval_seconds:
            return False, "min_interval"
        return True, "allowed"
