"""New York-time daily simulation quota manager with hourly pacing."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol
from zoneinfo import ZoneInfo

from alpha_factory.storage.models import DailyQuotaStatus

if TYPE_CHECKING:
    from alpha_factory.storage.db import AlphaDatabase  # noqa: F401

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_PACE_BUFFER = 1.25


class QuotaDatabaseProtocol(Protocol):
    """Duck-typed storage interface required by DailyQuotaManager."""

    def get_daily_quota(self, date_ny: str) -> Any: ...

    def increment_daily_quota_submitted(
        self, date_ny: str, count: int, warning_threshold: int
    ) -> Any: ...

    def increment_daily_quota_outcome(
        self, date_ny: str, completed: int, failed: int
    ) -> Any: ...


class DailyQuotaManager:
    """Enforces BRAIN's 5,000 simulations/day NY-midnight quota and hourly pacing."""

    def __init__(
        self,
        db: QuotaDatabaseProtocol | Any,
        daily_sim_limit: int = 5000,
        warning_threshold: int = 4000,
        hourly_pacing: bool = True,
        pace_buffer: float = DEFAULT_PACE_BUFFER,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db = db
        self.daily_sim_limit = int(daily_sim_limit)
        self.warning_threshold = int(warning_threshold)
        self.hourly_pacing = bool(hourly_pacing)
        self.pace_buffer = float(pace_buffer)
        self._now_fn = now_fn
        self._lock = threading.RLock()
        self._hourly_counts: dict[tuple[str, int], int] = defaultdict(int)
        self._warned_dates: set[str] = set()

    @classmethod
    def from_settings(
        cls,
        db: QuotaDatabaseProtocol | Any,
        settings: Mapping[str, Any],
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> DailyQuotaManager:
        """Create a DailyQuotaManager from loaded settings or its ``execution`` section."""
        exec_cfg = settings.get("execution", settings)
        if not isinstance(exec_cfg, Mapping):
            exec_cfg = {}
        return cls(
            db=db,
            daily_sim_limit=int(exec_cfg.get("daily_sim_limit", 5000)),
            warning_threshold=int(exec_cfg.get("warning_threshold", 4000)),
            hourly_pacing=bool(exec_cfg.get("hourly_pacing", True)),
            now_fn=now_fn,
        )

    def _now_ny(self) -> datetime:
        now = self._now_fn() if self._now_fn is not None else datetime.now(NY_TZ)
        if now.tzinfo is None:
            return now.replace(tzinfo=NY_TZ)
        return now.astimezone(NY_TZ)

    def get_ny_date(self) -> str:
        """Return current calendar date in ``America/New_York`` as ``YYYY-MM-DD``."""
        return self._now_ny().strftime("%Y-%m-%d")

    def seconds_until_ny_midnight(self) -> float:
        """Return exact seconds until the next ``America/New_York`` midnight (DST-aware)."""
        now_ny = self._now_ny()
        next_day = now_ny.date() + timedelta(days=1)
        next_midnight = datetime.combine(next_day, time(0, 0), tzinfo=NY_TZ)
        delta = next_midnight.astimezone(timezone.utc) - now_ny.astimezone(timezone.utc)
        return max(0.0, delta.total_seconds())

    def _normalize_status(self, date_ny: str, raw: Any) -> DailyQuotaStatus:
        if raw is None:
            submitted = 0
            completed = 0
            failed = 0
            stored_warning = False
        elif isinstance(raw, DailyQuotaStatus):
            submitted = int(raw.submitted_count)
            completed = int(raw.completed_count)
            failed = int(raw.failed_count)
            stored_warning = bool(raw.warning_triggered)
        elif isinstance(raw, Mapping):
            submitted = int(raw.get("submitted_count", 0))
            completed = int(raw.get("completed_count", 0))
            failed = int(raw.get("failed_count", 0))
            stored_warning = bool(raw.get("warning_triggered", False))
        else:
            submitted = int(getattr(raw, "submitted_count", 0))
            completed = int(getattr(raw, "completed_count", 0))
            failed = int(getattr(raw, "failed_count", 0))
            stored_warning = bool(getattr(raw, "warning_triggered", False))

        remaining = max(0, self.daily_sim_limit - submitted)
        warning_triggered = stored_warning or (submitted >= self.warning_threshold)
        can_submit = submitted < self.daily_sim_limit
        return DailyQuotaStatus(
            date_ny=date_ny,
            submitted_count=submitted,
            completed_count=completed,
            failed_count=failed,
            limit=self.daily_sim_limit,
            remaining=remaining,
            warning_triggered=warning_triggered,
            can_submit=can_submit,
        )

    def get_status(self) -> DailyQuotaStatus:
        """Return current NY-date simulation quota status."""
        with self._lock:
            date_ny = self.get_ny_date()
            raw = self.db.get_daily_quota(date_ny)
            return self._normalize_status(date_ny, raw)

    def can_submit(self, n: int = 1) -> bool:
        """Return True if submitting ``n`` simulations will stay within daily_sim_limit."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        with self._lock:
            status = self.get_status()
            if n == 0:
                return status.can_submit
            return (status.submitted_count + n) <= self.daily_sim_limit

    def record_submitted(self, n: int) -> DailyQuotaStatus:
        """Record ``n`` simulations after HTTP 201 acceptance and return updated status."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        with self._lock:
            now_ny = self._now_ny()
            date_ny = now_ny.strftime("%Y-%m-%d")
            if n == 0:
                return self._normalize_status(date_ny, self.db.get_daily_quota(date_ny))

            raw = self.db.increment_daily_quota_submitted(
                date_ny, n, self.warning_threshold
            )
            stale_keys = [key for key in self._hourly_counts if key[0] != date_ny]
            for key in stale_keys:
                del self._hourly_counts[key]
            self._hourly_counts[(date_ny, now_ny.hour)] += n

            if raw is None:
                raw = self.db.get_daily_quota(date_ny)
            status = self._normalize_status(date_ny, raw)

            if status.warning_triggered and date_ny not in self._warned_dates:
                self._warned_dates.add(date_ny)
                logger.warning(
                    "Daily BRAIN simulation quota warning for %s: %d/%d submitted (%d remaining)",
                    date_ny,
                    status.submitted_count,
                    status.limit,
                    status.remaining,
                )
            return status

    def record_outcome(self, completed: int = 0, failed: int = 0) -> None:
        """Record terminal simulation outcomes for the current NY date."""
        if completed < 0 or failed < 0:
            raise ValueError(
                f"completed and failed must be non-negative, got completed={completed}, failed={failed}"
            )
        if completed == 0 and failed == 0:
            return
        with self._lock:
            date_ny = self.get_ny_date()
            self.db.increment_daily_quota_outcome(date_ny, completed, failed)

    def get_current_hour_submitted(self) -> int:
        """Return submissions recorded during the current NY wall-clock hour."""
        with self._lock:
            now_ny = self._now_ny()
            return self._hourly_counts.get((now_ny.strftime("%Y-%m-%d"), now_ny.hour), 0)

    def should_pace(self) -> bool:
        """Return True when hourly_pacing is enabled and current-hour rate exceeds pace."""
        if not self.hourly_pacing:
            return False
        with self._lock:
            now_ny = self._now_ny()
            date_ny = now_ny.strftime("%Y-%m-%d")
            status = self._normalize_status(date_ny, self.db.get_daily_quota(date_ny))
            remaining_hours = max(1.0, self.seconds_until_ny_midnight() / 3600.0)
            pace = (status.remaining / remaining_hours) * self.pace_buffer
            current_hour_submitted = self._hourly_counts.get((date_ny, now_ny.hour), 0)
            return current_hour_submitted > pace


__all__ = [
    "DEFAULT_PACE_BUFFER",
    "DailyQuotaManager",
    "NY_TZ",
    "QuotaDatabaseProtocol",
]
