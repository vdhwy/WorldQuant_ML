"""Thread-safe atomic slot pool for concurrent WorldQuant BRAIN simulations."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

SLOT_IDLE = "IDLE"
SLOT_RUNNING = "RUNNING"

_DEFAULT_REGION_SLOTS: dict[str, int] = {
    "GLB": 2,
}


@dataclass
class _SlotState:
    """Mutable internal state for a single simulation execution slot."""

    slot_id: int
    status: str = SLOT_IDLE
    job_id: Optional[str] = None
    region: Optional[str] = None
    acquired_at: Optional[float] = None


class SlotPool:
    """Atomic simulation slot allocator backed by a single ``threading.Condition``.

    Supports multi-slot regions (e.g. ``GLB`` requiring 2 slots) without requiring
    slots to be contiguous, eliminating fragmented-slot starvation.
    """

    def __init__(
        self,
        max_slots: int = 3,
        region_slots: Optional[Mapping[str, int]] = None,
    ) -> None:
        if int(max_slots) < 1:
            raise ValueError(f"max_slots must be >= 1, got {max_slots}")
        self.max_slots = int(max_slots)

        resolved_region_slots: dict[str, int] = dict(_DEFAULT_REGION_SLOTS)
        if region_slots is not None:
            for region, count in region_slots.items():
                key = str(region).strip().upper()
                slots_needed = int(count)
                if slots_needed < 1:
                    raise ValueError(
                        f"slots required for region {region!r} must be >= 1, got {count}"
                    )
                if key:
                    resolved_region_slots[key] = slots_needed
        self.region_slots: dict[str, int] = resolved_region_slots

        self._cond = threading.Condition(threading.Lock())
        self._slots: list[_SlotState] = [_SlotState(slot_id=i) for i in range(self.max_slots)]

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> SlotPool:
        """Construct a ``SlotPool`` from loaded settings YAML."""
        exec_cfg = settings.get("execution", settings)
        if not isinstance(exec_cfg, Mapping):
            exec_cfg = {}
        max_slots = int(exec_cfg.get("max_slots", 3))

        region_slots: dict[str, int] = dict(_DEFAULT_REGION_SLOTS)
        markets_cfg = settings.get("markets")
        if isinstance(markets_cfg, Mapping):
            for region, mcfg in markets_cfg.items():
                if isinstance(mcfg, Mapping) and "slots_required" in mcfg:
                    region_slots[str(region).strip().upper()] = int(mcfg["slots_required"])

        return cls(max_slots=max_slots, region_slots=region_slots)

    def slots_required(self, region: str) -> int:
        """Return the number of slots required to run a job in ``region``."""
        key = str(region).strip().upper()
        return self.region_slots.get(key, 1)

    def can_acquire(self, region: str) -> bool:
        """Return True if enough idle slots exist right now to accommodate ``region``."""
        needed = self.slots_required(region)
        if needed > self.max_slots:
            return False
        with self._cond:
            idle = sum(1 for slot in self._slots if slot.status == SLOT_IDLE)
            return idle >= needed

    def acquire(
        self,
        region: str,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[list[int]]:
        """Atomically acquire slot(s) for ``region``, blocking up to ``timeout`` seconds.

        ``GLB`` (and any multi-slot region) claims any ``needed`` idle slots regardless
        of slot-index contiguity. Returns the acquired ``slot_id`` list or ``None`` if
        the timeout expires or ``slots_required(region) > max_slots``.
        """
        needed = self.slots_required(region)
        if needed > self.max_slots:
            logger.warning(
                "Region %s requires %d slots, exceeding max_slots=%d",
                region,
                needed,
                self.max_slots,
            )
            return None

        deadline: Optional[float] = None
        if timeout is not None:
            timeout_f = float(timeout)
            deadline = time.monotonic() + max(0.0, timeout_f)

        with self._cond:
            while True:
                idle_slots = [slot for slot in self._slots if slot.status == SLOT_IDLE]
                if len(idle_slots) >= needed:
                    chosen = idle_slots[:needed]
                    now = time.time()
                    for slot in chosen:
                        slot.status = SLOT_RUNNING
                        slot.job_id = str(job_id)
                        slot.region = str(region)
                        slot.acquired_at = now
                    return [slot.slot_id for slot in chosen]

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()

    def release(self, slot_ids: Sequence[int]) -> None:
        """Immediately mark ``slot_ids`` as ``IDLE`` (0s delay) and notify waiters."""
        if not slot_ids:
            return
        with self._cond:
            for raw_id in slot_ids:
                sid = int(raw_id)
                if sid < 0 or sid >= self.max_slots:
                    raise ValueError(f"Invalid slot_id {sid}; expected 0..{self.max_slots - 1}")
                slot = self._slots[sid]
                slot.status = SLOT_IDLE
                slot.job_id = None
                slot.region = None
                slot.acquired_at = None
            self._cond.notify_all()

    def active_count(self) -> int:
        """Return count of slots currently occupied by running jobs."""
        with self._cond:
            return sum(1 for slot in self._slots if slot.status != SLOT_IDLE)

    def idle_count(self) -> int:
        """Return count of currently idle slots."""
        with self._cond:
            return sum(1 for slot in self._slots if slot.status == SLOT_IDLE)

    def get_snapshot(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of all slots for live telemetry and watchdog."""
        with self._cond:
            return [
                {
                    "slot_id": slot.slot_id,
                    "status": slot.status,
                    "job_id": slot.job_id,
                    "region": slot.region,
                    "acquired_at": slot.acquired_at,
                }
                for slot in self._slots
            ]


__all__ = [
    "SLOT_IDLE",
    "SLOT_RUNNING",
    "SlotPool",
]
