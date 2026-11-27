"""Multi-simulation dispatcher with atomic slot acquisition, 400 isolation, and crash recovery."""

from __future__ import annotations

import inspect
import json
import logging
import math
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import requests

from alpha_factory.brain.auth import AuthenticationError, BrainAPIError, RateLimitError
from alpha_factory.brain.client import (
    MAX_MULTI_SIMULATIONS,
    BrainClient,
    PollingTimeoutError,
    SimulationError,
    SimulationValidationError,
    simulation_id_from_url,
)
from alpha_factory.execution.quota import DailyQuotaManager
from alpha_factory.execution.slot_pool import SlotPool
from alpha_factory.generation.validator import ExpressionValidator
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import (
    DEFAULT_SETTINGS,
    CandidateItem,
    SimulationRecord,
    SimulationSettings,
)

logger = logging.getLogger(__name__)

_INDEXED_ERROR_RE = re.compile(r"^\s*\[?(\d+)\]?\s*[.:]")

_SNAKE_TO_CAMEL_SETTING: dict[str, str] = {
    "instrument_type": "instrumentType",
    "test_period": "testPeriod",
    "unit_handling": "unitHandling",
    "nan_handling": "nanHandling",
    "max_trade": "maxTrade",
}


def _opt_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _opt_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _BatchOutcome:
    """Internal outcome summary for a submitted or split batch."""

    completed: int = 0
    failed: int = 0
    rate_limited: bool = False
    auth_error: bool = False
    transport_error: bool = False

    def merge(self, other: _BatchOutcome) -> None:
        self.completed += other.completed
        self.failed += other.failed
        self.rate_limited = self.rate_limited or other.rate_limited
        self.auth_error = self.auth_error or other.auth_error
        self.transport_error = self.transport_error or other.transport_error


class SimulationDispatcher:
    """Schedules pending alpha candidates across BRAIN slots with multi-sim and recovery."""

    def __init__(
        self,
        db: AlphaDB,
        client: BrainClient,
        quota: DailyQuotaManager,
        slot_pool: SlotPool,
        settings: Mapping[str, Any],
        qualification_queue: Optional[queue.Queue[str]] = None,
        validator: Optional[ExpressionValidator] = None,
        alert_callback: Optional[Callable[[str], Any]] = None,
        *,
        poll_interval_s: float = 0.5,
        acquire_timeout_s: float = 0.5,
    ) -> None:
        self.db = db
        self.client = client
        self.quota = quota
        self.slot_pool = slot_pool
        self.settings: dict[str, Any] = dict(settings)
        self.qualification_queue = qualification_queue
        self.validator = validator
        self.alert_callback = alert_callback

        exec_cfg = self.settings.get("execution")
        if not isinstance(exec_cfg, Mapping):
            exec_cfg = {}
        ops_cfg = self.settings.get("ops")
        if not isinstance(ops_cfg, Mapping):
            ops_cfg = {}

        raw_limit = int(exec_cfg.get("multi_sim_limit", MAX_MULTI_SIMULATIONS))
        self.multi_sim_limit = max(1, min(MAX_MULTI_SIMULATIONS, raw_limit))
        self.circuit_breaker_threshold = max(1, int(exec_cfg.get("circuit_breaker_threshold", 3)))
        self.region_cooldown_s = max(0.0, float(exec_cfg.get("region_cooldown_s", 1800)))
        self.shutdown_drain_timeout_s = max(
            0.0, float(ops_cfg.get("shutdown_drain_timeout_s", 300.0))
        )
        self._poll_interval_s = max(0.02, float(poll_interval_s))
        self.acquire_timeout_s = max(0.0, float(acquire_timeout_s))

        self._lock = threading.RLock()
        self._dispatch_lock = threading.Lock()
        self._workers_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._active_workers: set[threading.Thread] = set()

        self._region_cursor = 0
        self._transient_requeue_gen = 0
        self._batch_reservations: dict[str, int] = {}
        self._consecutive_failures: dict[str, int] = {}
        self._region_cooldowns: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Configuration & payload helpers
    # ------------------------------------------------------------------

    def _active_region_markets(self) -> list[tuple[str, list[tuple[str, int]]]]:
        """Return ordered ``[(region, [(universe, delay), ...]), ...]`` from settings."""
        markets_cfg = self.settings.get("markets")
        if not isinstance(markets_cfg, Mapping) or not markets_cfg:
            markets_cfg = DEFAULT_SETTINGS["markets"]

        result: list[tuple[str, list[tuple[str, int]]]] = []
        for raw_region, raw_cfg in markets_cfg.items():
            if not isinstance(raw_cfg, Mapping):
                continue
            if raw_cfg.get("enabled") is False:
                continue
            region = str(raw_region).strip()
            if not region:
                continue

            default_u = str(raw_cfg.get("default_universe") or "").strip()
            raw_universes = raw_cfg.get("universes")
            universes: list[str] = []
            if default_u:
                universes.append(default_u)
            if isinstance(raw_universes, Sequence) and not isinstance(raw_universes, (str, bytes)):
                for entry in raw_universes:
                    u = str(entry).strip()
                    if u and u not in universes:
                        universes.append(u)
            if not universes:
                universes = ["TOP3000"]

            raw_delays = raw_cfg.get("delays")
            if (
                isinstance(raw_delays, Sequence)
                and not isinstance(raw_delays, (str, bytes))
                and raw_delays
            ):
                delays = list(dict.fromkeys(int(d) for d in raw_delays))
            else:
                delays = [int(raw_cfg.get("delay", 1))]

            pairs: list[tuple[str, int]] = [
                (universe, delay) for delay in delays for universe in universes
            ]
            result.append((region, pairs))
        return result

    def _discover_pending_markets_for_region(self, region: str) -> list[tuple[str, int]]:
        """Fallback lookup for pending ``(universe, delay)`` tuples in an active region."""
        if not hasattr(self.db, "_read_conn"):
            return []
        try:
            with self.db._read_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT universe, delay
                    FROM candidate_queue
                    WHERE status = 'PENDING' AND region = ?
                    ORDER BY delay ASC, universe ASC
                    """,
                    (region,),
                ).fetchall()
            return [(str(row["universe"]), int(row["delay"])) for row in rows]
        except Exception:
            logger.debug("Fallback pending market discovery failed for %s", region, exc_info=True)
            return []

    def _build_sim_dict(self, item: CandidateItem) -> dict[str, Any]:
        """Build a single WorldQuant BRAIN ``/simulations`` request dict for ``item``."""
        exec_cfg = self.settings.get("execution")
        exec_defaults = (
            exec_cfg.get("default_settings")
            if isinstance(exec_cfg, Mapping) and isinstance(exec_cfg.get("default_settings"), Mapping)
            else {}
        )
        markets_cfg = self.settings.get("markets")
        mkt_cfg = (
            markets_cfg.get(item.region)
            if isinstance(markets_cfg, Mapping) and isinstance(markets_cfg.get(item.region), Mapping)
            else {}
        )

        region = str(item.region or "USA")
        universe = str(item.universe or mkt_cfg.get("default_universe") or "TOP3000")
        delay = int(item.delay if item.delay is not None else mkt_cfg.get("delay", 1))
        neutralization = str(
            mkt_cfg.get("default_neutralization")
            or exec_defaults.get("neutralization")
            or "INDUSTRY"
        )

        base_settings = SimulationSettings(
            instrument_type=str(
                exec_defaults.get("instrument_type", exec_defaults.get("instrumentType", "EQUITY"))
            ),
            region=region,
            universe=universe,
            delay=delay,
            decay=int(exec_defaults.get("decay", 4)),
            neutralization=neutralization,
            truncation=float(exec_defaults.get("truncation", 0.08)),
            pasteurization=str(exec_defaults.get("pasteurization", "ON")),
            test_period=str(
                exec_defaults.get("test_period", exec_defaults.get("testPeriod", "P2Y0M0D"))
            ),
            unit_handling=str(
                exec_defaults.get("unit_handling", exec_defaults.get("unitHandling", "VERIFY"))
            ),
            nan_handling=str(
                exec_defaults.get("nan_handling", exec_defaults.get("nanHandling", "OFF"))
            ),
            max_trade=str(exec_defaults.get("max_trade", exec_defaults.get("maxTrade", "OFF"))),
            language=str(exec_defaults.get("language", "FASTEXPR")),
            visualization=bool(exec_defaults.get("visualization", False)),
        ).to_api_dict()

        alpha_type = str(exec_defaults.get("alpha_type", "REGULAR"))

        if item.settings_json and item.settings_json.strip() not in ("", "{}"):
            try:
                parsed = json.loads(item.settings_json)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, Mapping):
                for raw_key, raw_val in parsed.items():
                    key_str = str(raw_key)
                    if key_str in ("alpha_type", "type"):
                        if raw_val:
                            alpha_type = str(raw_val)
                        continue
                    mapped_key = _SNAKE_TO_CAMEL_SETTING.get(key_str, key_str)
                    if raw_val is not None:
                        base_settings[mapped_key] = raw_val

        base_settings.setdefault("region", region)
        base_settings.setdefault("universe", universe)
        base_settings.setdefault("delay", delay)
        if not base_settings.get("region"):
            base_settings["region"] = region
        if not base_settings.get("universe"):
            base_settings["universe"] = universe
        if base_settings.get("delay") is None:
            base_settings["delay"] = delay

        return {
            "type": alpha_type,
            "settings": base_settings,
            "regular": item.expression,
        }

    def _build_payload(
        self, items: Sequence[CandidateItem]
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Build single-sim dict when ``len(items) == 1`` or multi-sim list for 2..10 items."""
        if len(items) == 1:
            return self._build_sim_dict(items[0])
        return [self._build_sim_dict(item) for item in items]

    # ------------------------------------------------------------------
    # Quota reservation & circuit breaker
    # ------------------------------------------------------------------

    def _reserve_batch_quota(self, batch_id: str, count: int) -> None:
        with self._lock:
            self._batch_reservations[batch_id] = int(count)

    def _update_batch_reservation(self, batch_id: str, count: int) -> None:
        with self._lock:
            if batch_id in self._batch_reservations:
                if count <= 0:
                    del self._batch_reservations[batch_id]
                else:
                    self._batch_reservations[batch_id] = int(count)

    def _consume_batch_reservation(self, batch_id: str, count: int) -> None:
        with self._lock:
            existing = self._batch_reservations.get(batch_id)
            if existing is None:
                return
            remaining = existing - int(count)
            if remaining <= 0:
                del self._batch_reservations[batch_id]
            else:
                self._batch_reservations[batch_id] = remaining

    def _release_batch_reservation(self, batch_id: str) -> None:
        with self._lock:
            self._batch_reservations.pop(batch_id, None)

    def _total_reserved_quota(self) -> int:
        with self._lock:
            return sum(self._batch_reservations.values())

    def _compute_pop_limit(self) -> int:
        """Compute maximum batch size allowed right now without exceeding daily quota."""
        if not self.quota.can_submit(1) or self.quota.should_pace():
            return 0
        reserved = self._total_reserved_quota()
        if reserved > 0 and not self.quota.can_submit(reserved + 1):
            return 0

        pop_limit = self.multi_sim_limit
        if hasattr(self.quota, "get_status"):
            try:
                status = self.quota.get_status()
                remaining = getattr(status, "remaining", None)
                if type(remaining) is int:
                    avail = remaining - reserved
                    if avail <= 0:
                        return 0
                    pop_limit = min(pop_limit, avail)
            except Exception:
                logger.debug("Unable to read quota status remaining count", exc_info=True)
        return pop_limit

    def _record_region_success(self, region: str) -> None:
        with self._lock:
            self._consecutive_failures[region] = 0

    def _record_region_failure(self, region: str) -> None:
        alert_msg: Optional[str] = None
        with self._lock:
            failures = self._consecutive_failures.get(region, 0) + 1
            self._consecutive_failures[region] = failures
            if failures >= self.circuit_breaker_threshold:
                cooldown_until = time.time() + self.region_cooldown_s
                self._region_cooldowns[region] = cooldown_until
                self._consecutive_failures[region] = 0
                alert_msg = (
                    f"Circuit breaker tripped for region {region} after {failures} "
                    f"consecutive failures; cooling down for {self.region_cooldown_s:.0f}s"
                )
        if alert_msg is not None:
            logger.warning(alert_msg)
            self._emit_alert(alert_msg)

    def _emit_alert(self, message: str) -> None:
        if self.alert_callback is None:
            return
        try:
            self.alert_callback(message)
        except Exception:
            logger.exception("alert_callback raised while delivering alert: %s", message)

    def get_region_cooldowns(self) -> dict[str, float]:
        """Return active ``region -> epoch_cooldown_until`` entries, pruning expired ones."""
        now = time.time()
        with self._lock:
            expired = [reg for reg, until in self._region_cooldowns.items() if until <= now]
            for reg in expired:
                del self._region_cooldowns[reg]
            return dict(self._region_cooldowns)

    # ------------------------------------------------------------------
    # Compiler learning, alpha parsing, and persistence
    # ------------------------------------------------------------------

    def _learn_from_error(self, expression: str, error_msg: str) -> None:
        if self.validator is None or not expression or not error_msg:
            return
        learn_fn = getattr(self.validator, "learn_from_error", None)
        if not callable(learn_fn):
            return
        try:
            sig = inspect.signature(learn_fn)
            params = sig.parameters
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_var_kw or "db" in params:
                learn_fn(expression, error_msg, db=self.db)
            else:
                learn_fn(expression, error_msg)
        except Exception:
            logger.exception("ExpressionValidator.learn_from_error failed for %r", expression)

    @staticmethod
    def _parse_indexed_validation_errors(
        errors: Sequence[str],
        batch_len: int,
    ) -> dict[int, list[str]]:
        """Extract 0-based batch indices from flattened 400 validation errors."""
        grouped: dict[int, list[str]] = {}
        for raw_err in errors:
            err_str = str(raw_err or "")
            match = _INDEXED_ERROR_RE.match(err_str)
            if not match:
                continue
            idx = int(match.group(1))
            if 0 <= idx < batch_len:
                grouped.setdefault(idx, []).append(err_str)
        return grouped

    @staticmethod
    def _extract_alpha_id(payload: Mapping[str, Any]) -> str:
        raw = payload.get("alpha")
        if isinstance(raw, (list, tuple)):
            return str(raw[0]).strip() if raw else ""
        return str(raw).strip() if raw else ""

    @staticmethod
    def _build_simulation_record(
        item: CandidateItem,
        alpha_id: str,
        alpha_data: Mapping[str, Any],
        sim_payload: Optional[Mapping[str, Any]] = None,
    ) -> SimulationRecord:
        """Build a ``SimulationRecord`` from BRAIN ``GET /alphas/{id}`` JSON."""
        is_data = alpha_data.get("is")
        if not isinstance(is_data, Mapping):
            is_data = {}

        raw_checks = is_data.get("checks")
        if raw_checks is None:
            raw_checks = alpha_data.get("checks")
        checks_list: list[Any] = list(raw_checks) if isinstance(raw_checks, list) else []

        fail_count = 0
        warn_count = 0
        for check in checks_list:
            if isinstance(check, Mapping):
                result = str(check.get("result") or "").strip().upper()
                if result in ("FAIL", "FAILED", "ERROR"):
                    fail_count += 1
                elif result in ("WARNING", "WARN"):
                    warn_count += 1

        regular_block = alpha_data.get("regular")
        fallback_expr = ""
        if isinstance(regular_block, Mapping):
            fallback_expr = str(regular_block.get("code") or "")

        sim_status = str((sim_payload or {}).get("status") or "COMPLETE").strip().upper()
        record_status = "COMPLETE" if sim_status in ("COMPLETE", "WARNING", "") else sim_status
        sim_message = str((sim_payload or {}).get("message") or "").strip() or None

        return SimulationRecord(
            alpha_id=str(alpha_id),
            config_hash=str(item.config_hash or ""),
            ast_fingerprint=str(item.ast_fingerprint or ""),
            expression=str(item.expression or fallback_expr),
            region=str(item.region or "USA"),
            universe=str(item.universe or "TOP3000"),
            delay=int(item.delay if item.delay is not None else 1),
            settings_json=str(item.settings_json or "{}"),
            status=record_status,
            sharpe=_opt_float(is_data.get("sharpe")),
            fitness=_opt_float(is_data.get("fitness")),
            turnover=_opt_float(is_data.get("turnover")),
            returns=_opt_float(is_data.get("returns")),
            drawdown=_opt_float(is_data.get("drawdown")),
            margin=_opt_float(is_data.get("margin")),
            long_count=_opt_int(is_data.get("longCount", is_data.get("long_count"))),
            short_count=_opt_int(is_data.get("shortCount", is_data.get("short_count"))),
            checks_json=json.dumps(checks_list, sort_keys=True),
            fail_count=fail_count,
            warn_count=warn_count,
            error_message=sim_message,
            simulated_at=str(time.time()),
        )

    @staticmethod
    def _item_ids(items: Sequence[CandidateItem]) -> list[int]:
        return [int(item.id) for item in items if item.id is not None]

    def _attribute_simulation_error(
        self,
        items: Sequence[CandidateItem],
        parent_payload: Optional[Mapping[str, Any]],
        exc: SimulationError,
    ) -> None:
        """Feed ``SimulationError`` back into compiler knowledge for the responsible item."""
        err_msg = exc.message or str(exc)
        children = (parent_payload or {}).get("children")
        if isinstance(children, list) and exc.simulation_id:
            target_sim_id = simulation_id_from_url(str(exc.simulation_id))
            for idx, child in enumerate(children):
                if idx >= len(items):
                    break
                child_id = simulation_id_from_url(str(child))
                if str(child) == str(exc.simulation_id) or child_id == target_sim_id:
                    self._learn_from_error(items[idx].expression, err_msg)
                    return
        if len(items) == 1:
            self._learn_from_error(items[0].expression, err_msg)

    def _poll_and_finalize(
        self,
        items: Sequence[CandidateItem],
        progress_url: str,
    ) -> _BatchOutcome:
        """Poll ``progress_url`` (and children if multi-sim), persist alphas, update DB/quota."""
        item_ids = self._item_ids(items)
        parent_payload: Optional[dict[str, Any]] = None
        try:
            parent_payload = self.client.poll_simulation(progress_url)
        except SimulationError as exc:
            logger.warning("Simulation failed polling %s: %s", progress_url, exc)
            self._attribute_simulation_error(items, parent_payload, exc)
            if item_ids:
                self.db.mark_queue_failed(item_ids, str(exc))
            self.quota.record_outcome(completed=0, failed=len(items))
            return _BatchOutcome(completed=0, failed=len(items))
        except RateLimitError as exc:
            logger.warning("Rate-limited polling %s: %s", progress_url, exc)
            if item_ids:
                self.db.mark_queue_failed(item_ids, str(exc))
            self.quota.record_outcome(completed=0, failed=len(items))
            return _BatchOutcome(completed=0, failed=len(items), rate_limited=True)
        except (PollingTimeoutError, BrainAPIError, requests.RequestException) as exc:
            logger.error("Polling error for %s: %s", progress_url, exc)
            if item_ids:
                self.db.mark_queue_failed(item_ids, str(exc))
            self.quota.record_outcome(completed=0, failed=len(items))
            return _BatchOutcome(completed=0, failed=len(items))
        except Exception as exc:
            logger.exception("Unexpected error polling %s", progress_url)
            if item_ids:
                self.db.mark_queue_failed(item_ids, str(exc))
            self.quota.record_outcome(completed=0, failed=len(items))
            return _BatchOutcome(completed=0, failed=len(items))

        done_ids: list[int] = []
        completed = 0
        failed = 0

        children = parent_payload.get("children")
        if isinstance(children, list) and children:
            api_url = str(getattr(self.client, "api_url", "https://api.worldquantbrain.com")).rstrip("/")
            paired_count = min(len(items), len(children))
            for item, child_ref in zip(items[:paired_count], children[:paired_count]):
                child_str = str(child_ref)
                child_url = child_str if child_str.startswith("http") else f"{api_url}/simulations/{child_str}"
                try:
                    sim_payload = self.client.poll_simulation(child_url)
                except SimulationError as exc:
                    err_msg = exc.message or str(exc)
                    logger.warning("Child simulation failed (%s, item id=%s): %s", child_url, item.id, exc)
                    self._learn_from_error(item.expression, err_msg)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], str(exc))
                    failed += 1
                    continue
                except (PollingTimeoutError, BrainAPIError, requests.RequestException) as exc:
                    logger.error("Child polling error (%s, item id=%s): %s", child_url, item.id, exc)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], str(exc))
                    failed += 1
                    continue
                except Exception as exc:
                    logger.exception("Unexpected child polling error (%s, item id=%s)", child_url, item.id)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], str(exc))
                    failed += 1
                    continue

                alpha_id = self._extract_alpha_id(sim_payload)
                if not alpha_id:
                    err_msg = "No alpha ID in completed simulation response"
                    logger.error("%s for item id=%s", err_msg, item.id)
                    self._learn_from_error(item.expression, err_msg)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], err_msg)
                    failed += 1
                    continue

                try:
                    alpha_data = self.client.get_alpha(alpha_id)
                    record = self._build_simulation_record(item, alpha_id, alpha_data, sim_payload)
                    self.db.save_simulation(record)
                    if self.qualification_queue is not None:
                        self.qualification_queue.put(alpha_id)
                    if item.id is not None:
                        done_ids.append(item.id)
                    completed += 1
                except Exception as exc:
                    logger.exception("Failed fetching/saving alpha %s (item id=%s)", alpha_id, item.id)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], str(exc))
                    failed += 1

            if paired_count < len(items):
                missing_items = items[paired_count:]
                missing_ids = self._item_ids(missing_items)
                if missing_ids:
                    self.db.mark_queue_failed(missing_ids, "Missing child simulation result")
                failed += len(missing_items)
        else:
            terminal_payloads = [parent_payload]
            for item, sim_payload in zip(items, terminal_payloads):
                alpha_id = self._extract_alpha_id(sim_payload)
                if not alpha_id:
                    err_msg = "No alpha ID in completed simulation response"
                    logger.error("%s for item id=%s", err_msg, item.id)
                    self._learn_from_error(item.expression, err_msg)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], err_msg)
                    failed += 1
                    continue

                try:
                    alpha_data = self.client.get_alpha(alpha_id)
                    record = self._build_simulation_record(item, alpha_id, alpha_data, sim_payload)
                    self.db.save_simulation(record)
                    if self.qualification_queue is not None:
                        self.qualification_queue.put(alpha_id)
                    if item.id is not None:
                        done_ids.append(item.id)
                    completed += 1
                except Exception as exc:
                    logger.exception("Failed fetching/saving alpha %s (item id=%s)", alpha_id, item.id)
                    if item.id is not None:
                        self.db.mark_queue_failed([item.id], str(exc))
                    failed += 1

        if done_ids:
            self.db.mark_queue_done(done_ids)

        if completed or failed:
            self.quota.record_outcome(completed=completed, failed=failed)

        return _BatchOutcome(completed=completed, failed=failed)

    # ------------------------------------------------------------------
    # Startup crash recovery
    # ------------------------------------------------------------------

    def _find_unsubmitted_dispatching_ids(self) -> list[int]:
        """Find ``DISPATCHING`` queue rows that lack a ``progress_url``."""
        ids: list[int] = []
        if hasattr(self.db, "_read_conn"):
            with self.db._read_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT id, progress_url
                    FROM candidate_queue
                    WHERE status = 'DISPATCHING'
                    ORDER BY id ASC
                    """
                ).fetchall()
            for row in rows:
                if not str(row["progress_url"] or "").strip():
                    ids.append(int(row["id"]))
        elif hasattr(self.db, "get_dispatching_jobs"):
            for item in self.db.get_dispatching_jobs():
                if item.id is not None and not str(item.progress_url or "").strip():
                    ids.append(int(item.id))
        return ids

    def recover_in_flight(self) -> int:
        """Resume polling any ``IN_FLIGHT`` jobs and requeue orphaned ``DISPATCHING`` rows."""
        dispatching_requeue_ids = self._find_unsubmitted_dispatching_ids()
        if dispatching_requeue_ids:
            logger.info(
                "Requeuing %d unsubmitted DISPATCHING row(s) to PENDING.",
                len(dispatching_requeue_ids),
            )
            self.db.requeue_items(dispatching_requeue_ids)

        in_flight_rows = self.db.get_in_flight_jobs()
        if not in_flight_rows:
            return 0

        missing_url_ids: list[int] = []
        grouped: dict[tuple[str, str], list[CandidateItem]] = {}
        for item in in_flight_rows:
            progress_url = str(item.progress_url or "").strip()
            if not progress_url:
                if item.id is not None:
                    missing_url_ids.append(int(item.id))
                continue
            key = (str(item.batch_id or ""), progress_url)
            grouped.setdefault(key, []).append(item)

        if missing_url_ids:
            logger.warning(
                "Requeuing %d IN_FLIGHT row(s) missing progress_url to PENDING.",
                len(missing_url_ids),
            )
            self.db.requeue_items(missing_url_ids)

        recovered_jobs = 0
        for (batch_id, progress_url), group_items in grouped.items():
            logger.info(
                "Recovering in-flight job batch_id=%s (%d item(s)) from %s",
                batch_id,
                len(group_items),
                progress_url,
            )
            self._poll_and_finalize(group_items, progress_url)
            recovered_jobs += 1

        return recovered_jobs

    # ------------------------------------------------------------------
    # Job submission, 400 isolation, and worker execution
    # ------------------------------------------------------------------

    def _submit_and_poll_items(
        self,
        items: list[CandidateItem],
        batch_id: str,
    ) -> _BatchOutcome:
        """Submit ``items`` (isolating 400 poison items if needed) and poll on 201."""
        if not items:
            return _BatchOutcome()

        payload = self._build_payload(items)
        item_ids = self._item_ids(items)

        try:
            location = self.client.submit_simulation(payload)
        except SimulationValidationError as exc:
            err_list = list(exc.errors)
            if len(items) == 1:
                err_msg = "; ".join(err_list) if err_list else str(exc)
                logger.warning(
                    "Simulation 400 validation failed for single item id=%s: %s",
                    items[0].id,
                    err_msg,
                )
                self._learn_from_error(items[0].expression, err_msg)
                if item_ids:
                    self.db.mark_queue_failed(item_ids, err_msg)
                self._consume_batch_reservation(batch_id, 1)
                return _BatchOutcome(failed=1)

            indexed_errors = self._parse_indexed_validation_errors(err_list, len(items))
            if indexed_errors:
                bad_count = 0
                for idx, msgs in sorted(indexed_errors.items()):
                    bad_item = items[idx]
                    err_msg = "; ".join(msgs)
                    self._learn_from_error(bad_item.expression, err_msg)
                    if bad_item.id is not None:
                        self.db.mark_queue_failed([bad_item.id], err_msg)
                    bad_count += 1
                self._consume_batch_reservation(batch_id, bad_count)

                valid_items = [
                    item for idx, item in enumerate(items) if idx not in indexed_errors
                ]
                if not valid_items:
                    return _BatchOutcome(failed=bad_count)

                outcome = _BatchOutcome(failed=bad_count)
                outcome.merge(self._submit_and_poll_items(valid_items, batch_id))
                return outcome

            # Non-indexed 400 on a multi-sim batch: split in half recursively to isolate bad item(s).
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]
            combined = self._submit_and_poll_items(left, batch_id)
            if combined.rate_limited or combined.auth_error or combined.transport_error:
                right_ids = self._item_ids(right)
                if right_ids:
                    self.db.requeue_items(right_ids)
                self._consume_batch_reservation(batch_id, len(right))
                return combined
            combined.merge(self._submit_and_poll_items(right, batch_id))
            return combined

        except RateLimitError as exc:
            logger.warning(
                "Rate-limited on POST /simulations for batch %s; requeuing %d unsubmitted item(s): %s",
                batch_id,
                len(items),
                exc,
            )
            with self._lock:
                self._transient_requeue_gen += 1
            if item_ids:
                self.db.requeue_items(item_ids)
            self._consume_batch_reservation(batch_id, len(items))
            return _BatchOutcome(rate_limited=True)

        except AuthenticationError as exc:
            logger.error(
                "Authentication error on POST /simulations for batch %s; requeuing %d item(s): %s",
                batch_id,
                len(items),
                exc,
            )
            with self._lock:
                self._transient_requeue_gen += 1
            if item_ids:
                self.db.requeue_items(item_ids)
            self._consume_batch_reservation(batch_id, len(items))
            self._emit_alert(f"BRAIN authentication error during simulation dispatch: {exc}")
            return _BatchOutcome(auth_error=True)

        except (requests.RequestException, BrainAPIError) as exc:
            logger.warning(
                "Transient error on POST /simulations for batch %s; requeuing %d unsubmitted item(s): %s",
                batch_id,
                len(items),
                exc,
            )
            with self._lock:
                self._transient_requeue_gen += 1
            if item_ids:
                self.db.requeue_items(item_ids)
            self._consume_batch_reservation(batch_id, len(items))
            return _BatchOutcome(transport_error=True)

        except Exception:
            logger.exception(
                "Unexpected error submitting batch %s; requeuing %d unsubmitted item(s).",
                batch_id,
                len(items),
            )
            with self._lock:
                self._transient_requeue_gen += 1
            if item_ids:
                self.db.requeue_items(item_ids)
            self._consume_batch_reservation(batch_id, len(items))
            return _BatchOutcome(transport_error=True)

        # HTTP 201 accepted: record submitted quota & persist IN_FLIGHT state immediately.
        with self._lock:
            self.quota.record_submitted(len(items))
            self._consume_batch_reservation(batch_id, len(items))
        if item_ids:
            self.db.mark_in_flight(item_ids, location, batch_id)

        return self._poll_and_finalize(items, location)

    def _execute_job(
        self,
        items: list[CandidateItem],
        slot_ids: list[int],
        batch_id: str,
    ) -> None:
        """Worker thread entrypoint: submit, isolate/poll, record outcomes, and release slots."""
        try:
            if not items:
                return
            region = str(items[0].region or "USA")
            outcome = self._submit_and_poll_items(items, batch_id)
            if outcome.completed > 0:
                self._record_region_success(region)
            elif outcome.failed > 0 and not outcome.rate_limited and not outcome.auth_error:
                self._record_region_failure(region)
        finally:
            self._release_batch_reservation(batch_id)
            self.slot_pool.release(slot_ids)

    def _run_worker(
        self,
        items: list[CandidateItem],
        slot_ids: list[int],
        batch_id: str,
    ) -> None:
        current = threading.current_thread()
        try:
            self._execute_job(items, slot_ids, batch_id)
        finally:
            with self._workers_lock:
                self._active_workers.discard(current)

    # ------------------------------------------------------------------
    # Dispatch loop & lifecycle
    # ------------------------------------------------------------------

    def dispatch_once(self) -> int:
        """Claim available pending batches into idle slots and spawn 1 worker per job."""
        with self._dispatch_lock:
            return self._dispatch_once_locked()

    def _dispatch_once_locked(self) -> int:
        if self._stop_event.is_set():
            return 0

        regions = self._active_region_markets()
        if not regions:
            return 0

        with self._lock:
            start_requeue_gen = self._transient_requeue_gen

        dispatched = 0
        max_passes = max(1, self.slot_pool.max_slots)
        for _ in range(max_passes):
            if self._stop_event.is_set() or self.slot_pool.idle_count() <= 0:
                break
            with self._lock:
                if self._transient_requeue_gen != start_requeue_gen:
                    break
            pop_limit = self._compute_pop_limit()
            if pop_limit <= 0:
                break

            dispatched_this_pass = False
            num_regions = len(regions)
            start_idx = self._region_cursor % num_regions

            for offset in range(num_regions):
                if self._stop_event.is_set():
                    break
                idx = (start_idx + offset) % num_regions
                region, market_pairs = regions[idx]

                cooldowns = self.get_region_cooldowns()
                if region in cooldowns:
                    continue
                if not self.slot_pool.can_acquire(region):
                    continue
                if self.db.count_queue(status="PENDING", region=region) <= 0:
                    continue

                popped_items: list[CandidateItem] = []
                for universe, delay in market_pairs:
                    popped_items = self.db.pop_pending_batch(
                        region=region,
                        universe=universe,
                        delay=delay,
                        limit=pop_limit,
                    )
                    if popped_items:
                        break

                if not popped_items:
                    for universe, delay in self._discover_pending_markets_for_region(region):
                        if (universe, delay) in market_pairs:
                            continue
                        popped_items = self.db.pop_pending_batch(
                            region=region,
                            universe=universe,
                            delay=delay,
                            limit=pop_limit,
                        )
                        if popped_items:
                            break

                if not popped_items:
                    continue

                batch_id = f"{region}-{uuid.uuid4().hex[:12]}"
                self._reserve_batch_quota(batch_id, len(popped_items))
                slot_ids = self.slot_pool.acquire(
                    region=region,
                    job_id=batch_id,
                    timeout=self.acquire_timeout_s,
                )
                if slot_ids is None:
                    self._release_batch_reservation(batch_id)
                    requeue_ids = self._item_ids(popped_items)
                    if requeue_ids:
                        self.db.requeue_items(requeue_ids)
                    continue

                spawned = False
                try:
                    with self._workers_lock:
                        if self._stop_event.is_set():
                            raise InterruptedError("Dispatcher stopping before worker start")
                        worker = threading.Thread(
                            target=self._run_worker,
                            args=(popped_items, slot_ids, batch_id),
                            name=f"alpha-factory-sim-{batch_id}",
                            daemon=True,
                        )
                        self._active_workers.add(worker)
                        worker.start()
                        spawned = True
                except Exception:
                    if not spawned:
                        self._release_batch_reservation(batch_id)
                        self.slot_pool.release(slot_ids)
                        requeue_ids = self._item_ids(popped_items)
                        if requeue_ids:
                            self.db.requeue_items(requeue_ids)
                    if self._stop_event.is_set():
                        return dispatched
                    logger.exception("Failed spawning worker for batch %s", batch_id)
                    continue

                self._region_cursor = (idx + 1) % num_regions
                dispatched += 1
                dispatched_this_pass = True
                break

            if not dispatched_this_pass:
                break

        return dispatched

    @property
    def is_running(self) -> bool:
        """Return True while the background dispatch loop thread is alive."""
        with self._lock:
            return self._loop_thread is not None and self._loop_thread.is_alive()

    def active_worker_count(self) -> int:
        """Return number of currently running job worker threads."""
        with self._workers_lock:
            return sum(1 for t in self._active_workers if t.is_alive())

    def _run_loop(self) -> None:
        try:
            if not self._stop_event.is_set():
                self.recover_in_flight()
        except Exception:
            logger.exception("Error during startup in-flight crash recovery")

        while not self._stop_event.is_set():
            try:
                self.dispatch_once()
            except Exception:
                logger.exception("Error in SimulationDispatcher loop")
            self._stop_event.wait(self._poll_interval_s)

    def start(self) -> None:
        """Start the background dispatch loop thread (idempotent)."""
        with self._lock:
            if self._loop_thread is not None and self._loop_thread.is_alive():
                return
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run_loop,
                name="alpha-factory-dispatcher",
                daemon=True,
            )
            self._loop_thread = thread
            thread.start()

    def stop(self, drain_timeout_s: Optional[float] = None) -> None:
        """Stop claiming new batches and wait up to ``drain_timeout_s`` for active workers."""
        self._stop_event.set()
        timeout_total = (
            self.shutdown_drain_timeout_s
            if drain_timeout_s is None
            else max(0.0, float(drain_timeout_s))
        )
        deadline = time.monotonic() + timeout_total

        with self._lock:
            loop_thread = self._loop_thread
        if (
            loop_thread is not None
            and loop_thread.is_alive()
            and loop_thread is not threading.current_thread()
        ):
            remaining = max(0.0, deadline - time.monotonic())
            loop_thread.join(timeout=remaining)

        with self._lock:
            if self._loop_thread is loop_thread and (
                loop_thread is None or not loop_thread.is_alive()
            ):
                self._loop_thread = None

        current = threading.current_thread()
        while True:
            with self._workers_lock:
                workers = [
                    t for t in self._active_workers if t.is_alive() and t is not current
                ]
            if not workers:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for worker in workers:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                worker.join(timeout=rem)


__all__ = ["SimulationDispatcher"]
