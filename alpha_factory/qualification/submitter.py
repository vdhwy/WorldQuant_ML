"""Sequential live re-validation and submission of top deduplicated vault alphas."""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

import requests

from alpha_factory.brain.auth import AuthenticationError, BrainAPIError, RateLimitError
from alpha_factory.brain.client import BrainClient, PollingTimeoutError
from alpha_factory.qualification.checks import parse_checks
from alpha_factory.qualification.correlations import fetch_prod_corr, fetch_self_corr
from alpha_factory.qualification.dedup_vault import CorrelationDedupVault
from alpha_factory.storage.db import NY_TZ, AlphaDB
from alpha_factory.storage.models import DEFAULT_SETTINGS, VaultEntry

logger = logging.getLogger(__name__)

_FAIL_STATUSES = frozenset({"FAIL", "FAILED", "ERROR"})
_SUBMISSION_STATE_PREFIX = "submissions:"


def _unwrap_corr_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, tuple) and raw:
        raw = raw[0]
    elif isinstance(raw, Mapping):
        for key in ("max", "max_corr", "value", "corr"):
            if key in raw and raw[key] is not None:
                raw = raw[key]
                break
        else:
            return None
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _raw_check_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        is_block = payload.get("is")
        if isinstance(is_block, Mapping) and isinstance(is_block.get("checks"), list):
            return list(is_block["checks"])
        if isinstance(payload.get("checks"), list):
            return list(payload["checks"])
    return []


def _extract_failed_checks(payload: Any) -> list[str]:
    """Parse BRAIN check payload via ``parse_checks`` and return failing check names."""
    raw_list = _raw_check_list(payload)
    target = raw_list if isinstance(payload, Mapping) else payload

    parsed: Any = None
    try:
        parsed = parse_checks(target)
    except (TypeError, AttributeError):
        if target is not payload:
            try:
                parsed = parse_checks(payload)
            except (TypeError, AttributeError):
                parsed = raw_list
        else:
            parsed = raw_list

    if isinstance(parsed, tuple) and parsed:
        first = parsed[0]
        items = first if isinstance(first, list) else raw_list
    elif isinstance(parsed, Mapping):
        items = _raw_check_list(parsed)
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = raw_list

    failures: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            name = str(item.get("name") or item.get("test") or "UNKNOWN")
            result = str(item.get("result") or "").strip().upper()
        else:
            name = str(getattr(item, "name", "UNKNOWN"))
            result = str(getattr(item, "result", "") or "").strip().upper()
        if result in _FAIL_STATUSES:
            failures.append(name)
    return failures


def _submission_response_rejected(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    status = str(payload.get("status") or "").strip().upper()
    if status in _FAIL_STATUSES:
        msg = str(payload.get("message") or payload.get("detail") or status)
        return f"submit_status:{msg}"
    failed_checks = _extract_failed_checks(payload)
    if failed_checks:
        return f"submit_check_fail:{','.join(failed_checks)}"
    return None


class SequentialSubmitter:
    """Processes ``KEPT`` vault alphas strictly one-by-one with live pre-submit checks."""

    def __init__(
        self,
        db: AlphaDB,
        client: BrainClient,
        vault: CorrelationDedupVault,
        qual_cfg: Optional[Mapping[str, Any]] = None,
        max_submissions_per_day: int = 2,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db = db
        self.client = client
        self.vault = vault
        self.max_submissions_per_day = max(0, int(max_submissions_per_day))
        self._now_fn = now_fn
        self._lock = threading.RLock()

        defaults = DEFAULT_SETTINGS["qualification"]
        raw_cfg = dict(qual_cfg or {})
        if isinstance(raw_cfg.get("qualification"), Mapping):
            cfg = dict(raw_cfg["qualification"])
        else:
            cfg = raw_cfg

        self.max_self_corr = float(cfg.get("max_self_corr", defaults["max_self_corr"]))
        self.max_prod_corr = float(cfg.get("max_prod_corr", defaults["max_prod_corr"]))

    def _ny_date_str(self) -> str:
        now = self._now_fn() if self._now_fn is not None else datetime.now(NY_TZ)
        if now.tzinfo is None:
            now = now.replace(tzinfo=NY_TZ)
        else:
            now = now.astimezone(NY_TZ)
        return now.strftime("%Y-%m-%d")

    def _get_submitted_today(self, date_ny: str) -> int:
        raw = self.db.get_runtime_state(f"{_SUBMISSION_STATE_PREFIX}{date_ny}", "0")
        try:
            return max(0, int(raw or 0))
        except (TypeError, ValueError):
            return 0

    def _increment_submitted_today(self, date_ny: str, count: int = 1) -> int:
        updated = self._get_submitted_today(date_ny) + max(0, int(count))
        self.db.set_runtime_state(f"{_SUBMISSION_STATE_PREFIX}{date_ny}", str(updated))
        return updated

    def _mark_vault_status(
        self,
        entry: VaultEntry,
        vault_status: str,
        *,
        duplicate_of: Optional[str] = None,
        prod_corr_max: Optional[float] = None,
    ) -> None:
        self.db.upsert_vault_entry(
            VaultEntry(
                alpha_id=entry.alpha_id,
                region=entry.region,
                delay=int(entry.delay),
                expression=entry.expression,
                composite_score=float(entry.composite_score),
                vault_status=vault_status,
                duplicate_of=duplicate_of,
                max_local_corr=entry.max_local_corr,
                max_local_corr_peer=entry.max_local_corr_peer,
                sharpe=entry.sharpe,
                fitness=entry.fitness,
                turnover=entry.turnover,
                prod_corr_max=entry.prod_corr_max if prod_corr_max is None else prod_corr_max,
            )
        )

    def _run_live_recheck(
        self,
        alpha_id: str,
    ) -> tuple[bool, list[str], Optional[float], Optional[float]]:
        check_payload = self.client.get_alpha_check(alpha_id)
        failed_checks = _extract_failed_checks(check_payload)
        if failed_checks:
            return False, [f"check_fail:{','.join(failed_checks)}"], None, None

        self_corr = _unwrap_corr_value(fetch_self_corr(self.client, alpha_id))
        if self_corr is not None and self_corr >= self.max_self_corr:
            return (
                False,
                [f"self_corr={self_corr:.4f}>={self.max_self_corr:.4f}"],
                self_corr,
                None,
            )

        prod_corr = _unwrap_corr_value(fetch_prod_corr(self.client, alpha_id))
        if prod_corr is not None and prod_corr >= self.max_prod_corr:
            return (
                False,
                [f"prod_corr={prod_corr:.4f}>={self.max_prod_corr:.4f}"],
                self_corr,
                prod_corr,
            )

        return True, [], self_corr, prod_corr

    def submit_top_kept(
        self,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Submit up to ``limit`` (capped by daily limit) top ``KEPT`` alphas sequentially."""
        with self._lock:
            date_ny = self._ny_date_str()
            submitted_today = self._get_submitted_today(date_ny)
            remaining_today = max(0, self.max_submissions_per_day - submitted_today)

            if limit is not None:
                target_count = min(max(0, int(limit)), remaining_today)
            else:
                target_count = remaining_today

            if target_count <= 0:
                return []

            results: list[dict[str, Any]] = []
            processed_ids: set[str] = set()
            accepted_count = 0

            while accepted_count < target_count:
                kept_entries = self.db.get_vault_entries(statuses="KEPT")
                next_entry = next(
                    (e for e in kept_entries if e.alpha_id not in processed_ids),
                    None,
                )
                if next_entry is None:
                    break

                processed_ids.add(next_entry.alpha_id)

                try:
                    passed, reasons, self_corr, prod_corr = self._run_live_recheck(
                        next_entry.alpha_id
                    )
                except (AuthenticationError, RateLimitError) as exc:
                    logger.warning(
                        "Halting submission loop during live re-check of %s: %s",
                        next_entry.alpha_id,
                        exc,
                    )
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "ERROR",
                            "submitted": False,
                            "self_corr": None,
                            "prod_corr": None,
                            "reason": str(exc),
                        }
                    )
                    break
                except (PollingTimeoutError, requests.RequestException, BrainAPIError) as exc:
                    logger.warning(
                        "Transient error during live re-check of %s: %s",
                        next_entry.alpha_id,
                        exc,
                    )
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "ERROR",
                            "submitted": False,
                            "self_corr": None,
                            "prod_corr": None,
                            "reason": str(exc),
                        }
                    )
                    continue

                if not passed:
                    reason_str = "; ".join(reasons)
                    logger.info(
                        "Alpha %s failed live submission re-check (%s); rejecting and reconciling.",
                        next_entry.alpha_id,
                        reason_str,
                    )
                    if not dry_run:
                        self._mark_vault_status(next_entry, "REJECTED")
                        self.vault.reconcile_clusters(next_entry.region, int(next_entry.delay))
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "DRY_RUN_REJECTED" if dry_run else "REJECTED",
                            "submitted": False,
                            "self_corr": self_corr,
                            "prod_corr": prod_corr,
                            "reason": reason_str,
                        }
                    )
                    continue

                if dry_run:
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "DRY_RUN",
                            "submitted": False,
                            "self_corr": self_corr,
                            "prod_corr": prod_corr,
                            "reason": "",
                        }
                    )
                    accepted_count += 1
                    continue

                try:
                    submit_resp = self.client.submit_alpha(next_entry.alpha_id)
                except (AuthenticationError, RateLimitError) as exc:
                    logger.warning(
                        "Halting submission loop submitting %s: %s",
                        next_entry.alpha_id,
                        exc,
                    )
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "ERROR",
                            "submitted": False,
                            "self_corr": self_corr,
                            "prod_corr": prod_corr,
                            "reason": str(exc),
                        }
                    )
                    break
                except (PollingTimeoutError, requests.RequestException, BrainAPIError) as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code in (400, 403, 422):
                        logger.info(
                            "Submission rejected by BRAIN (HTTP %s) for %s: %s",
                            status_code,
                            next_entry.alpha_id,
                            exc,
                        )
                        self._mark_vault_status(next_entry, "REJECTED")
                        self.vault.reconcile_clusters(next_entry.region, int(next_entry.delay))
                        results.append(
                            {
                                "alpha_id": next_entry.alpha_id,
                                "region": next_entry.region,
                                "delay": next_entry.delay,
                                "composite_score": next_entry.composite_score,
                                "status": "REJECTED",
                                "submitted": False,
                                "self_corr": self_corr,
                                "prod_corr": prod_corr,
                                "reason": str(exc),
                            }
                        )
                        continue
                    logger.warning(
                        "Transient error submitting %s: %s",
                        next_entry.alpha_id,
                        exc,
                    )
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "ERROR",
                            "submitted": False,
                            "self_corr": self_corr,
                            "prod_corr": prod_corr,
                            "reason": str(exc),
                        }
                    )
                    continue

                reject_reason = _submission_response_rejected(submit_resp)
                if reject_reason is not None:
                    logger.info(
                        "Submission response rejected %s (%s); rejecting and reconciling.",
                        next_entry.alpha_id,
                        reject_reason,
                    )
                    self._mark_vault_status(next_entry, "REJECTED")
                    self.vault.reconcile_clusters(next_entry.region, int(next_entry.delay))
                    results.append(
                        {
                            "alpha_id": next_entry.alpha_id,
                            "region": next_entry.region,
                            "delay": next_entry.delay,
                            "composite_score": next_entry.composite_score,
                            "status": "REJECTED",
                            "submitted": False,
                            "self_corr": self_corr,
                            "prod_corr": prod_corr,
                            "reason": reject_reason,
                        }
                    )
                    continue

                self._mark_vault_status(
                    next_entry,
                    "SUBMITTED",
                    duplicate_of=None,
                    prod_corr_max=prod_corr,
                )
                already_submitted = bool(
                    isinstance(submit_resp, Mapping) and submit_resp.get("already_submitted")
                )
                if not already_submitted:
                    self._increment_submitted_today(date_ny, 1)
                    accepted_count += 1

                results.append(
                    {
                        "alpha_id": next_entry.alpha_id,
                        "region": next_entry.region,
                        "delay": next_entry.delay,
                        "composite_score": next_entry.composite_score,
                        "status": "SUBMITTED",
                        "submitted": True,
                        "self_corr": self_corr,
                        "prod_corr": prod_corr,
                        "reason": "",
                    }
                )

            return results


__all__ = ["SequentialSubmitter"]
