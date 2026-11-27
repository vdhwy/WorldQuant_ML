"""Telegram alert notifications with thread-safe per-key rate limiting."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

import requests

from alpha_factory.storage.models import DailyQuotaStatus, VaultEntry

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
BRAIN_ALPHA_URL_PREFIX = "https://platform.worldquantbrain.com/alpha"


def _clean_opt_str(value: Optional[str]) -> Optional[str]:
    """Normalize an optional string, returning None when empty or whitespace."""
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _fmt_float(value: Optional[float], precision: int = 2) -> str:
    """Format an optional float for human-readable alert output."""
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


class AlertManager:
    """Thread-safe Telegram notifier with per-key cooldown deduplication."""

    def __init__(
        self,
        enabled: bool = False,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        cooldown_s: float = 300.0,
    ) -> None:
        self.enabled = bool(enabled)
        env_token = os.environ.get("TELEGRAM_BOT_TOKEN") if token is None else token
        env_chat_id = os.environ.get("TELEGRAM_CHAT_ID") if chat_id is None else chat_id
        self.token: Optional[str] = _clean_opt_str(env_token)
        self.chat_id: Optional[str] = _clean_opt_str(env_chat_id)
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> AlertManager:
        """Construct an AlertManager from loaded settings dictionary."""
        ops_cfg = settings.get("ops", {}) if isinstance(settings, dict) else {}
        return cls(
            enabled=bool(ops_cfg.get("telegram_enabled", False)),
            token=_clean_opt_str(ops_cfg.get("telegram_token")),
            chat_id=_clean_opt_str(ops_cfg.get("telegram_chat_id")),
            cooldown_s=float(ops_cfg.get("alert_cooldown_s", 300.0)),
        )

    def _reserve_cooldown_slot(
        self, key: Optional[str], force: bool
    ) -> tuple[bool, Optional[float], Optional[float]]:
        """Check and reserve a cooldown slot for ``key`` under lock.

        Returns ``(allowed, reserved_ts, previous_ts)``.
        """
        if key is None or force or self.cooldown_s <= 0.0:
            return (True, None, None)

        with self._lock:
            now = time.monotonic()
            prev = self._last_sent.get(key)
            if prev is not None and (now - prev) < self.cooldown_s:
                return (False, None, prev)
            self._last_sent[key] = now
            return (True, now, prev)

    def _rollback_cooldown_slot(
        self, key: Optional[str], reserved_ts: Optional[float], previous_ts: Optional[float]
    ) -> None:
        """Restore prior cooldown timestamp if an alert send failed."""
        if key is None or reserved_ts is None:
            return
        with self._lock:
            if self._last_sent.get(key) == reserved_ts:
                if previous_ts is None:
                    self._last_sent.pop(key, None)
                else:
                    self._last_sent[key] = previous_ts

    def send(self, message: str, key: Optional[str] = None, force: bool = False) -> bool:
        """Send a Telegram message if enabled, configured, and outside key cooldown."""
        if not self.enabled:
            return False

        if not self.token or not self.chat_id:
            logger.warning(
                "Telegram alerts enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
            )
            return False

        allowed, reserved_ts, previous_ts = self._reserve_cooldown_slot(key, force)
        if not allowed:
            logger.debug("Suppressed duplicate alert for key=%s within cooldown", key)
            return False

        url = f"{TELEGRAM_API_BASE}/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            try:
                body = resp.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and body.get("ok") is False:
                logger.warning("Telegram API returned ok=false: %s", body.get("description", body))
                self._rollback_cooldown_slot(key, reserved_ts, previous_ts)
                return False
            return True
        except Exception as exc:
            logger.warning("Failed to send Telegram alert: %s", exc)
            self._rollback_cooldown_slot(key, reserved_ts, previous_ts)
            return False

    def alert_new_kept_alpha(self, entry: VaultEntry) -> bool:
        """Notify when a new qualified alpha is kept in the deduplicated vault."""
        alpha_url = f"{BRAIN_ALPHA_URL_PREFIX}/{entry.alpha_id}"
        message = (
            f"[Alpha Factory] New Kept Alpha\n"
            f"URL: {alpha_url}\n"
            f"Region: {entry.region} | Delay: {entry.delay}\n"
            f"Score: {_fmt_float(entry.composite_score, 4)}\n"
            f"Sharpe: {_fmt_float(entry.sharpe, 2)} | "
            f"Fitness: {_fmt_float(entry.fitness, 2)}\n"
            f"Turnover: {_fmt_float(entry.turnover, 4)} | "
            f"ProdCorrMax: {_fmt_float(entry.prod_corr_max, 4)}"
        )
        return self.send(message, key=f"kept_alpha:{entry.alpha_id}")

    def alert_biometric_required(self, verify_url: str) -> bool:
        """Notify operator that BRAIN biometric / 2FA verification is required."""
        message = (
            "[Alpha Factory] Biometric / 2FA Verification Required\n"
            f"Complete verification in browser to resume BRAIN session:\n{verify_url}"
        )
        return self.send(message, key="biometric_required")

    def alert_quota_warning(self, status: DailyQuotaStatus) -> bool:
        """Notify when daily BRAIN simulation quota approaches or hits its cap."""
        message = (
            f"[Alpha Factory] Daily Quota Warning ({status.date_ny})\n"
            f"Submitted: {status.submitted_count}/{status.limit} "
            f"(Remaining: {status.remaining})\n"
            f"Completed: {status.completed_count} | Failed: {status.failed_count}"
        )
        return self.send(message, key=f"quota_warning:{status.date_ny}")

    def alert_circuit_breaker(self, region: str, reason: str) -> bool:
        """Notify when consecutive errors trip a region's execution circuit breaker."""
        message = (
            f"[Alpha Factory] Circuit Breaker Tripped\n"
            f"Region: {region}\n"
            f"Reason: {reason}"
        )
        return self.send(message, key=f"circuit_breaker:{region}")

    def send_daily_digest(self, stats: dict[str, Any]) -> bool:
        """Send a daily summary digest of simulation, qualification, and vault metrics."""
        date_label = str(stats.get("date_ny") or stats.get("date") or "today")
        lines = [f"[Alpha Factory] Daily Digest ({date_label})"]
        ordered_keys = [
            ("submitted", "Submitted"),
            ("completed", "Completed"),
            ("failed", "Failed"),
            ("gate1_pass", "Gate 1 Pass"),
            ("gate2_pass", "Gate 2 Pass"),
            ("kept", "Kept in Vault"),
            ("submitted_to_brain", "Submitted to BRAIN"),
            ("vault_active", "Active Vault Total"),
        ]
        seen = {"date_ny", "date"}
        for key, label in ordered_keys:
            if key in stats:
                lines.append(f"{label}: {stats[key]}")
                seen.add(key)
        for key, value in stats.items():
            if key not in seen:
                lines.append(f"{key}: {value}")
        return self.send("\n".join(lines), key=f"daily_digest:{date_label}")


__all__ = ["AlertManager"]
