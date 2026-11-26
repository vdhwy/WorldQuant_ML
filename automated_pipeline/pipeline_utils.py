"""
pipeline_utils.py — Shared utilities for the WorldQuant BRAIN automated pipeline.

Provides:
  • safe_poll()        — robust polling loop with Retry-After fix & timeout ceiling
  • hash_alpha()       — deterministic hash for alpha-config deduplication
  • SimulationCache    — parquet-backed already-simulated cache
  • send_telegram()    — optional Telegram notification helper
  • load_config()      — YAML config loader with sane defaults
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger("ace.pipeline")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "market": {
        "instrument_type": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
    },
    "dataset_filters": {
        "coverage_min": 0.0,
        "coverage_max": 1.0,
        "field_count_min": 1,
        "field_count_max": 1000,
        "user_count_min": 1,
        "user_count_max": 1000,
        "value_score_min": 0.0,
        "value_score_max": 10.0,
        "sort_by": "valueScore",
        "sort_ascending": False,
        "category": "",
    },
    "expressions": {
        "lookback_days": [5, 10, 20, 60],
        "templates": {
            "sentiment_comparison": True,
            "vector_ops": True,
            "matrix_ops": True,
        },
    },
    "simulation": {
        "alpha_type": "REGULAR",
        "decay": 4,
        "neutralization": "INDUSTRY",
        "truncation": 0.1,
        "pasteurization": "ON",
        "test_period": "P2Y",
        "unit_handling": "VERIFY",
        "nan_handling": "OFF",
        "max_trade": "OFF",
        "visualization": False,
        "batch_size": 200,
        "multi_sim_limit": 10,
        "concurrent_limit": 3,
        "relogin_threshold_s": 2000,
    },
    "polling": {
        "max_wait_s": 3600,
        "default_interval_s": 5,
        "max_http_retries": 3,
    },
    "output": {
        "results_dir": "pipeline_results",
        "cache_file": "simulation_cache.parquet",
        "save_pnl": False,
        "save_stats": False,
        "save_result_file": False,
        "check_submission": False,
        "check_self_corr": False,
        "check_prod_corr": False,
    },
    "telegram": {
        "enabled": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str = "pipeline_config.yaml") -> dict:
    """
    Load a YAML config file and merge it on top of sane defaults.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        The fully-resolved configuration dictionary.
    """
    if os.path.exists(path):
        with open(path, "r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    else:
        logger.warning("Config file %s not found — using defaults", path)
        user_cfg = {}
    return _deep_merge(_DEFAULTS, user_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Safe Polling  (fixes the Retry-After hang from the changelog)
# ──────────────────────────────────────────────────────────────────────────────


class PollingTimeoutError(Exception):
    """Raised when polling exceeds the max_wait_s ceiling."""

    def __init__(self, url: str, elapsed: float, max_wait: float):
        self.url = url
        self.elapsed = elapsed
        self.max_wait = max_wait
        super().__init__(
            f"Polling timed out after {elapsed:.0f}s (ceiling {max_wait}s) for {url}"
        )


class SimulationError(Exception):
    """Raised when a simulation finishes with a non-success status."""

    def __init__(
        self,
        status: str,
        message: str = "",
        location: dict | None = None,
        simulation_id: str | None = None,
    ):
        self.status = status
        self.message = message
        self.location = location
        self.simulation_id = simulation_id
        super().__init__(
            f"Simulation {simulation_id or '?'} ended with status={status}: {message}"
        )


_TERMINAL_SUCCESS = {"COMPLETE", "WARNING"}
_TERMINAL_FAILURE = {"ERROR", "FAIL", "TIMEOUT", "CANCELLED"}


def _parse_retry_after(headers: dict, default: float) -> float | None:
    """
    Parse the Retry-After header defensively.

    Returns
    -------
    None   — header absent or == 0  → polling is **finished**
    float  — positive seconds to sleep before next poll
    """
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None  # absent → finished

    try:
        value = float(raw)
    except (ValueError, TypeError):
        logger.warning("Unparseable Retry-After=%r, using default %ss", raw, default)
        return default

    if value <= 0:
        return None  # 0 or negative → finished

    return value


def safe_poll(
    session: requests.Session,
    url: str,
    *,
    max_wait_s: float = 3600,
    default_interval_s: float = 5,
    max_http_retries: int = 3,
) -> requests.Response:
    """
    Poll *url* until Retry-After signals completion, with a hard time ceiling.

    This fixes the hang documented in the changelog: the original code compared
    the **string** ``"0"`` to the **int** ``0``, which is always ``False`` in
    Python, causing an infinite tight loop.

    Parameters
    ----------
    session : requests.Session
        Authenticated session.
    url : str
        The simulation-progress URL (from the Location header).
    max_wait_s : float
        Hard ceiling in seconds.
    default_interval_s : float
        Fallback sleep when Retry-After is present but unparseable.
    max_http_retries : int
        Number of transient HTTP errors (non-2xx) tolerated.

    Returns
    -------
    requests.Response
        The final, terminal response.

    Raises
    ------
    PollingTimeoutError
        If the ceiling is exceeded.
    SimulationError
        If the simulation terminates with a failure status.
    """
    start = time.monotonic()
    http_failures = 0

    while True:
        elapsed = time.monotonic() - start
        if elapsed >= max_wait_s:
            raise PollingTimeoutError(url, elapsed, max_wait_s)

        resp = session.get(url)

        # ── transient HTTP error ──────────────────────────────────────────
        if resp.status_code // 100 != 2:
            http_failures += 1
            if http_failures > max_http_retries:
                logger.error(
                    "Polling %s: %d consecutive HTTP errors (last %d) — giving up",
                    url,
                    http_failures,
                    resp.status_code,
                )
                resp.raise_for_status()
            logger.warning(
                "Polling %s: HTTP %d (attempt %d/%d), retrying in %ss",
                url,
                resp.status_code,
                http_failures,
                max_http_retries,
                default_interval_s,
            )
            time.sleep(default_interval_s)
            continue

        # reset on success
        http_failures = 0

        # ── check Retry-After ─────────────────────────────────────────────
        wait = _parse_retry_after(resp.headers, default_interval_s)
        if wait is None:
            # Finished (absent or 0) — validate status before returning.
            break

        logger.debug("Polling %s: Retry-After=%s, sleeping…", url, wait)
        time.sleep(wait)

    # ── validate terminal status ──────────────────────────────────────────
    try:
        body = resp.json()
    except Exception:
        return resp  # non-JSON terminal — caller decides

    status = body.get("status", "")
    if status in _TERMINAL_FAILURE:
        msg = body.get("message", "")
        # Log the full body so we can diagnose why simulations fail
        logger.warning(
            "Simulation %s ended with status=%s. Full response: %s",
            body.get("id", "?"),
            status,
            json.dumps(body, default=str)[:1000],
        )
        raise SimulationError(
            status=status,
            message=msg,
            location=body.get("location"),
            simulation_id=body.get("id"),
        )
    if status and status not in _TERMINAL_SUCCESS:
        # Non-terminal state reached without Retry-After — treat as error
        raise SimulationError(
            status=status,
            message=f"Unexpected non-terminal status without Retry-After: {status}",
            simulation_id=body.get("id"),
        )

    return resp


def safe_poll_children(
    session: requests.Session,
    children: list[str],
    brain_api_url: str,
    *,
    max_wait_s: float = 3600,
    default_interval_s: float = 5,
    max_http_retries: int = 3,
) -> list[dict]:
    """
    Poll every child simulation individually (fixes changelog item #5).

    Returns a list of child JSON bodies.
    """
    results = []
    for child_id in children:
        child_url = f"{brain_api_url}/simulations/{child_id}"
        try:
            resp = safe_poll(
                session,
                child_url,
                max_wait_s=max_wait_s,
                default_interval_s=default_interval_s,
                max_http_retries=max_http_retries,
            )
            results.append(resp.json())
        except (SimulationError, PollingTimeoutError) as exc:
            logger.error("Child simulation %s failed: %s", child_id, exc)
            results.append({"id": child_id, "status": "ERROR", "error": str(exc)})
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Alpha Hashing & Caching
# ──────────────────────────────────────────────────────────────────────────────


def hash_alpha(alpha_dict: dict) -> str:
    """
    Produce a deterministic SHA-256 hex digest for an alpha configuration.

    Sorts keys so that insertion order doesn't affect the hash.
    """
    canonical = json.dumps(alpha_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class SimulationCache:
    """
    Parquet-backed cache that tracks which alpha configs have already been
    simulated so we can skip them on re-runs.
    """

    def __init__(self, path: str = "simulation_cache.parquet"):
        self.path = path
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not os.path.exists(self.path):
            df = pd.DataFrame(
                columns=["alpha_hashed", "alpha_id", "expression", "date_created"]
            )
            df.to_parquet(self.path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def contains(self, alpha_dict: dict) -> bool:
        h = hash_alpha(alpha_dict)
        df = self.load()
        return h in df["alpha_hashed"].values

    def add(
        self,
        alpha_dict: dict,
        alpha_id: str | None,
        expression: str = "",
    ) -> None:
        h = hash_alpha(alpha_dict)
        df = self.load()
        if h in df["alpha_hashed"].values:
            return
        new_row = {
            "alpha_hashed": h,
            "alpha_id": alpha_id,
            "expression": expression,
            "date_created": datetime.now(timezone.utc).isoformat(),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_parquet(self.path, index=False)

    def already_simulated_hashes(self) -> set[str]:
        df = self.load()
        return set(df["alpha_hashed"].values)


# ──────────────────────────────────────────────────────────────────────────────
# Telegram Helper
# ──────────────────────────────────────────────────────────────────────────────


def send_telegram(
    message: str,
    token: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """
    Send a message via Telegram Bot API.  Falls back to env vars.

    Returns True on success.
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.debug("Telegram not configured — skipping notification")
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        return resp.ok
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────


def fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ──────────────────────────────────────────────────────────────────────────────
# Intra-batch Similarity Deduplication
# ──────────────────────────────────────────────────────────────────────────────

import re as _re

# Operators commonly found in alpha expressions — kept out of field detection
_KNOWN_OPS = frozenset({
    # Time-series
    "ts_rank", "ts_delta", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max",
    "ts_skewness", "ts_kurtosis", "ts_zscore", "ts_regression", "ts_decay_linear",
    "ts_decay_exp_window", "ts_moment", "ts_entropy", "ts_median",
    "ts_arg_max", "ts_arg_min", "ts_product", "ts_corr", "ts_covariance",
    "ts_count_nans", "ts_av_diff", "ts_returns", "ts_scale",
    "ts_percentage", "ts_ir",
    # Vector
    "rank", "zscore", "winsorize", "normalize", "sigmoid", "sign", "log",
    "abs", "sqrt", "inverse", "power", "signed_power", "scale", "truncate",
    # Group
    "group_rank", "group_zscore", "group_neutralize", "group_mean",
    "group_sum", "group_count", "group_max", "group_min", "group_median",
    "group_std", "group_normalize", "group_percentage",
    # Multi-arg
    "correlation", "add", "subtract", "multiply", "divide",
    "min", "max", "delta", "reverse",
    # Neut
    "vector_neut",
    # Misc keywords
    "true", "false", "filter", "if", "else", "nan",
    "sector", "industry", "subindustry", "market", "country",
})

# Regex: a "field token" is a lowercase+digit+underscore identifier ≥5 chars
# that is NOT a known operator.  Shorter tokens are usually params/keywords.
_FIELD_TOKEN_RE = _re.compile(r"\b([a-z][a-z0-9_]{4,})\b")

# Regex: numeric literal (int or float, possibly negative)
_NUM_RE = _re.compile(r"-?\b\d+(?:\.\d+)?\b")


def _structural_fingerprint(expr: str) -> str:
    """
    Build a structural fingerprint for an expression.

    Replaces each unique field-like token with a canonical placeholder (F0, F1, …)
    in order of first appearance.  Operators, punctuation, and numeric parameters
    are preserved, so expressions differing only in which datafields they reference
    collapse to the same fingerprint.

    Example
    -------
    >>> _structural_fingerprint("ts_rank(model_abc, 10) - ts_mean(model_xyz, 10)")
    'ts_rank(F0, 10) - ts_mean(F1, 10)'
    >>> _structural_fingerprint("ts_rank(model_xyz, 10) - ts_mean(model_abc, 10)")
    'ts_rank(F0, 10) - ts_mean(F1, 10)'
    """
    # Collect unique field tokens in order of first appearance
    seen: dict[str, str] = {}
    counter = 0

    def _replace(match: _re.Match) -> str:
        nonlocal counter
        token = match.group(1)
        if token in _KNOWN_OPS:
            return token          # keep operators as-is
        if token not in seen:
            seen[token] = f"F{counter}"
            counter += 1
        return seen[token]

    return _FIELD_TOKEN_RE.sub(_replace, expr)


def deduplicate_similar_expressions(
    expressions: list[str],
    max_per_structure: int = 1,
) -> tuple[list[str], int]:
    """
    Fast O(n) intra-batch similarity deduplication for pipeline expressions.

    Instead of expensive pairwise similarity (O(n²)), this function groups
    expressions by their *structural fingerprint* — the operator skeleton
    with field names replaced by canonical placeholders.  Expressions that
    share the same structure but differ only in which datafields they
    reference are considered "similar".

    Within each structural group, at most ``max_per_structure`` expressions
    are kept (the first ones encountered).

    Parameters
    ----------
    expressions : list[str]
        The full list of concrete expressions.
    max_per_structure : int
        How many expressions to keep per unique structure (default 1).

    Returns
    -------
    (kept, skipped_count) : tuple[list[str], int]
        The deduplicated list and how many were removed.
    """
    structure_counts: dict[str, int] = {}
    kept: list[str] = []
    skipped = 0

    for expr in expressions:
        fp = _structural_fingerprint(expr)
        count = structure_counts.get(fp, 0)
        if count < max_per_structure:
            kept.append(expr)
            structure_counts[fp] = count + 1
        else:
            skipped += 1

    return kept, skipped

