"""Two-stage submittable qualification gates and PnL flatline checks for BRAIN alphas."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from alpha_factory.brain.client import BrainClient, parse_recordset
from alpha_factory.qualification.correlations import (
    fetch_power_pool_corr,
    fetch_prod_corr,
    fetch_self_corr,
)
from alpha_factory.storage.models import CheckItem, QualificationResult
from alpha_factory.storage.pnl_store import extract_daily_returns_from_recordset

try:
    from requests import HTTPError as _RequestsHTTPError
except Exception:  # pragma: no cover - defensive fallback
    _RequestsHTTPError = OSError  # type: ignore[misc, assignment]


def _opt_finite_float(value: Any) -> Optional[float]:
    """Convert value to a finite float, returning None for missing, bool, or NaN/Inf."""
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
    """Convert value to int, returning None for missing, bool, or non-finite values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_qual_cfg(qual_cfg: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Accept either the ``qualification`` sub-dict or a full settings dict."""
    if not isinstance(qual_cfg, Mapping):
        return {}
    nested = qual_cfg.get("qualification")
    if isinstance(nested, Mapping):
        return nested
    return qual_cfg


def _format_match_item(item: Any) -> Optional[str]:
    """Extract a readable identifier/name string from a pyramid, theme, or competition entry."""
    if isinstance(item, str):
        cleaned = item.strip()
        return cleaned or None
    if isinstance(item, Mapping):
        for key in ("name", "id", "title", "theme", "pyramid", "competition"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if val is not None and not isinstance(val, (dict, list, bool)):
                cleaned = str(val).strip()
                if cleaned:
                    return cleaned
    return None


def _extend_unique(target: list[str], seen: set[str], raw_items: Any) -> None:
    """Append unique formatted match strings from a list or single entry."""
    if raw_items is None:
        return
    items = raw_items if isinstance(raw_items, (list, tuple)) else [raw_items]
    for item in items:
        label = _format_match_item(item)
        if label and label not in seen:
            seen.add(label)
            target.append(label)


def _extract_matches(
    payload: Mapping[str, Any],
    is_data: Mapping[str, Any],
    checks_raw: Sequence[Any],
) -> tuple[list[str], list[str], list[str]]:
    """Extract deduplicated pyramid, theme, and competition matches from BRAIN payloads."""
    pyramids: list[str] = []
    themes: list[str] = []
    competitions: list[str] = []
    seen_pyramids: set[str] = set()
    seen_themes: set[str] = set()
    seen_competitions: set[str] = set()

    def _consume_pyramid_themes(raw_pt: Any) -> None:
        if isinstance(raw_pt, Mapping):
            _extend_unique(pyramids, seen_pyramids, raw_pt.get("pyramids"))
            _extend_unique(themes, seen_themes, raw_pt.get("themes"))
        elif isinstance(raw_pt, (list, tuple)):
            _extend_unique(pyramids, seen_pyramids, raw_pt)

    for container in (payload, is_data):
        if not isinstance(container, Mapping):
            continue
        _extend_unique(pyramids, seen_pyramids, container.get("pyramids"))
        _consume_pyramid_themes(container.get("pyramidThemes"))
        _extend_unique(themes, seen_themes, container.get("themes"))
        _extend_unique(competitions, seen_competitions, container.get("competitions"))

    for entry in checks_raw:
        if not isinstance(entry, Mapping):
            continue
        _extend_unique(pyramids, seen_pyramids, entry.get("pyramids"))
        _consume_pyramid_themes(entry.get("pyramidThemes"))
        _extend_unique(themes, seen_themes, entry.get("themes"))
        _extend_unique(competitions, seen_competitions, entry.get("competitions"))

        check_name = str(entry.get("name") or entry.get("test") or "").strip().upper()
        if "PYRAMID" in check_name:
            _extend_unique(pyramids, seen_pyramids, entry.get("matches"))
        elif "THEME" in check_name:
            _extend_unique(themes, seen_themes, entry.get("matches"))
        elif "COMPETITION" in check_name:
            _extend_unique(competitions, seen_competitions, entry.get("matches"))

    return pyramids, themes, competitions


def parse_checks(checks_raw: Sequence[Mapping[str, Any]] | list[dict[str, Any]]) -> list[CheckItem]:
    """Parse BRAIN ``is.checks`` entries using the ``result`` key (PASS/FAIL/WARNING/PENDING)."""
    if not isinstance(checks_raw, Sequence) or isinstance(checks_raw, (str, bytes)):
        return []

    parsed: list[CheckItem] = []
    for entry in checks_raw:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("test") or "").strip() or "UNKNOWN"
        raw_result = str(entry.get("result") or "").strip().upper()
        if raw_result == "FAILED":
            result = "FAIL"
        elif raw_result == "WARN":
            result = "WARNING"
        else:
            result = raw_result
        parsed.append(
            CheckItem(
                name=name,
                result=result,
                limit=_opt_finite_float(entry.get("limit")),
                value=_opt_finite_float(entry.get("value")),
            )
        )
    return parsed


def check_pnl_flatline(
    daily_returns: np.ndarray,
    zero_ratio_max: float = 0.30,
    tail_days: int = 252,
) -> tuple[bool, str]:
    """Check whether a daily returns series is non-empty, active, and non-flatlined."""
    arr = np.asarray(daily_returns, dtype=np.float64).ravel()
    if arr.size < 20:
        return False, f"INSUFFICIENT_PNL_DAYS ({arr.size} < 20)"
    if not np.all(np.isfinite(arr)):
        return False, "NON_FINITE_RETURNS"

    zero_ratio = float(np.mean(np.abs(arr) < 1e-12))
    if zero_ratio > float(zero_ratio_max):
        return False, f"HIGH_ZERO_RETURN_RATIO ({zero_ratio:.3f} > {float(zero_ratio_max):.3f})"

    tail_len = int(tail_days)
    if tail_len > 0:
        tail = arr[-tail_len:] if arr.size >= tail_len else arr
        tail_zero_ratio = float(np.mean(np.abs(tail) < 1e-12))
        if tail_zero_ratio > 0.85:
            return False, f"TAIL_ZERO_RETURN_RATIO ({tail_zero_ratio:.3f} > 0.850)"
        tail_std = float(np.std(tail))
        if tail_std < 1e-12:
            return False, f"TAIL_ZERO_VARIANCE ({tail_std:.3e} < 1e-12)"

    return True, "OK"


def evaluate_gate1(
    alpha_payload: Mapping[str, Any],
    qual_cfg: Optional[Mapping[str, Any]] = None,
) -> QualificationResult:
    """Evaluate Gate 1 checks, numeric performance thresholds, and sanity bounds."""
    payload = alpha_payload if isinstance(alpha_payload, Mapping) else {}
    alpha_id = str(payload.get("id") or payload.get("alpha_id") or "")
    raw_is = payload.get("is")
    is_data = raw_is if isinstance(raw_is, Mapping) else {}

    raw_checks = is_data.get("checks")
    if raw_checks is None:
        raw_checks = payload.get("checks")
    checks_list = list(raw_checks) if isinstance(raw_checks, list) else []
    checks = parse_checks(checks_list)

    pyramids, themes, competitions = _extract_matches(payload, is_data, checks_list)
    warn_reasons = [c.name for c in checks if c.result == "WARNING"]
    failed_checks = [c.name for c in checks if c.result == "FAIL"]

    if failed_checks:
        return QualificationResult(
            alpha_id=alpha_id,
            passed=False,
            needs_review=False,
            fail_reasons=failed_checks,
            warn_reasons=warn_reasons,
            checks=checks,
            pyramid_matches=pyramids,
            theme_matches=themes,
            competition_matches=competitions,
        )

    cfg = _resolve_qual_cfg(qual_cfg)
    min_sharpe = float(cfg.get("min_sharpe", 1.58))
    min_fitness = float(cfg.get("min_fitness", 1.0))
    min_turnover = float(cfg.get("min_turnover", 0.01))
    max_turnover = float(cfg.get("max_turnover", 0.70))
    min_margin = float(cfg.get("min_margin", 0.0008))
    max_sanity_sharpe = float(cfg.get("max_sanity_sharpe", 6.0))
    max_sanity_returns = float(cfg.get("max_sanity_returns", 0.50))

    sharpe = _opt_finite_float(is_data.get("sharpe"))
    fitness = _opt_finite_float(is_data.get("fitness"))
    turnover = _opt_finite_float(is_data.get("turnover"))
    margin = _opt_finite_float(is_data.get("margin"))
    returns = _opt_finite_float(is_data.get("returns"))
    long_count = _opt_int(is_data.get("longCount", is_data.get("long_count")))
    short_count = _opt_int(is_data.get("shortCount", is_data.get("short_count")))

    fail_reasons: list[str] = []
    if sharpe is None or sharpe < min_sharpe:
        fail_reasons.append(f"LOW_SHARPE ({sharpe} < {min_sharpe})")
    if fitness is None or fitness < min_fitness:
        fail_reasons.append(f"LOW_FITNESS ({fitness} < {min_fitness})")
    if turnover is None:
        fail_reasons.append("MISSING_TURNOVER")
    elif turnover < min_turnover:
        fail_reasons.append(f"LOW_TURNOVER ({turnover} < {min_turnover})")
    elif turnover > max_turnover:
        fail_reasons.append(f"HIGH_TURNOVER ({turnover} > {max_turnover})")
    if margin is None or margin < min_margin:
        fail_reasons.append(f"LOW_MARGIN ({margin} < {min_margin})")
    if long_count is None or long_count <= 0:
        fail_reasons.append(f"NO_LONG_POSITIONS ({long_count})")
    if short_count is None or short_count <= 0:
        fail_reasons.append(f"NO_SHORT_POSITIONS ({short_count})")

    needs_review = False
    if (sharpe is not None and sharpe > max_sanity_sharpe) or (
        returns is not None and returns > max_sanity_returns
    ):
        needs_review = True
        fail_reasons.append("SANITY_TOO_GOOD_TO_BE_TRUE")

    passed = (len(fail_reasons) == 0) and (not needs_review)
    return QualificationResult(
        alpha_id=alpha_id,
        passed=passed,
        needs_review=needs_review,
        fail_reasons=fail_reasons,
        warn_reasons=warn_reasons,
        checks=checks,
        pyramid_matches=pyramids,
        theme_matches=themes,
        competition_matches=competitions,
    )


def _extract_pnl_series(
    client: BrainClient,
    alpha_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch ``pnl`` recordset (falling back to ``daily-pnl``) and extract daily returns."""
    parser = getattr(client, "parse_recordset", None)
    parse_fn = parser if callable(parser) else parse_recordset

    def _fetch_and_parse(recordset_name: str) -> tuple[np.ndarray, np.ndarray]:
        rs = client.fetch_recordset(alpha_id, recordset_name)
        if not isinstance(rs, Mapping) or not rs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        if "records" in rs or "schema" in rs:
            parsed_raw = parse_fn(dict(rs))
            parsed = dict(parsed_raw) if isinstance(parsed_raw, Mapping) else {}
            if "schema" in rs and "schema" not in parsed:
                parsed["schema"] = rs["schema"]
            if "name" in rs and "name" not in parsed:
                parsed["name"] = rs["name"]
        elif "rows" in rs:
            parsed = dict(rs)
        else:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        if (
            recordset_name == "daily-pnl"
            and parsed.get("name") is None
            and not (isinstance(parsed.get("schema"), Mapping) and parsed["schema"].get("name"))
        ):
            parsed["recordset_name"] = "daily-pnl"

        return extract_daily_returns_from_recordset(parsed)

    try:
        dates, daily_returns = _fetch_and_parse("pnl")
    except _RequestsHTTPError as exc:
        resp = getattr(exc, "response", None)
        if getattr(resp, "status_code", None) == 404:
            return _fetch_and_parse("daily-pnl")
        raise

    if dates.size == 0:
        return _fetch_and_parse("daily-pnl")
    return dates, daily_returns


def evaluate_gate2(
    client: BrainClient,
    alpha_id: str,
    qual_cfg: Optional[Mapping[str, Any]] = None,
) -> tuple[QualificationResult, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Evaluate Gate 2 submission checks, correlations, and PnL flatline health."""
    cfg = _resolve_qual_cfg(qual_cfg)
    max_self_corr = float(cfg.get("max_self_corr", 0.70))
    max_prod_corr = float(cfg.get("max_prod_corr", 0.70))
    check_power_pool = bool(cfg.get("check_power_pool", False))
    max_power_pool_corr = float(cfg.get("max_power_pool_corr", 0.50))
    zero_ratio_max = float(cfg.get("flatline_zero_ratio_max", 0.30))
    tail_days = int(cfg.get("flatline_tail_days", 252))

    check_payload = client.get_alpha_check(alpha_id)
    if not isinstance(check_payload, Mapping):
        check_payload = {}

    raw_is = check_payload.get("is")
    is_data = raw_is if isinstance(raw_is, Mapping) else {}
    raw_checks = is_data.get("checks")
    if raw_checks is None:
        raw_checks = check_payload.get("checks")
    checks_list = list(raw_checks) if isinstance(raw_checks, list) else []
    checks = parse_checks(checks_list)

    pyramids, themes, competitions = _extract_matches(check_payload, is_data, checks_list)
    warn_reasons = [c.name for c in checks if c.result == "WARNING"]
    failed_checks = [c.name for c in checks if c.result == "FAIL"]

    result = QualificationResult(
        alpha_id=alpha_id,
        passed=False,
        needs_review=False,
        fail_reasons=list(failed_checks),
        warn_reasons=warn_reasons,
        checks=checks,
        pyramid_matches=pyramids,
        theme_matches=themes,
        competition_matches=competitions,
    )

    if failed_checks:
        return result, None

    self_corr = fetch_self_corr(client, alpha_id)
    result.self_corr_max = self_corr
    if self_corr >= max_self_corr:
        result.fail_reasons.append(f"SELF_CORRELATION ({self_corr:.4f} >= {max_self_corr:.4f})")
        return result, None

    prod_corr, prod_risk_tier = fetch_prod_corr(client, alpha_id)
    result.prod_corr_max = prod_corr
    result.prod_risk_tier = prod_risk_tier
    if prod_corr >= max_prod_corr:
        result.fail_reasons.append(f"PROD_CORRELATION ({prod_corr:.4f} >= {max_prod_corr:.4f})")
        return result, None

    if check_power_pool:
        power_pool_corr = fetch_power_pool_corr(client, alpha_id)
        result.power_pool_corr_max = power_pool_corr
        if power_pool_corr >= max_power_pool_corr:
            result.fail_reasons.append(
                f"POWER_POOL_CORRELATION ({power_pool_corr:.4f} >= {max_power_pool_corr:.4f})"
            )
            return result, None

    dates, daily_returns = _extract_pnl_series(client, alpha_id)
    is_healthy, flatline_reason = check_pnl_flatline(
        daily_returns,
        zero_ratio_max=zero_ratio_max,
        tail_days=tail_days,
    )
    if not is_healthy:
        result.fail_reasons.append(f"PNL_FLATLINE: {flatline_reason}")
        pnl_tuple = (dates, daily_returns) if dates.size > 0 else None
        return result, pnl_tuple

    result.passed = True
    return result, (dates, daily_returns)


__all__ = [
    "check_pnl_flatline",
    "evaluate_gate1",
    "evaluate_gate2",
    "parse_checks",
]
