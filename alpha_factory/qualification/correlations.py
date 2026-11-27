"""BRAIN correlation fetchers for self, production, and power-pool checks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from alpha_factory.brain.client import BrainClient, parse_recordset


def _to_finite_float(value: Any) -> Optional[float]:
    """Convert value to a finite float, returning None for missing, bool, or non-finite inputs."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _row_value_ci(row: Mapping[str, Any], *keys: str) -> Any:
    """Lookup the first matching key in row (exact match first, then case-insensitive)."""
    for key in keys:
        if key in row:
            return row[key]
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        lowered = key.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None


def _parse_rows(client: BrainClient, payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract row dicts from a BRAIN correlation payload via client.parse_recordset."""
    records = payload.get("records")
    if isinstance(records, list) and records:
        if all(isinstance(item, Mapping) for item in records):
            return [item for item in records if isinstance(item, Mapping)]
        parser = getattr(client, "parse_recordset", None)
        parsed = parser(dict(payload)) if callable(parser) else parse_recordset(dict(payload))
        rows = parsed.get("rows") if isinstance(parsed, Mapping) else None
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, Mapping)]
        return []

    raw_rows = payload.get("rows")
    if isinstance(raw_rows, list):
        return [r for r in raw_rows if isinstance(r, Mapping)]
    return []


def _extract_histogram_max(
    rows: list[Mapping[str, Any]],
    top_max: Optional[float],
) -> Optional[float]:
    """Extract max correlation from histogram buckets (alphas > 0 -> max) with top_max refinement."""
    occupied_buckets: list[tuple[float, Optional[float]]] = []
    saw_histogram_col = False

    for row in rows:
        alphas_raw = _row_value_ci(row, "alphas", "alpha_count", "count")
        if alphas_raw is not None or any(
            str(k).lower() in {"alphas", "alpha_count", "count"} for k in row
        ):
            saw_histogram_col = True
        alphas_val = _to_finite_float(alphas_raw)
        if alphas_val is None or alphas_val <= 0:
            continue

        bucket_max = _to_finite_float(_row_value_ci(row, "max", "correlation", "corr"))
        if bucket_max is None:
            continue
        bucket_min = _to_finite_float(_row_value_ci(row, "min"))
        occupied_buckets.append((bucket_max, bucket_min))

    if occupied_buckets:
        best_max, best_min = max(occupied_buckets, key=lambda item: item[0])
        if top_max is not None and (best_min is None or top_max >= best_min):
            return top_max
        return best_max

    if saw_histogram_col:
        return 0.0
    return top_max


def _classify_prod_risk_tier(corr: float) -> str:
    """Map max production correlation to LOW (<0.5), MEDIUM (0.5..0.7), or HIGH (>=0.7)."""
    if corr < 0.5:
        return "LOW"
    if corr < 0.7:
        return "MEDIUM"
    return "HIGH"


def _extract_record_max_corr(
    client: BrainClient,
    payload: Mapping[str, Any],
) -> float:
    """Extract max correlation from record-based correlation payloads (self / power-pool)."""
    if not isinstance(payload, Mapping) or not payload:
        return 0.0

    top_max = _to_finite_float(payload.get("max"))
    if "records" in payload:
        raw_records = payload.get("records")
        if not raw_records:
            return 0.0

    rows = _parse_rows(client, payload)
    if rows:
        corr_vals: list[float] = []
        for row in rows:
            val = _to_finite_float(_row_value_ci(row, "correlation", "corr"))
            if val is not None:
                corr_vals.append(val)
        if corr_vals:
            rec_max = max(corr_vals)
            if top_max is not None and top_max > rec_max:
                return top_max
            return rec_max

        hist_max = _extract_histogram_max(rows, top_max)
        if hist_max is not None:
            return hist_max

    return top_max if top_max is not None else 0.0


def fetch_self_corr(client: BrainClient, alpha_id: str) -> float:
    """Fetch self-correlation for ``alpha_id`` and return max correlation (or 0.0 if empty)."""
    payload = client.get_alpha_correlations(alpha_id, "self")
    return _extract_record_max_corr(client, payload)


def fetch_prod_corr(client: BrainClient, alpha_id: str) -> tuple[float, str]:
    """Fetch production correlation histogram and return ``(max_prod_corr, risk_tier)``."""
    payload = client.get_alpha_correlations(alpha_id, "prod")
    if not isinstance(payload, Mapping) or not payload:
        return 0.0, "LOW"

    top_max = _to_finite_float(payload.get("max"))
    rows = _parse_rows(client, payload)
    if rows:
        hist_max = _extract_histogram_max(rows, top_max)
        if hist_max is not None:
            return hist_max, _classify_prod_risk_tier(hist_max)

    if top_max is not None:
        return top_max, _classify_prod_risk_tier(top_max)
    return 0.0, "LOW"


def fetch_power_pool_corr(client: BrainClient, alpha_id: str) -> float:
    """Fetch power-pool correlation for ``alpha_id`` and return max correlation (or 0.0 if empty)."""
    payload = client.get_alpha_correlations(alpha_id, "power-pool")
    return _extract_record_max_corr(client, payload)


__all__ = [
    "fetch_power_pool_corr",
    "fetch_prod_corr",
    "fetch_self_corr",
]
