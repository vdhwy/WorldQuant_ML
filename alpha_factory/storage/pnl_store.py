"""NumPy compressed PnL time-series store and BRAIN recordset parser."""

from __future__ import annotations

import datetime as dt
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

_DATE_CANDIDATES = ("date", "dt", "timestamp", "trade_date")
_DAILY_VALUE_CANDIDATES = (
    "returns",
    "return",
    "daily_pnl",
    "dailypnl",
    "daily-pnl",
    "daily_returns",
    "dailyreturns",
    "daily_return",
    "dailyreturn",
)
_CUMULATIVE_VALUE_CANDIDATES = (
    "pnl",
    "cum_pnl",
    "cumulative_pnl",
    "value",
)
_DAILY_RECORDSET_NAMES = {"daily-pnl", "daily_pnl", "dailypnl", "returns"}
_CUMULATIVE_RECORDSET_NAMES = {"pnl", "cumulative-pnl", "cumulative_pnl"}


def _empty_series() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    cleaned = value.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return cleaned


def _resolve_rows_and_names(
    parsed_recordset: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[str], list[Mapping[str, Any]]]:
    schema = parsed_recordset.get("schema") or {}
    properties = parsed_recordset.get("properties")
    if properties is None and isinstance(schema, Mapping):
        properties = schema.get("properties")
    prop_list = list(properties or [])

    raw_names = parsed_recordset.get("names")
    if raw_names is not None:
        names = [str(n) for n in raw_names if n is not None]
    else:
        names = [str(p.get("name")) for p in prop_list if isinstance(p, Mapping) and p.get("name")]

    raw_rows = parsed_recordset.get("rows")
    if raw_rows is not None:
        rows = [r for r in raw_rows if isinstance(r, Mapping)]
        if not names and rows:
            names = [str(k) for k in rows[0].keys()]
        return rows, names, prop_list

    records = parsed_recordset.get("records") or []
    rows = [dict(zip(names, rec)) for rec in records]
    return rows, names, prop_list


def _pick_column(names: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lower_map = {name.lower(): name for name in names}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _infer_is_cumulative(
    parsed_recordset: Mapping[str, Any],
    value_col: str,
    prop_list: list[Mapping[str, Any]],
    cumulative: Optional[bool],
) -> bool:
    if cumulative is not None:
        return bool(cumulative)
    if parsed_recordset.get("cumulative") is not None:
        return bool(parsed_recordset["cumulative"])
    if parsed_recordset.get("is_cumulative") is not None:
        return bool(parsed_recordset["is_cumulative"])

    schema = parsed_recordset.get("schema") or {}
    rec_name = (
        parsed_recordset.get("name")
        or parsed_recordset.get("recordset")
        or parsed_recordset.get("recordset_name")
        or (schema.get("name") if isinstance(schema, Mapping) else None)
    )
    if isinstance(rec_name, str) and rec_name.strip():
        lowered_rec = rec_name.strip().lower()
        if lowered_rec in _DAILY_RECORDSET_NAMES:
            return False
        if lowered_rec in _CUMULATIVE_RECORDSET_NAMES:
            return True

    if value_col.lower() in _DAILY_VALUE_CANDIDATES:
        return False

    for prop in prop_list:
        if isinstance(prop, Mapping) and str(prop.get("name", "")).lower() == value_col.lower():
            meta_text = f"{prop.get('title', '')} {prop.get('description', '')}".lower()
            if "daily" in meta_text:
                return False

    return True


def _parse_date_ordinal(raw_date: Any) -> int:
    if isinstance(raw_date, dt.datetime):
        return raw_date.date().toordinal()
    if isinstance(raw_date, dt.date):
        return raw_date.toordinal()
    return dt.date.fromisoformat(str(raw_date).strip()[:10]).toordinal()


def extract_daily_returns_from_recordset(
    parsed_recordset: Mapping[str, Any],
    cumulative: Optional[bool] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted ``(dates[int64], daily_returns[float64])`` from a BRAIN recordset.

    Supports both ``daily-pnl`` (e.g. ``date`` + ``returns`` / ``pnl``) and cumulative
    ``pnl`` recordsets (sorts by date ascending and computes ``np.diff`` prepending
    the first cumulative value).
    """
    if not isinstance(parsed_recordset, Mapping) or not parsed_recordset:
        return _empty_series()

    rows, names, prop_list = _resolve_rows_and_names(parsed_recordset)
    if not rows:
        return _empty_series()

    date_col = _pick_column(names, _DATE_CANDIDATES)
    if date_col is None:
        raise ValueError(f"Could not find date column in recordset columns: {names!r}")

    value_col = _pick_column(names, _DAILY_VALUE_CANDIDATES)
    if value_col is None:
        value_col = _pick_column(names, _CUMULATIVE_VALUE_CANDIDATES)
    if value_col is None:
        other_cols = [n for n in names if n != date_col]
        if len(other_cols) == 1:
            value_col = other_cols[0]
    if value_col is None:
        raise ValueError(f"Could not find PnL/returns column in recordset columns: {names!r}")

    is_cumulative = _infer_is_cumulative(parsed_recordset, value_col, prop_list, cumulative)

    ordinals: list[int] = []
    values: list[float] = []
    for row in rows:
        raw_date = row.get(date_col)
        raw_val = row.get(value_col)
        if raw_date is None or raw_date == "" or raw_val is None:
            continue
        ordinals.append(_parse_date_ordinal(raw_date))
        values.append(float(raw_val))

    if not ordinals:
        return _empty_series()

    dates_arr = np.asarray(ordinals, dtype=np.int64)
    vals_arr = np.asarray(values, dtype=np.float64)

    order = np.argsort(dates_arr, kind="mergesort")
    dates_arr = dates_arr[order]
    vals_arr = vals_arr[order]

    if is_cumulative:
        daily_returns = np.diff(vals_arr, prepend=0.0).astype(np.float64, copy=False)
    else:
        daily_returns = vals_arr.astype(np.float64, copy=False)

    return dates_arr, daily_returns


class PnlStore:
    """Thread-safe compressed ``.npz`` store for daily alpha PnL arrays with LRU cache."""

    def __init__(self, pnl_dir: str | Path, max_cache_entries: int = 2000) -> None:
        self.pnl_dir = Path(pnl_dir).expanduser()
        self.pnl_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_entries = max(0, int(max_cache_entries))
        self._cache: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lock = threading.RLock()

    def _path_for(self, alpha_id: str, region: str) -> Path:
        safe_region = _validate_identifier(region, "region")
        safe_alpha = _validate_identifier(alpha_id, "alpha_id")
        return self.pnl_dir / safe_region / f"{safe_alpha}.npz"

    def _cache_put(
        self,
        key: tuple[str, str],
        dates: np.ndarray,
        daily_returns: np.ndarray,
    ) -> None:
        if self.max_cache_entries <= 0:
            return
        self._cache[key] = (dates, daily_returns)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    def has(self, alpha_id: str, region: str) -> bool:
        """Return True if PnL arrays for ``(region, alpha_id)`` are cached or on disk."""
        path = self._path_for(alpha_id, region)
        key = (region.strip(), alpha_id.strip())
        with self._lock:
            if key in self._cache:
                return True
        return path.is_file()

    def save(
        self,
        alpha_id: str,
        region: str,
        dates: np.ndarray,
        daily_returns: np.ndarray,
    ) -> Path:
        """Atomically persist ``(dates, daily_returns)`` as compressed ``.npz`` and cache it."""
        path = self._path_for(alpha_id, region)
        dates_arr = np.asarray(dates, dtype=np.int64)
        returns_arr = np.asarray(daily_returns, dtype=np.float64)
        if dates_arr.ndim != 1 or returns_arr.ndim != 1:
            raise ValueError("dates and daily_returns must be 1D arrays")
        if dates_arr.shape != returns_arr.shape:
            raise ValueError(
                f"Shape mismatch: dates {dates_arr.shape} vs daily_returns {returns_arr.shape}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
        try:
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(fh, dates=dates_arr, daily_returns=returns_arr)
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        key = (region.strip(), alpha_id.strip())
        with self._lock:
            self._cache_put(key, dates_arr.copy(), returns_arr.copy())
        return path

    def load(
        self,
        alpha_id: str,
        region: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Load ``(dates, daily_returns)`` from the LRU cache or ``.npz`` file on disk."""
        path = self._path_for(alpha_id, region)
        key = (region.strip(), alpha_id.strip())
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        if not path.is_file():
            return None

        with np.load(path, allow_pickle=False) as data:
            dates_arr = np.asarray(data["dates"], dtype=np.int64)
            returns_key = "daily_returns" if "daily_returns" in data else "returns"
            returns_arr = np.asarray(data[returns_key], dtype=np.float64)

        with self._lock:
            self._cache_put(key, dates_arr, returns_arr)
        return dates_arr, returns_arr

    def delete(self, alpha_id: str, region: str) -> bool:
        """Remove a stored PnL entry from both LRU cache and disk."""
        path = self._path_for(alpha_id, region)
        key = (region.strip(), alpha_id.strip())
        with self._lock:
            self._cache.pop(key, None)
        if path.is_file():
            path.unlink()
            return True
        return False

    def clear_cache(self) -> None:
        """Evict all in-memory LRU cache entries without deleting disk files."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


__all__ = [
    "PnlStore",
    "extract_daily_returns_from_recordset",
]
