"""BRAIN metadata catalog synchronization, local cache bootstrap, and field helpers."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from alpha_factory.brain.auth import _parse_json_dict
from alpha_factory.brain.client import BrainClient, _handle_response
from alpha_factory.storage.db import AlphaDB, _extract_label
from alpha_factory.storage.models import OperatorRecord

logger = logging.getLogger(__name__)

_ORIG_GET_DATASETS = BrainClient.get_datasets
_ORIG_GET_DATAFIELDS_PAGE = BrainClient.get_datafields_page

_VEC_CALL_RE = re.compile(r"^vec_[a-zA-Z0-9_]+\s*\(")
_INFIX_RE = re.compile(r"\binput1\b\s*(?:==|!=|>=|<=|>|<)\s*\binput2\b")
_IDENT_CALL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(")

_OPTIONAL_DATA_INPUT_NAMES = frozenset(
    {"x", "y", "z", "input", "input1", "input2", "input3", "weight", "g1", "g2", "other"}
)
_LOOKBACK_PARAM_NAMES = frozenset({"d", "lookback"})


def wrap_vector_field(field_id: str, op: str = "vec_avg") -> str:
    """Wrap a VECTOR field identifier with ``op(...)`` unless already wrapped in a ``vec_*`` call."""
    cleaned = field_id.strip()
    if not cleaned:
        return ""
    op_name = op.strip() or "vec_avg"

    if _has_outer_call(cleaned, op_name) or _has_outer_vec_call(cleaned):
        return cleaned
    return f"{op_name}({cleaned})"


def _has_outer_vec_call(expr: str) -> bool:
    match = _VEC_CALL_RE.match(expr)
    if not match:
        return False
    return _parens_span_whole(expr, match.end() - 1)


def _has_outer_call(expr: str, op_name: str) -> bool:
    match = re.match(rf"^{re.escape(op_name)}\s*\(", expr)
    if not match:
        return False
    return _parens_span_whole(expr, match.end() - 1)


def _parens_span_whole(expr: str, open_idx: int) -> bool:
    if open_idx < 0 or open_idx >= len(expr) or expr[open_idx] != "(":
        return False
    depth = 0
    in_quote: Optional[str] = None
    for idx in range(open_idx, len(expr)):
        ch = expr[idx]
        if in_quote is not None:
            if ch == in_quote and (idx == 0 or expr[idx - 1] != "\\"):
                in_quote = None
            continue
        if ch in ("'", '"', "“", "”"):
            in_quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx == len(expr) - 1
    return False


def _extract_first_call_args(definition: str) -> Optional[str]:
    match = _IDENT_CALL_RE.match(definition.strip())
    if not match:
        return None
    open_idx = definition.strip().find("(", match.start())
    text = definition.strip()
    depth = 0
    in_quote: Optional[str] = None
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if in_quote is not None:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"', "“", "”"):
            in_quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx]
    return None


def _split_top_level_args(args_str: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote: Optional[str] = None
    for ch in args_str:
        if in_quote is not None:
            buf.append(ch)
            if (in_quote == "“" and ch == "”") or (ch == in_quote and in_quote != "“"):
                in_quote = None
            continue
        if ch in ("'", '"', "“"):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf.clear()
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_operator_definition(name: str, definition: str) -> tuple[int, int, bool]:
    """Parse ``(num_inputs, min_inputs, uses_lookback)`` from an operator definition string."""
    defn = (definition or "").strip()
    if not defn:
        return (1, 1, False)

    first_line = defn.splitlines()[0].strip()
    if _INFIX_RE.search(first_line):
        return (2, 2, False)

    args_str = _extract_first_call_args(first_line)
    if args_str is None:
        token = first_line.lower()
        if token == "in":
            return (2, 2, False)
        if token == "universe_size":
            return (0, 0, False)
        return (1, 1, False)

    if not args_str.strip():
        return (0, 0, False)

    required_inputs = 0
    optional_inputs = 0
    uses_lookback = False

    for raw_arg in _split_top_level_args(args_str):
        arg = raw_arg.strip()
        if not arg or arg in ("...", ".."):
            continue
        if arg.endswith(".."):
            arg = arg[:-2].rstrip()
            if not arg:
                continue

        if "=" in arg:
            param_raw, default_raw = arg.split("=", 1)
            param_norm = re.sub(r"\s+", "", param_raw).lower()
            default_norm = default_raw.strip().lower()
            if param_norm in _LOOKBACK_PARAM_NAMES or default_norm == "d":
                uses_lookback = True
            elif param_norm in _OPTIONAL_DATA_INPUT_NAMES:
                optional_inputs += 1
            continue

        arg_norm = re.sub(r"\s+", "", arg).lower()
        if arg_norm in _LOOKBACK_PARAM_NAMES:
            uses_lookback = True
            continue
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", arg_norm):
            continue
        required_inputs += 1

    num_inputs = required_inputs + optional_inputs
    min_inputs = required_inputs
    return (num_inputs, min_inputs, uses_lookback)


def _parse_rate_limit_wait(headers: Optional[Mapping[str, Any]]) -> float:
    """Determine wait duration in seconds from ``x-ratelimit-remaining-*`` headers if low."""
    if not headers:
        return 0.0

    lowered = {str(k).lower(): str(v) for k, v in headers.items() if v is not None}
    max_wait = 0.0

    for key, raw_val in lowered.items():
        if not key.startswith("x-ratelimit-remaining"):
            continue
        try:
            remaining = float(raw_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(remaining) or remaining > 1.0:
            continue

        suffix = key[len("x-ratelimit-remaining") :]
        reset_raw = (
            lowered.get(f"x-ratelimit-reset{suffix}")
            or lowered.get("x-ratelimit-reset")
            or lowered.get("retry-after")
        )
        wait_s: Optional[float] = None
        if reset_raw is not None:
            try:
                reset_val = float(reset_raw)
                if not math.isnan(reset_val) and not math.isinf(reset_val):
                    if reset_val > 1_000_000_000:
                        wait_s = max(0.0, reset_val - time.time())
                    elif reset_val > 0:
                        wait_s = reset_val
            except (TypeError, ValueError):
                wait_s = None

        if remaining <= 0.0:
            chosen = 1.0 if wait_s is None else min(max(wait_s, 0.1), 60.0)
        else:
            chosen = 0.25 if wait_s is None else min(max(wait_s, 0.1), 2.0)

        if chosen > max_wait:
            max_wait = chosen

    return max_wait


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    """Safely read a JSON file containing a list of dict records."""
    try:
        raw_text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("Failed to read cache file %s: %s", path, exc)
        return []
    if not raw_text:
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("Skipping invalid JSON cache file %s: %s", path, exc)
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        parsed = parsed["results"]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


class CatalogSync:
    """Synchronizes BRAIN operators, datasets, and datafields into ``AlphaDB``."""

    def __init__(self, db: AlphaDB, client: Optional[BrainClient] = None) -> None:
        self.db = db
        self.client = client

    def _require_client(self) -> BrainClient:
        if self.client is None:
            raise RuntimeError("BrainClient is required for remote catalog synchronization")
        return self.client

    def _respect_ratelimit_headers(self, headers: Optional[Mapping[str, Any]]) -> None:
        wait_s = _parse_rate_limit_wait(headers)
        if wait_s > 0.0:
            logger.info("Rate-limit header remaining low; sleeping %.2fs", wait_s)
            time.sleep(wait_s)

    def _can_use_direct_brain_request(self, method_name: str, orig_method: Any) -> bool:
        client = self.client
        if client is None or not isinstance(client, BrainClient):
            return False
        if method_name in client.__dict__:
            return False
        return getattr(type(client), method_name, None) is orig_method

    def _fetch_datasets_page(
        self,
        *,
        instrument_type: str,
        region: str,
        delay: int,
        universe: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        client = self._require_client()
        if self._can_use_direct_brain_request("get_datasets", _ORIG_GET_DATASETS):
            params = {
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "universe": universe,
                "limit": limit,
                "offset": offset,
            }
            resp = client._request("GET", "/data-sets", params=params)
            _handle_response(resp, context="GET /data-sets")
            self._respect_ratelimit_headers(getattr(resp, "headers", None))
            return _parse_json_dict(resp)
        return client.get_datasets(
            instrument_type=instrument_type,
            region=region,
            delay=delay,
            universe=universe,
            limit=limit,
            offset=offset,
        )

    def _fetch_datafields_page(
        self,
        *,
        instrument_type: str,
        region: str,
        delay: int,
        universe: str,
        dataset_id: str,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        client = self._require_client()
        if self._can_use_direct_brain_request("get_datafields_page", _ORIG_GET_DATAFIELDS_PAGE):
            params: dict[str, Any] = {
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "universe": universe,
                "limit": limit,
                "offset": offset,
            }
            if dataset_id:
                params["dataset.id"] = dataset_id
            resp = client._request("GET", "/data-fields", params=params)
            _handle_response(resp, context="GET /data-fields")
            self._respect_ratelimit_headers(getattr(resp, "headers", None))
            return _parse_json_dict(resp)
        return client.get_datafields_page(
            instrument_type=instrument_type,
            region=region,
            delay=delay,
            universe=universe,
            dataset_id=dataset_id,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _build_operator_record(raw: Mapping[str, Any]) -> Optional[OperatorRecord]:
        name = str(raw.get("name") or "").strip()
        if not name:
            return None
        definition = str(raw.get("definition") or "")
        parsed_num, parsed_min, parsed_lookback = parse_operator_definition(name, definition)

        num_inputs = raw.get("num_inputs", raw.get("numInputs"))
        min_inputs = raw.get("min_inputs", raw.get("minInputs"))
        uses_lookback = raw.get("uses_lookback", raw.get("usesLookback"))

        scope_val = raw.get("scope_json", raw.get("scope", []))
        if isinstance(scope_val, str):
            scope_json = scope_val or "[]"
        else:
            scope_json = json.dumps(scope_val if scope_val is not None else [], sort_keys=True)

        return OperatorRecord(
            name=name,
            category=_extract_label(raw.get("category")),
            scope_json=scope_json,
            definition=definition,
            description=str(raw.get("description") or ""),
            num_inputs=parsed_num if num_inputs is None else int(num_inputs),
            min_inputs=parsed_min if min_inputs is None else int(min_inputs),
            uses_lookback=parsed_lookback if uses_lookback is None else bool(uses_lookback),
        )

    def sync_operators(self) -> int:
        """Fetch BRAIN ``/operators``, parse signatures, and upsert into ``AlphaDB``."""
        client = self._require_client()
        raw_ops = client.get_operators()
        records: list[OperatorRecord] = []
        for item in raw_ops:
            if isinstance(item, Mapping):
                rec = self._build_operator_record(item)
                if rec is not None:
                    records.append(rec)
        return self.db.upsert_operators(records)

    def sync_region(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        universe: str = "TOP3000",
        max_datasets: Optional[int] = None,
    ) -> tuple[int, int]:
        """Paginate ``/data-sets`` and each dataset's ``/data-fields`` into ``AlphaDB``."""
        self._require_client()
        if max_datasets is not None and max_datasets <= 0:
            return (0, 0)

        page_limit = 50
        all_datasets: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._fetch_datasets_page(
                instrument_type=instrument_type,
                region=region,
                delay=delay,
                universe=universe,
                limit=page_limit,
                offset=offset,
            )
            results = payload.get("results") or []
            if not results:
                break
            all_datasets.extend(results)
            if max_datasets is not None and len(all_datasets) >= max_datasets:
                all_datasets = all_datasets[:max_datasets]
                break

            offset += page_limit
            total_count = payload.get("count")
            if total_count is not None:
                if offset >= int(total_count):
                    break
            elif len(results) < page_limit:
                break

        datasets_synced = self.db.upsert_datasets(
            all_datasets,
            region=region,
            universe=universe,
            delay=delay,
            instrument_type=instrument_type,
        )

        datafields_synced = 0
        for ds in all_datasets:
            ds_id = str(ds.get("id") or ds.get("dataset_id") or "").strip()
            if not ds_id:
                continue
            ds_fields: list[dict[str, Any]] = []
            field_offset = 0
            while True:
                page = self._fetch_datafields_page(
                    instrument_type=instrument_type,
                    region=region,
                    delay=delay,
                    universe=universe,
                    dataset_id=ds_id,
                    limit=page_limit,
                    offset=field_offset,
                )
                results = page.get("results") or []
                if not results:
                    break
                ds_fields.extend(results)
                field_offset += page_limit
                total_count = page.get("count")
                if total_count is not None:
                    if field_offset >= int(total_count):
                        break
                elif len(results) < page_limit:
                    break

            if ds_fields:
                datafields_synced += self.db.upsert_datafields(
                    ds_fields,
                    region=region,
                    universe=universe,
                    delay=delay,
                    dataset_id=ds_id,
                )

        return (datasets_synced, datafields_synced)

    @staticmethod
    def _parse_cache_filename_market(path: Path) -> tuple[Optional[str], int, Optional[str]]:
        name = path.stem
        if name.startswith("data_fields_cache_"):
            rest = name[len("data_fields_cache_") :]
            parts = rest.split("_", 2)
            if len(parts) == 3:
                region_str, delay_str, universe_str = parts
                try:
                    delay_val = int(delay_str)
                except ValueError:
                    delay_val = 1
                return (region_str or None, delay_val, universe_str or None)
        elif name.startswith("datafields_cache_"):
            region_str = name[len("datafields_cache_") :]
            return (region_str or None, 1, None)
        return (None, 1, None)

    def bootstrap_from_local_cache(self, workspace_root: str = ".") -> dict[str, int]:
        """Import local operator and datafield JSON caches into ``AlphaDB`` and synthesize datasets."""
        root = Path(workspace_root).expanduser()

        # 1. Operators from local operatorRAW.json files
        op_files = [
            root / "worldquant-miner" / "generation_two" / "constants" / "operatorRAW.json",
            root / "alpha_mining" / "constants" / "operatorRAW.json",
        ]
        op_by_name: dict[str, OperatorRecord] = {}
        for op_path in op_files:
            if not op_path.is_file():
                continue
            for item in _load_json_list(op_path):
                rec = self._build_operator_record(item)
                if rec is not None:
                    op_by_name[rec.name] = rec
        operators_count = self.db.upsert_operators(list(op_by_name.values()))

        # 2. Optional datasets_cache.json
        datasets_count = 0
        datasets_cache_path = root / "constants" / "datasets_cache.json"
        if datasets_cache_path.is_file():
            cached_datasets = _load_json_list(datasets_cache_path)
            if cached_datasets:
                datasets_count += self.db.upsert_datasets(cached_datasets)

        # 3. Datafields caches (legacy first, primary constants/ overrides)
        cache_files: list[Path] = []
        legacy_dir = root / "worldquant-miner" / "generation_two" / "constants"
        if legacy_dir.is_dir():
            cache_files.extend(sorted(legacy_dir.glob("data_fields_cache_*.json")))
        constants_dir = root / "constants"
        if constants_dir.is_dir():
            cache_files.extend(sorted(constants_dir.glob("datafields_cache_*.json")))
            cache_files.extend(sorted(constants_dir.glob("data_fields_cache_*.json")))

        deduped_fields: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        markets_seen: set[tuple[str, str, int]] = set()

        for cache_path in cache_files:
            file_region, file_delay, file_universe = self._parse_cache_filename_market(cache_path)
            for item in _load_json_list(cache_path):
                field_id = str(item.get("field_id") or item.get("id") or "").strip()
                if not field_id:
                    continue
                region = str(item.get("region") or file_region or "USA")
                universe = str(item.get("universe") or file_universe or "TOP3000")
                raw_delay = item.get("delay")
                delay = int(raw_delay) if raw_delay is not None else int(file_delay)

                normalized = dict(item)
                normalized["region"] = region
                normalized["universe"] = universe
                normalized["delay"] = delay
                deduped_fields[(field_id, region, universe, delay)] = normalized
                markets_seen.add((region, universe, delay))

        datafields_list = list(deduped_fields.values())
        datafields_count = self.db.upsert_datafields(datafields_list)

        # 4. Synthesize missing dataset records from loaded datafields
        existing_ds_keys: set[tuple[str, str, str, int]] = set()
        for m_region, m_universe, m_delay in markets_seen:
            for row in self.db.get_datasets(m_region, m_universe, m_delay):
                existing_ds_keys.add(
                    (
                        str(row["id"]),
                        str(row["region"]),
                        str(row["universe"]),
                        int(row["delay"]),
                    )
                )

        grouped_fields: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for item in datafields_list:
            raw_ds = item.get("dataset_id") or item.get("dataset")
            if isinstance(raw_ds, Mapping):
                ds_id = str(raw_ds.get("id") or "").strip()
            else:
                ds_id = str(raw_ds or "").strip()
            if not ds_id:
                continue
            key = (ds_id, str(item["region"]), str(item["universe"]), int(item["delay"]))
            grouped_fields[key].append(item)

        synthesized: list[dict[str, Any]] = []
        for (ds_id, ds_region, ds_universe, ds_delay), items in sorted(grouped_fields.items()):
            if (ds_id, ds_region, ds_universe, ds_delay) in existing_ds_keys:
                continue

            ds_name = ds_id
            categories: Counter[str] = Counter()
            subcategories: Counter[str] = Counter()
            coverages: list[float] = []
            user_counts: list[int] = []
            alpha_counts: list[int] = []

            for f in items:
                raw_ds = f.get("dataset")
                if isinstance(raw_ds, Mapping) and raw_ds.get("name"):
                    ds_name = str(raw_ds["name"])
                cat = _extract_label(f.get("category"))
                if cat:
                    categories[cat] += 1
                subcat = _extract_label(f.get("subcategory"))
                if subcat:
                    subcategories[subcat] += 1
                coverages.append(float(f.get("coverage") or 0.0))
                user_counts.append(int(f.get("user_count", f.get("userCount")) or 0))
                alpha_counts.append(int(f.get("alpha_count", f.get("alphaCount")) or 0))

            synthesized.append(
                {
                    "id": ds_id,
                    "region": ds_region,
                    "universe": ds_universe,
                    "delay": ds_delay,
                    "instrument_type": "EQUITY",
                    "name": ds_name,
                    "category": categories.most_common(1)[0][0] if categories else "",
                    "subcategory": subcategories.most_common(1)[0][0] if subcategories else "",
                    "coverage": sum(coverages) / len(coverages) if coverages else 0.0,
                    "field_count": len(items),
                    "user_count": max(user_counts) if user_counts else 0,
                    "value_score": 0.0,
                    "alpha_count": sum(alpha_counts),
                }
            )

        if synthesized:
            datasets_count += self.db.upsert_datasets(synthesized)

        return {
            "operators": operators_count,
            "datasets": datasets_count,
            "datafields": datafields_count,
        }


__all__ = [
    "CatalogSync",
    "parse_operator_definition",
    "wrap_vector_field",
]
