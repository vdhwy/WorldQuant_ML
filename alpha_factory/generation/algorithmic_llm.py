"""Algorithmic and LLM-assisted FastExpr alpha generator with zero-stall fallback."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import requests

from alpha_factory.storage.models import DataFieldRecord, OperatorRecord

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 10, 20, 60)
_MAX_POOL_SIZE = 40

_DISALLOWED_OPERATORS: frozenset[str] = frozenset(
    {
        "and",
        "or",
        "not",
        "equal",
        "not_equal",
        "less",
        "greater",
        "less_equal",
        "greater_equal",
        "if_else",
        "trade_when",
        "bucket",
        "pasteurize",
        "purify",
        "group_cartesian_product",
        "group_extra",
        "group_loop",
        "is_nan",
        "is_not_nan",
        "is_finite",
        "filter",
        "days_from_last_change",
        "humandate",
        "inst_tvr",
        "vec_choose",
        "vec_filter",
        "vec_percentage",
        "vec_powersum",
        "vec_norm",
    }
)

_SAFE_UNARY_CS_PRIORITY: frozenset[str] = frozenset(
    {
        "rank",
        "zscore",
        "normalize",
        "scale",
        "winsorize",
        "quantile",
        "sigmoid",
        "tanh",
        "arc_tan",
        "sign",
        "abs",
        "truncate",
    }
)

_RISKY_UNARY_CS: frozenset[str] = frozenset(
    {
        "log",
        "ln",
        "sqrt",
        "inverse",
        "s_log_1p",
        "rank_gmean_amean_diff",
    }
)

_SAFE_GROUP_OPS: frozenset[str] = frozenset(
    {
        "group_neutralize",
        "group_rank",
        "group_zscore",
        "group_scale",
        "group_demean",
        "group_mean",
        "group_std_dev",
        "group_median",
        "group_max",
        "group_min",
    }
)

_PREFERRED_TS_OPS: frozenset[str] = frozenset(
    {
        "ts_rank",
        "ts_zscore",
        "ts_mean",
        "ts_delta",
        "ts_std_dev",
        "ts_decay_linear",
        "ts_backfill",
        "ts_sum",
        "ts_ir",
        "ts_av_diff",
        "ts_corr",
        "ts_covariance",
    }
)

_PREFERRED_BINARY_CS_OPS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "min",
        "max",
        "signed_power",
    }
)

_LOOKBACK_PARAM_NAMES: frozenset[str] = frozenset(
    {"d", "days", "window", "period", "lookback", "n", "k", "lag"}
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_OP_PLACEHOLDER_RE = re.compile(r"\bOPERATOR\d+\b")
_FIELD_PLACEHOLDER_RE = re.compile(r"\bDATA_FIELD\d+\b")

_DEFAULT_FALLBACK_OPERATORS: tuple[OperatorRecord, ...] = (
    OperatorRecord(name="rank", category="Cross Sectional", num_inputs=1, min_inputs=1, uses_lookback=False),
    OperatorRecord(name="zscore", category="Cross Sectional", num_inputs=1, min_inputs=1, uses_lookback=False),
    OperatorRecord(name="normalize", category="Cross Sectional", num_inputs=1, min_inputs=1, uses_lookback=False),
    OperatorRecord(name="scale", category="Cross Sectional", num_inputs=1, min_inputs=1, uses_lookback=False),
    OperatorRecord(name="winsorize", category="Cross Sectional", num_inputs=1, min_inputs=1, uses_lookback=False),
    OperatorRecord(name="ts_rank", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_zscore", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_mean", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_delta", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_std_dev", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_decay_linear", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_backfill", category="Time Series", num_inputs=1, min_inputs=1, uses_lookback=True),
    OperatorRecord(name="ts_corr", category="Time Series", num_inputs=2, min_inputs=2, uses_lookback=True),
    OperatorRecord(name="ts_covariance", category="Time Series", num_inputs=2, min_inputs=2, uses_lookback=True),
    OperatorRecord(name="add", category="Arithmetic", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="subtract", category="Arithmetic", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="multiply", category="Arithmetic", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="group_neutralize", category="Group", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="group_rank", category="Group", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="group_zscore", category="Group", num_inputs=2, min_inputs=2, uses_lookback=False),
    OperatorRecord(name="vec_avg", category="Vector", num_inputs=1, min_inputs=1, uses_lookback=False),
)


@dataclass(frozen=True)
class _OperatorBuckets:
    unary_ts: tuple[OperatorRecord, ...]
    unary_cs: tuple[OperatorRecord, ...]
    binary_ts: tuple[OperatorRecord, ...]
    binary_cs: tuple[OperatorRecord, ...]
    group_ops: tuple[OperatorRecord, ...]
    vector_ops: tuple[OperatorRecord, ...]
    slot_lookup: dict[str, str]


@dataclass(frozen=True)
class _SkeletonInstance:
    skeleton_id: int
    template: str
    op_slots: dict[str, str]
    field_slots: tuple[str, ...]
    is_contrast: bool = False
    lookbacks: tuple[int, ...] = ()


def _parse_scopes(scope_json: str) -> set[str]:
    if not scope_json:
        return set()
    try:
        parsed = json.loads(scope_json)
    except (TypeError, ValueError):
        return set()
    if isinstance(parsed, list):
        return {str(item).strip().upper() for item in parsed if item}
    return set()


def _infer_expr_arity(op: OperatorRecord) -> int:
    """Infer required expression-input arity from OperatorRecord fields and definition."""
    min_in = max(1, int(op.min_inputs or 1))
    if not op.uses_lookback:
        return min_in
    definition = (op.definition or "").strip()
    if "(" in definition and ")" in definition:
        inner = definition.split("(", 1)[1].rsplit(")", 1)[0]
        raw_params = [p.strip() for p in inner.split(",") if p.strip()]
        required_params = [p for p in raw_params if "=" not in p]
        if required_params and len(required_params) == min_in and min_in >= 2:
            tail = required_params[-1].lower().split()[0]
            if tail in _LOOKBACK_PARAM_NAMES:
                return min_in - 1
    return min_in


def _classify_operator(op: OperatorRecord) -> Optional[str]:
    """Map an OperatorRecord to its compatible skeleton slot bucket, or None if unsuitable."""
    name = (op.name or "").strip()
    if not name or not _IDENT_RE.fullmatch(name) or name in _DISALLOWED_OPERATORS:
        return None
    scopes = _parse_scopes(op.scope_json)
    if scopes and "REGULAR" not in scopes:
        return None

    category_l = (op.category or "").lower()
    if name.startswith("vec_") or "vector" in category_l:
        if int(op.min_inputs or 1) <= 1 and not op.uses_lookback:
            return "vector"
        return None

    if name in _SAFE_GROUP_OPS or name.startswith("group_") or "group" in category_l:
        if op.uses_lookback:
            return None
        if name in _SAFE_GROUP_OPS or (1 <= int(op.min_inputs or 1) <= 2 and "cartesian" not in name):
            return "group"
        return None

    arity = _infer_expr_arity(op)
    if arity == 1:
        return "unary_ts" if op.uses_lookback else "unary_cs"
    if arity == 2:
        return "binary_ts" if op.uses_lookback else "binary_cs"
    return None


def _build_buckets(operators: Sequence[OperatorRecord]) -> _OperatorBuckets:
    source_ops = list(operators) if operators else list(_DEFAULT_FALLBACK_OPERATORS)
    unary_ts: list[OperatorRecord] = []
    unary_cs: list[OperatorRecord] = []
    binary_ts: list[OperatorRecord] = []
    binary_cs: list[OperatorRecord] = []
    group_ops: list[OperatorRecord] = []
    vector_ops: list[OperatorRecord] = []
    slot_lookup: dict[str, str] = {}

    seen_names: set[str] = set()
    for op in source_ops:
        if op.name in seen_names:
            continue
        bucket = _classify_operator(op)
        if bucket is None:
            continue
        seen_names.add(op.name)
        slot_lookup[op.name] = bucket
        if bucket == "unary_ts":
            unary_ts.append(op)
        elif bucket == "unary_cs":
            unary_cs.append(op)
        elif bucket == "binary_ts":
            binary_ts.append(op)
        elif bucket == "binary_cs":
            binary_cs.append(op)
        elif bucket == "group":
            group_ops.append(op)
        elif bucket == "vector":
            vector_ops.append(op)

    # Guarantee at least one fallback bucket if caller supplied only exotic/unusable operators.
    if not any((unary_ts, unary_cs, binary_ts, binary_cs, group_ops)):
        for op in _DEFAULT_FALLBACK_OPERATORS:
            bucket = _classify_operator(op)
            if bucket is None:
                continue
            slot_lookup[op.name] = bucket
            if bucket == "unary_ts":
                unary_ts.append(op)
            elif bucket == "unary_cs":
                unary_cs.append(op)
            elif bucket == "binary_ts":
                binary_ts.append(op)
            elif bucket == "binary_cs":
                binary_cs.append(op)
            elif bucket == "group":
                group_ops.append(op)
            elif bucket == "vector":
                vector_ops.append(op)

    return _OperatorBuckets(
        unary_ts=tuple(unary_ts),
        unary_cs=tuple(unary_cs),
        binary_ts=tuple(binary_ts),
        binary_cs=tuple(binary_cs),
        group_ops=tuple(group_ops),
        vector_ops=tuple(vector_ops),
        slot_lookup=slot_lookup,
    )


def _operator_weight(op: OperatorRecord, bucket: str) -> float:
    if bucket == "unary_cs":
        if op.name in _SAFE_UNARY_CS_PRIORITY:
            return 4.0
        if op.name in _RISKY_UNARY_CS:
            return 0.35
        return 1.5
    if bucket == "unary_ts":
        return 3.5 if op.name in _PREFERRED_TS_OPS else 1.5
    if bucket == "binary_ts":
        return 3.0 if op.name in _PREFERRED_TS_OPS else 1.2
    if bucket == "binary_cs":
        if op.name in _PREFERRED_BINARY_CS_OPS:
            return 3.0
        if op.name == "divide":
            return 0.6
        return 1.2
    if bucket == "group":
        if op.name == "group_neutralize":
            return 4.0
        if op.name in {"group_rank", "group_zscore"}:
            return 3.0
        return 1.5
    if bucket == "vector":
        return 6.0 if op.name == "vec_avg" else 1.5
    return 1.0


def _field_weight(field: DataFieldRecord) -> float:
    ftype = (field.field_type or "MATRIX").upper()
    if ftype == "MATRIX":
        type_mult = 2.0
    elif ftype == "VECTOR":
        type_mult = 1.0
    else:
        type_mult = 0.2
    if bool(field.is_event_input) or ftype == "EVENT":
        type_mult *= 0.4
    cov = max(0.0, min(1.0, float(field.coverage or 0.0)))
    cov_mult = 0.05 if cov <= 0.0 else (0.25 + cov)
    user_mult = 1.0 + 0.15 * math.log1p(max(0, int(field.user_count or 0)))
    alpha_mult = 1.0 + 0.10 * math.log1p(min(500, max(0, int(field.alpha_count or 0))))
    return max(1e-4, type_mult * cov_mult * user_mult * alpha_mult)


def _weighted_choice(rng: random.Random, items: Sequence[Any], weights: Sequence[float]) -> Any:
    if len(items) == 1:
        return items[0]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(items)
    return rng.choices(items, weights=weights, k=1)[0]


def _extract_json_payload(raw_text: str) -> Any:
    """Parse JSON from raw LLM output, tolerating <think> tags and markdown fences."""
    cleaned = _THINK_RE.sub("", raw_text or "").strip()
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    for candidate in _json_substrings(cleaned):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _json_substrings(text: str) -> list[str]:
    candidates: list[str] = [text]
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj : last_obj + 1])
    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr : last_arr + 1])
    return candidates


def _is_oom_or_missing_response(status_code: int, body_text: str) -> bool:
    if status_code == 404:
        return True
    lower = (body_text or "").lower()
    if any(
        token in lower
        for token in (
            "out of memory",
            "not enough memory",
            "requires more system memory",
            "oom",
            "model not found",
            "not found",
        )
    ):
        return True
    return status_code in (500, 502, 503, 504)


class AlgorithmicLLMGenerator:
    """Hybrid structural-skeleton generator using Ollama/OpenAI-compat LLM or algorithmic fallback."""

    def __init__(
        self,
        llm_cfg: Optional[Mapping[str, Any]],
        operators: Sequence[OperatorRecord],
        datafields: Sequence[DataFieldRecord],
        seed: Optional[int] = None,
    ) -> None:
        self.llm_cfg: dict[str, Any] = dict(llm_cfg or {})
        self.operators: list[OperatorRecord] = list(operators)
        self.datafields: list[DataFieldRecord] = list(datafields)
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._buckets = _build_buckets(self.operators)
        self._model_index = 0

    def _effective_backend(self) -> str:
        if "backend" in self.llm_cfg:
            raw = str(self.llm_cfg.get("backend") or "none").strip().lower()
            return raw or "none"
        return "ollama" if self.llm_cfg else "none"

    def _select_candidate_fields(
        self,
        region: str,
        universe: str,
        delay: int,
        datafields: Sequence[DataFieldRecord],
    ) -> list[DataFieldRecord]:
        pool = list(datafields) if datafields else list(self.datafields)
        if not pool:
            return []
        exact = [
            f
            for f in pool
            if f.region == region and f.universe == universe and int(f.delay) == int(delay)
        ]
        if exact:
            pool = exact
        else:
            by_region = [f for f in pool if f.region == region]
            if by_region:
                pool = by_region

        non_group = [f for f in pool if (f.field_type or "MATRIX").upper() != "GROUP"]
        return non_group if non_group else pool

    def _build_field_pool(self, fields: Sequence[DataFieldRecord]) -> list[DataFieldRecord]:
        """Select up to 40 high-value, dataset-diverse fields and shuffle them."""
        if len(fields) <= _MAX_POOL_SIZE:
            chosen = list(fields)
            self._rng.shuffle(chosen)
            return chosen

        # Efraimidis-Spirakis weighted sampling without replacement.
        scored: list[tuple[float, DataFieldRecord]] = []
        for f in fields:
            u = max(1e-12, self._rng.random())
            w = _field_weight(f)
            key = u ** (1.0 / w)
            scored.append((key, f))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [pair[1] for pair in scored[:_MAX_POOL_SIZE]]
        self._rng.shuffle(chosen)
        return chosen

    def _build_operator_pool(self) -> list[OperatorRecord]:
        """Select up to 40 diverse operators covering all available buckets and shuffle them."""
        buckets = [
            list(self._buckets.unary_ts),
            list(self._buckets.unary_cs),
            list(self._buckets.group_ops),
            list(self._buckets.binary_ts),
            list(self._buckets.binary_cs),
        ]
        chosen: list[OperatorRecord] = []
        seen: set[str] = set()

        # Guarantee representation from each bucket first.
        for bucket in buckets:
            if not bucket:
                continue
            shuffled_bucket = list(bucket)
            self._rng.shuffle(shuffled_bucket)
            for op in shuffled_bucket[:8]:
                if op.name not in seen and len(chosen) < _MAX_POOL_SIZE:
                    seen.add(op.name)
                    chosen.append(op)

        # Top up to _MAX_POOL_SIZE from remaining usable operators.
        if len(chosen) < _MAX_POOL_SIZE:
            remaining: list[OperatorRecord] = []
            for bucket in buckets:
                for op in bucket:
                    if op.name not in seen:
                        remaining.append(op)
            self._rng.shuffle(remaining)
            for op in remaining:
                if len(chosen) >= _MAX_POOL_SIZE:
                    break
                if op.name not in seen:
                    seen.add(op.name)
                    chosen.append(op)

        self._rng.shuffle(chosen)
        return chosen

    def _valid_groups(self, region: str) -> list[str]:
        groups = ["industry", "subindustry", "sector", "market"]
        if region.upper() in {"GLB", "EUR", "ASI", "USA", "IND"}:
            groups.append("country")
        return groups

    def _eligible_skeleton_specs(self, num_fields_available: int) -> list[dict[str, Any]]:
        b = self._buckets
        has_uts = bool(b.unary_ts)
        has_ucs = bool(b.unary_cs)
        has_bts = bool(b.binary_ts)
        has_bcs = bool(b.binary_cs)
        has_grp = bool(b.group_ops)
        has_group_neut = any(op.name == "group_neutralize" for op in b.group_ops)
        multi_field = num_fields_available >= 2

        specs: list[dict[str, Any]] = []
        if has_uts and has_ucs:
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(DATA_FIELD1), {lb1})",
                    "op_slots": {"OPERATOR1": "unary_ts", "OPERATOR2": "unary_cs"},
                    "num_fields": 1,
                    "num_lbs": 1,
                    "uses_group": False,
                    "is_contrast": False,
                }
            )
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(DATA_FIELD1, {lb1}))",
                    "op_slots": {"OPERATOR1": "unary_cs", "OPERATOR2": "unary_ts"},
                    "num_fields": 1,
                    "num_lbs": 1,
                    "uses_group": False,
                    "is_contrast": False,
                }
            )
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(DATA_FIELD1, {lb1}) - OPERATOR3(DATA_FIELD1, {lb2}))",
                    "op_slots": {
                        "OPERATOR1": "unary_cs",
                        "OPERATOR2": "unary_ts",
                        "OPERATOR3": "unary_ts",
                    },
                    "num_fields": 1,
                    "num_lbs": 2,
                    "uses_group": False,
                    "is_contrast": True,
                }
            )
        if has_group_neut and has_uts:
            specs.append(
                {
                    "template": "group_neutralize(OPERATOR1(DATA_FIELD1, {lb1}), {group})",
                    "op_slots": {"OPERATOR1": "unary_ts"},
                    "num_fields": 1,
                    "num_lbs": 1,
                    "uses_group": True,
                    "is_contrast": False,
                }
            )
        if has_grp and has_uts and has_ucs:
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(OPERATOR3(DATA_FIELD1, {lb1}), {group}))",
                    "op_slots": {
                        "OPERATOR1": "unary_cs",
                        "OPERATOR2": "group",
                        "OPERATOR3": "unary_ts",
                    },
                    "num_fields": 1,
                    "num_lbs": 1,
                    "uses_group": True,
                    "is_contrast": False,
                }
            )
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(OPERATOR3(DATA_FIELD1, {lb1})), {group})",
                    "op_slots": {
                        "OPERATOR1": "group",
                        "OPERATOR2": "unary_cs",
                        "OPERATOR3": "unary_ts",
                    },
                    "num_fields": 1,
                    "num_lbs": 1,
                    "uses_group": True,
                    "is_contrast": False,
                }
            )
        if multi_field and has_bts:
            specs.append(
                {
                    "template": "OPERATOR1(DATA_FIELD1, DATA_FIELD2, {lb1})",
                    "op_slots": {"OPERATOR1": "binary_ts"},
                    "num_fields": 2,
                    "num_lbs": 1,
                    "uses_group": False,
                    "is_contrast": False,
                }
            )
            if has_ucs:
                specs.append(
                    {
                        "template": "OPERATOR1(OPERATOR2(DATA_FIELD1, DATA_FIELD2, {lb1}))",
                        "op_slots": {"OPERATOR1": "unary_cs", "OPERATOR2": "binary_ts"},
                        "num_fields": 2,
                        "num_lbs": 1,
                        "uses_group": False,
                        "is_contrast": False,
                    }
                )
        if multi_field and has_bcs and has_uts:
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(DATA_FIELD1, DATA_FIELD2), {lb1})",
                    "op_slots": {"OPERATOR1": "unary_ts", "OPERATOR2": "binary_cs"},
                    "num_fields": 2,
                    "num_lbs": 1,
                    "uses_group": False,
                    "is_contrast": False,
                }
            )
            if has_ucs:
                specs.append(
                    {
                        "template": (
                            "OPERATOR1(OPERATOR2(OPERATOR3(DATA_FIELD1, {lb1}), "
                            "OPERATOR4(DATA_FIELD2, {lb1})))"
                        ),
                        "op_slots": {
                            "OPERATOR1": "unary_cs",
                            "OPERATOR2": "binary_cs",
                            "OPERATOR3": "unary_ts",
                            "OPERATOR4": "unary_ts",
                        },
                        "num_fields": 2,
                        "num_lbs": 1,
                        "uses_group": False,
                        "is_contrast": False,
                    }
                )
        if multi_field and has_grp and has_uts:
            specs.append(
                {
                    "template": "OPERATOR1(OPERATOR2(DATA_FIELD1, {lb1}) - OPERATOR3(DATA_FIELD2, {lb1}), {group})",
                    "op_slots": {
                        "OPERATOR1": "group",
                        "OPERATOR2": "unary_ts",
                        "OPERATOR3": "unary_ts",
                    },
                    "num_fields": 2,
                    "num_lbs": 1,
                    "uses_group": True,
                    "is_contrast": False,
                }
            )

        if not specs:
            if has_uts:
                specs.append(
                    {
                        "template": "OPERATOR1(DATA_FIELD1, {lb1})",
                        "op_slots": {"OPERATOR1": "unary_ts"},
                        "num_fields": 1,
                        "num_lbs": 1,
                        "uses_group": False,
                        "is_contrast": False,
                    }
                )
            if has_ucs:
                specs.append(
                    {
                        "template": "OPERATOR1(DATA_FIELD1)",
                        "op_slots": {"OPERATOR1": "unary_cs"},
                        "num_fields": 1,
                        "num_lbs": 0,
                        "uses_group": False,
                        "is_contrast": False,
                    }
                )
            if multi_field and has_bcs:
                specs.append(
                    {
                        "template": "OPERATOR1(DATA_FIELD1, DATA_FIELD2)",
                        "op_slots": {"OPERATOR1": "binary_cs"},
                        "num_fields": 2,
                        "num_lbs": 0,
                        "uses_group": False,
                        "is_contrast": False,
                    }
                )
            if has_grp:
                specs.append(
                    {
                        "template": "OPERATOR1(DATA_FIELD1, {group})",
                        "op_slots": {"OPERATOR1": "group"},
                        "num_fields": 1,
                        "num_lbs": 0,
                        "uses_group": True,
                        "is_contrast": False,
                    }
                )

        return specs

    def build_skeletons(
        self,
        count: int,
        lookbacks: Sequence[int] = _DEFAULT_LOOKBACKS,
        region: str = "USA",
        num_fields_available: int = 2,
    ) -> list[_SkeletonInstance]:
        """Create concrete structural operator skeletons with bound lookbacks and groups."""
        lb_list = [int(x) for x in lookbacks if int(x) > 0] or list(_DEFAULT_LOOKBACKS)
        specs = self._eligible_skeleton_specs(num_fields_available)
        if not specs:
            return []

        groups = self._valid_groups(region)
        skeletons: list[_SkeletonInstance] = []
        for idx in range(max(0, count)):
            spec = specs[idx % len(specs)] if idx < len(specs) else self._rng.choice(specs)
            lb1 = self._rng.choice(lb_list)
            if spec["num_lbs"] >= 2:
                other_lbs = [lb for lb in lb_list if lb != lb1]
                lb2 = self._rng.choice(other_lbs) if other_lbs else lb1
            else:
                lb2 = lb1
            group = self._rng.choice(groups)
            rendered = spec["template"].format(lb1=lb1, lb2=lb2, group=group)
            field_slots = tuple(f"DATA_FIELD{i + 1}" for i in range(spec["num_fields"]))
            skeletons.append(
                _SkeletonInstance(
                    skeleton_id=idx,
                    template=rendered,
                    op_slots=dict(spec["op_slots"]),
                    field_slots=field_slots,
                    is_contrast=bool(spec["is_contrast"]),
                    lookbacks=(lb1, lb2) if spec["num_lbs"] >= 2 else (lb1,),
                )
            )
        return skeletons

    def _format_field(self, field: DataFieldRecord) -> str:
        if (field.field_type or "MATRIX").upper() == "VECTOR":
            vec_ops = self._buckets.vector_ops
            if vec_ops:
                weights = [_operator_weight(op, "vector") for op in vec_ops]
                vec_op = _weighted_choice(self._rng, vec_ops, weights).name
            else:
                vec_op = "vec_avg"
            return f"{vec_op}({field.field_id})"
        return field.field_id

    def _pick_operator_for_slot(
        self,
        bucket: str,
        op_pool: Sequence[OperatorRecord],
        disallow_name: Optional[str] = None,
    ) -> Optional[str]:
        candidates = [
            op
            for op in op_pool
            if self._buckets.slot_lookup.get(op.name) == bucket and op.name != disallow_name
        ]
        if not candidates:
            full_bucket = getattr(self._buckets, "group_ops" if bucket == "group" else bucket, ())
            candidates = [op for op in full_bucket if op.name != disallow_name]
            if not candidates:
                candidates = list(full_bucket)
        if not candidates:
            return None
        weights = [_operator_weight(op, bucket) for op in candidates]
        return str(_weighted_choice(self._rng, candidates, weights).name)

    def _pick_field_for_slot(
        self,
        field_pool: Sequence[DataFieldRecord],
        disallow_id: Optional[str] = None,
    ) -> DataFieldRecord:
        candidates = [f for f in field_pool if f.field_id != disallow_id]
        if not candidates:
            candidates = list(field_pool)
        weights = [_field_weight(f) for f in candidates]
        return _weighted_choice(self._rng, candidates, weights)

    def _render_skeleton_with_fallback(
        self,
        skel: _SkeletonInstance,
        op_pool: Sequence[OperatorRecord],
        field_pool: Sequence[DataFieldRecord],
        chosen_op_indices: Optional[Mapping[str, Any]] = None,
        chosen_field_indices: Optional[Mapping[str, Any]] = None,
    ) -> Optional[tuple[str, str]]:
        expr = skel.template

        # Resolve operators, strictly checking arity/lookback compatibility against slot bucket.
        assigned_ops: dict[str, str] = {}
        op_index_map = chosen_op_indices or {}
        for slot_name, bucket in skel.op_slots.items():
            picked_name: Optional[str] = None
            raw_idx = op_index_map.get(slot_name)
            if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
                if 0 <= raw_idx < len(op_pool):
                    candidate_op = op_pool[raw_idx]
                    if self._buckets.slot_lookup.get(candidate_op.name) == bucket:
                        picked_name = candidate_op.name
            if picked_name is None:
                disallow = None
                if (
                    skel.is_contrast
                    and slot_name == "OPERATOR3"
                    and len(skel.lookbacks) >= 2
                    and skel.lookbacks[0] == skel.lookbacks[1]
                ):
                    disallow = assigned_ops.get("OPERATOR2")
                picked_name = self._pick_operator_for_slot(bucket, op_pool, disallow_name=disallow)
            if picked_name is None:
                return None
            assigned_ops[slot_name] = picked_name

        # Prevent identical canceling contrast branch when lookbacks match.
        if (
            skel.is_contrast
            and len(skel.lookbacks) >= 2
            and skel.lookbacks[0] == skel.lookbacks[1]
            and assigned_ops.get("OPERATOR2") == assigned_ops.get("OPERATOR3")
        ):
            alt = self._pick_operator_for_slot(
                skel.op_slots["OPERATOR3"],
                op_pool,
                disallow_name=assigned_ops.get("OPERATOR2"),
            )
            if alt:
                assigned_ops["OPERATOR3"] = alt

        # Resolve fields.
        assigned_fields: dict[str, DataFieldRecord] = {}
        used_field_ids: set[str] = set()
        field_index_map = chosen_field_indices or {}
        for slot_name in skel.field_slots:
            picked_field: Optional[DataFieldRecord] = None
            raw_idx = field_index_map.get(slot_name)
            if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
                if 0 <= raw_idx < len(field_pool):
                    candidate_f = field_pool[raw_idx]
                    if candidate_f.field_id not in used_field_ids or len(field_pool) == 1:
                        picked_field = candidate_f
            if picked_field is None:
                disallow = next(iter(used_field_ids), None) if len(field_pool) > 1 else None
                picked_field = self._pick_field_for_slot(field_pool, disallow_id=disallow)
            used_field_ids.add(picked_field.field_id)
            assigned_fields[slot_name] = picked_field

        # Replace placeholders deterministically via regex boundary matching.
        expr = _OP_PLACEHOLDER_RE.sub(
            lambda m: assigned_ops.get(m.group(0), m.group(0)),
            expr,
        )
        expr = _FIELD_PLACEHOLDER_RE.sub(
            lambda m: self._format_field(assigned_fields[m.group(0)])
            if m.group(0) in assigned_fields
            else m.group(0),
            expr,
        )

        primary_field = assigned_fields.get("DATA_FIELD1") or next(iter(assigned_fields.values()))
        return expr, str(primary_field.dataset_id or "")

    def _build_llm_prompt(
        self,
        region: str,
        universe: str,
        delay: int,
        skeletons: Sequence[_SkeletonInstance],
        op_pool: Sequence[OperatorRecord],
        field_pool: Sequence[DataFieldRecord],
    ) -> str:
        op_lines = [
            (
                f"[{i}] {op.name} | slot={self._buckets.slot_lookup.get(op.name, 'other')} "
                f"| inputs={_infer_expr_arity(op)} | lookback={op.uses_lookback}"
            )
            for i, op in enumerate(op_pool)
        ]
        field_lines = [
            (
                f"[{i}] {f.field_id} | type={f.field_type} | dataset={f.dataset_id} "
                f"| cov={f.coverage:.2f} | cat={f.category}"
            )
            for i, f in enumerate(field_pool)
        ]
        skel_lines = [
            (
                f"{s.skeleton_id}: {s.template} "
                f"(op_slots={json.dumps(s.op_slots)}, fields={list(s.field_slots)})"
            )
            for s in skeletons
        ]
        return (
            f"Design FastExpr alpha expressions for market ({region}, {universe}, delay={delay}).\n"
            f"Choose integer indices from the shuffled operator and datafield pools below.\n"
            f"Each OPERATORn index must match the required slot type.\n\n"
            f"OPERATORS (0..{len(op_pool) - 1}):\n" + "\n".join(op_lines) + "\n\n"
            f"DATAFIELDS (0..{len(field_pool) - 1}):\n" + "\n".join(field_lines) + "\n\n"
            f"SKELETONS:\n" + "\n".join(skel_lines) + "\n\n"
            "Return ONLY valid JSON with key 'selections':\n"
            '{"selections": [{"skeleton_id": 0, "operators": {"OPERATOR1": 0}, "fields": {"DATA_FIELD1": 0}}]}'
        )

    def _call_ollama(self, prompt: str) -> Optional[str]:
        base_url = str(self.llm_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        models = [str(m) for m in (self.llm_cfg.get("models") or []) if str(m).strip()]
        if not models:
            return None
        timeout_s = float(self.llm_cfg.get("timeout_s", 45))
        url = f"{base_url}/api/generate"

        with self._lock:
            start_idx = min(self._model_index, len(models) - 1)

        for idx in range(start_idx, len(models)):
            model_name = models[idx]
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.4},
            }
            try:
                resp = requests.post(url, json=payload, timeout=timeout_s)
            except requests.exceptions.Timeout:
                logger.warning("Ollama timeout on model %s; downgrading ladder", model_name)
                continue
            except requests.exceptions.RequestException as exc:
                logger.debug("Ollama unreachable (%s); falling back to algorithmic mode", exc)
                return None

            if resp.status_code != 200:
                if _is_oom_or_missing_response(resp.status_code, resp.text):
                    logger.warning(
                        "Ollama model %s failed (%s); downgrading ladder",
                        model_name,
                        resp.status_code,
                    )
                    continue
                logger.warning("Ollama HTTP %s on model %s; falling back", resp.status_code, model_name)
                return None

            try:
                body = resp.json()
            except ValueError:
                return resp.text

            if isinstance(body, Mapping) and body.get("error"):
                err_msg = str(body.get("error"))
                if _is_oom_or_missing_response(500, err_msg):
                    logger.warning("Ollama model %s error (%s); downgrading ladder", model_name, err_msg)
                    continue
                return None

            with self._lock:
                self._model_index = idx
            if isinstance(body, Mapping) and "response" in body:
                return str(body["response"])
            return json.dumps(body)

        return None

    def _call_openai_compat(self, prompt: str) -> Optional[str]:
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        api_base_url = str(self.llm_cfg.get("api_base_url") or "").strip()
        if not api_key or not api_base_url:
            return None

        endpoint = (
            api_base_url
            if api_base_url.endswith("/chat/completions")
            else f"{api_base_url.rstrip('/')}/chat/completions"
        )
        models = [str(m) for m in (self.llm_cfg.get("models") or []) if str(m).strip()]
        model = str(self.llm_cfg.get("api_model") or (models[0] if models else "gpt-4o-mini"))
        timeout_s = float(self.llm_cfg.get("timeout_s", 45))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quantitative alpha researcher. Output valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
        except requests.exceptions.RequestException as exc:
            logger.debug("OpenAI-compat endpoint error (%s); falling back", exc)
            return None

        if resp.status_code != 200:
            logger.warning("OpenAI-compat HTTP %s; falling back", resp.status_code)
            return None

        try:
            body = resp.json()
            choices = body.get("choices") or []
            if choices and isinstance(choices[0], Mapping):
                msg = choices[0].get("message") or {}
                if isinstance(msg, Mapping) and "content" in msg:
                    return str(msg["content"])
        except (ValueError, TypeError, KeyError, IndexError):
            return None
        return None

    def _parse_llm_selections(
        self,
        raw_text: Optional[str],
        skeletons: Sequence[_SkeletonInstance],
    ) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
        if not raw_text:
            return {}
        parsed = _extract_json_payload(raw_text)
        if parsed is None:
            return {}

        if isinstance(parsed, Mapping):
            raw_items = parsed.get("selections")
            if raw_items is None and "items" in parsed:
                raw_items = parsed.get("items")
            if raw_items is None and ("operators" in parsed or "fields" in parsed):
                raw_items = [parsed]
        elif isinstance(parsed, list):
            raw_items = parsed
        else:
            return {}

        if not isinstance(raw_items, list):
            return {}

        results: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for pos, entry in enumerate(raw_items):
            if not isinstance(entry, Mapping):
                continue
            raw_sid = entry.get("skeleton_id", pos)
            if not isinstance(raw_sid, int) or isinstance(raw_sid, bool):
                continue
            if not (0 <= raw_sid < len(skeletons)):
                continue
            skel = skeletons[raw_sid]

            op_map: dict[str, Any] = {}
            raw_ops = entry.get("operators")
            if isinstance(raw_ops, Mapping):
                op_map.update(raw_ops)
            elif isinstance(raw_ops, list):
                for slot_name, val in zip(skel.op_slots.keys(), raw_ops):
                    op_map[slot_name] = val
            for slot_name in skel.op_slots:
                if slot_name in entry and slot_name not in op_map:
                    op_map[slot_name] = entry[slot_name]

            field_map: dict[str, Any] = {}
            raw_fields = entry.get("fields")
            if isinstance(raw_fields, Mapping):
                field_map.update(raw_fields)
            elif isinstance(raw_fields, list):
                for slot_name, val in zip(skel.field_slots, raw_fields):
                    field_map[slot_name] = val
            for slot_name in skel.field_slots:
                if slot_name in entry and slot_name not in field_map:
                    field_map[slot_name] = entry[slot_name]

            results[raw_sid] = (op_map, field_map)
        return results

    def generate_batch(
        self,
        region: str,
        universe: str,
        delay: int,
        datafields: Sequence[DataFieldRecord],
        count: int = 10,
        lookbacks: Sequence[int] = _DEFAULT_LOOKBACKS,
    ) -> list[tuple[str, str]]:
        """Generate up to ``count`` valid ``(expression, dataset_id)`` alpha candidates."""
        if count <= 0:
            return []

        with self._lock:
            eligible_fields = self._select_candidate_fields(region, universe, delay, datafields)
            if not eligible_fields:
                return []

            field_pool = self._build_field_pool(eligible_fields)
            op_pool = self._build_operator_pool()
            skeletons = self.build_skeletons(
                count=count,
                lookbacks=lookbacks,
                region=region,
                num_fields_available=len(field_pool),
            )
            if not skeletons:
                return []

            backend = self._effective_backend()
            prompt = (
                self._build_llm_prompt(region, universe, delay, skeletons, op_pool, field_pool)
                if backend in {"ollama", "openai_compat"}
                else ""
            )

        llm_raw: Optional[str] = None
        if backend == "ollama":
            llm_raw = self._call_ollama(prompt)
        elif backend == "openai_compat":
            llm_raw = self._call_openai_compat(prompt)

        with self._lock:
            selections = self._parse_llm_selections(llm_raw, skeletons)
            results: list[tuple[str, str]] = []
            seen_exprs: set[str] = set()

            for skel in skeletons:
                op_sel, field_sel = selections.get(skel.skeleton_id, ({}, {}))
                rendered = self._render_skeleton_with_fallback(
                    skel,
                    op_pool=op_pool,
                    field_pool=field_pool,
                    chosen_op_indices=op_sel,
                    chosen_field_indices=field_sel,
                )
                if rendered is not None and rendered[0] not in seen_exprs:
                    seen_exprs.add(rendered[0])
                    results.append(rendered)

            # Top up via algorithmic fallback if collisions prevented reaching count.
            attempts = 0
            max_extra_attempts = max(10, count * 4)
            while len(results) < count and attempts < max_extra_attempts:
                attempts += 1
                extra_skels = self.build_skeletons(
                    count=1,
                    lookbacks=lookbacks,
                    region=region,
                    num_fields_available=len(field_pool),
                )
                if not extra_skels:
                    break
                rendered = self._render_skeleton_with_fallback(
                    extra_skels[0],
                    op_pool=op_pool,
                    field_pool=field_pool,
                )
                if rendered is not None and rendered[0] not in seen_exprs:
                    seen_exprs.add(rendered[0])
                    results.append(rendered)

            return results[:count]


__all__ = ["AlgorithmicLLMGenerator"]
