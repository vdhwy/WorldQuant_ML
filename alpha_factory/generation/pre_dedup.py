"""Pre-simulation exact config hashing, canonical AST fingerprinting, and deduplication."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import CandidateItem

GROUP_NAMES: frozenset[str] = frozenset(
    {
        "sector",
        "industry",
        "subindustry",
        "market",
        "country",
        "exchange",
    }
)

BOOLEAN_LITERALS: frozenset[str] = frozenset({"true", "false"})

DEFAULT_KNOWN_OPERATORS: frozenset[str] = frozenset(
    {
        # Arithmetic
        "abs",
        "add",
        "arc_cos",
        "arc_sin",
        "arc_tan",
        "ceiling",
        "cosh",
        "densify",
        "divide",
        "exp",
        "floor",
        "fraction",
        "inverse",
        "log",
        "max",
        "min",
        "mod",
        "multiply",
        "power",
        "reverse",
        "round",
        "round_down",
        "s_log_1p",
        "sign",
        "signed_power",
        "sinh",
        "sqrt",
        "subtract",
        "tanh",
        "to_nan",
        # Logical
        "and",
        "equal",
        "greater",
        "greater_equal",
        "if_else",
        "is_finite",
        "is_nan",
        "is_not_finite",
        "is_not_nan",
        "less",
        "less_equal",
        "nan_mask",
        "not",
        "not_equal",
        "or",
        "replace",
        # Time Series
        "days_from_last_change",
        "hump",
        "hump_decay",
        "inst_tvr",
        "jump_decay",
        "kth_element",
        "last_diff_value",
        "ts_arg_max",
        "ts_arg_min",
        "ts_av_diff",
        "ts_backfill",
        "ts_co_kurtosis",
        "ts_co_skewness",
        "ts_corr",
        "ts_count_nans",
        "ts_covariance",
        "ts_decay_exp_window",
        "ts_decay_linear",
        "ts_delay",
        "ts_delta",
        "ts_delta_limit",
        "ts_entropy",
        "ts_ir",
        "ts_kurtosis",
        "ts_max",
        "ts_mean",
        "ts_median",
        "ts_min",
        "ts_moment",
        "ts_partial_corr",
        "ts_poly_regression",
        "ts_product",
        "ts_quantile",
        "ts_rank",
        "ts_regression",
        "ts_returns",
        "ts_scale",
        "ts_skewness",
        "ts_std_dev",
        "ts_step",
        "ts_sum",
        "ts_target_tvr_decay",
        "ts_target_tvr_delta_limit",
        "ts_theilsen",
        "ts_triple_corr",
        "ts_vector_neut",
        "ts_vector_proj",
        "ts_zscore",
        # Cross Sectional
        "generalized_rank",
        "normalize",
        "quantile",
        "rank",
        "rank_by_side",
        "rank_gmean_amean_diff",
        "regression_neut",
        "scale",
        "scale_down",
        "truncate",
        "vector_neut",
        "winsorize",
        "zscore",
        # Vector
        "vec_avg",
        "vec_choose",
        "vec_count",
        "vec_ir",
        "vec_kurtosis",
        "vec_max",
        "vec_min",
        "vec_norm",
        "vec_percentage",
        "vec_powersum",
        "vec_range",
        "vec_skewness",
        "vec_stddev",
        "vec_sum",
        # Transformational
        "bucket",
        "clamp",
        "filter",
        "generate_stats",
        "keep",
        "left_tail",
        "pasteurize",
        "purify",
        "right_tail",
        "sigmoid",
        "tail",
        "trade_when",
        # Group
        "combo_a",
        "group_backfill",
        "group_cartesian_product",
        "group_count",
        "group_extra",
        "group_max",
        "group_mean",
        "group_median",
        "group_min",
        "group_neutralize",
        "group_normalize",
        "group_percentage",
        "group_power_sum",
        "group_rank",
        "group_scale",
        "group_std_dev",
        "group_sum",
        "group_vector_neut",
        "group_vector_proj",
        "group_zscore",
        # Special & Reduce
        "convert",
        "in",
        "reduce_avg",
        "reduce_choose",
        "reduce_count",
        "reduce_ir",
        "reduce_kurtosis",
        "reduce_max",
        "reduce_min",
        "reduce_norm",
        "reduce_percentage",
        "reduce_powersum",
        "reduce_range",
        "reduce_skewness",
        "reduce_stddev",
        "reduce_sum",
        "self_corr",
        "universe_size",
    }
)

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
    | (?P<STR>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
    | (?P<NUM>(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
    | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<MULTI_OP>==|!=|<=|>=|&&|\|\|)
    | (?P<PUNCT>[(),;=+\-*/^<>!~])
    | (?P<OTHER>\S)
    """,
    re.VERBOSE,
)


def _normalize_settings_payload(settings_dict: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(settings_dict, "to_api_dict") and callable(settings_dict.to_api_dict):
        return dict(settings_dict.to_api_dict())
    return dict(settings_dict)


def _normalize_expression_ws(expression: str) -> str:
    """Normalize whitespace in an expression while preserving quoted string literals."""
    parts: list[str] = []
    pos = 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', expression):
        non_str = expression[pos : match.start()]
        if non_str:
            collapsed = re.sub(r"\s+", " ", non_str)
            collapsed = re.sub(r"\s*([(),;])\s*", r"\1", collapsed)
            collapsed = re.sub(r"(?<![=!<>])\s*=\s*(?!=)", "=", collapsed)
            parts.append(collapsed)
        parts.append(match.group(0))
        pos = match.end()
    tail = expression[pos:]
    if tail:
        collapsed = re.sub(r"\s+", " ", tail)
        collapsed = re.sub(r"\s*([(),;])\s*", r"\1", collapsed)
        collapsed = re.sub(r"(?<![=!<>])\s*=\s*(?!=)", "=", collapsed)
        parts.append(collapsed)
    return "".join(parts).strip()


def compute_config_hash(expression: str, settings_dict: Mapping[str, Any] | Any) -> str:
    """Compute SHA-256 hex digest from whitespace-normalized expression and sorted settings JSON."""
    norm_expr = _normalize_expression_ws(expression)
    settings_json = json.dumps(_normalize_settings_payload(settings_dict), sort_keys=True)
    payload = f"{norm_expr}|{settings_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bucket_number(raw: str) -> str:
    if "." in raw or "e" in raw or "E" in raw:
        return "N_F"
    val = int(raw)
    if val <= 15:
        return "N_S"
    if val <= 60:
        return "N_M"
    return "N_L"


def canonical_ast_fingerprint(
    expression: str,
    known_operators: set[str] | frozenset[str] = DEFAULT_KNOWN_OPERATORS,
) -> str:
    """Return canonical structural fingerprint with anonymized fields and bucketed constants."""
    if not expression or not expression.strip():
        return ""

    raw_tokens: list[tuple[str, str]] = []
    for match in _TOKEN_RE.finditer(expression):
        kind = match.lastgroup
        if kind == "WS" or kind is None:
            continue
        raw_tokens.append((kind, match.group(0)))

    if not raw_tokens:
        return ""

    field_map: dict[str, str] = {}
    out_tokens: list[tuple[str, str]] = []
    paren_depth = 0

    for idx, (kind, tok) in enumerate(raw_tokens):
        if kind == "STR":
            out_tokens.append(("STR", tok))
            continue

        if kind == "NUM":
            out_tokens.append(("NUM", _bucket_number(tok)))
            continue

        if kind == "MULTI_OP" or kind == "OTHER":
            out_tokens.append(("BINOP", tok))
            continue

        if kind == "PUNCT":
            if tok == "(":
                paren_depth += 1
                out_tokens.append(("LPAREN", tok))
            elif tok == ")":
                paren_depth = max(0, paren_depth - 1)
                out_tokens.append(("RPAREN", tok))
            elif tok == ",":
                out_tokens.append(("COMMA", tok))
            elif tok == ";":
                out_tokens.append(("SEMI", tok))
            elif tok == "=":
                if paren_depth > 0:
                    out_tokens.append(("KW_EQ", tok))
                else:
                    out_tokens.append(("BINOP", tok))
            elif tok in ("!", "~"):
                out_tokens.append(("UNARY", tok))
            elif tok in ("+", "-"):
                prev_kind = out_tokens[-1][0] if out_tokens else None
                if prev_kind in (None, "LPAREN", "COMMA", "SEMI", "BINOP", "KW_EQ", "UNARY"):
                    out_tokens.append(("UNARY", tok))
                else:
                    out_tokens.append(("BINOP", tok))
            else:
                out_tokens.append(("BINOP", tok))
            continue

        # IDENT classification
        next_tok = raw_tokens[idx + 1] if idx + 1 < len(raw_tokens) else None
        prev_out_kind = out_tokens[-1][0] if out_tokens else None

        # Named call parameter (e.g., filter=true, std=4) inside parentheses
        if (
            paren_depth > 0
            and next_tok == ("PUNCT", "=")
            and prev_out_kind in ("LPAREN", "COMMA")
        ):
            out_tokens.append(("PARAM", tok))
            continue

        lowered = tok.lower()
        if tok in known_operators:
            out_tokens.append(("OP", tok))
        elif lowered in GROUP_NAMES:
            out_tokens.append(("GROUP", lowered))
        elif lowered in BOOLEAN_LITERALS:
            out_tokens.append(("BOOL", lowered))
        else:
            canonical_field = field_map.get(tok)
            if canonical_field is None:
                canonical_field = f"F{len(field_map)}"
                field_map[tok] = canonical_field
            out_tokens.append(("FIELD", canonical_field))

    return _format_canonical_tokens(out_tokens)


def _format_canonical_tokens(tokens: Sequence[tuple[str, str]]) -> str:
    parts: list[str] = []
    prev_kind: Optional[str] = None

    for kind, text in tokens:
        if prev_kind is None:
            parts.append(text)
        elif prev_kind in ("LPAREN", "UNARY", "KW_EQ"):
            parts.append(text)
        elif kind in ("RPAREN", "COMMA", "SEMI", "KW_EQ"):
            parts.append(text)
        elif kind == "LPAREN" and prev_kind in ("OP", "FIELD"):
            parts.append(text)
        elif prev_kind in ("COMMA", "SEMI", "BINOP") or kind == "BINOP":
            parts.append(f" {text}")
        else:
            parts.append(f" {text}")
        prev_kind = kind

    return "".join(parts)


class PreSimDeduplicator:
    """Filters duplicate and structurally over-represented candidates before queuing."""

    def __init__(
        self,
        db: AlphaDB,
        known_operators: Optional[set[str]] = None,
        max_per_structure: int = 3,
        max_per_dataset_structure: int = 2,
    ) -> None:
        self.db = db
        if known_operators is not None:
            self.known_operators: frozenset[str] = frozenset(known_operators)
        else:
            db_ops: set[str] = set()
            if hasattr(db, "get_operators"):
                try:
                    db_ops = {op.name for op in db.get_operators() if getattr(op, "name", "")}
                except Exception:
                    db_ops = set()
            self.known_operators = DEFAULT_KNOWN_OPERATORS | frozenset(db_ops)

        self.max_per_structure = int(max_per_structure)
        self.max_per_dataset_structure = int(max_per_dataset_structure)

    def filter_candidates(
        self,
        candidates: Sequence[tuple[str, str]],
        settings_dict: Mapping[str, Any] | Any,
        region: str,
        universe: str,
        delay: int,
        source: str,
    ) -> list[CandidateItem]:
        """Deduplicate candidate ``(expression, dataset_id)`` pairs against DB and batch budgets."""
        if not candidates or self.max_per_structure <= 0 or self.max_per_dataset_structure <= 0:
            return []

        settings_payload = _normalize_settings_payload(settings_dict)
        settings_json = json.dumps(settings_payload, sort_keys=True)

        seen_hashes: set[str] = set()
        batch_struct_counts: dict[str, int] = defaultdict(int)
        batch_ds_struct_counts: dict[tuple[str, str], int] = defaultdict(int)

        db_struct_cache: dict[str, int] = {}
        db_ds_struct_cache: dict[tuple[str, str], int] = {}

        accepted: list[CandidateItem] = []

        for raw_expr, raw_ds_id in candidates:
            expr = (raw_expr or "").strip()
            if not expr:
                continue
            ds_id = str(raw_ds_id or "").strip()

            cfg_hash = compute_config_hash(expr, settings_payload)
            if cfg_hash in seen_hashes or self.db.has_config_hash(cfg_hash):
                continue

            ast_fp = canonical_ast_fingerprint(expr, known_operators=self.known_operators)

            if ast_fp not in db_struct_cache:
                db_struct_cache[ast_fp] = self.db.count_fingerprint(ast_fp)
            if db_struct_cache[ast_fp] + batch_struct_counts[ast_fp] >= self.max_per_structure:
                continue

            if ds_id:
                ds_key = (ast_fp, ds_id)
                if ds_key not in db_ds_struct_cache:
                    db_ds_struct_cache[ds_key] = self.db.count_fingerprint(ast_fp, dataset_id=ds_id)
                if (
                    db_ds_struct_cache[ds_key] + batch_ds_struct_counts[ds_key]
                    >= self.max_per_dataset_structure
                ):
                    continue

            seen_hashes.add(cfg_hash)
            batch_struct_counts[ast_fp] += 1
            if ds_id:
                batch_ds_struct_counts[(ast_fp, ds_id)] += 1

            accepted.append(
                CandidateItem(
                    config_hash=cfg_hash,
                    ast_fingerprint=ast_fp,
                    expression=expr,
                    region=region,
                    universe=universe,
                    delay=int(delay),
                    settings_json=settings_json,
                    source=source,
                    dataset_id=ds_id,
                    status="PENDING",
                )
            )

        return accepted


__all__ = [
    "BOOLEAN_LITERALS",
    "DEFAULT_KNOWN_OPERATORS",
    "GROUP_NAMES",
    "PreSimDeduplicator",
    "canonical_ast_fingerprint",
    "compute_config_hash",
]
