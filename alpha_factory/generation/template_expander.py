"""Template loader and bounded FastExpr template expansion."""

from __future__ import annotations

import itertools
import math
import random
import re
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from alpha_factory.brain.catalog import wrap_vector_field
from alpha_factory.storage.models import DataFieldRecord

_DEFAULT_A_VALUES: tuple[str, ...] = ("0.5", "1.0", "2.0")

_POS_TOKEN_RE = re.compile(r"(?:^|_)(?:pos|positive|bull|bullish|upward|up)(?:_|$)", re.IGNORECASE)
_NEG_TOKEN_RE = re.compile(r"(?:^|_)(?:neg|negative|bear|bearish|downward|down)(?:_|$)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")
_VEC_WRAP_PREFIX_RE = re.compile(r"(?:\{vec_op\}|vec_[a-zA-Z0-9_]+)\s*\(\s*$")

_PAIR_SWAPS: tuple[tuple[str, str], ...] = (
    ("positive", "negative"),
    ("pos", "neg"),
    ("bullish", "bearish"),
    ("bull", "bear"),
    ("upward", "downward"),
    ("up", "down"),
    ("buy", "sell"),
)


def load_templates_yaml(path: str | Path) -> list[str]:
    """Load template pattern strings from a YAML file while preserving order."""
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Templates YAML file not found: {cfg_path}")

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if loaded is None:
        return []

    patterns: list[str] = []
    seen: set[str] = set()

    def _collect(node: Any) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text and text not in seen:
                seen.add(text)
                patterns.append(text)
        elif isinstance(node, dict):
            if "templates" in node and len(node) == 1:
                _collect(node["templates"])
                return
            for key in ("pattern", "template", "expression"):
                val = node.get(key)
                if isinstance(val, str):
                    text = val.strip()
                    if text and text not in seen:
                        seen.add(text)
                        patterns.append(text)
                    return
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(loaded)
    return patterns


def _compute_dataset_stats(
    datafields: Sequence[DataFieldRecord],
) -> dict[str, tuple[float, float]]:
    """Return ``{dataset_id: (avg_coverage, max_user_count)}``."""
    by_ds: dict[str, list[DataFieldRecord]] = defaultdict(list)
    for df in datafields:
        by_ds[df.dataset_id].append(df)

    stats: dict[str, tuple[float, float]] = {}
    for ds_id, items in by_ds.items():
        avg_cov = sum(float(f.coverage) for f in items) / len(items)
        ds_users = max(int(f.user_count) for f in items)
        stats[ds_id] = (avg_cov, float(ds_users))
    return stats


def _prioritize_fields(
    fields: Sequence[DataFieldRecord],
    dataset_stats: dict[str, tuple[float, float]],
) -> list[DataFieldRecord]:
    """Sort fields prioritizing high coverage and lower dataset/field user_count."""
    if not fields:
        return []

    def _base_score(f: DataFieldRecord) -> float:
        ds_cov, ds_users = dataset_stats.get(f.dataset_id, (float(f.coverage), float(f.user_count)))
        cov_term = 0.5 * float(f.coverage) + 0.5 * ds_cov
        user_penalty = 0.08 * math.log1p(max(0, int(f.user_count))) + 0.08 * math.log1p(
            max(0.0, ds_users)
        )
        return cov_term - user_penalty

    by_ds: dict[str, list[tuple[float, DataFieldRecord]]] = defaultdict(list)
    for f in fields:
        by_ds[f.dataset_id].append((_base_score(f), f))

    ranked: list[tuple[float, float, int, str, DataFieldRecord]] = []
    for ds_items in by_ds.values():
        ds_items.sort(key=lambda item: (-item[0], -float(item[1].coverage), int(item[1].user_count), item[1].field_id))
        for within_idx, (base_sc, f) in enumerate(ds_items):
            # Mild intra-dataset decay promotes dataset diversity among comparably strong datasets
            # without allowing low-coverage/crowded datasets to outrank strong ones.
            effective = base_sc - 0.03 * within_idx
            ranked.append((effective, float(f.coverage), -int(f.user_count), f.field_id, f))

    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    return [row[4] for row in ranked]


def _is_risk_matrix(field: DataFieldRecord) -> bool:
    if field.field_type.upper() != "MATRIX":
        return False
    haystack = " ".join(
        (
            field.category,
            field.subcategory,
            field.dataset_id,
            field.field_id,
            field.description,
        )
    ).lower()
    return "risk" in haystack or "volatility" in haystack


def _is_pos_field(field: DataFieldRecord) -> bool:
    fid = field.field_id
    if _NEG_TOKEN_RE.search(fid):
        return False
    if _POS_TOKEN_RE.search(fid):
        return True
    desc = field.description.lower()
    return "positive" in desc and "negative" not in desc


def _is_neg_field(field: DataFieldRecord) -> bool:
    fid = field.field_id
    if _POS_TOKEN_RE.search(fid):
        return False
    if _NEG_TOKEN_RE.search(fid):
        return True
    desc = field.description.lower()
    return "negative" in desc and "positive" not in desc


def _swap_polarity_name(field_id: str) -> list[str]:
    candidates: list[str] = []
    for pos_tok, neg_tok in _PAIR_SWAPS:
        pattern = re.compile(rf"(^|_){re.escape(pos_tok)}(_|$)", re.IGNORECASE)
        if pattern.search(field_id):
            candidates.append(pattern.sub(rf"\g<1>{neg_tok}\g<2>", field_id))
    return candidates


def _pair_pos_neg_fields(
    prioritized_fields: Sequence[DataFieldRecord],
    require_vector: bool,
) -> list[tuple[DataFieldRecord, DataFieldRecord]]:
    """Pair positive and negative sentiment fields, preferring stem matches in the same dataset."""
    pool = [
        f
        for f in prioritized_fields
        if (not require_vector or f.field_type.upper() == "VECTOR")
    ]
    if not pool:
        return []

    by_ds_type: dict[tuple[str, str], dict[str, DataFieldRecord]] = defaultdict(dict)
    for f in pool:
        by_ds_type[(f.dataset_id, f.field_type.upper())][f.field_id.lower()] = f

    pairs: list[tuple[DataFieldRecord, DataFieldRecord]] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Pass 1: exact stem-swapped pairs within the same dataset and field type
    for f in pool:
        if not _is_pos_field(f):
            continue
        ds_map = by_ds_type[(f.dataset_id, f.field_type.upper())]
        for cand_name in _swap_polarity_name(f.field_id):
            neg_match = ds_map.get(cand_name.lower())
            if neg_match is not None and neg_match.field_id != f.field_id:
                pair_key = (f.field_id, neg_match.field_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((f, neg_match))

    # Pass 2: general pos/neg pairs within the same dataset and field type
    pos_by_ds: dict[tuple[str, str], list[DataFieldRecord]] = defaultdict(list)
    neg_by_ds: dict[tuple[str, str], list[DataFieldRecord]] = defaultdict(list)
    for f in pool:
        key = (f.dataset_id, f.field_type.upper())
        if _is_pos_field(f):
            pos_by_ds[key].append(f)
        elif _is_neg_field(f):
            neg_by_ds[key].append(f)

    for key, pos_list in pos_by_ds.items():
        neg_list = neg_by_ds.get(key, [])
        for p_field in pos_list:
            for n_field in neg_list:
                if p_field.field_id == n_field.field_id:
                    continue
                pair_key = (p_field.field_id, n_field.field_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append((p_field, n_field))

    return pairs


def _is_wrapped_by_vec_context(template: str, match_start: int, vec_op: str) -> bool:
    """Return True if the placeholder at ``match_start`` is directly inside a ``vec_*`` call."""
    prefix = template[:match_start]
    if not _VEC_WRAP_PREFIX_RE.search(prefix):
        return False
    stripped = prefix.rstrip()
    # If preceded literally by `{vec_op}(`, check whether the chosen vec_op is actually `vec_*`.
    if re.search(r"\{vec_op\}\s*\($", stripped):
        return vec_op.strip().startswith("vec_")
    return True


def _render_template(
    template: str,
    field_bindings: dict[str, DataFieldRecord],
    param_bindings: dict[str, str],
) -> str:
    """Substitute all placeholders while applying context-aware vector wrapping."""
    vec_op = param_bindings.get("vec_op", "")

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in param_bindings:
            return param_bindings[name]
        if name in field_bindings:
            rec = field_bindings[name]
            is_vector = rec.field_type.upper() == "VECTOR"
            if is_vector and not _is_wrapped_by_vec_context(template, match.start(), vec_op):
                return wrap_vector_field(rec.field_id, "vec_avg")
            return rec.field_id
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


def _needs_vector_arg(template: str, placeholder_name: str) -> bool:
    pattern = re.compile(rf"(?:\{{vec_op\}}|vec_[a-zA-Z0-9_]+)\s*\(\s*\{{{placeholder_name}\}}")
    return bool(pattern.search(template))


def expand_template(
    template: str,
    datafields: Sequence[DataFieldRecord],
    vector_ops: Sequence[str],
    time_series_ops: Sequence[str],
    lookbacks: Sequence[int],
    groups: Sequence[str] = ("sector", "industry", "subindustry"),
    max_per_template: int = 200,
) -> list[tuple[str, str]]:
    """Expand a FastExpr template over prioritized datafields and parameters up to ``max_per_template``.

    Returns a deduplicated list of ``(expression, dataset_id)`` tuples.
    """
    if max_per_template <= 0 or not template or not template.strip():
        return []

    placeholders = set(_PLACEHOLDER_RE.findall(template))

    # Build non-field parameter dimensions required by this template
    param_names: list[str] = []
    param_lists: list[list[str]] = []

    if "ts_op" in placeholders:
        vals = [str(x).strip() for x in time_series_ops if str(x).strip()]
        if not vals:
            return []
        param_names.append("ts_op")
        param_lists.append(vals)

    if "vec_op" in placeholders:
        vals = [str(x).strip() for x in vector_ops if str(x).strip()]
        if not vals:
            return []
        param_names.append("vec_op")
        param_lists.append(vals)

    if "lookback" in placeholders:
        vals = [str(int(lb)) for lb in lookbacks]
        if not vals:
            return []
        param_names.append("lookback")
        param_lists.append(vals)

    if "a" in placeholders:
        param_names.append("a")
        param_lists.append(list(_DEFAULT_A_VALUES))

    if "group" in placeholders:
        vals = [str(g).strip() for g in groups if str(g).strip()]
        if not vals:
            return []
        param_names.append("group")
        param_lists.append(vals)

    # Check if any field placeholders are present
    field_placeholders = placeholders & {
        "field",
        "vector_field",
        "matrix_field",
        "risk_matrix",
        "pos",
        "neg",
    }

    if not field_placeholders:
        results: list[tuple[str, str]] = []
        seen_expr: set[str] = set()
        for combo in itertools.product(*param_lists) if param_lists else [()]:
            p_map = dict(zip(param_names, combo))
            expr = _render_template(template, {}, p_map)
            if expr not in seen_expr:
                seen_expr.add(expr)
                results.append((expr, ""))
                if len(results) >= max_per_template:
                    break
        return results

    if not datafields:
        return []

    ds_stats = _compute_dataset_stats(datafields)
    prioritized_all = _prioritize_fields(datafields, ds_stats)
    prioritized_matrix = [f for f in prioritized_all if f.field_type.upper() == "MATRIX"]
    prioritized_vector = [f for f in prioritized_all if f.field_type.upper() == "VECTOR"]
    prioritized_risk = [f for f in prioritized_matrix if _is_risk_matrix(f)]

    # Build primary field slot assignments lazily without full secondary Cartesian blowup
    field_assignments: list[tuple[dict[str, DataFieldRecord], str]] = []

    if "pos" in field_placeholders and "neg" in field_placeholders:
        require_vec = _needs_vector_arg(template, "pos") or _needs_vector_arg(template, "neg")
        paired = _pair_pos_neg_fields(prioritized_all, require_vector=require_vec)
        if not paired:
            return []
        for pos_f, neg_f in paired:
            binding: dict[str, DataFieldRecord] = {"pos": pos_f, "neg": neg_f}
            if "matrix_field" in field_placeholders:
                if not prioritized_matrix:
                    return []
                binding["matrix_field"] = prioritized_matrix[0]
            if "vector_field" in field_placeholders:
                if not prioritized_vector:
                    return []
                binding["vector_field"] = prioritized_vector[0]
            if "risk_matrix" in field_placeholders:
                if not prioritized_risk:
                    return []
                binding["risk_matrix"] = prioritized_risk[0]
            if "field" in field_placeholders:
                binding["field"] = prioritized_matrix[0] if prioritized_matrix else prioritized_all[0]
            field_assignments.append((binding, pos_f.dataset_id))
    else:
        # Collect candidate lists per independent field slot present
        slot_pools: dict[str, list[DataFieldRecord]] = {}
        if "matrix_field" in field_placeholders:
            if not prioritized_matrix:
                return []
            slot_pools["matrix_field"] = prioritized_matrix
        if "vector_field" in field_placeholders:
            if not prioritized_vector:
                return []
            slot_pools["vector_field"] = prioritized_vector
        if "risk_matrix" in field_placeholders:
            if not prioritized_risk:
                return []
            slot_pools["risk_matrix"] = prioritized_risk
        if "pos" in field_placeholders:
            require_vec = _needs_vector_arg(template, "pos")
            pos_pool = [
                f
                for f in prioritized_all
                if _is_pos_field(f) and (not require_vec or f.field_type.upper() == "VECTOR")
            ]
            if not pos_pool:
                return []
            slot_pools["pos"] = pos_pool
        if "neg" in field_placeholders:
            require_vec = _needs_vector_arg(template, "neg")
            neg_pool = [
                f
                for f in prioritized_all
                if _is_neg_field(f) and (not require_vec or f.field_type.upper() == "VECTOR")
            ]
            if not neg_pool:
                return []
            slot_pools["neg"] = neg_pool
        if "field" in field_placeholders:
            if _needs_vector_arg(template, "field"):
                if not prioritized_vector:
                    return []
                slot_pools["field"] = prioritized_vector
            else:
                slot_pools["field"] = prioritized_matrix + prioritized_vector

        slot_order = [
            name
            for name in ("vector_field", "matrix_field", "risk_matrix", "pos", "neg", "field")
            if name in slot_pools
        ]

        if len(slot_order) == 1:
            only_slot = slot_order[0]
            for f in slot_pools[only_slot]:
                field_assignments.append(({only_slot: f}, f.dataset_id))
        else:
            # Pair multi-slot templates without materializing quadratic products
            primary_slot = slot_order[0]
            primary_pool = slot_pools[primary_slot]
            secondary_slots = slot_order[1:]

            total_field_prod = math.prod(len(slot_pools[s]) for s in slot_order)
            if total_field_prod <= max_per_template * 2:
                for combo in itertools.product(*(slot_pools[s] for s in slot_order)):
                    binding = dict(zip(slot_order, combo))
                    # Skip trivial self-pairing when multiple options exist for a secondary slot
                    if (
                        len(slot_order) == 2
                        and binding[slot_order[0]].field_id == binding[slot_order[1]].field_id
                        and len(slot_pools[slot_order[1]]) > 1
                    ):
                        continue
                    field_assignments.append((binding, binding[primary_slot].dataset_id))
            else:
                # Stream bounded diverse pairings prioritizing top primary fields
                max_field_combos = max(max_per_template * 2, 64)
                rng_pair = random.Random(zlib.crc32(f"{template}:pairs".encode("utf-8")))
                seen_bindings: set[tuple[str, ...]] = set()
                idx = 0
                attempts = 0
                max_attempts = max_field_combos * 4
                while len(field_assignments) < max_field_combos and attempts < max_attempts:
                    attempts += 1
                    p_rec = primary_pool[idx % len(primary_pool)]
                    idx += 1
                    binding = {primary_slot: p_rec}
                    used_ids = {p_rec.field_id}
                    for sec in secondary_slots:
                        sec_pool = slot_pools[sec]
                        # Prefer a distinct field from the top half of the prioritized secondary pool
                        upper = min(len(sec_pool), max(16, max_per_template))
                        chosen = sec_pool[rng_pair.randrange(upper)]
                        if chosen.field_id in used_ids and len(sec_pool) > 1:
                            for offset in range(1, min(len(sec_pool), 10) + 1):
                                alt = sec_pool[(rng_pair.randrange(upper) + offset) % len(sec_pool)]
                                if alt.field_id not in used_ids:
                                    chosen = alt
                                    break
                        binding[sec] = chosen
                        used_ids.add(chosen.field_id)
                    sig = tuple(binding[s].field_id for s in slot_order)
                    if sig not in seen_bindings:
                        seen_bindings.add(sig)
                        field_assignments.append((binding, p_rec.dataset_id))

    if not field_assignments:
        return []

    param_total = math.prod(len(pl) for pl in param_lists) if param_lists else 1
    total_space = len(field_assignments) * param_total

    results = []
    seen_expr = set()

    if total_space <= max_per_template:
        param_combos = list(itertools.product(*param_lists)) if param_lists else [()]
        for f_bind, ds_id in field_assignments:
            for p_combo in param_combos:
                p_map = dict(zip(param_names, p_combo))
                expr = _render_template(template, f_bind, p_map)
                if expr not in seen_expr:
                    seen_expr.add(expr)
                    results.append((expr, ds_id))
                    if len(results) >= max_per_template:
                        return results
        return results

    # Bounded sampling when total Cartesian space exceeds max_per_template:
    # walk prioritized field assignments from highest priority downward, pairing each
    # with deterministically sampled parameter tuples so we never materialize the full product.
    rng = random.Random(zlib.crc32(template.encode("utf-8")))
    shuffled_lists = [list(pl) for pl in param_lists]
    for pl in shuffled_lists:
        rng.shuffle(pl)

    def _sample_param_map(step_idx: int) -> dict[str, str]:
        if not shuffled_lists:
            return {}
        if step_idx < param_total:
            # Mixed-radix indexing covers diverse parameter combos uniformly without repetition
            rem = step_idx
            chosen_vals: list[str] = []
            for pl in reversed(shuffled_lists):
                chosen_vals.append(pl[rem % len(pl)])
                rem //= len(pl)
            chosen_vals.reverse()
            return dict(zip(param_names, chosen_vals))
        return {name: rng.choice(pl) for name, pl in zip(param_names, shuffled_lists)}

    pool_cap = min(len(field_assignments), max(max_per_template, 1))
    top_fields = field_assignments[:pool_cap]
    step = 0
    round_num = 0
    max_rounds = max(1, (max_per_template // len(top_fields)) + 3)

    while len(results) < max_per_template and round_num < max_rounds:
        for f_bind, ds_id in top_fields:
            p_map = _sample_param_map(step)
            step += 1
            expr = _render_template(template, f_bind, p_map)
            if expr not in seen_expr:
                seen_expr.add(expr)
                results.append((expr, ds_id))
                if len(results) >= max_per_template:
                    return results
        round_num += 1

    return results


__all__ = [
    "expand_template",
    "load_templates_yaml",
]
