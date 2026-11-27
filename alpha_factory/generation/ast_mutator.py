"""Safe AST-only expression mutator and crossover engine for FASTEXPR."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional

from alpha_factory.generation.validator import (
    BUILTIN_FIELDS,
    GROUP_FIELDS,
    BinaryOpNode,
    BoolNode,
    ExpressionValidator,
    ExprNode,
    FASTEXPRSyntaxError,
    FieldNode,
    FunctionNode,
    KeywordArgNode,
    NumberNode,
    StringNode,
    UnaryOpNode,
    parse_expression,
)

BUILTIN_NUMERIC_FIELDS: tuple[str, ...] = tuple(
    sorted(BUILTIN_FIELDS - GROUP_FIELDS)
)

_LOOKBACK_KWARG_NAMES: frozenset[str] = frozenset(
    {"d", "lookback", "window", "nlength", "days", "period"}
)

_LOOKBACK_OPERATORS: frozenset[str] = frozenset(
    {
        "last_diff_value",
        "jump_decay",
        "kth_element",
        "group_backfill",
        "ts_step",
    }
)

_GROUP_ARG_POSITIONS: dict[str, frozenset[int]] = {
    "group_rank": frozenset({1}),
    "group_zscore": frozenset({1}),
    "group_neutralize": frozenset({1}),
    "group_scale": frozenset({1}),
    "group_min": frozenset({1}),
    "group_max": frozenset({1}),
    "group_backfill": frozenset({1}),
    "group_mean": frozenset({2}),
    "group_cartesian_product": frozenset({0, 1}),
}

_ENUM_KWARG_NAMES: frozenset[str] = frozenset({"driver", "mode", "distribution", "method"})

OPERATOR_SWAP_GROUPS: tuple[tuple[str, ...], ...] = (
    ("rank", "zscore", "scale", "normalize", "winsorize", "scale_down", "quantile"),
    ("abs", "sign", "reverse", "inverse"),
    (
        "ts_rank",
        "ts_zscore",
        "ts_delta",
        "ts_mean",
        "ts_sum",
        "ts_std_dev",
        "ts_av_diff",
        "ts_scale",
        "ts_decay_linear",
        "ts_max",
        "ts_min",
        "ts_arg_max",
        "ts_arg_min",
        "ts_delay",
    ),
    ("ts_corr", "ts_covariance", "ts_regression"),
    ("group_rank", "group_zscore", "group_neutralize", "group_scale", "group_min", "group_max"),
    ("vec_avg", "vec_sum", "vec_min", "vec_max"),
    ("add", "subtract", "multiply", "divide", "min", "max"),
)

_OPERATOR_TO_GROUP: dict[str, tuple[str, ...]] = {
    op: group for group in OPERATOR_SWAP_GROUPS for op in group
}


@dataclass(frozen=True)
class _NodeSlot:
    """Reference to an AST node by path of child accessors from the root."""

    path: tuple[tuple[str, int | str], ...]
    node: ExprNode
    parent: Optional[ExprNode]
    arg_index: Optional[int]
    in_unary: bool
    in_enum_kwarg: bool
    kwarg_name: Optional[str]


def _collect_slots(root: ExprNode) -> list[_NodeSlot]:
    """Traverse AST and collect slots with parent/context metadata."""
    slots: list[_NodeSlot] = []

    def walk(
        node: ExprNode,
        path: tuple[tuple[str, int | str], ...],
        parent: Optional[ExprNode],
        arg_index: Optional[int],
        in_unary: bool,
        in_enum_kwarg: bool,
        kwarg_name: Optional[str],
    ) -> None:
        slots.append(
            _NodeSlot(
                path=path,
                node=node,
                parent=parent,
                arg_index=arg_index,
                in_unary=in_unary,
                in_enum_kwarg=in_enum_kwarg,
                kwarg_name=kwarg_name,
            )
        )
        if isinstance(node, FunctionNode):
            for idx, arg in enumerate(node.args):
                walk(
                    arg,
                    path + (("arg", idx),),
                    parent=node,
                    arg_index=idx,
                    in_unary=False,
                    in_enum_kwarg=False,
                    kwarg_name=None,
                )
        elif isinstance(node, KeywordArgNode):
            is_enum = node.name.lower() in _ENUM_KWARG_NAMES and isinstance(
                node.value, FieldNode
            )
            walk(
                node.value,
                path + (("kwval", node.name),),
                parent=parent,
                arg_index=arg_index,
                in_unary=False,
                in_enum_kwarg=is_enum,
                kwarg_name=node.name,
            )
        elif isinstance(node, BinaryOpNode):
            walk(node.left, path + (("left", 0),), node, None, False, False, None)
            walk(node.right, path + (("right", 0),), node, None, False, False, None)
        elif isinstance(node, UnaryOpNode):
            walk(node.operand, path + (("operand", 0),), node, None, True, False, None)

    walk(root, (), None, None, False, False, None)
    return slots


def _get_at_path(root: ExprNode, path: tuple[tuple[str, int | str], ...]) -> ExprNode:
    cur = root
    for kind, key in path:
        if kind == "arg":
            assert isinstance(cur, FunctionNode) and isinstance(key, int)
            cur = cur.args[key]
        elif kind == "kwval":
            assert isinstance(cur, KeywordArgNode)
            cur = cur.value
        elif kind == "left":
            assert isinstance(cur, BinaryOpNode)
            cur = cur.left
        elif kind == "right":
            assert isinstance(cur, BinaryOpNode)
            cur = cur.right
        elif kind == "operand":
            assert isinstance(cur, UnaryOpNode)
            cur = cur.operand
    return cur


def _set_at_path(
    root: ExprNode,
    path: tuple[tuple[str, int | str], ...],
    new_node: ExprNode,
) -> ExprNode:
    if not path:
        return new_node
    parent = _get_at_path(root, path[:-1])
    kind, key = path[-1]
    if kind == "arg":
        assert isinstance(parent, FunctionNode) and isinstance(key, int)
        parent.args[key] = new_node
    elif kind == "kwval":
        assert isinstance(parent, KeywordArgNode)
        parent.value = new_node
    elif kind == "left":
        assert isinstance(parent, BinaryOpNode)
        parent.left = new_node
    elif kind == "right":
        assert isinstance(parent, BinaryOpNode)
        parent.right = new_node
    elif kind == "operand":
        assert isinstance(parent, UnaryOpNode)
        parent.operand = new_node
    return root


class SafeASTMutator:
    """AST-level mutator and crossover engine that never corrupts field identifiers."""

    def __init__(
        self,
        validator: ExpressionValidator,
        lookback_grid: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 252),
    ) -> None:
        self.validator = validator
        self.lookback_grid: tuple[int, ...] = tuple(lookback_grid)

    def mutate(
        self,
        expression: str,
        fields_by_type: Optional[dict[str, list[str]]] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """Perform a single AST-safe mutation and return a validated FASTEXPR string.

        Mutation strategies:
        1. Lookback mutation: swap integer > 1 lookback literal from ``lookback_grid``.
        2. Operator swap: swap a ``FunctionNode`` with a compatible same-arity operator.
        3. Field swap: swap a non-group ``FieldNode`` with another field of the same type.
        """
        rand = rng if rng is not None else random.Random()
        try:
            ast = parse_expression(expression)
        except FASTEXPRSyntaxError:
            return None

        slots = _collect_slots(ast)
        by_strategy: list[list[str]] = [
            cands
            for cands in (
                self._lookback_candidates(ast, slots, rand),
                self._operator_candidates(ast, slots, rand),
                self._field_candidates(ast, slots, fields_by_type, rand),
            )
            if cands
        ]
        if not by_strategy:
            return None

        rand.shuffle(by_strategy)
        for group in by_strategy:
            for expr_cand in group:
                is_valid, _ = self.validator.validate(expr_cand)
                if is_valid:
                    return expr_cand
        return None

    def crossover(
        self,
        expr_a: str,
        expr_b: str,
        rng: Optional[random.Random] = None,
    ) -> Optional[str]:
        """Combine or cross over two valid FASTEXPR expressions via AST operations."""
        rand = rng if rng is not None else random.Random()
        try:
            ast_a = parse_expression(expr_a)
            ast_b = parse_expression(expr_b)
        except FASTEXPRSyntaxError:
            return None

        norm_a = ast_a.to_string()
        norm_b = ast_b.to_string()

        subtree_cands = self._subtree_crossover_candidates(ast_a, ast_b, norm_a, norm_b, rand)
        combine_cands = self._combine_crossover_candidates(ast_a, ast_b, rand)

        strategies = [subtree_cands, combine_cands]
        rand.shuffle(strategies)

        for group in strategies:
            for candidate in group:
                is_valid, _ = self.validator.validate(candidate)
                if is_valid:
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Mutation candidate builders
    # ------------------------------------------------------------------

    def _is_lookback_operator(self, op_name: str) -> bool:
        if op_name.startswith("ts_") or op_name in _LOOKBACK_OPERATORS:
            return True
        rec = self.validator.operators.get(op_name)
        return bool(rec is not None and rec.uses_lookback)

    def _lookback_candidates(
        self,
        ast: ExprNode,
        slots: list[_NodeSlot],
        rand: random.Random,
    ) -> list[str]:
        preferred: list[_NodeSlot] = []
        fallback: list[_NodeSlot] = []

        for slot in slots:
            node = slot.node
            if not isinstance(node, NumberNode):
                continue
            if not (isinstance(node.value, int) and not isinstance(node.value, bool) and node.value > 1):
                continue
            if slot.in_unary:
                continue
            if slot.kwarg_name is not None:
                if slot.kwarg_name.lower() in _LOOKBACK_KWARG_NAMES:
                    preferred.append(slot)
                continue
            if isinstance(slot.parent, FunctionNode):
                is_pos_arg = slot.arg_index is not None and (
                    slot.arg_index >= 1 or slot.parent.name == "ts_step"
                )
                if is_pos_arg:
                    if self._is_lookback_operator(slot.parent.name):
                        preferred.append(slot)
                    else:
                        fallback.append(slot)

        targets = preferred if preferred else fallback
        if not targets:
            return []

        valid_grid = [int(v) for v in self.lookback_grid if isinstance(v, int) and not isinstance(v, bool) and v > 1]
        if not valid_grid:
            return []

        shuffled_targets = list(targets)
        rand.shuffle(shuffled_targets)
        results: list[str] = []

        for target in shuffled_targets:
            assert isinstance(target.node, NumberNode)
            choices = [lb for lb in valid_grid if lb != target.node.value]
            rand.shuffle(choices)
            for new_lb in choices:
                cloned = copy.deepcopy(ast)
                mutated_node = _get_at_path(cloned, target.path)
                assert isinstance(mutated_node, NumberNode)
                mutated_node.value = new_lb
                results.append(cloned.to_string())
        return results

    def _operator_candidates(
        self,
        ast: ExprNode,
        slots: list[_NodeSlot],
        rand: random.Random,
    ) -> list[str]:
        fn_slots = [
            s
            for s in slots
            if isinstance(s.node, FunctionNode)
            and s.node.name in _OPERATOR_TO_GROUP
            and not any(isinstance(a, KeywordArgNode) for a in s.node.args)
        ]
        if not fn_slots:
            return []

        rand.shuffle(fn_slots)
        results: list[str] = []

        for slot in fn_slots:
            fn = slot.node
            assert isinstance(fn, FunctionNode)
            group = _OPERATOR_TO_GROUP[fn.name]
            positional_count = len(fn.args)

            has_event_child = any(
                isinstance(arg, FieldNode)
                and arg.name in self.validator.datafields
                and (
                    self.validator.datafields[arg.name].is_event_input
                    or self.validator.datafields[arg.name].field_type.upper() == "EVENT"
                )
                for arg in fn.args
            )

            replacements: list[str] = []
            for cand_op in group:
                if cand_op == fn.name:
                    continue
                if cand_op in self.validator.blocked_operators:
                    continue
                if self.validator.operators:
                    op_rec = self.validator.operators.get(cand_op)
                    if op_rec is None or positional_count < op_rec.min_inputs:
                        continue
                if has_event_child and cand_op.lower() in self.validator.event_incompatible_operators:
                    continue
                replacements.append(cand_op)

            rand.shuffle(replacements)
            for cand_op in replacements:
                cloned = copy.deepcopy(ast)
                target_fn = _get_at_path(cloned, slot.path)
                assert isinstance(target_fn, FunctionNode)
                target_fn.name = cand_op
                results.append(cloned.to_string())
        return results

    def _resolve_fields_by_type(
        self,
        fields_by_type: Optional[dict[str, list[str]]],
    ) -> dict[str, list[str]]:
        pool: dict[str, list[str]] = {}
        if fields_by_type is not None:
            for ftype, fids in fields_by_type.items():
                norm_type = str(ftype).upper()
                cleaned = [
                    fid
                    for fid in fids
                    if fid not in GROUP_FIELDS and fid not in self.validator.blocked_fields
                ]
                if cleaned:
                    pool[norm_type] = cleaned
            return pool

        if self.validator.datafields:
            for fid, rec in self.validator.datafields.items():
                if fid in GROUP_FIELDS or fid in self.validator.blocked_fields:
                    continue
                ftype = str(rec.field_type).upper()
                pool.setdefault(ftype, []).append(fid)
            return pool

        pool["MATRIX"] = list(BUILTIN_NUMERIC_FIELDS)
        return pool

    def _infer_field_type(
        self,
        slot: _NodeSlot,
        pool: dict[str, list[str]],
    ) -> str:
        assert isinstance(slot.node, FieldNode)
        fname = slot.node.name
        rec = self.validator.datafields.get(fname)
        if rec is not None:
            return str(rec.field_type).upper()
        for ftype, fids in pool.items():
            if fname in fids:
                return ftype
        if isinstance(slot.parent, FunctionNode) and slot.parent.name.startswith("vec_"):
            return "VECTOR"
        return "MATRIX"

    def _field_candidates(
        self,
        ast: ExprNode,
        slots: list[_NodeSlot],
        fields_by_type: Optional[dict[str, list[str]]],
        rand: random.Random,
    ) -> list[str]:
        pool = self._resolve_fields_by_type(fields_by_type)
        target_slots: list[_NodeSlot] = []

        for slot in slots:
            if not isinstance(slot.node, FieldNode):
                continue
            if slot.in_enum_kwarg or slot.node.name in GROUP_FIELDS:
                continue
            if isinstance(slot.parent, FunctionNode) and slot.arg_index is not None:
                group_indices = _GROUP_ARG_POSITIONS.get(slot.parent.name)
                if group_indices and slot.arg_index in group_indices:
                    continue
            target_slots.append(slot)

        if not target_slots:
            return []

        rand.shuffle(target_slots)
        results: list[str] = []

        for slot in target_slots:
            assert isinstance(slot.node, FieldNode)
            ftype = self._infer_field_type(slot, pool)
            candidates = list(pool.get(ftype, ()))

            if (
                not candidates
                and ftype == "MATRIX"
                and not self.validator.datafields
                and fields_by_type is None
            ):
                candidates = list(BUILTIN_NUMERIC_FIELDS)

            parent_op = slot.parent.name.lower() if isinstance(slot.parent, FunctionNode) else None
            filtered: list[str] = []
            for fid in candidates:
                if fid == slot.node.name or fid in GROUP_FIELDS:
                    continue
                if parent_op and parent_op in self.validator.event_incompatible_operators:
                    rec = self.validator.datafields.get(fid)
                    if rec is not None and (
                        rec.is_event_input or str(rec.field_type).upper() == "EVENT"
                    ):
                        continue
                filtered.append(fid)

            rand.shuffle(filtered)
            for new_fid in filtered:
                cloned = copy.deepcopy(ast)
                target_field = _get_at_path(cloned, slot.path)
                assert isinstance(target_field, FieldNode)
                target_field.name = new_fid
                results.append(cloned.to_string())
        return results

    # ------------------------------------------------------------------
    # Crossover candidate builders
    # ------------------------------------------------------------------

    def _classify_subtree_type(self, slot: _NodeSlot) -> Optional[str]:
        node = slot.node
        if not slot.path:
            return None
        if isinstance(node, (NumberNode, BoolNode, StringNode, KeywordArgNode)):
            return None
        if slot.in_enum_kwarg:
            return None
        if isinstance(slot.parent, FunctionNode) and slot.arg_index is not None:
            group_indices = _GROUP_ARG_POSITIONS.get(slot.parent.name)
            if group_indices and slot.arg_index in group_indices:
                return None

        if isinstance(node, FieldNode):
            if node.name in GROUP_FIELDS:
                return None
            rec = self.validator.datafields.get(node.name)
            if rec is not None and str(rec.field_type).upper() == "VECTOR":
                return "VECTOR"
            if isinstance(slot.parent, FunctionNode) and slot.parent.name.startswith("vec_"):
                return "VECTOR"
            return "MATRIX"

        if isinstance(slot.parent, FunctionNode) and slot.parent.name.startswith("vec_"):
            return None
        return "MATRIX"

    def _subtree_crossover_candidates(
        self,
        ast_a: ExprNode,
        ast_b: ExprNode,
        norm_a: str,
        norm_b: str,
        rand: random.Random,
    ) -> list[str]:
        slots_a = _collect_slots(ast_a)
        slots_b = _collect_slots(ast_b)

        typed_a: dict[str, list[_NodeSlot]] = {}
        typed_b: dict[str, list[_NodeSlot]] = {}

        for s in slots_a:
            stype = self._classify_subtree_type(s)
            if stype is not None:
                typed_a.setdefault(stype, []).append(s)

        for s in slots_b:
            stype = self._classify_subtree_type(s)
            if stype is not None:
                typed_b.setdefault(stype, []).append(s)

        pairs: list[tuple[ExprNode, _NodeSlot, ExprNode]] = []
        for stype, sa_list in typed_a.items():
            sb_list = typed_b.get(stype, [])
            for sa in sa_list:
                for sb in sb_list:
                    pairs.append((ast_a, sa, sb.node))
                    pairs.append((ast_b, sb, sa.node))

        if not pairs:
            return []

        rand.shuffle(pairs)
        results: list[str] = []
        seen: set[str] = {norm_a, norm_b}
        for base_ast, target_slot, donor_node in pairs:
            cloned = copy.deepcopy(base_ast)
            _set_at_path(cloned, target_slot.path, copy.deepcopy(donor_node))
            candidate = cloned.to_string()
            if candidate not in seen:
                seen.add(candidate)
                results.append(candidate)
        return results

    def _combine_crossover_candidates(
        self,
        ast_a: ExprNode,
        ast_b: ExprNode,
        rand: random.Random,
    ) -> list[str]:
        def _wrap_rank(node: ExprNode) -> ExprNode:
            cloned = copy.deepcopy(node)
            if isinstance(cloned, FunctionNode) and cloned.name == "rank":
                return cloned
            return FunctionNode(name="rank", args=[cloned])

        left_ranked = _wrap_rank(ast_a)
        right_ranked = _wrap_rank(ast_b)

        ops = ["add", "subtract"]
        rand.shuffle(ops)

        candidates: list[str] = [
            FunctionNode(name=op, args=[copy.deepcopy(left_ranked), copy.deepcopy(right_ranked)]).to_string()
            for op in ops
        ]

        # Fallbacks if validator has operators loaded without rank/add/subtract
        bin_ops = ["+", "-"]
        rand.shuffle(bin_ops)
        for bop in bin_ops:
            candidates.append(
                BinaryOpNode(op=bop, left=copy.deepcopy(left_ranked), right=copy.deepcopy(right_ranked)).to_string()
            )
            candidates.append(
                BinaryOpNode(op=bop, left=copy.deepcopy(ast_a), right=copy.deepcopy(ast_b)).to_string()
            )
        return candidates


__all__ = [
    "BUILTIN_NUMERIC_FIELDS",
    "OPERATOR_SWAP_GROUPS",
    "SafeASTMutator",
]
