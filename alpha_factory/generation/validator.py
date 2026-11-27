"""FASTEXPR recursive-descent parser, AST nodes, and expression validator."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import DataFieldRecord, OperatorRecord

logger = logging.getLogger(__name__)

MAX_AST_DEPTH = 80

BUILTIN_FIELDS: frozenset[str] = frozenset(
    {
        "close",
        "open",
        "high",
        "low",
        "volume",
        "vwap",
        "returns",
        "cap",
        "adv20",
        "sharesout",
        "split",
        "dividend",
        "sector",
        "industry",
        "subindustry",
        "market",
        "country",
        "exchange",
    }
)

GROUP_FIELDS: frozenset[str] = frozenset(
    {"sector", "industry", "subindustry", "market", "country", "exchange"}
)

_BINARY_OP_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
    "^": 8,
}

_BINARY_OP_TO_FUNC: dict[str, str] = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "^": "power",
    "==": "equal",
    "!=": "not_equal",
    ">": "greater",
    ">=": "greater_equal",
    "<": "less",
    "<=": "less_equal",
    "&&": "and",
    "||": "or",
}

_UNARY_OP_TO_FUNC: dict[str, str] = {
    "-": "reverse",
    "!": "not",
}

_DEFAULT_EVENT_REPLACEMENTS: dict[str, str] = {
    "multiply": "ts_multiply",
    "add": "ts_add",
    "subtract": "ts_subtract",
    "divide": "ts_divide",
    "power": "ts_power",
    "signed_power": "ts_signed_power",
    "abs": "ts_abs",
    "log": "ts_log",
    "sqrt": "ts_sqrt",
    "min": "ts_min",
    "max": "ts_max",
    "rank": "ts_rank",
    "delta": "ts_delta",
    "correlation": "ts_correlation",
    "reverse": "ts_reverse",
}

_ENUM_KWARG_NAMES: frozenset[str] = frozenset({"driver", "mode", "distribution", "method"})

_DEFAULT_COMPILER_JSON = (
    Path(__file__).resolve().parent.parent.parent
    / "worldquant-miner"
    / "generation_two"
    / "core"
    / "compiler_knowledge.json"
)


class FASTEXPRSyntaxError(ValueError):
    """Raised when a FASTEXPR expression has invalid lexical or syntactic structure."""

    def __init__(self, message: str, position: Optional[int] = None, expression: str = "") -> None:
        self.position = position
        self.expression = expression
        detail = f"{message} (at position {position})" if position is not None else message
        super().__init__(detail)


class ExprNode:
    """Base class for all FASTEXPR AST nodes."""

    def to_string(self) -> str:
        """Render this AST node back into valid FASTEXPR source text."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class FunctionNode(ExprNode):
    """Operator/function invocation node, e.g. ``ts_rank(close, 20)``."""

    name: str
    args: list[ExprNode] = field(default_factory=list)

    def to_string(self) -> str:
        args_str = ", ".join(arg.to_string() for arg in self.args)
        return f"{self.name}({args_str})"


@dataclass
class BinaryOpNode(ExprNode):
    """Infix binary expression node, e.g. ``close - open``."""

    op: str
    left: ExprNode
    right: ExprNode

    def to_string(self) -> str:
        self_prec = _BINARY_OP_PRECEDENCE.get(self.op, 0)

        left_str = self.left.to_string()
        if isinstance(self.left, BinaryOpNode):
            left_prec = _BINARY_OP_PRECEDENCE.get(self.left.op, 0)
            if left_prec < self_prec or (left_prec == self_prec and self.op == "^"):
                left_str = f"({left_str})"
        elif isinstance(self.left, UnaryOpNode) and self.op == "^":
            left_str = f"({left_str})"
        elif isinstance(self.left, NumberNode) and not isinstance(self.left.value, bool) and self.left.value < 0 and self.op == "^":
            left_str = f"({left_str})"

        right_str = self.right.to_string()
        if isinstance(self.right, BinaryOpNode):
            right_prec = _BINARY_OP_PRECEDENCE.get(self.right.op, 0)
            if right_prec < self_prec or (right_prec == self_prec and self.op != "^"):
                right_str = f"({right_str})"

        return f"{left_str} {self.op} {right_str}"


@dataclass
class UnaryOpNode(ExprNode):
    """Prefix unary expression node, e.g. ``-close`` or ``!is_nan(close)``."""

    op: str
    operand: ExprNode

    def to_string(self) -> str:
        inner = self.operand.to_string()
        if isinstance(self.operand, (BinaryOpNode, UnaryOpNode)):
            return f"{self.op}({inner})"
        return f"{self.op}{inner}"


@dataclass
class FieldNode(ExprNode):
    """Data field or built-in identifier node, e.g. ``close`` or ``anl49_1stfiscalquarter``."""

    name: str

    def to_string(self) -> str:
        return self.name


@dataclass
class NumberNode(ExprNode):
    """Numeric literal node (integer or float)."""

    value: int | float

    def to_string(self) -> str:
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)


@dataclass
class BoolNode(ExprNode):
    """Boolean literal node (``true`` / ``false``)."""

    value: bool

    def to_string(self) -> str:
        return "true" if self.value else "false"


@dataclass
class StringNode(ExprNode):
    """String literal node, e.g. ``"gaussian"``."""

    value: str
    quote: str = '"'

    def to_string(self) -> str:
        escaped = self.value.replace("\\", "\\\\").replace(self.quote, f"\\{self.quote}")
        return f"{self.quote}{escaped}{self.quote}"


@dataclass
class KeywordArgNode(ExprNode):
    """Named argument node inside a function call, e.g. ``filter=false``."""

    name: str
    value: ExprNode

    def to_string(self) -> str:
        return f"{self.name}={self.value.to_string()}"


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    pos: int


_TWO_CHAR_OPS = {"<=", ">=", "==", "!=", "&&", "||"}
_SINGLE_CHAR_OPS = set("+-*/^%<>!=")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _tokenize(expr: str) -> list[_Token]:
    """Tokenize a FASTEXPR string into a stream of typed tokens."""
    tokens: list[_Token] = []
    n = len(expr)
    i = 0

    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue

        if ch in "(),":
            tokens.append(_Token(ch, ch, i))
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            start = i
            i += 1
            chars: list[str] = []
            closed = False
            while i < n:
                cur = expr[i]
                if cur == "\\":
                    if i + 1 >= n:
                        raise FASTEXPRSyntaxError("Unterminated escape in string literal", start, expr)
                    chars.append(expr[i + 1])
                    i += 2
                    continue
                if cur == quote:
                    closed = True
                    i += 1
                    break
                chars.append(cur)
                i += 1
            if not closed:
                raise FASTEXPRSyntaxError("Unterminated string literal", start, expr)
            tokens.append(_Token("STRING", quote + "".join(chars), start))
            continue

        if i + 1 < n and expr[i : i + 2] in _TWO_CHAR_OPS:
            op = expr[i : i + 2]
            tokens.append(_Token("OP", op, i))
            i += 2
            continue

        if ch in _SINGLE_CHAR_OPS:
            kind = "EQ" if ch == "=" else "OP"
            tokens.append(_Token(kind, ch, i))
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            match = _NUMBER_RE.match(expr, i)
            if match is None:
                raise FASTEXPRSyntaxError(f"Invalid numeric literal at '{expr[i:]}'", i, expr)
            lexeme = match.group(0)
            tokens.append(_Token("NUMBER", lexeme, i))
            i = match.end()
            continue

        if ch.isalpha() or ch == "_":
            match = _IDENT_RE.match(expr, i)
            assert match is not None
            lexeme = match.group(0)
            lower = lexeme.lower()
            if lower in ("true", "false"):
                tokens.append(_Token("BOOL", lower, i))
            else:
                tokens.append(_Token("IDENT", lexeme, i))
            i = match.end()
            continue

        raise FASTEXPRSyntaxError(f"Unexpected character '{ch}'", i, expr)

    tokens.append(_Token("EOF", "", n))
    return tokens


class _RecursiveDescentParser:
    """Recursive-descent parser producing a typed FASTEXPR AST."""

    def __init__(self, tokens: list[_Token], expr: str) -> None:
        self._tokens = tokens
        self._expr = expr
        self._idx = 0
        self._depth = 0

    @property
    def _current(self) -> _Token:
        return self._tokens[self._idx]

    def _peek(self, offset: int = 1) -> _Token:
        pos = min(self._idx + offset, len(self._tokens) - 1)
        return self._tokens[pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._idx]
        if self._idx < len(self._tokens) - 1:
            self._idx += 1
        return tok

    def parse(self) -> ExprNode:
        node = self._parse_or()
        if self._current.kind != "EOF":
            raise FASTEXPRSyntaxError(
                f"Unexpected trailing token '{self._current.value}'",
                self._current.pos,
                self._expr,
            )
        return node

    def _enter(self, pos: int) -> None:
        self._depth += 1
        if self._depth > MAX_AST_DEPTH:
            raise FASTEXPRSyntaxError(
                f"Expression exceeds maximum nesting depth of {MAX_AST_DEPTH}",
                pos,
                self._expr,
            )

    def _exit(self) -> None:
        self._depth -= 1

    def _parse_or(self) -> ExprNode:
        node = self._parse_and()
        while self._current.kind == "OP" and self._current.value == "||":
            op = self._advance().value
            right = self._parse_and()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_and(self) -> ExprNode:
        node = self._parse_equality()
        while self._current.kind == "OP" and self._current.value == "&&":
            op = self._advance().value
            right = self._parse_equality()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_equality(self) -> ExprNode:
        node = self._parse_relational()
        while self._current.kind == "OP" and self._current.value in ("==", "!="):
            op = self._advance().value
            right = self._parse_relational()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_relational(self) -> ExprNode:
        node = self._parse_additive()
        while self._current.kind == "OP" and self._current.value in ("<", "<=", ">", ">="):
            op = self._advance().value
            right = self._parse_additive()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_additive(self) -> ExprNode:
        node = self._parse_multiplicative()
        while self._current.kind == "OP" and self._current.value in ("+", "-"):
            op = self._advance().value
            right = self._parse_multiplicative()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_multiplicative(self) -> ExprNode:
        node = self._parse_unary()
        while self._current.kind == "OP" and self._current.value in ("*", "/", "%"):
            op = self._advance().value
            right = self._parse_unary()
            node = BinaryOpNode(op=op, left=node, right=right)
        return node

    def _parse_unary(self) -> ExprNode:
        if self._current.kind == "OP" and self._current.value in ("+", "-", "!"):
            tok = self._advance()
            self._enter(tok.pos)
            try:
                operand = self._parse_unary()
            finally:
                self._exit()
            return UnaryOpNode(op=tok.value, operand=operand)
        return self._parse_power()

    def _parse_power(self) -> ExprNode:
        left = self._parse_primary()
        if self._current.kind == "OP" and self._current.value == "^":
            tok = self._advance()
            self._enter(tok.pos)
            try:
                right = self._parse_unary()
            finally:
                self._exit()
            return BinaryOpNode(op="^", left=left, right=right)
        return left

    def _parse_primary(self) -> ExprNode:
        tok = self._current

        if tok.kind == "(":
            self._advance()
            if self._current.kind == ")":
                raise FASTEXPRSyntaxError("Empty parentheses '()' are not valid", tok.pos, self._expr)
            self._enter(tok.pos)
            try:
                inner = self._parse_or()
            finally:
                self._exit()
            if self._current.kind != ")":
                raise FASTEXPRSyntaxError(
                    "Unclosed '(' - expected ')'",
                    self._current.pos,
                    self._expr,
                )
            self._advance()
            return inner

        if tok.kind == "IDENT":
            ident = self._advance()
            if self._current.kind == "(":
                return self._parse_function_call(ident)
            return FieldNode(name=ident.value)

        if tok.kind == "NUMBER":
            self._advance()
            raw = tok.value
            if any(ch in raw for ch in (".", "e", "E")):
                return NumberNode(value=float(raw))
            return NumberNode(value=int(raw))

        if tok.kind == "BOOL":
            self._advance()
            return BoolNode(value=(tok.value == "true"))

        if tok.kind == "STRING":
            self._advance()
            quote = tok.value[0]
            return StringNode(value=tok.value[1:], quote=quote)

        if tok.kind == "EOF":
            raise FASTEXPRSyntaxError("Unexpected end of expression", tok.pos, self._expr)

        raise FASTEXPRSyntaxError(
            f"Unexpected token '{tok.value}'",
            tok.pos,
            self._expr,
        )

    def _parse_function_call(self, ident: _Token) -> FunctionNode:
        self._advance()  # consume '('
        args: list[ExprNode] = []
        seen_kwarg = False
        seen_kw_names: set[str] = set()

        self._enter(ident.pos)
        try:
            if self._current.kind != ")":
                while True:
                    if self._current.kind == "EOF":
                        raise FASTEXPRSyntaxError(
                            f"Unclosed '(' in call to '{ident.value}' - expected ')'",
                            self._current.pos,
                            self._expr,
                        )
                    if self._current.kind == ")":
                        raise FASTEXPRSyntaxError(
                            f"Trailing comma in call to '{ident.value}'",
                            self._current.pos,
                            self._expr,
                        )
                    if self._current.kind == "IDENT" and self._peek().kind == "EQ":
                        kw_tok = self._advance()
                        self._advance()  # consume '='
                        if kw_tok.value in seen_kw_names:
                            raise FASTEXPRSyntaxError(
                                f"Duplicate keyword argument '{kw_tok.value}' in '{ident.value}'",
                                kw_tok.pos,
                                self._expr,
                            )
                        seen_kw_names.add(kw_tok.value)
                        seen_kwarg = True
                        val_node = self._parse_or()
                        args.append(KeywordArgNode(name=kw_tok.value, value=val_node))
                    else:
                        if seen_kwarg:
                            raise FASTEXPRSyntaxError(
                                f"Positional argument cannot follow keyword argument in '{ident.value}'",
                                self._current.pos,
                                self._expr,
                            )
                        args.append(self._parse_or())

                    if self._current.kind == ",":
                        self._advance()
                        continue
                    if self._current.kind == ")":
                        break
                    if self._current.kind == "EOF":
                        raise FASTEXPRSyntaxError(
                            f"Unclosed '(' in call to '{ident.value}' - expected ')'",
                            self._current.pos,
                            self._expr,
                        )
                    raise FASTEXPRSyntaxError(
                        f"Expected ',' or ')' in call to '{ident.value}', got '{self._current.value}'",
                        self._current.pos,
                        self._expr,
                    )

            self._advance()  # consume ')'
        finally:
            self._exit()

        return FunctionNode(name=ident.value, args=args)


def parse_expression(expr: str) -> ExprNode:
    """Parse a FASTEXPR expression string into an AST rooted at :class:`ExprNode`.

    Raises :class:`FASTEXPRSyntaxError` on empty input, lexical errors,
    unbalanced parentheses, or trailing tokens.
    """
    if not isinstance(expr, str):
        raise FASTEXPRSyntaxError("Expression must be a string")
    stripped = expr.strip()
    if not stripped:
        raise FASTEXPRSyntaxError("Expression cannot be empty", 0, expr)
    tokens = _tokenize(stripped)
    return _RecursiveDescentParser(tokens, stripped).parse()


def _coerce_operator_record(item: OperatorRecord | Mapping[str, Any]) -> OperatorRecord:
    if isinstance(item, OperatorRecord):
        return item
    scope_val = item.get("scope_json", item.get("scope", "[]"))
    if not isinstance(scope_val, str):
        scope_val = json.dumps(scope_val, sort_keys=True)
    return OperatorRecord(
        name=str(item.get("name", "")),
        category=str(item.get("category", "")),
        scope_json=scope_val,
        definition=str(item.get("definition", "")),
        description=str(item.get("description", "")),
        num_inputs=int(item.get("num_inputs", item.get("numInputs", 1))),
        min_inputs=int(item.get("min_inputs", item.get("minInputs", 1))),
        uses_lookback=bool(item.get("uses_lookback", item.get("usesLookback", False))),
    )


def _coerce_datafield_record(item: DataFieldRecord | Mapping[str, Any]) -> DataFieldRecord:
    if isinstance(item, DataFieldRecord):
        return item
    raw_ds = item.get("dataset_id", item.get("dataset", ""))
    if isinstance(raw_ds, Mapping):
        ds_id = str(raw_ds.get("id", ""))
    else:
        ds_id = str(raw_ds or "")
    return DataFieldRecord(
        field_id=str(item.get("field_id", item.get("id", ""))),
        dataset_id=ds_id,
        region=str(item.get("region", "USA")),
        universe=str(item.get("universe", "TOP3000")),
        delay=int(item.get("delay", 1)),
        field_type=str(item.get("field_type", item.get("type", "MATRIX"))).upper(),
        category=str(item.get("category", "")),
        subcategory=str(item.get("subcategory", "")),
        coverage=float(item.get("coverage", 0.0)),
        user_count=int(item.get("user_count", item.get("userCount", 0))),
        alpha_count=int(item.get("alpha_count", item.get("alphaCount", 0))),
        description=str(item.get("description", "")),
        is_event_input=bool(item.get("is_event_input", item.get("isEventInput", False))),
    )


class _DualBindMethod:
    """Descriptor allowing a method to work on both class and instance calls."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        wraps(func)(self)

    def __get__(self, instance: Any, owner: Any = None) -> Callable[..., Any]:
        @wraps(self._func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._func(instance, *args, **kwargs)

        return wrapper


class ExpressionValidator:
    """Semantic and compiler-rule validator for parsed FASTEXPR expressions."""

    BUILTIN_FIELDS: frozenset[str] = BUILTIN_FIELDS

    def __init__(
        self,
        operators: Optional[
            Sequence[OperatorRecord | Mapping[str, Any]]
            | Mapping[str, OperatorRecord | Mapping[str, Any]]
        ] = None,
        datafields: Optional[
            Sequence[DataFieldRecord | Mapping[str, Any]]
            | Mapping[str, DataFieldRecord | Mapping[str, Any]]
        ] = None,
        compiler_rules: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        self.operators: dict[str, OperatorRecord] = {}
        self.datafields: dict[str, DataFieldRecord] = {}
        self.event_incompatible_operators: set[str] = set()
        self.event_input_replacements: dict[str, str] = dict(_DEFAULT_EVENT_REPLACEMENTS)
        self.blocked_operators: set[str] = set()
        self.blocked_fields: set[str] = set()
        self.compiler_rules: list[dict[str, Any]] = []

        if operators:
            items = operators.values() if isinstance(operators, Mapping) else operators
            for raw in items:
                rec = _coerce_operator_record(raw)
                if rec.name:
                    self.operators[rec.name] = rec

        if datafields:
            items = datafields.values() if isinstance(datafields, Mapping) else datafields
            for raw in items:
                rec = _coerce_datafield_record(raw)
                if rec.field_id:
                    self.datafields[rec.field_id] = rec

        if compiler_rules:
            self.load_compiler_rules(compiler_rules)

    def load_compiler_rules(self, compiler_rules: Sequence[Mapping[str, Any]]) -> None:
        """Load compiler rules (from ``db.get_compiler_rules()``) into validator state."""
        self.compiler_rules = [dict(r) for r in compiler_rules]
        for rule in self.compiler_rules:
            rule_type = str(rule.get("rule_type") or rule.get("knowledge_type") or "").strip().lower()
            op_name = str(rule.get("operator_name") or "").strip()
            field_id = str(rule.get("field_id") or "").strip()
            replacement = str(rule.get("replacement") or "").strip()

            if rule_type in {
                "event_input_incompatible",
                "event_input_incompatibility",
                "event_input",
            }:
                if op_name:
                    lower_op = op_name.lower()
                    self.event_incompatible_operators.add(lower_op)
                    if replacement:
                        self.event_input_replacements[lower_op] = replacement
            elif rule_type in {"unknown_field", "invalid_field"}:
                if field_id:
                    self.blocked_fields.add(field_id)
            elif rule_type in {"unknown_operator", "invalid_operator", "inaccessible_operator"}:
                if op_name:
                    self.blocked_operators.add(op_name)

    def validate(self, expression: str) -> tuple[bool, list[str]]:
        """Parse and semantically validate a FASTEXPR expression.

        Returns ``(is_valid, error_messages)``.
        """
        try:
            ast = parse_expression(expression)
        except FASTEXPRSyntaxError as exc:
            return False, [str(exc)]

        errors: list[str] = []
        self._validate_node(ast, parent=None, errors=errors, in_enum_kwarg=False)
        return len(errors) == 0, errors

    def _validate_node(
        self,
        node: ExprNode,
        parent: Optional[ExprNode],
        errors: list[str],
        in_enum_kwarg: bool,
    ) -> None:
        if isinstance(node, FunctionNode):
            op_name = node.name
            if op_name in self.blocked_operators:
                errors.append(f"Operator '{op_name}' is blocked by compiler rules")
            elif self.operators:
                op_rec = self.operators.get(op_name)
                if op_rec is None:
                    errors.append(f"Unknown operator '{op_name}'")
                else:
                    positional_args = sum(
                        1 for arg in node.args if not isinstance(arg, KeywordArgNode)
                    )
                    if positional_args < op_rec.min_inputs:
                        errors.append(
                            f"Operator '{op_name}' requires at least {op_rec.min_inputs} "
                            f"positional input(s), got {positional_args}"
                        )

            for arg in node.args:
                if isinstance(arg, KeywordArgNode):
                    is_enum = arg.name.lower() in _ENUM_KWARG_NAMES and isinstance(
                        arg.value, FieldNode
                    )
                    self._validate_node(
                        arg.value,
                        parent=node,
                        errors=errors,
                        in_enum_kwarg=is_enum,
                    )
                else:
                    self._validate_node(arg, parent=node, errors=errors, in_enum_kwarg=False)
            return

        if isinstance(node, BinaryOpNode):
            self._validate_node(node.left, parent=node, errors=errors, in_enum_kwarg=False)
            self._validate_node(node.right, parent=node, errors=errors, in_enum_kwarg=False)
            return

        if isinstance(node, UnaryOpNode):
            self._validate_node(node.operand, parent=node, errors=errors, in_enum_kwarg=False)
            return

        if isinstance(node, KeywordArgNode):
            is_enum = node.name.lower() in _ENUM_KWARG_NAMES and isinstance(node.value, FieldNode)
            self._validate_node(node.value, parent=parent, errors=errors, in_enum_kwarg=is_enum)
            return

        if isinstance(node, FieldNode):
            if in_enum_kwarg:
                return
            self._validate_field(node.name, parent=parent, errors=errors)

    def _validate_field(
        self,
        field_name: str,
        parent: Optional[ExprNode],
        errors: list[str],
    ) -> None:
        if field_name in self.blocked_fields:
            errors.append(f"Field '{field_name}' is blocked by compiler rules")

        field_rec = self.datafields.get(field_name)
        if self.datafields and field_rec is None and field_name not in BUILTIN_FIELDS:
            errors.append(f"Unknown field '{field_name}'")
            return

        if field_rec is None:
            return

        field_type = str(field_rec.field_type).upper()
        if field_type == "VECTOR":
            if not (isinstance(parent, FunctionNode) and parent.name.startswith("vec_")):
                parent_desc = (
                    f"'{parent.name}'"
                    if isinstance(parent, FunctionNode)
                    else (
                        f"'{parent.op}'"
                        if isinstance(parent, (BinaryOpNode, UnaryOpNode))
                        else "top-level expression"
                    )
                )
                errors.append(
                    f"VECTOR field '{field_name}' must be wrapped directly by a vec_* operator, "
                    f"found parent {parent_desc}"
                )

        is_event = bool(field_rec.is_event_input) or field_type == "EVENT"
        if is_event:
            enclosing_op = self._enclosing_operator_name(parent)
            if enclosing_op and enclosing_op.lower() in self.event_incompatible_operators:
                replacement = self.event_input_replacements.get(enclosing_op.lower(), "")
                hint = f" (try '{replacement}')" if replacement else ""
                errors.append(
                    f"Operator '{enclosing_op}' does not support event input field "
                    f"'{field_name}'{hint}"
                )

    @staticmethod
    def _enclosing_operator_name(parent: Optional[ExprNode]) -> Optional[str]:
        if isinstance(parent, FunctionNode):
            return parent.name
        if isinstance(parent, BinaryOpNode):
            return _BINARY_OP_TO_FUNC.get(parent.op)
        if isinstance(parent, UnaryOpNode):
            return _UNARY_OP_TO_FUNC.get(parent.op)
        return None

    @_DualBindMethod
    def seed_compiler_knowledge(
        self: Optional[ExpressionValidator],
        db: AlphaDB,
        json_path: Optional[str] = None,
    ) -> int:
        """Seed initial rules from ``compiler_knowledge.json`` if ``compiler_knowledge`` is empty."""
        if db.get_compiler_rules():
            if self is not None:
                self.load_compiler_rules(db.get_compiler_rules())
            return 0

        if json_path is not None:
            path = Path(json_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Compiler knowledge JSON not found: {path}")
        else:
            path = _DEFAULT_COMPILER_JSON
            if not path.is_file():
                logger.warning("Default compiler_knowledge.json not found at %s", path)
                return 0

        payload = json.loads(path.read_text(encoding="utf-8"))
        seeded = 0

        event_section = payload.get("event_input_compatibility", {})
        replacements = {
            str(k).strip().lower(): str(v).strip()
            for k, v in (event_section.get("replacements") or {}).items()
            if str(k).strip()
        }
        incompatible_ops: list[str] = []
        seen_ops: set[str] = set()
        for raw_op in (event_section.get("incompatible_operators") or []):
            op = str(raw_op).strip().lower()
            if op and op not in seen_ops:
                seen_ops.add(op)
                incompatible_ops.append(op)
        for op in replacements:
            if op not in seen_ops:
                seen_ops.add(op)
                incompatible_ops.append(op)

        for op in incompatible_ops:
            db.record_compiler_rule(
                rule_type="event_input_incompatible",
                operator_name=op,
                field_id="",
                error_pattern="does not support event inputs",
                replacement=replacements.get(op, ""),
            )
            seeded += 1

        for raw_rule in (payload.get("learned_rules") or {}).get("rules", []):
            if not isinstance(raw_rule, Mapping):
                continue
            rule_type = str(raw_rule.get("rule_type") or "").strip()
            if not rule_type:
                continue
            db.record_compiler_rule(
                rule_type=rule_type,
                operator_name=str(raw_rule.get("operator_name") or ""),
                field_id=str(raw_rule.get("field_id") or ""),
                error_pattern=str(raw_rule.get("error_pattern") or ""),
                replacement=str(raw_rule.get("replacement") or ""),
            )
            seeded += 1

        if self is not None:
            self.load_compiler_rules(db.get_compiler_rules())

        return seeded

    def learn_from_error(
        self,
        expression: str,
        error_msg: str,
        db: Optional[AlphaDB] = None,
    ) -> list[dict[str, Any]]:
        """Parse a BRAIN API error message and record compiler rules in memory and ``db``."""
        msg = (error_msg or "").strip()
        if not msg:
            return []

        lower = msg.lower()
        recorded: list[dict[str, Any]] = []

        def _persist(
            rule_type: str,
            operator_name: str = "",
            field_id: str = "",
            error_pattern: str = "",
            replacement: str = "",
        ) -> None:
            entry = {
                "rule_type": rule_type,
                "operator_name": operator_name,
                "field_id": field_id,
                "error_pattern": error_pattern,
                "replacement": replacement,
            }
            recorded.append(entry)
            if db is not None:
                db.record_compiler_rule(
                    rule_type=rule_type,
                    operator_name=operator_name,
                    field_id=field_id,
                    error_pattern=error_pattern,
                    replacement=replacement,
                )

        # 1. Event-input incompatibility or requirement
        if "event input" in lower:
            expects_only = "expects only event input" in lower
            op_match = re.search(
                r"(?:operator\s+)?['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+"
                r"(?:does\s+not\s+support|expects\s+only|is\s+not\s+supported\s+for)\s+event\s+inputs?",
                msg,
                re.IGNORECASE,
            )
            op_name = op_match.group(1).lower() if op_match else ""
            if not op_name:
                op_name = self._infer_event_operator_from_expr(expression) or ""

            if expects_only:
                if op_name:
                    _persist(
                        rule_type="event_input_required",
                        operator_name=op_name,
                        error_pattern="expects only event inputs",
                    )
            else:
                if op_name:
                    self.event_incompatible_operators.add(op_name)
                    replacement = self.event_input_replacements.get(op_name, "")
                    _persist(
                        rule_type="event_input_incompatible",
                        operator_name=op_name,
                        error_pattern="does not support event inputs",
                        replacement=replacement,
                    )

        # 2. Unknown / inaccessible variable or field
        var_match = re.search(
            r"(?:attempted\s+to\s+use\s+)?(?:inaccessible\s+or\s+)?"
            r"(?:unknown|undefined|invalid)\s+(?:variable|field)\s*[:\s]*['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
            msg,
            re.IGNORECASE,
        )
        if var_match:
            field_id = var_match.group(1)
            self.blocked_fields.add(field_id)
            _persist(
                rule_type="unknown_field",
                field_id=field_id,
                error_pattern="unknown variable",
            )

        # 3. Unknown / inaccessible operator
        op_err_match = re.search(
            r"(?:attempted\s+to\s+use\s+)?(?:inaccessible\s+or\s+)?"
            r"(?:unknown|inaccessible|invalid)\s+operator\s*[:\s]*['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?",
            msg,
            re.IGNORECASE,
        )
        if op_err_match:
            op_name = op_err_match.group(1)
            self.blocked_operators.add(op_name)
            replacement = self.event_input_replacements.get(op_name.lower(), "")
            _persist(
                rule_type="unknown_operator",
                operator_name=op_name,
                error_pattern="unknown operator",
                replacement=replacement,
            )

        # 4. Missing lookback / arity / vector mismatch fallback
        if not recorded:
            if "lookback" in lower:
                op_name = self._first_operator_name(expression) or ""
                _persist(
                    rule_type="missing_lookback",
                    operator_name=op_name,
                    error_pattern="required lookback missing",
                )
            elif "vector" in lower:
                op_name = self._first_operator_name(expression) or ""
                _persist(
                    rule_type="vector_type_error",
                    operator_name=op_name,
                    error_pattern="vector field type mismatch",
                )
            elif "number of inputs" in lower or "arguments" in lower or "expects" in lower:
                op_name = self._first_operator_name(expression) or ""
                _persist(
                    rule_type="arity_mismatch",
                    operator_name=op_name,
                    error_pattern=msg[:120],
                )

        return recorded

    def _infer_event_operator_from_expr(self, expression: str) -> Optional[str]:
        try:
            ast = parse_expression(expression)
        except FASTEXPRSyntaxError:
            return None

        candidate_op: Optional[str] = None
        fallback_op: Optional[str] = None

        def walk(node: ExprNode, parent: Optional[ExprNode]) -> None:
            nonlocal candidate_op, fallback_op
            if isinstance(node, FunctionNode):
                if fallback_op is None:
                    fallback_op = node.name.lower()
                for arg in node.args:
                    walk(arg, node)
            elif isinstance(node, BinaryOpNode):
                if fallback_op is None and node.op in _BINARY_OP_TO_FUNC:
                    fallback_op = _BINARY_OP_TO_FUNC[node.op]
                walk(node.left, node)
                walk(node.right, node)
            elif isinstance(node, UnaryOpNode):
                walk(node.operand, node)
            elif isinstance(node, KeywordArgNode):
                walk(node.value, parent)
            elif isinstance(node, FieldNode):
                rec = self.datafields.get(node.name)
                if rec is not None and (rec.is_event_input or rec.field_type.upper() == "EVENT"):
                    enclosing = self._enclosing_operator_name(parent)
                    if enclosing and candidate_op is None:
                        candidate_op = enclosing.lower()

        walk(ast, None)
        return candidate_op or fallback_op

    @staticmethod
    def _first_operator_name(expression: str) -> Optional[str]:
        try:
            ast = parse_expression(expression)
        except FASTEXPRSyntaxError:
            return None
        if isinstance(ast, FunctionNode):
            return ast.name
        return None


__all__ = [
    "BUILTIN_FIELDS",
    "GROUP_FIELDS",
    "FASTEXPRSyntaxError",
    "ExprNode",
    "FunctionNode",
    "BinaryOpNode",
    "UnaryOpNode",
    "FieldNode",
    "NumberNode",
    "BoolNode",
    "StringNode",
    "KeywordArgNode",
    "parse_expression",
    "ExpressionValidator",
]
