"""Phase 2 tests for catalog sync, FASTEXPR parser/validator, AST mutator, pre-dedup, bandit, and producer."""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import requests

from alpha_factory.brain import catalog as catalog_mod
from alpha_factory.brain.catalog import CatalogSync, parse_operator_definition, wrap_vector_field
from alpha_factory.brain.client import BrainClient
from alpha_factory.generation.algorithmic_llm import AlgorithmicLLMGenerator
from alpha_factory.generation.ast_mutator import SafeASTMutator
from alpha_factory.generation.bandit import (
    ThompsonBanditScheduler,
    format_arm_key,
    parse_arm_key,
)
from alpha_factory.generation.pre_dedup import (
    PreSimDeduplicator,
    canonical_ast_fingerprint,
    compute_config_hash,
)
from alpha_factory.generation.producer import CandidateProducer
from alpha_factory.generation.template_expander import expand_template, load_templates_yaml
from alpha_factory.generation.validator import (
    BinaryOpNode,
    ExpressionValidator,
    ExprNode,
    FASTEXPRSyntaxError,
    FieldNode,
    FunctionNode,
    KeywordArgNode,
    UnaryOpNode,
    parse_expression,
)
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import (
    DataFieldRecord,
    OperatorRecord,
    SimulationRecord,
    SimulationSettings,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Offline & Zero-Sleep Safeguards + HTTP Stubs
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_and_fast_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure zero real network calls and instant sleeps across all Phase 2 tests."""
    monkeypatch.setattr(time, "sleep", lambda _secs: None)

    def _forbid_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Real network call attempted in offline unit test")

    monkeypatch.setattr(requests.Session, "request", _forbid_network)
    monkeypatch.setattr(requests, "request", _forbid_network)
    monkeypatch.setattr(requests, "get", _forbid_network)
    monkeypatch.setattr(requests, "post", _forbid_network)


class FakeResponse:
    """Minimal in-memory HTTP response stub compatible with BrainClient."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        headers: Optional[dict[str, Any]] = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = dict(headers or {})
        self.text = text

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        if self.text:
            return json.loads(self.text)
        raise ValueError("No JSON payload")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Thread-safe mocked session capturing HTTP methods, URLs, and query params."""

    def __init__(self, handler: Callable[[str, str, dict[str, Any]], FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self.cookies = requests.cookies.RequestsCookieJar()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._handler = handler

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        with self._lock:
            self.calls.append((method.upper(), url, dict(kwargs)))
        return self._handler(method.upper(), url, kwargs)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("POST", url, **kwargs)


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------


def _sample_operators() -> list[OperatorRecord]:
    return [
        OperatorRecord(
            name="rank",
            category="Cross Sectional",
            scope_json='["REGULAR"]',
            definition="rank(x, rate=2)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="zscore",
            category="Cross Sectional",
            scope_json='["REGULAR"]',
            definition="zscore(x)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="normalize",
            category="Cross Sectional",
            scope_json='["REGULAR"]',
            definition="normalize(x, useStd=false, limit=0.0)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="scale",
            category="Cross Sectional",
            scope_json='["REGULAR"]',
            definition="scale(x, scale=1, longscale=1, shortscale=1)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="winsorize",
            category="Cross Sectional",
            scope_json='["REGULAR"]',
            definition="winsorize(x, std=4)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="ts_rank",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_rank(x, d, constant=0)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_zscore",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_zscore(x, d)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_mean",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_mean(x, d)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_delta",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_delta(x, d)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_std_dev",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_std_dev(x, d)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_decay_linear",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_decay_linear(x, d, dense=false)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_backfill",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_backfill(x, lookback=d, k=1, ignore='NAN')",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_corr",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_corr(x, y, d)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="ts_regression",
            category="Time Series",
            scope_json='["REGULAR"]',
            definition="ts_regression(y, x, d, lag=0, rettype=0)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=True,
        ),
        OperatorRecord(
            name="add",
            category="Arithmetic",
            scope_json='["REGULAR"]',
            definition="add(x, y, filter=false)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="subtract",
            category="Arithmetic",
            scope_json='["REGULAR"]',
            definition="subtract(x, y, filter=false)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="multiply",
            category="Arithmetic",
            scope_json='["REGULAR"]',
            definition="multiply(x, y, filter=false)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="group_neutralize",
            category="Group",
            scope_json='["REGULAR"]',
            definition="group_neutralize(x, group)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="group_rank",
            category="Group",
            scope_json='["REGULAR"]',
            definition="group_rank(x, group)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="group_zscore",
            category="Group",
            scope_json='["REGULAR"]',
            definition="group_zscore(x, group)",
            num_inputs=2,
            min_inputs=2,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="vec_avg",
            category="Vector",
            scope_json='["REGULAR"]',
            definition="vec_avg(x)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="vec_sum",
            category="Vector",
            scope_json='["REGULAR"]',
            definition="vec_sum(x)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
        OperatorRecord(
            name="is_nan",
            category="Logical",
            scope_json='["REGULAR"]',
            definition="is_nan(x)",
            num_inputs=1,
            min_inputs=1,
            uses_lookback=False,
        ),
    ]


def _collect_field_names(node: ExprNode) -> list[str]:
    names: list[str] = []
    if isinstance(node, FieldNode):
        names.append(node.name)
    elif isinstance(node, FunctionNode):
        for arg in node.args:
            names.extend(_collect_field_names(arg))
    elif isinstance(node, KeywordArgNode):
        names.extend(_collect_field_names(node.value))
    elif isinstance(node, BinaryOpNode):
        names.extend(_collect_field_names(node.left))
        names.extend(_collect_field_names(node.right))
    elif isinstance(node, UnaryOpNode):
        names.extend(_collect_field_names(node.operand))
    return names


# ---------------------------------------------------------------------------
# 1. CatalogSync: Local Cache Bootstrap & Paginated Remote Sync
# ---------------------------------------------------------------------------


def test_catalog_bootstrap_from_real_workspace_cache(tmp_path: Path) -> None:
    op_cache = WORKSPACE_ROOT / "worldquant-miner" / "generation_two" / "constants" / "operatorRAW.json"
    usa_fields_cache = WORKSPACE_ROOT / "constants" / "data_fields_cache_USA_1_TOP3000.json"
    assert op_cache.is_file()
    assert usa_fields_cache.is_file()

    db = AlphaDB(tmp_path / "catalog_real.db")
    sync = CatalogSync(db=db)
    counts = sync.bootstrap_from_local_cache(str(WORKSPACE_ROOT))

    assert counts["operators"] > 50
    assert counts["datasets"] > 0
    assert counts["datafields"] > 100

    operators = {op.name: op for op in db.get_operators()}
    assert {"add", "rank", "ts_rank", "vec_avg", "group_neutralize"}.issubset(operators.keys())
    assert operators["add"].min_inputs == 2
    assert operators["ts_rank"].uses_lookback is True
    assert operators["rank"].min_inputs == 1

    usa_fields = db.get_datafields("USA", "TOP3000", 1)
    assert len(usa_fields) > 50
    field_ids = {f.field_id for f in usa_fields}
    assert "fnd69_capxy2q" in field_ids

    usa_datasets = db.get_datasets("USA", "TOP3000", 1)
    assert len(usa_datasets) > 0
    assert "fundamental69" in {ds["id"] for ds in usa_datasets}


def test_catalog_bootstrap_synthesizes_datasets_from_datafields_cache(tmp_path: Path) -> None:
    ws = tmp_path / "isolated_workspace"
    op_dst = ws / "worldquant-miner" / "generation_two" / "constants" / "operatorRAW.json"
    df_dst = ws / "constants" / "data_fields_cache_USA_1_TOP3000.json"
    op_dst.parent.mkdir(parents=True, exist_ok=True)
    df_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        WORKSPACE_ROOT / "worldquant-miner" / "generation_two" / "constants" / "operatorRAW.json",
        op_dst,
    )
    shutil.copyfile(
        WORKSPACE_ROOT / "constants" / "data_fields_cache_USA_1_TOP3000.json",
        df_dst,
    )

    db = AlphaDB(tmp_path / "catalog_synth.db")
    counts = CatalogSync(db=db).bootstrap_from_local_cache(str(ws))

    assert counts["operators"] > 50
    assert counts["datafields"] > 50
    assert counts["datasets"] > 0

    usa_datasets = db.get_datasets("USA", "TOP3000", 1)
    ds_map = {ds["id"]: ds for ds in usa_datasets}
    assert "fundamental69" in ds_map
    assert ds_map["fundamental69"]["field_count"] > 0


def test_catalog_sync_region_pagination_with_mocked_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_datasets = 65
    first_ds_fields = 120

    datasets_payload = [
        {
            "id": f"ds_{i:03d}",
            "name": f"Dataset {i}",
            "category": {"id": "fundamental", "name": "Fundamental"},
            "subcategory": {"id": "sub", "name": "Sub"},
            "coverage": 0.9,
            "fieldCount": first_ds_fields if i == 0 else 1,
            "userCount": 10 + i,
            "valueScore": float(100 - i),
            "alphaCount": 5,
        }
        for i in range(total_datasets)
    ]

    fields_by_ds: dict[str, list[dict[str, Any]]] = {
        "ds_000": [
            {
                "id": f"f_000_{j:03d}",
                "dataset": {"id": "ds_000"},
                "category": {"name": "Fundamental"},
                "subcategory": {"name": "Sub"},
                "type": "MATRIX",
                "coverage": 0.95,
                "userCount": 5,
                "alphaCount": 2,
                "description": f"Field {j}",
            }
            for j in range(first_ds_fields)
        ]
    }
    for i in range(1, total_datasets):
        ds_id = f"ds_{i:03d}"
        fields_by_ds[ds_id] = [
            {
                "id": f"f_{i:03d}_000",
                "dataset": {"id": ds_id},
                "category": {"name": "Fundamental"},
                "type": "MATRIX",
                "coverage": 0.85,
            }
        ]

    dataset_offsets: list[int] = []
    ds0_field_offsets: list[int] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(catalog_mod.time, "sleep", lambda sec: sleep_calls.append(sec))

    def handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        params = kwargs.get("params") or {}
        if url.endswith("/operators"):
            return FakeResponse(
                200,
                json_data=[
                    {"name": "ts_rank", "category": "Time Series", "definition": "ts_rank(x, d)"},
                    {"name": "rank", "category": "Cross Sectional", "definition": "rank(x)"},
                ],
            )
        if url.endswith("/data-sets"):
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 50))
            dataset_offsets.append(offset)
            headers = {"x-ratelimit-remaining-1m": "1", "x-ratelimit-reset-1m": "0.2"} if offset == 0 else {}
            return FakeResponse(
                200,
                json_data={
                    "count": total_datasets,
                    "results": datasets_payload[offset : offset + limit],
                },
                headers=headers,
            )
        if url.endswith("/data-fields"):
            ds_id = str(params.get("dataset.id", ""))
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 50))
            if ds_id == "ds_000":
                ds0_field_offsets.append(offset)
            items = fields_by_ds.get(ds_id, [])
            return FakeResponse(
                200,
                json_data={
                    "count": len(items),
                    "results": items[offset : offset + limit],
                },
            )
        raise AssertionError(f"Unexpected endpoint: {method} {url}")

    session = FakeSession(handler)
    client = BrainClient(session=session, api_url="https://api.worldquantbrain.com")
    db = AlphaDB(tmp_path / "sync_region.db")
    sync = CatalogSync(db=db, client=client)

    ops_count = sync.sync_operators()
    ds_count, df_count = sync.sync_region(region="USA", delay=1, universe="TOP3000")

    assert ops_count == 2
    assert ds_count == total_datasets
    assert df_count == first_ds_fields + (total_datasets - 1)
    assert dataset_offsets == [0, 50]
    assert ds0_field_offsets == [0, 50, 100]
    assert len(sleep_calls) == 1

    assert len(db.get_datasets("USA", "TOP3000", 1)) == total_datasets
    assert len(db.get_datafields("USA", "TOP3000", 1)) == first_ds_fields + (total_datasets - 1)
    assert wrap_vector_field("snt_vec") == "vec_avg(snt_vec)"
    assert wrap_vector_field("vec_sum(snt_vec)") == "vec_sum(snt_vec)"
    assert parse_operator_definition("ts_corr", "ts_corr(x, y, d)") == (2, 2, True)


# ---------------------------------------------------------------------------
# 2. FASTEXPR Parser & ExpressionValidator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "ts_rank(close, 20)",
        "rank(ts_backfill(anl49_1stfiscalquarter, 60))",
        "group_neutralize(ts_zscore(close, 20), industry)",
        "ts_rank(vec_avg(snt1_vec), 10)",
        "add(close, open, filter=true)",
        "winsorize(close, std=4)",
        "-(close - open) * volume / adv20",
        "(close + open) * vwap",
        "rank(close) > 0.5 && !is_nan(volume)",
        "2 ^ 3 ^ 2",
        "(2 ^ 3) ^ 2",
        'ts_regression(close, open, 20, rettype="resid")',
    ],
)
def test_fastexpr_parser_round_trip_valid_expressions(expr: str) -> None:
    ast = parse_expression(expr)
    rendered = ast.to_string()
    reparsed = parse_expression(rendered)
    assert reparsed == ast
    assert rendered == expr


@pytest.mark.parametrize(
    "bad_expr",
    [
        "",
        "   ",
        "ts_rank(close, 20",
        "ts_rank(close, 20))",
        "(close + open",
        "close + open)",
        "()",
        "ts_rank(close, )",
        "add(filter=true, close)",
        'ts_rank("unterminated, 20)',
    ],
)
def test_fastexpr_parser_and_validator_reject_syntax_errors(bad_expr: str) -> None:
    with pytest.raises(FASTEXPRSyntaxError):
        parse_expression(bad_expr)

    validator = ExpressionValidator(operators=_sample_operators())
    ok, errors = validator.validate(bad_expr)
    assert ok is False
    assert errors


def test_expression_validator_rejects_unknown_op_bad_arity_and_unwrapped_vector() -> None:
    datafields = [
        DataFieldRecord(field_id="close", dataset_id="pv1", field_type="MATRIX"),
        DataFieldRecord(field_id="open", dataset_id="pv1", field_type="MATRIX"),
        DataFieldRecord(field_id="snt_vec", dataset_id="snt1", field_type="VECTOR"),
    ]
    validator = ExpressionValidator(operators=_sample_operators(), datafields=datafields)

    # Unknown operator
    ok_unk, err_unk = validator.validate("nonexistent_op(close, 20)")
    assert ok_unk is False
    assert any("Unknown operator 'nonexistent_op'" in e for e in err_unk)

    # Insufficient positional args (keyword arg does not count toward positional min_inputs)
    for bad_arity_expr in ("add(close)", "add(close, filter=true)", "ts_corr(close)"):
        ok_arg, err_arg = validator.validate(bad_arity_expr)
        assert ok_arg is False
        assert any("positional input" in e for e in err_arg)

    # Unwrapped VECTOR field (top-level, non-vec function, and binary op)
    for unwrapped in ("snt_vec", "ts_rank(snt_vec, 20)", "snt_vec + close"):
        ok_vec, err_vec = validator.validate(unwrapped)
        assert ok_vec is False
        assert any("VECTOR field 'snt_vec'" in e for e in err_vec)

    # Properly wrapped VECTOR field succeeds
    ok_wrapped, err_wrapped = validator.validate("ts_rank(vec_avg(snt_vec), 20)")
    assert ok_wrapped is True
    assert err_wrapped == []


def test_expression_validator_event_input_incompatibility_and_learn_from_error(
    tmp_path: Path,
) -> None:
    db = AlphaDB(tmp_path / "compiler_rules.db")
    datafields = [
        DataFieldRecord(
            field_id="evt_surprise",
            dataset_id="event1",
            field_type="EVENT",
            is_event_input=True,
        ),
        DataFieldRecord(field_id="close", dataset_id="pv1", field_type="MATRIX"),
    ]
    validator = ExpressionValidator(operators=_sample_operators(), datafields=datafields)

    # Initially no rule blocks rank(evt_surprise)
    assert validator.validate("rank(evt_surprise)") == (True, [])

    # Learn explicit operator event incompatibility
    recorded = validator.learn_from_error(
        expression="rank(evt_surprise)",
        error_msg='Operator "rank" does not support event inputs',
        db=db,
    )
    assert len(recorded) == 1
    assert recorded[0]["rule_type"] == "event_input_incompatible"
    assert recorded[0]["operator_name"] == "rank"
    assert recorded[0]["replacement"] == "ts_rank"

    ok_after, errors_after = validator.validate("rank(evt_surprise)")
    assert ok_after is False
    assert any("does not support event input" in e and "ts_rank" in e for e in errors_after)

    # Compatible time-series operator remains valid
    assert validator.validate("ts_rank(evt_surprise, 20)") == (True, [])

    # Also test infix inference when operator name is omitted from error string
    recorded_infix = validator.learn_from_error(
        expression="evt_surprise + close",
        error_msg="Event input is not supported in this context",
        db=db,
    )
    assert recorded_infix[0]["operator_name"] == "add"
    ok_infix, errors_infix = validator.validate("evt_surprise + close")
    assert ok_infix is False
    assert any("add" in e and "event input" in e for e in errors_infix)

    # Reload rules from SQLite into a fresh validator
    reloaded = ExpressionValidator(
        operators=_sample_operators(),
        datafields=datafields,
        compiler_rules=db.get_compiler_rules(),
    )
    assert reloaded.validate("rank(evt_surprise)")[0] is False
    assert reloaded.validate("evt_surprise + close")[0] is False
    assert reloaded.validate("ts_rank(evt_surprise, 20)")[0] is True


# ---------------------------------------------------------------------------
# 3. SafeASTMutator
# ---------------------------------------------------------------------------


def test_safe_ast_mutator_never_corrupts_digits_inside_field_names() -> None:
    target_field = "anl49_1stfiscalquarter"
    alt_fields = [target_field, "anl49_2ndfiscalquarter", "fnd69_capxy2q"]
    datafields = [
        DataFieldRecord(field_id=fid, dataset_id="analyst49", field_type="MATRIX")
        for fid in alt_fields
    ]
    validator = ExpressionValidator(operators=_sample_operators(), datafields=datafields)
    mutator = SafeASTMutator(validator=validator, lookback_grid=(5, 10, 20, 40, 60, 120))

    expr = f"ts_rank({target_field}, 20)"

    # Case 1: Only the target field exists -> every mutation must preserve `anl49_1stfiscalquarter` verbatim
    single_field_mutator = SafeASTMutator(
        validator=ExpressionValidator(
            operators=_sample_operators(),
            datafields=[datafields[0]],
        ),
        lookback_grid=(5, 10, 20, 40, 60, 120),
    )
    seen_single: set[str] = set()
    for seed in range(50):
        mutated = single_field_mutator.mutate(
            expr,
            fields_by_type={"MATRIX": [target_field]},
            rng=random.Random(seed),
        )
        assert mutated is not None
        assert mutated != expr
        assert target_field in mutated
        assert single_field_mutator.validator.validate(mutated)[0] is True
        seen_single.add(mutated)
    assert len(seen_single) >= 3

    # Case 2: Multiple fields available -> every FieldNode in mutated AST must be a known uncorrupted field
    allowed_set = set(alt_fields)
    seen_multi: set[str] = set()
    for seed in range(50):
        mutated = mutator.mutate(
            expr,
            fields_by_type={"MATRIX": alt_fields},
            rng=random.Random(seed),
        )
        assert mutated is not None
        assert mutated != expr
        assert validator.validate(mutated)[0] is True
        parsed_fields = _collect_field_names(parse_expression(mutated))
        assert parsed_fields
        assert all(fid in allowed_set for fid in parsed_fields)
        seen_multi.add(mutated)
    assert len(seen_multi) >= 5


def test_safe_ast_mutator_crossover_produces_valid_expressions() -> None:
    datafields = [
        DataFieldRecord(field_id="close", dataset_id="pv1", field_type="MATRIX"),
        DataFieldRecord(field_id="vwap", dataset_id="pv1", field_type="MATRIX"),
        DataFieldRecord(field_id="snt_vec", dataset_id="snt1", field_type="VECTOR"),
    ]
    validator = ExpressionValidator(operators=_sample_operators(), datafields=datafields)
    mutator = SafeASTMutator(validator=validator)

    expr_a = "ts_rank(close, 20)"
    expr_b = "group_neutralize(ts_zscore(vwap, 60), industry)"
    crossovers: set[str] = set()
    for seed in range(30):
        child = mutator.crossover(expr_a, expr_b, rng=random.Random(seed))
        assert child is not None
        ok, errs = validator.validate(child)
        assert ok is True, f"Invalid crossover {child}: {errs}"
        assert child not in {expr_a, expr_b}
        crossovers.add(child)
    assert len(crossovers) >= 2

    # Ensure VECTOR field safety is preserved during crossover
    vec_expr = "ts_rank(vec_avg(snt_vec), 20)"
    mat_expr = "rank(ts_zscore(close, 10))"
    for seed in range(25):
        child = mutator.crossover(vec_expr, mat_expr, rng=random.Random(seed))
        assert child is not None
        ok, errs = validator.validate(child)
        assert ok is True, f"Invalid vector crossover {child}: {errs}"


# ---------------------------------------------------------------------------
# 4. Template Expander, canonical_ast_fingerprint & PreSimDeduplicator
# ---------------------------------------------------------------------------


def test_template_expander_bounds_and_vector_wrapping() -> None:
    templates = load_templates_yaml(WORKSPACE_ROOT / "alpha_factory" / "config" / "templates.yaml")
    assert len(templates) >= 5

    fields = [
        DataFieldRecord(field_id="fnd_a", dataset_id="fnd1", field_type="MATRIX", coverage=0.95),
        DataFieldRecord(field_id="fnd_b", dataset_id="fnd1", field_type="MATRIX", coverage=0.90),
        DataFieldRecord(
            field_id="snt_pos_vec",
            dataset_id="snt1",
            field_type="VECTOR",
            coverage=0.85,
        ),
        DataFieldRecord(
            field_id="snt_neg_vec",
            dataset_id="snt1",
            field_type="VECTOR",
            coverage=0.85,
        ),
    ]
    expanded = expand_template(
        template="{ts_op}({vec_op}({neg}) - {vec_op}({pos}), {lookback})",
        datafields=fields,
        vector_ops=["vec_avg", "vec_sum"],
        time_series_ops=["ts_rank", "ts_zscore"],
        lookbacks=[5, 20],
        max_per_template=4,
    )
    assert 1 <= len(expanded) <= 4
    assert all("snt_neg_vec" in expr and "snt_pos_vec" in expr for expr, _ in expanded)

    # {field} receives a VECTOR field outside a vec_* call -> auto-wrapped via vec_avg
    wrapped = expand_template(
        template="rank(ts_backfill({field}, {lookback}))",
        datafields=[fields[2]],
        vector_ops=["vec_avg"],
        time_series_ops=["ts_backfill"],
        lookbacks=[20],
        max_per_template=5,
    )
    assert wrapped == [("rank(ts_backfill(vec_avg(snt_pos_vec), 20))", "snt1")]


def test_canonical_ast_fingerprint_normalizes_short_fields() -> None:
    fp_close = canonical_ast_fingerprint("ts_rank(close, 20)")
    fp_vwap = canonical_ast_fingerprint("ts_rank(vwap, 20)")
    fp_cap = canonical_ast_fingerprint("ts_rank(cap, 20)")
    fp_adv20 = canonical_ast_fingerprint("ts_rank(adv20, 20)")

    assert fp_close == "ts_rank(F0, N_M)"
    assert fp_close == fp_vwap == fp_cap == fp_adv20

    # Multi-field short identifiers map deterministically to F0, F1
    assert canonical_ast_fingerprint("ts_corr(close, adv20, 20)") == "ts_corr(F0, F1, N_M)"
    assert canonical_ast_fingerprint("ts_corr(vwap, cap, 20)") == "ts_corr(F0, F1, N_M)"

    # Lookback buckets distinguish short, medium, and long windows
    assert canonical_ast_fingerprint("ts_rank(close, 10)") == "ts_rank(F0, N_S)"
    assert canonical_ast_fingerprint("ts_rank(close, 120)") == "ts_rank(F0, N_L)"


def test_pre_sim_deduplicator_enforces_structure_and_config_hash_uniqueness(
    tmp_path: Path,
) -> None:
    db = AlphaDB(tmp_path / "dedup.db")
    dedup = PreSimDeduplicator(db=db, max_per_structure=2, max_per_dataset_structure=2)
    s_decay4 = SimulationSettings(region="USA", universe="TOP3000", delay=1, decay=4)
    s_decay0 = SimulationSettings(region="USA", universe="TOP3000", delay=1, decay=0)

    # Whitespace-equivalent expressions share the exact config_hash
    assert compute_config_hash("ts_rank(close, 20)", s_decay4) == compute_config_hash(
        "ts_rank( close ,  20 )", s_decay4
    )
    assert compute_config_hash("ts_rank(close, 20)", s_decay4) != compute_config_hash(
        "ts_rank(close, 20)", s_decay0
    )

    batch = [
        ("ts_rank(close, 20)", "pv1"),
        ("ts_rank( close ,  20 )", "pv1"),  # exact config_hash duplicate
        ("ts_rank(vwap, 20)", "pv2"),       # same AST fingerprint, 2nd slot
        ("ts_rank(cap, 20)", "pv3"),        # exceeds max_per_structure=2
    ]
    accepted = dedup.filter_candidates(
        candidates=batch,
        settings_dict=s_decay4,
        region="USA",
        universe="TOP3000",
        delay=1,
        source="template",
    )
    assert [c.expression for c in accepted] == ["ts_rank(close, 20)", "ts_rank(vwap, 20)"]
    assert all(c.ast_fingerprint == "ts_rank(F0, N_M)" for c in accepted)

    # Persist accepted candidates -> DB structure cap now blocks further same-structure expressions
    assert db.enqueue_candidates(accepted) == 2
    more = dedup.filter_candidates(
        candidates=[("ts_rank(adv20, 20)", "pv4")],
        settings_dict=s_decay4,
        region="USA",
        universe="TOP3000",
        delay=1,
        source="template",
    )
    assert more == []

    # Per-dataset structure cap check
    ds_dedup = PreSimDeduplicator(db=db, max_per_structure=5, max_per_dataset_structure=1)
    ds_accepted = ds_dedup.filter_candidates(
        candidates=[
            ("ts_zscore(close, 10)", "ds_alpha"),
            ("ts_zscore(vwap, 10)", "ds_alpha"),  # same dataset -> blocked by max_per_dataset_structure=1
            ("ts_zscore(cap, 10)", "ds_beta"),    # different dataset -> allowed
        ],
        settings_dict=s_decay4,
        region="USA",
        universe="TOP3000",
        delay=1,
        source="template",
    )
    assert [c.expression for c in ds_accepted] == ["ts_zscore(close, 10)", "ts_zscore(cap, 10)"]

    # Config hash present in simulations table blocks re-queuing with same settings, allows new decay
    sim_hash = compute_config_hash("rank(close)", s_decay4)
    db.save_simulation(
        SimulationRecord(
            alpha_id="alpha_existing",
            config_hash=sim_hash,
            ast_fingerprint=canonical_ast_fingerprint("rank(close)"),
            expression="rank(close)",
            region="USA",
            universe="TOP3000",
            delay=1,
            settings_json=s_decay4.to_json(),
            status="COMPLETE",
        )
    )
    assert (
        ds_dedup.filter_candidates(
            candidates=[("rank(close)", "pv1")],
            settings_dict=s_decay4,
            region="USA",
            universe="TOP3000",
            delay=1,
            source="template",
        )
        == []
    )
    diff_settings = ds_dedup.filter_candidates(
        candidates=[("rank(close)", "pv1")],
        settings_dict=s_decay0,
        region="USA",
        universe="TOP3000",
        delay=1,
        source="template",
    )
    assert len(diff_settings) == 1


# ---------------------------------------------------------------------------
# 5. ThompsonBanditScheduler
# ---------------------------------------------------------------------------


def _bandit_settings() -> dict[str, Any]:
    return {
        "markets": {
            "USA": {
                "default_universe": "TOP3000",
                "universes": ["TOP3000"],
                "neutralizations": ["INDUSTRY"],
                "default_neutralization": "INDUSTRY",
                "delay": 1,
            },
            "EUR": {
                "default_universe": "TOP2500",
                "universes": ["TOP2500"],
                "neutralizations": ["INDUSTRY"],
                "default_neutralization": "INDUSTRY",
                "delay": 1,
            },
        },
        "generation": {
            "mix": {"template": 0.5, "algorithmic_llm": 0.5},
            "categories": ["fundamental"],
            "neutralizations": ["INDUSTRY"],
            "decays": [4],
        },
    }


def test_thompson_bandit_reward_updates_and_persistence(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "bandit_rewards.db")
    settings = _bandit_settings()
    scheduler = ThompsonBanditScheduler(db=db, settings=settings, exploration_floor=0.0, seed=123)

    usa_tmpl = format_arm_key("USA", "TOP3000", "fundamental", "template", "INDUSTRY", 4)
    usa_llm = format_arm_key("USA", "TOP3000", "fundamental", "algorithmic_llm", "INDUSTRY", 4)
    assert parse_arm_key(usa_tmpl)["region"] == "USA"

    scheduler.record_outcome(usa_tmpl, reward=1.5)   # clamped to 1.0
    scheduler.record_outcome(usa_tmpl, reward=0.8)
    scheduler.record_outcome(usa_tmpl, reward=-0.3)  # clamped to 0.0

    arm_state = scheduler.get_arm(usa_tmpl)
    assert arm_state is not None
    assert arm_state["pulls"] == 3
    assert arm_state["alpha_param"] == pytest.approx(1.0 + 1.0 + 0.8 + 0.0)
    assert arm_state["beta_param"] == pytest.approx(1.0 + 0.0 + 0.2 + 1.0)
    assert arm_state["total_reward"] == pytest.approx(1.8)

    # Strong posterior separation inside USA should strongly favor usa_tmpl
    for _ in range(20):
        scheduler.record_outcome(usa_tmpl, reward=1.0)
        scheduler.record_outcome(usa_llm, reward=0.0)

    reloaded = ThompsonBanditScheduler(db=db, settings=settings, exploration_floor=0.0, seed=99)
    picks = [reloaded.select_arm(region="USA")["arm_key"] for _ in range(25)]
    assert picks.count(usa_tmpl) >= 22


def test_thompson_bandit_region_and_arm_cooldown_skipping(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "bandit_cooldown.db")
    settings = _bandit_settings()
    scheduler = ThompsonBanditScheduler(db=db, settings=settings, exploration_floor=0.2, seed=7)

    now = time.time()
    # External region cooldown skips USA in favor of EUR
    for _ in range(20):
        chosen = scheduler.select_arm(region_cooldowns={"USA": now + 1800.0})
        assert chosen["region"] == "EUR"

    # Per-arm cooldown skips cooled-down EUR arms too
    for arm in scheduler.list_arms():
        if arm["region"] == "EUR":
            scheduler.record_outcome(arm["arm_key"], reward=0.9, cooldown_until=now + 1800.0)

    for _ in range(20):
        chosen = scheduler.select_arm()
        assert chosen["region"] == "USA"

    # When all regions are cooled down, scheduler falls back to the soonest-expiring region
    fallback = scheduler.select_arm(
        region_cooldowns={"USA": now + 60.0, "EUR": now + 3600.0}
    )
    assert fallback["region"] == "USA"


# ---------------------------------------------------------------------------
# 6. CandidateProducer.produce_once() Offline Integration
# ---------------------------------------------------------------------------


def _producer_datafields() -> list[DataFieldRecord]:
    return [
        DataFieldRecord(
            field_id="fnd69_capxy2q",
            dataset_id="fundamental69",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="MATRIX",
            category="fundamental",
            coverage=0.95,
            user_count=12,
        ),
        DataFieldRecord(
            field_id="fnd69_dltry2q",
            dataset_id="fundamental69",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="MATRIX",
            category="fundamental",
            coverage=0.90,
            user_count=8,
        ),
        DataFieldRecord(
            field_id="fnd28_ebitda",
            dataset_id="fundamental28",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="MATRIX",
            category="fundamental",
            coverage=0.92,
            user_count=15,
        ),
        DataFieldRecord(
            field_id="pv1_close_adj",
            dataset_id="pv1",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="MATRIX",
            category="pv",
            coverage=0.99,
            user_count=40,
        ),
        DataFieldRecord(
            field_id="snt1_positive_vec",
            dataset_id="sentiment1",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="VECTOR",
            category="sentiment",
            coverage=0.80,
            user_count=5,
        ),
        DataFieldRecord(
            field_id="snt1_negative_vec",
            dataset_id="sentiment1",
            region="USA",
            universe="TOP3000",
            delay=1,
            field_type="VECTOR",
            category="sentiment",
            coverage=0.80,
            user_count=5,
        ),
    ]


def test_candidate_producer_produce_once_offline_no_llm(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "producer_none.db")
    db.upsert_operators(_sample_operators())
    fields = _producer_datafields()
    db.upsert_datafields(fields)

    settings = {
        "markets": {
            "USA": {
                "default_universe": "TOP3000",
                "universes": ["TOP3000"],
                "neutralizations": ["INDUSTRY", "SUBINDUSTRY"],
                "default_neutralization": "INDUSTRY",
                "delay": 1,
            }
        },
        "execution": {
            "default_settings": {
                "decay": 4,
                "truncation": 0.08,
                "pasteurization": "ON",
                "unit_handling": "VERIFY",
                "nan_handling": "OFF",
                "max_trade": "OFF",
                "test_period": "P2Y0M0D",
                "visualization": False,
            }
        },
        "generation": {
            "queue_low_watermark": 20,
            "queue_high_watermark": 50,
            "mix": {"template": 0.5, "algorithmic_llm": 0.5, "evolution": 0.0},
            "categories": ["fundamental", "pv", "sentiment"],
            "neutralizations": ["INDUSTRY"],
            "decays": [4],
            "max_per_structure": 3,
            "max_per_dataset_structure": 2,
            "lookback_days": [5, 20, 60],
            "llm": {"backend": "none"},
        },
    }

    templates = load_templates_yaml(WORKSPACE_ROOT / "alpha_factory" / "config" / "templates.yaml")
    producer = CandidateProducer(db=db, settings=settings, templates=templates)

    inserted = producer.produce_once(region="USA", batch_target=8)
    assert 1 <= inserted <= 8
    assert db.count_queue(status="PENDING") == inserted

    # Verify all queued candidates are valid and respect dedup invariants
    full_validator = ExpressionValidator(operators=db.get_operators(), datafields=fields)
    with db._read_conn() as conn:
        rows = conn.execute("SELECT * FROM candidate_queue ORDER BY id ASC").fetchall()

    assert len(rows) == inserted
    cfg_hashes = [str(r["config_hash"]) for r in rows]
    assert len(set(cfg_hashes)) == len(cfg_hashes)

    fp_counts: dict[str, int] = {}
    ds_fp_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        expr = str(row["expression"])
        fp = str(row["ast_fingerprint"])
        ds_id = str(row["dataset_id"])
        assert row["region"] == "USA"
        assert row["universe"] == "TOP3000"
        assert int(row["delay"]) == 1
        assert row["status"] == "PENDING"
        assert fp != ""
        ok, errs = full_validator.validate(expr)
        assert ok is True, f"Produced invalid expression {expr}: {errs}"

        fp_counts[fp] = fp_counts.get(fp, 0) + 1
        assert fp_counts[fp] <= 3
        if ds_id:
            ds_fp_counts[(fp, ds_id)] = ds_fp_counts.get((fp, ds_id), 0) + 1
            assert ds_fp_counts[(fp, ds_id)] <= 2


def test_candidate_producer_produce_once_ollama_unreachable_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = AlphaDB(tmp_path / "producer_fallback.db")
    db.upsert_operators(_sample_operators())
    fields = _producer_datafields()
    db.upsert_datafields(fields)

    post_attempts: list[str] = []

    def _unreachable_ollama(url: str, **kwargs: Any) -> Any:
        post_attempts.append(url)
        raise requests.exceptions.ConnectionError("Ollama service disabled")

    monkeypatch.setattr(requests, "post", _unreachable_ollama)

    settings = {
        "markets": {
            "USA": {
                "default_universe": "TOP3000",
                "universes": ["TOP3000"],
                "neutralizations": ["INDUSTRY"],
                "default_neutralization": "INDUSTRY",
                "delay": 1,
            }
        },
        "execution": {"default_settings": {"decay": 4}},
        "generation": {
            "queue_low_watermark": 20,
            "queue_high_watermark": 50,
            "mix": {"template": 0.0, "algorithmic_llm": 1.0, "evolution": 0.0},
            "categories": ["fundamental"],
            "neutralizations": ["INDUSTRY"],
            "decays": [4],
            "max_per_structure": 3,
            "max_per_dataset_structure": 2,
            "lookback_days": [5, 20, 60],
            "llm": {
                "backend": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "models": ["qwen2.5-coder:7b"],
                "timeout_s": 2,
            },
        },
    }

    algo_llm = AlgorithmicLLMGenerator(
        llm_cfg=settings["generation"]["llm"],
        operators=db.get_operators(),
        datafields=fields,
        seed=42,
    )
    producer = CandidateProducer(db=db, settings=settings, algo_llm=algo_llm, templates=[])

    inserted = producer.produce_once(region="USA", batch_target=6)
    assert 1 <= inserted <= 6
    assert len(post_attempts) == 1
    assert post_attempts[0] == "http://127.0.0.1:11434/api/generate"
    assert db.count_queue(status="PENDING") == inserted

    full_validator = ExpressionValidator(operators=db.get_operators(), datafields=fields)
    with db._read_conn() as conn:
        rows = conn.execute("SELECT expression, config_hash FROM candidate_queue").fetchall()
    assert len({str(r["config_hash"]) for r in rows}) == inserted
    for row in rows:
        ok, errs = full_validator.validate(str(row["expression"]))
        assert ok is True, f"Invalid fallback expression {row['expression']}: {errs}"
