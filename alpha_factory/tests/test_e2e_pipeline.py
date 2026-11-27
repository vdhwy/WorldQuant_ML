"""Phase 5C end-to-end integration tests for the full alpha_factory pipeline and CLI."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest
import requests
import yaml

from alpha_factory.brain.auth import BrainAuthManager
from alpha_factory.brain.catalog import CatalogSync
from alpha_factory.brain.client import BrainClient
from alpha_factory.ops import cli
from alpha_factory.ops.daemon import AlphaMiningDaemon
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import DEFAULT_SETTINGS, CandidateItem

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_YAML = REPO_ROOT / "alpha_factory" / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Offline & Fast Execution Guards + Fake HTTP Session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_and_fast_guards(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Block all live network calls, make time.sleep instant, and clean logger handlers."""
    monkeypatch.setattr(time, "sleep", lambda _secs: None)

    def _forbid_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Real network call attempted in offline E2E test")

    monkeypatch.setattr(requests.Session, "request", _forbid_network)
    monkeypatch.setattr(requests, "request", _forbid_network)
    monkeypatch.setattr(requests, "get", _forbid_network)
    monkeypatch.setattr(requests, "post", _forbid_network)
    monkeypatch.setattr(requests, "patch", _forbid_network)
    monkeypatch.setattr(requests, "delete", _forbid_network)

    yield

    target_logger = logging.getLogger("alpha_factory")
    for handler in list(target_logger.handlers):
        target_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


class FakeResponse:
    """In-memory HTTP response compatible with BrainClient and BrainAuthManager."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        headers: Optional[dict[str, Any]] = None,
        text: str = "",
        url: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = dict(headers or {})
        self.text = text
        self.url = url

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        if self.text:
            return json.loads(self.text)
        raise ValueError("No JSON body available")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Thread-safe mocked requests.Session routing calls to an in-memory handler."""

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

    def patch(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("DELETE", url, **kwargs)


# ---------------------------------------------------------------------------
# Deterministic Payload & Test Data Helpers
# ---------------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    handler: Callable[[str, str, dict[str, Any]], FakeResponse],
    *,
    max_http_retries: int = 0,
) -> tuple[BrainClient, FakeSession]:
    auth = BrainAuthManager(
        api_url="https://api.worldquantbrain.com",
        credentials_path=tmp_path / "creds.json",
        session_cache_path=tmp_path / "session.json",
        email="e2e@example.com",
        password="secret",
    )
    session = FakeSession(handler)
    client = BrainClient(
        auth_manager=auth,
        session=session,  # type: ignore[arg-type]
        api_url="https://api.worldquantbrain.com",
        max_wait_s=60.0,
        recordset_max_wait_s=60.0,
        default_interval_s=0.01,
        max_http_retries=max_http_retries,
    )
    return client, session


def _make_isolated_settings(
    tmp_path: Path,
    *,
    max_slots: int = 2,
    multi_sim_limit: int = 5,
    queue_low_watermark: int = 5,
    queue_high_watermark: int = 5,
    qualification_workers: int = 1,
) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_SETTINGS)
    cfg["brain"]["credentials_path"] = str(tmp_path / "creds.json")
    cfg["brain"]["session_cache_path"] = str(tmp_path / "session.json")
    cfg["polling"]["default_interval_s"] = 0.01
    cfg["polling"]["max_http_retries"] = 0

    cfg["storage"] = {
        "db_path": str(tmp_path / "alpha_factory.db"),
        "pnl_dir": str(tmp_path / "pnl"),
        "backup_dir": str(tmp_path / "backups"),
    }
    cfg["markets"] = {
        "USA": {
            "default_universe": "TOP3000",
            "universes": ["TOP3000"],
            "neutralizations": ["INDUSTRY"],
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        }
    }
    cfg["execution"]["max_slots"] = max_slots
    cfg["execution"]["multi_sim_limit"] = multi_sim_limit
    cfg["execution"]["hourly_pacing"] = False
    cfg["generation"]["queue_low_watermark"] = queue_low_watermark
    cfg["generation"]["queue_high_watermark"] = queue_high_watermark
    cfg["generation"]["llm"]["backend"] = "none"
    cfg["ops"] = {
        "log_dir": str(tmp_path / "logs"),
        "workspace_root": str(REPO_ROOT),
        "shutdown_drain_timeout_s": 5.0,
        "watchdog_extra_s": 60,
        "telegram_enabled": False,
        "qualification_workers": qualification_workers,
    }
    return cfg


def _make_dates(n_days: int = 504) -> np.ndarray:
    start = dt.date(2023, 1, 2).toordinal()
    return np.arange(start, start + n_days, dtype=np.int64)


def _orthonormal_vectors(n_days: int = 504, count: int = 4, seed: int = 42) -> list[np.ndarray]:
    """Generate zero-mean unit-norm mutually orthogonal daily vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(n_days, count))
    basis: list[np.ndarray] = []
    for col in range(count):
        vec = raw[:, col].astype(np.float64)
        vec -= vec.mean()
        for prev in basis:
            vec -= np.dot(vec, prev) * prev
        vec -= vec.mean()
        vec /= np.linalg.norm(vec)
        basis.append(vec)
    return basis


def _correlated_vector(
    base_unit: np.ndarray,
    ortho_unit: np.ndarray,
    target_corr: float,
    scale: float = 0.01,
    drift: float = 0.0002,
) -> np.ndarray:
    """Construct a daily return vector with exact Pearson correlation ``target_corr`` to ``base_unit``."""
    r = float(target_corr)
    rem = float(np.sqrt(max(0.0, 1.0 - r * r)))
    return scale * (r * base_unit + rem * ortho_unit) + drift


def _make_cumulative_pnl_payload(dates: np.ndarray, daily_returns: np.ndarray) -> dict[str, Any]:
    cum = np.cumsum(daily_returns)
    records = [[dt.date.fromordinal(int(d)).isoformat(), float(v)] for d, v in zip(dates, cum)]
    return {
        "schema": {
            "name": "pnl",
            "properties": [
                {"name": "date", "type": "date"},
                {"name": "pnl", "type": "decimal"},
            ],
        },
        "records": records,
    }


def _healthy_checks() -> list[dict[str, Any]]:
    return [
        {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.4},
        {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 1.6},
        {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.18},
        {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.70, "value": 0.18},
        {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "limit": 0.75, "value": 1.20},
    ]


def _failing_checks(fail_name: str = "LOW_SUB_UNIVERSE_SHARPE") -> list[dict[str, Any]]:
    return [
        {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.1},
        {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 1.3},
        {"name": fail_name, "result": "FAIL", "limit": 0.75, "value": 0.42},
    ]


def _make_alpha_payload(
    alpha_id: str,
    *,
    expression: str = "rank(close)",
    sharpe: float = 2.4,
    fitness: float = 1.6,
    turnover: float = 0.18,
    margin: float = 0.0018,
    returns: float = 0.15,
    checks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "id": alpha_id,
        "regular": {"code": expression},
        "is": {
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "margin": margin,
            "returns": returns,
            "drawdown": 0.05,
            "longCount": 250,
            "shortCount": 240,
            "checks": checks if checks is not None else _healthy_checks(),
        },
    }


def _make_self_corr_payload(max_corr: float) -> dict[str, Any]:
    return {
        "schema": {
            "name": "selfCorrelation",
            "properties": [
                {"name": "id", "type": "text"},
                {"name": "correlation", "type": "decimal"},
            ],
        },
        "records": [["old_alpha_ref", float(max_corr)]],
        "max": float(max_corr),
    }


def _make_prod_corr_payload(max_corr: float) -> dict[str, Any]:
    return {
        "schema": {
            "name": "prodCorrelation",
            "properties": [
                {"name": "min", "type": "decimal"},
                {"name": "max", "type": "decimal"},
                {"name": "alphas", "type": "integer"},
            ],
        },
        "records": [
            [-1.0, 0.5, 40],
            [0.5, 0.7, 2 if max_corr >= 0.5 else 0],
        ],
        "max": float(max_corr),
    }


# ---------------------------------------------------------------------------
# Scenario 1: Full AlphaMiningDaemon E2E Pipeline + CLI CSV Export
# ---------------------------------------------------------------------------


def test_e2e_full_daemon_pipeline_and_csv_export(tmp_path: Path) -> None:
    """Exercise offline cache bootstrap -> generation/pre-dedup -> multi-sim + 400/child fail -> Gate1/2 + PnL dedup -> CSV export."""
    assert (REPO_ROOT / "worldquant-miner" / "generation_two" / "constants" / "operatorRAW.json").is_file()
    assert (REPO_ROOT / "constants" / "data_fields_cache_USA_1_TOP3000.json").is_file()

    settings = _make_isolated_settings(
        tmp_path,
        max_slots=2,
        multi_sim_limit=5,
        queue_low_watermark=5,
        queue_high_watermark=5,
        qualification_workers=1,
    )
    cfg_path = tmp_path / "e2e_settings.yaml"
    cfg_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")

    dates = _make_dates(504)
    basis = _orthonormal_vectors(n_days=504, count=3, seed=77)
    pnl_pass_1 = _correlated_vector(basis[0], basis[1], target_corr=1.0)
    pnl_pass_2 = _correlated_vector(basis[0], basis[1], target_corr=0.88)
    pnl_pass_3 = _correlated_vector(basis[0], basis[2], target_corr=0.15)

    alpha_specs: dict[str, dict[str, Any]] = {
        "E2E_FAIL_G1": {
            "sharpe": 2.1,
            "fitness": 1.3,
            "turnover": 0.20,
            "checks": _failing_checks("LOW_SUB_UNIVERSE_SHARPE"),
        },
        "E2E_PASS_1": {
            "sharpe": 2.6,
            "fitness": 1.8,
            "turnover": 0.15,
            "self_corr": 0.10,
            "prod_corr": 0.20,
            "pnl": pnl_pass_1,
        },
        "E2E_PASS_2": {
            "sharpe": 2.1,
            "fitness": 1.3,
            "turnover": 0.20,
            "self_corr": 0.12,
            "prod_corr": 0.25,
            "pnl": pnl_pass_2,
        },
        "E2E_PASS_3": {
            "sharpe": 2.3,
            "fitness": 1.5,
            "turnover": 0.18,
            "self_corr": 0.08,
            "prod_corr": 0.18,
            "pnl": pnl_pass_3,
        },
        "E2E_FAIL_G1_B2": {
            "sharpe": 2.0,
            "fitness": 1.2,
            "turnover": 0.22,
            "checks": _failing_checks("LOW_SHARPE"),
        },
        "E2E_FAIL_G2_SELF": {
            "sharpe": 2.2,
            "fitness": 1.4,
            "turnover": 0.19,
            "self_corr": 0.85,
            "prod_corr": 0.20,
            "pnl": pnl_pass_3,
        },
        "E2E_FAIL_G1_B3": {
            "sharpe": 1.9,
            "fitness": 1.1,
            "turnover": 0.21,
            "checks": _failing_checks("LOW_FITNESS"),
        },
        "E2E_FAIL_G1_B4": {
            "sharpe": 1.8,
            "fitness": 1.1,
            "turnover": 0.24,
            "checks": _failing_checks("CONCENTRATED_WEIGHT"),
        },
    }

    state_lock = threading.Lock()
    authenticated = False
    sim_post_count = 0
    sim_poll_counts: dict[str, int] = {}
    posted_batches: list[list[str]] = []
    alpha_expressions: dict[str, str] = {}
    patch_calls: list[tuple[str, dict[str, Any]]] = []

    child_to_outcome: dict[str, dict[str, Any]] = {
        "b1-c0": {"status": "ERROR", "message": "Attempted to use unknown variable 'bad_runtime_col'"},
        "b1-c1": {"status": "COMPLETE", "alpha": "E2E_FAIL_G1"},
        "b1-c2": {"status": "COMPLETE", "alpha": "E2E_PASS_1"},
        "b1-c3": {"status": "COMPLETE", "alpha": "E2E_PASS_2"},
        "b2-c0": {"status": "COMPLETE", "alpha": "E2E_PASS_3"},
        "b2-c1": {"status": "COMPLETE", "alpha": "E2E_FAIL_G1_B2"},
        "b2-c2": {"status": "COMPLETE", "alpha": "E2E_FAIL_G2_SELF"},
        "b2-c3": {"status": "COMPLETE", "alpha": "E2E_FAIL_G1_B3"},
        "b2-c4": {"status": "COMPLETE", "alpha": "E2E_FAIL_G1_B4"},
    }

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        nonlocal authenticated, sim_post_count

        if method == "GET" and url.endswith("/authentication"):
            if not authenticated:
                return FakeResponse(204)
            return FakeResponse(200, json_data={"user": {"id": "e2e"}, "token": {"expiry": 14400}})

        if method == "POST" and url.endswith("/authentication"):
            authenticated = True
            return FakeResponse(
                201,
                headers={"X-WQB-Session-Token": "e2e-session-token"},
                json_data={"user": {"id": "e2e"}, "token": {"expiry": 14400}},
            )

        if method == "GET" and url.endswith("/operators"):
            # Force _startup_sync offline fallback to real local workspace caches.
            return FakeResponse(500, text="Simulated API downtime for catalog")

        if method == "GET" and "/users/self/alphas" in url:
            return FakeResponse(200, json_data={"count": 0, "results": []})

        if method == "POST" and url.endswith("/simulations"):
            payload = kwargs["json"]
            assert isinstance(payload, list)
            exprs = [str(item["regular"]) for item in payload]
            with state_lock:
                sim_post_count += 1
                call_num = sim_post_count
                posted_batches.append(exprs)

            if call_num == 1:
                assert len(exprs) == 5
                return FakeResponse(
                    400,
                    json_data={"2": {"regular": ["Simulated syntax error in poison item"]}},
                )
            if call_num == 2:
                assert len(exprs) == 4
                for cid, expr in zip(["b1-c0", "b1-c1", "b1-c2", "b1-c3"], exprs):
                    alpha_id = child_to_outcome[cid].get("alpha")
                    if alpha_id:
                        alpha_expressions[alpha_id] = expr
                return FakeResponse(
                    201,
                    headers={"Location": "https://api.worldquantbrain.com/simulations/sim-batch-1"},
                )
            if call_num == 3:
                assert len(exprs) == 5
                for cid, expr in zip(["b2-c0", "b2-c1", "b2-c2", "b2-c3", "b2-c4"], exprs):
                    alpha_id = child_to_outcome[cid].get("alpha")
                    if alpha_id:
                        alpha_expressions[alpha_id] = expr
                return FakeResponse(
                    201,
                    headers={"Location": "https://api.worldquantbrain.com/simulations/sim-batch-2"},
                )
            raise AssertionError(f"Unexpected POST /simulations call #{call_num}")

        if method == "GET" and url.endswith("/simulations/sim-batch-1"):
            with state_lock:
                polls = sim_poll_counts.get("sim-batch-1", 0) + 1
                sim_poll_counts["sim-batch-1"] = polls
            if polls == 1:
                return FakeResponse(
                    200,
                    headers={"Retry-After": "0.01"},
                    json_data={"id": "sim-batch-1", "status": "SIMULATING"},
                )
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-batch-1",
                    "status": "COMPLETE",
                    "children": ["b1-c0", "b1-c1", "b1-c2", "b1-c3"],
                },
            )

        if method == "GET" and url.endswith("/simulations/sim-batch-2"):
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-batch-2",
                    "status": "COMPLETE",
                    "children": ["b2-c0", "b2-c1", "b2-c2", "b2-c3", "b2-c4"],
                },
            )

        if method == "GET" and "/simulations/" in url:
            child_id = url.rsplit("/", 1)[-1]
            if child_id in child_to_outcome:
                return FakeResponse(200, json_data={"id": child_id, **child_to_outcome[child_id]})

        if method == "GET" and "/alphas/" in url:
            tail = url.split("/alphas/", 1)[1]
            parts = tail.split("/")
            alpha_id = parts[0]
            spec = alpha_specs[alpha_id]

            if len(parts) == 1:
                return FakeResponse(
                    200,
                    json_data=_make_alpha_payload(
                        alpha_id,
                        expression=alpha_expressions.get(alpha_id, f"rank(close) + {alpha_id}"),
                        sharpe=spec["sharpe"],
                        fitness=spec["fitness"],
                        turnover=spec["turnover"],
                        checks=spec.get("checks"),
                    ),
                )
            if parts[1] == "check":
                return FakeResponse(200, json_data={"is": {"checks": spec.get("checks") or _healthy_checks()}})
            if parts[1] == "correlations" and parts[2] == "self":
                return FakeResponse(200, json_data=_make_self_corr_payload(spec["self_corr"]))
            if parts[1] == "correlations" and parts[2] == "prod":
                return FakeResponse(200, json_data=_make_prod_corr_payload(spec["prod_corr"]))
            if parts[1] == "recordsets" and parts[2] == "pnl":
                return FakeResponse(200, json_data=_make_cumulative_pnl_payload(dates, spec["pnl"]))

        if method == "PATCH" and "/alphas/" in url:
            alpha_id = url.split("/alphas/", 1)[1]
            with state_lock:
                patch_calls.append((alpha_id, dict(kwargs.get("json") or {})))
            return FakeResponse(200, json_data={})

        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    client, _session = _make_client(tmp_path, _handler, max_http_retries=0)
    daemon = AlphaMiningDaemon(settings=settings, client=client)

    # Confirm WAL mode before running.
    with daemon.db._read_conn() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(journal_mode).lower() == "wal"

    summary = daemon.run(dry_run=False, max_cycles=2, install_signals=False)

    # 1. Offline catalog bootstrap from real workspace caches.
    assert len(daemon.db.get_operators()) >= 100
    assert len(daemon.db.get_datafields("USA", "TOP3000", 1)) > 500
    assert len(daemon.db.get_datasets("USA", "TOP3000", 1)) > 0

    # 2. Producer generated & pre-deduped 10 unique candidates across 2 cycles.
    assert summary["dry_run"] is False
    assert summary["cycles_completed"] == 2
    assert summary["candidates_queued"] == 10
    assert daemon.db.count_queue(status=None) == 10
    assert daemon.db.count_queue(status="DONE") == 8
    assert daemon.db.count_queue(status="FAILED") == 2
    assert daemon.db.count_queue(status="PENDING") == 0

    with daemon.db._read_conn() as conn:
        queued_rows = conn.execute(
            "SELECT expression, config_hash, ast_fingerprint, dataset_id FROM candidate_queue"
        ).fetchall()
    assert len({row["config_hash"] for row in queued_rows}) == 10
    assert all(str(row["ast_fingerprint"]).strip() for row in queued_rows)

    # Pre-dedup rejects exact duplicate of an already-queued candidate.
    sample_row = queued_rows[0]
    with daemon.db._read_conn() as conn:
        full_row = conn.execute("SELECT settings_json FROM candidate_queue LIMIT 1").fetchone()
    assert full_row is not None
    assert (
        daemon.pre_dedup.filter_candidates(
            [(str(sample_row["expression"]), str(sample_row["dataset_id"]))],
            json.loads(str(full_row["settings_json"])),
            "USA",
            "TOP3000",
            1,
            "template",
        )
        == []
    )

    # 3. Multi-sim batches, 400 poison isolation, failing child, and NY quota accuracy.
    assert len(posted_batches) == 3
    assert len(posted_batches[0]) == 5  # rejected with 400
    assert len(posted_batches[1]) == 4  # isolated clean batch 1
    assert len(posted_batches[2]) == 5  # clean batch 2
    assert "bad_runtime_col" in daemon.validator.blocked_fields
    assert daemon.slot_pool.active_count() == 0

    quota_status = daemon.quota.get_status()
    assert quota_status.submitted_count == 9
    assert quota_status.completed_count == 8
    assert quota_status.failed_count == 1

    # 4. Qualification filtering and correlation dedup vault (2 KEPT + 1 CORR_DUP).
    all_sims = daemon.db.get_simulations()
    assert len(all_sims) == 8
    assert sum(1 for s in all_sims if s.gate1_pass is True) == 4
    assert sum(1 for s in all_sims if s.gate1_pass is False) == 4
    assert sum(1 for s in all_sims if s.gate2_pass is True) == 3

    g2_fail = daemon.db.get_simulation("E2E_FAIL_G2_SELF")
    assert g2_fail is not None and g2_fail.gate1_pass is True and g2_fail.gate2_pass is False

    vault_entries = daemon.db.get_vault_entries(region="USA", delay=1)
    by_id = {entry.alpha_id: entry for entry in vault_entries}
    assert set(by_id.keys()) == {"E2E_PASS_1", "E2E_PASS_2", "E2E_PASS_3"}

    assert by_id["E2E_PASS_1"].vault_status == "KEPT"
    assert by_id["E2E_PASS_3"].vault_status == "KEPT"
    assert by_id["E2E_PASS_2"].vault_status == "CORR_DUP"
    assert by_id["E2E_PASS_2"].duplicate_of == "E2E_PASS_1"
    assert by_id["E2E_PASS_2"].max_local_corr == pytest.approx(0.88, abs=1e-3)
    assert by_id["E2E_PASS_3"].max_local_corr == pytest.approx(0.15, abs=1e-3)

    patch_map = {aid: body.get("color") for aid, body in patch_calls}
    assert patch_map["E2E_FAIL_G1"] == "RED"
    assert patch_map["E2E_FAIL_G2_SELF"] == "RED"
    assert patch_map["E2E_PASS_1"] == "GREEN"
    assert patch_map["E2E_PASS_2"] == "YELLOW"
    assert patch_map["E2E_PASS_3"] == "GREEN"

    assert (tmp_path / "pnl" / "USA" / "E2E_PASS_1.npz").is_file()
    assert (tmp_path / "pnl" / "USA" / "E2E_PASS_2.npz").is_file()
    assert (tmp_path / "pnl" / "USA" / "E2E_PASS_3.npz").is_file()
    assert (tmp_path / "backups" / f"alpha_factory_{quota_status.date_ny}.db").is_file()

    # 5. CLI export-submittable writes CSV containing only the 2 uncorrelated KEPT alphas.
    csv_path = tmp_path / "exports" / "submittable.csv"
    rc = cli.main(["export-submittable", "--config", str(cfg_path), "--out", str(csv_path)])
    assert rc == 0
    assert csv_path.is_file()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        exported_rows = list(csv.DictReader(fh))
    assert [row["alpha_id"] for row in exported_rows] == ["E2E_PASS_1", "E2E_PASS_3"]
    assert all(row["vault_status"] == "KEPT" for row in exported_rows)
    assert not (REPO_ROOT / "alpha_factory" / "data").exists()


# ---------------------------------------------------------------------------
# Scenario 2: Graceful Shutdown & Restart Crash Recovery
# ---------------------------------------------------------------------------


def test_e2e_graceful_shutdown_and_restart_crash_recovery(tmp_path: Path) -> None:
    """Verify clean thread drain on request_stop() and restart recovery of IN_FLIGHT multi-sim jobs."""
    dates = _make_dates(300)
    basis = _orthonormal_vectors(n_days=300, count=2, seed=99)
    pnl_a = _correlated_vector(basis[0], basis[1], target_corr=1.0)
    pnl_b = _correlated_vector(basis[0], basis[1], target_corr=0.10)

    # --- Part A: Graceful shutdown while daemon loop is active ---
    shutdown_dir = tmp_path / "graceful"
    shutdown_settings = _make_isolated_settings(
        shutdown_dir,
        max_slots=2,
        multi_sim_limit=5,
        queue_low_watermark=2,
        queue_high_watermark=2,
        qualification_workers=1,
    )
    shutdown_db = AlphaDB(shutdown_dir / "alpha_factory.db")
    CatalogSync(shutdown_db).bootstrap_from_local_cache(workspace_root=str(REPO_ROOT))
    shutdown_db.enqueue_candidates(
        [
            CandidateItem(
                config_hash="grace-cfg-0",
                ast_fingerprint="grace-fp-0",
                expression="rank(close)",
                region="USA",
                universe="TOP3000",
                delay=1,
                settings_json="{}",
                source="template",
                dataset_id="pv1",
                status="PENDING",
            ),
            CandidateItem(
                config_hash="grace-cfg-1",
                ast_fingerprint="grace-fp-1",
                expression="rank(open)",
                region="USA",
                universe="TOP3000",
                delay=1,
                settings_json="{}",
                source="template",
                dataset_id="pv1",
                status="PENDING",
            ),
        ]
    )

    live_daemon_ref: list[AlphaMiningDaemon] = []

    def _shutdown_handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "GET" and url.endswith("/authentication"):
            return FakeResponse(200, json_data={"user": {"id": "u1"}, "token": {"expiry": 14400}})
        if method == "POST" and url.endswith("/authentication"):
            return FakeResponse(
                201,
                headers={"X-WQB-Session-Token": "tok"},
                json_data={"user": {"id": "u1"}, "token": {"expiry": 14400}},
            )
        if method == "GET" and url.endswith("/operators"):
            return FakeResponse(500, text="offline")
        if method == "GET" and "/users/self/alphas" in url:
            return FakeResponse(200, json_data={"count": 0, "results": []})
        if method == "POST" and url.endswith("/simulations"):
            return FakeResponse(
                201,
                headers={"Location": "https://api.worldquantbrain.com/simulations/sim-graceful"},
            )
        if method == "GET" and url.endswith("/simulations/sim-graceful"):
            if live_daemon_ref:
                live_daemon_ref[0].request_stop()
            return FakeResponse(
                200,
                json_data={"id": "sim-graceful", "status": "COMPLETE", "children": ["g-c0", "g-c1"]},
            )
        if method == "GET" and url.endswith("/simulations/g-c0"):
            return FakeResponse(200, json_data={"id": "g-c0", "status": "COMPLETE", "alpha": "GRACE_0"})
        if method == "GET" and url.endswith("/simulations/g-c1"):
            return FakeResponse(200, json_data={"id": "g-c1", "status": "COMPLETE", "alpha": "GRACE_1"})
        if method == "GET" and "/alphas/" in url:
            tail = url.split("/alphas/", 1)[1].split("/")
            aid = tail[0]
            if len(tail) == 1:
                return FakeResponse(200, json_data=_make_alpha_payload(aid))
            if tail[1] == "check":
                return FakeResponse(200, json_data={"is": {"checks": _healthy_checks()}})
            if tail[1] == "correlations" and tail[2] == "self":
                return FakeResponse(200, json_data=_make_self_corr_payload(0.10))
            if tail[1] == "correlations" and tail[2] == "prod":
                return FakeResponse(200, json_data=_make_prod_corr_payload(0.15))
            if tail[1] == "recordsets" and tail[2] == "pnl":
                vec = pnl_a if aid == "GRACE_0" else pnl_b
                return FakeResponse(200, json_data=_make_cumulative_pnl_payload(dates, vec))
        if method == "PATCH" and "/alphas/" in url:
            return FakeResponse(200, json_data={})
        raise AssertionError(f"Unexpected HTTP call in graceful shutdown: {method} {url}")

    shutdown_client, _ = _make_client(shutdown_dir, _shutdown_handler, max_http_retries=0)
    daemon_live = AlphaMiningDaemon(settings=shutdown_settings, client=shutdown_client, db=shutdown_db)
    daemon_live.dispatcher._poll_interval_s = 0.02
    daemon_live.producer._poll_interval_s = 0.02
    live_daemon_ref.append(daemon_live)

    summary_live = daemon_live.run(dry_run=False, max_cycles=None, install_signals=False)
    assert summary_live["stopped"] is True
    assert not daemon_live.producer.is_running
    assert not daemon_live.dispatcher.is_running
    assert daemon_live._qual_threads == []
    assert daemon_live._housekeeping_thread is None
    assert daemon_live.slot_pool.active_count() == 0
    assert shutdown_db.count_queue(status="DONE") == 2
    assert shutdown_db.get_simulation("GRACE_0") is not None
    assert shutdown_db.get_simulation("GRACE_1") is not None
    assert {e.alpha_id for e in shutdown_db.get_vault_entries(statuses="KEPT")} == {
        "GRACE_0",
        "GRACE_1",
    }

    # --- Part B: Restart crash recovery on pre-populated IN_FLIGHT multi-sim job ---
    rec_dir = tmp_path / "recovery"
    rec_settings = _make_isolated_settings(
        rec_dir,
        max_slots=2,
        multi_sim_limit=5,
        queue_low_watermark=0,
        queue_high_watermark=0,
        qualification_workers=1,
    )
    rec_db = AlphaDB(rec_dir / "alpha_factory.db")
    rec_db.enqueue_candidates(
        [
            CandidateItem(
                config_hash="rec-cfg-0",
                ast_fingerprint="rec-fp-0",
                expression="rank(ts_mean(close, 5))",
                region="USA",
                universe="TOP3000",
                delay=1,
                settings_json="{}",
                source="template",
                dataset_id="pv1",
                status="IN_FLIGHT",
                progress_url="https://api.worldquantbrain.com/simulations/sim-crashed-multi",
                batch_id="USA-crashed-batch",
            ),
            CandidateItem(
                config_hash="rec-cfg-1",
                ast_fingerprint="rec-fp-1",
                expression="rank(ts_mean(open, 10))",
                region="USA",
                universe="TOP3000",
                delay=1,
                settings_json="{}",
                source="template",
                dataset_id="pv1",
                status="IN_FLIGHT",
                progress_url="https://api.worldquantbrain.com/simulations/sim-crashed-multi",
                batch_id="USA-crashed-batch",
            ),
        ]
    )

    def _recovery_handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            raise AssertionError("POST /simulations must NOT be called during crash recovery")
        if method == "GET" and url.endswith("/authentication"):
            return FakeResponse(200, json_data={"user": {"id": "u1"}, "token": {"expiry": 14400}})
        if method == "POST" and url.endswith("/authentication"):
            return FakeResponse(
                201,
                headers={"X-WQB-Session-Token": "tok-rec"},
                json_data={"user": {"id": "u1"}, "token": {"expiry": 14400}},
            )
        if method == "GET" and url.endswith("/operators"):
            return FakeResponse(500, text="offline")
        if method == "GET" and "/users/self/alphas" in url:
            return FakeResponse(200, json_data={"count": 0, "results": []})
        if method == "GET" and url.endswith("/simulations/sim-crashed-multi"):
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-crashed-multi",
                    "status": "COMPLETE",
                    "children": ["rec-c0", "rec-c1"],
                },
            )
        if method == "GET" and url.endswith("/simulations/rec-c0"):
            return FakeResponse(200, json_data={"id": "rec-c0", "status": "COMPLETE", "alpha": "REC_ALPHA_0"})
        if method == "GET" and url.endswith("/simulations/rec-c1"):
            return FakeResponse(200, json_data={"id": "rec-c1", "status": "COMPLETE", "alpha": "REC_ALPHA_1"})
        if method == "GET" and "/alphas/" in url:
            tail = url.split("/alphas/", 1)[1].split("/")
            aid = tail[0]
            if len(tail) == 1:
                checks = _healthy_checks() if aid == "REC_ALPHA_0" else _failing_checks("LOW_SHARPE")
                expr = "rank(ts_mean(close, 5))" if aid == "REC_ALPHA_0" else "rank(ts_mean(open, 10))"
                return FakeResponse(200, json_data=_make_alpha_payload(aid, expression=expr, checks=checks))
            if tail[1] == "check":
                return FakeResponse(200, json_data={"is": {"checks": _healthy_checks()}})
            if tail[1] == "correlations" and tail[2] == "self":
                return FakeResponse(200, json_data=_make_self_corr_payload(0.11))
            if tail[1] == "correlations" and tail[2] == "prod":
                return FakeResponse(200, json_data=_make_prod_corr_payload(0.16))
            if tail[1] == "recordsets" and tail[2] == "pnl":
                return FakeResponse(200, json_data=_make_cumulative_pnl_payload(dates, pnl_a))
        if method == "PATCH" and "/alphas/" in url:
            return FakeResponse(200, json_data={})
        raise AssertionError(f"Unexpected HTTP call in recovery: {method} {url}")

    rec_client, rec_session = _make_client(rec_dir, _recovery_handler, max_http_retries=0)
    daemon_rec = AlphaMiningDaemon(settings=rec_settings, client=rec_client, db=rec_db)

    recovered_counts: list[int] = []
    orig_recover = daemon_rec.dispatcher.recover_in_flight

    def _spy_recover() -> int:
        res = orig_recover()
        recovered_counts.append(res)
        return res

    daemon_rec.dispatcher.recover_in_flight = _spy_recover  # type: ignore[method-assign]

    summary_rec = daemon_rec.run(dry_run=False, max_cycles=1, install_signals=False)
    assert summary_rec["cycles_completed"] == 1
    assert recovered_counts == [1]
    assert not [c for c in rec_session.calls if c[0] == "POST" and c[1].endswith("/simulations")]
    assert rec_db.count_queue(status="IN_FLIGHT") == 0
    assert rec_db.count_queue(status="DONE") == 2

    sim0 = rec_db.get_simulation("REC_ALPHA_0")
    sim1 = rec_db.get_simulation("REC_ALPHA_1")
    assert sim0 is not None and sim0.gate1_pass is True and sim0.gate2_pass is True
    assert sim1 is not None and sim1.gate1_pass is False

    vault_rec = rec_db.get_vault_entries(statuses="KEPT")
    assert [e.alpha_id for e in vault_rec] == ["REC_ALPHA_0"]
    assert daemon_rec.quota.get_status().submitted_count == 0
    assert daemon_rec.quota.get_status().completed_count == 2
    assert daemon_rec.slot_pool.active_count() == 0


# ---------------------------------------------------------------------------
# Scenario 3: CLI sync-catalog --offline, run --dry-run, and status
# ---------------------------------------------------------------------------


def test_e2e_cli_sync_catalog_dry_run_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise CLI sync-catalog --offline, run --dry-run --max-cycles 1, and status on tmp_path config."""
    raw_cfg = yaml.safe_load(DEFAULT_SETTINGS_YAML.read_text(encoding="utf-8"))
    raw_cfg["brain"]["credentials_path"] = str(tmp_path / "creds.json")
    raw_cfg["brain"]["session_cache_path"] = str(tmp_path / "session.json")
    raw_cfg["storage"]["db_path"] = str(tmp_path / "cli_e2e.db")
    raw_cfg["storage"]["pnl_dir"] = str(tmp_path / "cli_pnl")
    raw_cfg["storage"]["backup_dir"] = str(tmp_path / "cli_backups")
    raw_cfg["ops"]["log_dir"] = str(tmp_path / "cli_logs")
    raw_cfg["ops"]["telegram_enabled"] = False
    raw_cfg["generation"]["llm"]["backend"] = "none"

    cfg_path = tmp_path / "settings.yaml"
    cfg_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.chdir(REPO_ROOT)

    # 1. sync-catalog --offline
    rc_sync = cli.main(["sync-catalog", "--config", str(cfg_path), "--offline"])
    out_sync = capsys.readouterr().out
    assert rc_sync == 0
    assert "Offline catalog bootstrap complete:" in out_sync

    db = AlphaDB(tmp_path / "cli_e2e.db")
    stats_after_sync = db.get_dashboard_stats()
    assert stats_after_sync["catalog"]["operators"] >= 100
    assert stats_after_sync["catalog"]["datasets"] >= 1
    assert stats_after_sync["catalog"]["datafields"] >= 1000

    # 2. run --dry-run --max-cycles 1
    rc_run = cli.main(
        ["run", "--config", str(cfg_path), "--dry-run", "--max-cycles", "1", "--log-level", "INFO"]
    )
    out_run = capsys.readouterr().out
    assert rc_run == 0
    summary = json.loads(out_run)
    assert summary["dry_run"] is True
    assert summary["cycles_completed"] == 1
    assert summary["candidates_queued"] > 0
    assert summary["stats"]["queue"]["PENDING"] == summary["candidates_queued"]
    assert summary["stats"]["quota"]["submitted_count"] == 0
    assert (tmp_path / "cli_logs" / "alpha_factory.log").is_file()

    # 3. status
    rc_status = cli.main(["status", "--config", str(cfg_path)])
    out_status = capsys.readouterr().out
    assert rc_status == 0
    assert "Alpha Factory Status (NY Date:" in out_status
    assert "NY Quota:     0/5000" in out_status
    assert f"PENDING={summary['candidates_queued']}" in out_status
    assert (
        f"Catalog:      datasets={stats_after_sync['catalog']['datasets']} "
        f"datafields={stats_after_sync['catalog']['datafields']} "
        f"operators={stats_after_sync['catalog']['operators']}"
    ) in out_status
    assert not (REPO_ROOT / "alpha_factory" / "data").exists()
