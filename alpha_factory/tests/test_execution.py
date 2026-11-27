"""Phase 3 tests for SlotPool, DailyQuotaManager integration, and SimulationDispatcher."""

from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import requests

from alpha_factory.brain.auth import BrainAuthManager
from alpha_factory.brain.client import BrainClient
from alpha_factory.execution.dispatcher import SimulationDispatcher
from alpha_factory.execution.quota import DailyQuotaManager, NY_TZ
from alpha_factory.execution.slot_pool import SLOT_IDLE, SLOT_RUNNING, SlotPool
from alpha_factory.generation.validator import ExpressionValidator
from alpha_factory.storage.db import AlphaDB
from alpha_factory.storage.models import CandidateItem


# ---------------------------------------------------------------------------
# Offline & Fast Execution Guards + Fake HTTP Session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_and_fast_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all real network I/O and make time.sleep instantaneous."""
    monkeypatch.setattr(time, "sleep", lambda _secs: None)

    def _forbid_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Real network call attempted in offline execution test")

    monkeypatch.setattr(requests.Session, "request", _forbid_network)
    monkeypatch.setattr(requests, "request", _forbid_network)
    monkeypatch.setattr(requests, "get", _forbid_network)
    monkeypatch.setattr(requests, "post", _forbid_network)


class FakeResponse:
    """In-memory HTTP response compatible with BrainClient."""

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


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _fixed_now_ny() -> dt.datetime:
    return dt.datetime(2026, 6, 15, 11, 30, 0, tzinfo=NY_TZ)


def _make_settings(
    *,
    max_slots: int = 3,
    multi_sim_limit: int = 10,
    daily_sim_limit: int = 5000,
    circuit_breaker_threshold: int = 3,
    region_cooldown_s: float = 1800.0,
    markets: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    default_markets = {
        "USA": {
            "default_universe": "TOP3000",
            "universes": ["TOP3000"],
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "EUR": {
            "default_universe": "TOP2500",
            "universes": ["TOP2500"],
            "default_neutralization": "INDUSTRY",
            "slots_required": 1,
            "delay": 1,
        },
        "GLB": {
            "default_universe": "TOP3000",
            "universes": ["TOP3000"],
            "default_neutralization": "INDUSTRY",
            "slots_required": 2,
            "delay": 1,
        },
    }
    return {
        "markets": markets if markets is not None else default_markets,
        "execution": {
            "max_slots": max_slots,
            "multi_sim_limit": multi_sim_limit,
            "daily_sim_limit": daily_sim_limit,
            "warning_threshold": 4000,
            "hourly_pacing": True,
            "circuit_breaker_threshold": circuit_breaker_threshold,
            "region_cooldown_s": region_cooldown_s,
            "default_settings": {
                "alpha_type": "REGULAR",
                "delay": 1,
                "decay": 4,
                "neutralization": "INDUSTRY",
                "truncation": 0.08,
                "pasteurization": "ON",
                "unit_handling": "VERIFY",
                "nan_handling": "OFF",
                "max_trade": "OFF",
                "test_period": "P2Y0M0D",
                "visualization": False,
            },
        },
        "ops": {
            "shutdown_drain_timeout_s": 5.0,
        },
    }


def _make_client(tmp_path: Path, session: FakeSession) -> BrainClient:
    auth = BrainAuthManager(
        api_url="https://api.worldquantbrain.com",
        credentials_path=tmp_path / "creds.json",
        session_cache_path=tmp_path / "session.json",
        email="tester@example.com",
        password="secret",
    )
    return BrainClient(
        auth_manager=auth,
        session=session,  # type: ignore[arg-type]
        max_wait_s=60.0,
        default_interval_s=0.1,
        max_http_retries=1,
    )


def _make_candidate(
    idx: int,
    *,
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    expression: Optional[str] = None,
    status: str = "PENDING",
    progress_url: str = "",
    batch_id: str = "",
    priority: float = 0.0,
) -> CandidateItem:
    expr = expression or f"rank(close) + {idx}"
    return CandidateItem(
        config_hash=f"cfg-{region}-{idx}-{abs(hash(expr)) % 100000}",
        ast_fingerprint=f"fp-{idx}",
        expression=expr,
        region=region,
        universe=universe,
        delay=delay,
        settings_json="{}",
        source="template",
        dataset_id="pv1",
        priority=priority,
        status=status,
        progress_url=progress_url,
        batch_id=batch_id,
    )


def _drain_workers(dispatcher: SimulationDispatcher, timeout: float = 5.0) -> None:
    """Wait until all active worker threads spawned by dispatcher have exited."""
    deadline = time.monotonic() + timeout
    while True:
        with dispatcher._workers_lock:
            workers = [t for t in dispatcher._active_workers if t.is_alive()]
        if not workers:
            return
        for worker in workers:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("Timed out draining SimulationDispatcher workers")
            worker.join(timeout=remaining)


def _alpha_payload(
    alpha_id: str,
    expression: str = "rank(close)",
    checks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "id": alpha_id,
        "regular": {"code": expression},
        "is": {
            "sharpe": 2.15,
            "fitness": 1.42,
            "turnover": 0.18,
            "returns": 0.14,
            "drawdown": 0.06,
            "margin": 0.0015,
            "longCount": 850,
            "shortCount": 840,
            "checks": checks
            if checks is not None
            else [
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "LOW_FITNESS", "result": "PASS"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# 1. Concurrent SlotPool race test
# ---------------------------------------------------------------------------


def test_slot_pool_concurrent_race_20_threads_3_slots() -> None:
    pool = SlotPool(max_slots=3)
    num_threads = 20
    rounds_per_thread = 5

    entry_barrier = threading.Barrier(num_threads)
    wave_acquire_barrier = threading.Barrier(3)
    wave_release_barrier = threading.Barrier(3)
    state_lock = threading.Lock()
    wave_claimed = 0
    active_workers = 0
    max_observed_active = 0
    in_use_slots: set[int] = set()
    total_acquisitions = 0
    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        nonlocal wave_claimed, active_workers, max_observed_active, total_acquisitions
        try:
            entry_barrier.wait(timeout=5.0)
            for rnd in range(rounds_per_thread):
                slot_ids = pool.acquire("USA", f"job-{tid}-{rnd}", timeout=5.0)
                assert slot_ids is not None and len(slot_ids) == 1
                sid = slot_ids[0]
                assert 0 <= sid < 3

                join_wave = False
                with state_lock:
                    assert sid not in in_use_slots, f"Slot {sid} double-allocated!"
                    in_use_slots.add(sid)
                    active_workers += 1
                    total_acquisitions += 1
                    if active_workers > max_observed_active:
                        max_observed_active = active_workers
                    assert active_workers <= 3
                    assert pool.active_count() <= 3
                    if wave_claimed < 3:
                        wave_claimed += 1
                        join_wave = True

                if join_wave:
                    # Force all 3 slots to be held simultaneously while verifying active_count == 3.
                    wave_acquire_barrier.wait(timeout=5.0)
                    assert pool.active_count() == 3
                    wave_release_barrier.wait(timeout=5.0)

                with state_lock:
                    in_use_slots.remove(sid)
                    active_workers -= 1

                pool.release(slot_ids)
        except BaseException as exc:  # pragma: no cover - surfaced below
            with state_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Worker errors occurred: {errors}"
    assert total_acquisitions == num_threads * rounds_per_thread
    assert max_observed_active == 3
    assert pool.active_count() == 0
    assert pool.idle_count() == 3
    for slot in pool.get_snapshot():
        assert slot["status"] == SLOT_IDLE
        assert slot["job_id"] is None
        assert slot["region"] is None
        assert slot["acquired_at"] is None


# ---------------------------------------------------------------------------
# 2. Fragmented non-consecutive GLB 2-slot acquisition
# ---------------------------------------------------------------------------


def test_slot_pool_fragmented_non_consecutive_glb_acquisition() -> None:
    pool = SlotPool(max_slots=4)

    s0 = pool.acquire("USA", "usa-0", timeout=0.5)
    s1 = pool.acquire("USA", "usa-1", timeout=0.5)
    s2 = pool.acquire("USA", "usa-2", timeout=0.5)
    s3 = pool.acquire("USA", "usa-3", timeout=0.5)
    assert s0 == [0]
    assert s1 == [1]
    assert s2 == [2]
    assert s3 == [3]

    # Release non-consecutive slots 1 and 3 so [BUSY, IDLE, BUSY, IDLE] remains.
    pool.release(s1)
    pool.release(s3)

    statuses = [slot["status"] for slot in pool.get_snapshot()]
    assert statuses == [SLOT_RUNNING, SLOT_IDLE, SLOT_RUNNING, SLOT_IDLE]
    assert pool.active_count() == 2
    assert pool.idle_count() == 2
    assert pool.can_acquire("GLB") is True

    glb_slots = pool.acquire("GLB", "glb-0", timeout=0.5)
    assert glb_slots == [1, 3]
    assert pool.active_count() == 4
    assert pool.idle_count() == 0

    snapshot = pool.get_snapshot()
    for sid in (1, 3):
        assert snapshot[sid]["status"] == SLOT_RUNNING
        assert snapshot[sid]["job_id"] == "glb-0"
        assert snapshot[sid]["region"] == "GLB"

    pool.release(glb_slots)
    pool.release(s0)
    pool.release(s2)
    assert pool.active_count() == 0
    assert pool.idle_count() == 4


# ---------------------------------------------------------------------------
# 3. GLB double-submission regression test
# ---------------------------------------------------------------------------


def test_glb_dispatch_no_double_submission_and_releases_two_slots(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "glb.db")
    db.enqueue_candidates(
        [_make_candidate(i, region="GLB", universe="TOP3000", delay=1) for i in range(3)]
    )
    settings = _make_settings(max_slots=3)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)

    post_payloads: list[Any] = []
    observed_active_during_poll: list[int] = []
    observed_glb_slot_ids: list[list[int]] = []

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            post_payloads.append(kwargs["json"])
            return FakeResponse(
                201,
                headers={"Location": "https://api.worldquantbrain.com/simulations/sim-glb-parent"},
            )
        if method == "GET" and url.endswith("/simulations/sim-glb-parent"):
            observed_active_during_poll.append(slot_pool.active_count())
            observed_glb_slot_ids.append(
                [s["slot_id"] for s in slot_pool.get_snapshot() if s["region"] == "GLB"]
            )
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-glb-parent",
                    "status": "COMPLETE",
                    "children": ["glb-c0", "glb-c1", "glb-c2"],
                },
            )
        for idx in range(3):
            if method == "GET" and url.endswith(f"/simulations/glb-c{idx}"):
                return FakeResponse(
                    200,
                    json_data={"id": f"glb-c{idx}", "status": "COMPLETE", "alpha": f"GLB_A{idx}"},
                )
            if method == "GET" and url.endswith(f"/alphas/GLB_A{idx}"):
                return FakeResponse(200, json_data=_alpha_payload(f"GLB_A{idx}"))
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        dispatched = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    assert dispatched == 1
    assert len(post_payloads) == 1
    assert isinstance(post_payloads[0], list) and len(post_payloads[0]) == 3
    assert observed_active_during_poll == [2]
    assert len(observed_glb_slot_ids[0]) == 2
    assert quota.get_status().submitted_count == 3
    assert quota.get_status().completed_count == 3
    assert slot_pool.active_count() == 0
    assert slot_pool.idle_count() == 3
    assert db.count_queue(status="DONE", region="GLB") == 3
    assert len(db.get_simulations(region="GLB")) == 3


# ---------------------------------------------------------------------------
# 4. Multi-sim 10-alpha batch with mixed child outcomes
# ---------------------------------------------------------------------------


def test_multi_sim_10_alpha_mixed_child_outcomes(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "mixed_10.db")
    db.enqueue_candidates(
        [
            _make_candidate(
                i,
                region="USA",
                universe="TOP3000",
                delay=1,
                expression=f"rank(bad_field_{i})" if i == 4 else f"rank(close) + {i}",
            )
            for i in range(10)
        ]
    )
    settings = _make_settings(max_slots=3, multi_sim_limit=10)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)
    qual_queue: queue.Queue[str] = queue.Queue()
    validator = ExpressionValidator()

    checks_by_idx: dict[int, list[dict[str, Any]]] = {
        0: [
            {"name": "LOW_SHARPE", "result": "FAIL"},
            {"name": "LOW_FITNESS", "result": "FAIL"},
            {"name": "LOW_TURNOVER", "result": "WARNING"},
            {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
        ],
        1: [
            {"name": "LOW_SHARPE", "result": "FAIL"},
            {"name": "LOW_TURNOVER", "result": "PASS"},
            {"name": "LOW_FITNESS", "result": "PASS"},
        ],
        2: [
            {"name": "LOW_TURNOVER", "result": "WARNING"},
            {"name": "SUB_UNIVERSE_SHARPE", "result": "WARN"},
            {"name": "LOW_SHARPE", "result": "PASS"},
        ],
    }

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            return FakeResponse(
                201,
                headers={"Location": "https://api.worldquantbrain.com/simulations/parent-10"},
            )
        if method == "GET" and url.endswith("/simulations/parent-10"):
            return FakeResponse(
                200,
                json_data={
                    "id": "parent-10",
                    "status": "COMPLETE",
                    "children": [f"child-{i}" for i in range(10)],
                },
            )
        for idx in range(10):
            if method == "GET" and url.endswith(f"/simulations/child-{idx}"):
                if idx == 4:
                    return FakeResponse(
                        200,
                        json_data={
                            "id": "child-4",
                            "status": "ERROR",
                            "message": "Attempted to use unknown variable 'bad_field_4'",
                        },
                    )
                return FakeResponse(
                    200,
                    json_data={"id": f"child-{idx}", "status": "COMPLETE", "alpha": f"ALPHA_10_{idx}"},
                )
            if method == "GET" and url.endswith(f"/alphas/ALPHA_10_{idx}"):
                return FakeResponse(
                    200,
                    json_data=_alpha_payload(
                        f"ALPHA_10_{idx}",
                        expression=f"rank(close) + {idx}",
                        checks=checks_by_idx.get(idx),
                    ),
                )
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        qualification_queue=qual_queue,
        validator=validator,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        dispatched = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    assert dispatched == 1

    expected_alpha_ids = {f"ALPHA_10_{i}" for i in range(10) if i != 4}
    sims = db.get_simulations(region="USA")
    assert len(sims) == 9
    assert {sim.alpha_id for sim in sims} == expected_alpha_ids

    rec_0 = db.get_simulation("ALPHA_10_0")
    assert rec_0 is not None
    assert rec_0.fail_count == 2
    assert rec_0.warn_count == 1

    rec_1 = db.get_simulation("ALPHA_10_1")
    assert rec_1 is not None
    assert rec_1.fail_count == 1
    assert rec_1.warn_count == 0

    rec_2 = db.get_simulation("ALPHA_10_2")
    assert rec_2 is not None
    assert rec_2.fail_count == 0
    assert rec_2.warn_count == 2

    queued_ids: list[str] = []
    while not qual_queue.empty():
        queued_ids.append(qual_queue.get_nowait())
    assert len(queued_ids) == 9
    assert set(queued_ids) == expected_alpha_ids

    assert db.count_queue(status="DONE", region="USA") == 9
    assert db.count_queue(status="FAILED", region="USA") == 1
    status = quota.get_status()
    assert status.submitted_count == 10
    assert status.completed_count == 9
    assert status.failed_count == 1
    assert "bad_field_4" in validator.blocked_fields


# ---------------------------------------------------------------------------
# 5. Poison-batch 400 isolation (indexed and non-indexed)
# ---------------------------------------------------------------------------


def test_poison_batch_400_indexed_isolation(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "poison_indexed.db")
    expressions = ["rank(close)", "rank(open)", "bad_syntax((", "rank(volume)"]
    db.enqueue_candidates(
        [
            _make_candidate(i, region="USA", universe="TOP3000", delay=1, expression=expr, priority=10.0 - i)
            for i, expr in enumerate(expressions)
        ]
    )
    settings = _make_settings(max_slots=3)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)

    validator = ExpressionValidator()
    learned_calls: list[tuple[str, str]] = []
    orig_learn = validator.learn_from_error

    def _spy_learn(expression: str, error_msg: str, db: Optional[AlphaDB] = None) -> list[dict[str, Any]]:
        learned_calls.append((expression, error_msg))
        return orig_learn(expression, error_msg, db=db)

    validator.learn_from_error = _spy_learn  # type: ignore[method-assign]

    post_batches: list[list[str]] = []

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            payload = kwargs["json"]
            assert isinstance(payload, list)
            batch_exprs = [entry["regular"] for entry in payload]
            post_batches.append(batch_exprs)
            if "bad_syntax((" in batch_exprs:
                bad_idx = batch_exprs.index("bad_syntax((")
                return FakeResponse(400, json_data={str(bad_idx): {"regular": ["syntax error"]}})
            return FakeResponse(
                201,
                headers={"Location": "https://api.worldquantbrain.com/simulations/sim-clean-3"},
            )
        if method == "GET" and url.endswith("/simulations/sim-clean-3"):
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-clean-3",
                    "status": "COMPLETE",
                    "children": ["clean-0", "clean-1", "clean-2"],
                },
            )
        for idx in range(3):
            if method == "GET" and url.endswith(f"/simulations/clean-{idx}"):
                return FakeResponse(
                    200,
                    json_data={"id": f"clean-{idx}", "status": "COMPLETE", "alpha": f"CLEAN_{idx}"},
                )
            if method == "GET" and url.endswith(f"/alphas/CLEAN_{idx}"):
                return FakeResponse(200, json_data=_alpha_payload(f"CLEAN_{idx}"))
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        validator=validator,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        dispatched = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    assert dispatched == 1
    assert len(post_batches) == 2
    assert post_batches[0] == expressions
    assert post_batches[1] == ["rank(close)", "rank(open)", "rank(volume)"]
    assert learned_calls == [("bad_syntax((", "2.regular: syntax error")]
    assert db.count_queue(status="FAILED") == 1
    assert db.count_queue(status="DONE") == 3
    assert len(db.get_simulations()) == 3
    assert quota.get_status().submitted_count == 3


def test_poison_batch_400_non_indexed_isolation(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "poison_non_indexed.db")
    expressions = ["rank(close)", "rank(open)", "rank(bad_col_99)", "rank(volume)"]
    db.enqueue_candidates(
        [
            _make_candidate(i, region="USA", universe="TOP3000", delay=1, expression=expr, priority=10.0 - i)
            for i, expr in enumerate(expressions)
        ]
    )
    settings = _make_settings(max_slots=3)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)

    validator = ExpressionValidator()
    learned_calls: list[tuple[str, str]] = []
    orig_learn = validator.learn_from_error

    def _spy_learn(expression: str, error_msg: str, db: Optional[AlphaDB] = None) -> list[dict[str, Any]]:
        learned_calls.append((expression, error_msg))
        return orig_learn(expression, error_msg, db=db)

    validator.learn_from_error = _spy_learn  # type: ignore[method-assign]

    sim_counter = 0
    sim_lock = threading.Lock()
    sim_children: dict[str, list[str]] = {}
    child_to_alpha: dict[str, str] = {}

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        nonlocal sim_counter
        if method == "POST" and url.endswith("/simulations"):
            payload = kwargs["json"]
            items_in_req = payload if isinstance(payload, list) else [payload]
            exprs = [entry["regular"] for entry in items_in_req]
            if any("bad_col_99" in expr for expr in exprs):
                # Non-indexed 400 error forces binary-split isolation down to single item.
                return FakeResponse(
                    400,
                    json_data={"regular": ["Attempted to use unknown variable 'bad_col_99'"]},
                )
            with sim_lock:
                sim_counter += 1
                sim_id = f"sim-nonidx-{sim_counter}"
                if len(items_in_req) == 1:
                    child_to_alpha[sim_id] = f"ALPHA_SINGLE_{sim_counter}"
                else:
                    children = [f"{sim_id}-c{idx}" for idx in range(len(items_in_req))]
                    sim_children[sim_id] = children
                    for idx, child_id in enumerate(children):
                        child_to_alpha[child_id] = f"ALPHA_{sim_counter}_{idx}"
            return FakeResponse(
                201,
                headers={"Location": f"https://api.worldquantbrain.com/simulations/{sim_id}"},
            )

        if method == "GET" and "/simulations/" in url:
            sim_id = url.rsplit("/", 1)[-1]
            if sim_id in sim_children:
                return FakeResponse(
                    200,
                    json_data={"id": sim_id, "status": "COMPLETE", "children": sim_children[sim_id]},
                )
            if sim_id in child_to_alpha:
                return FakeResponse(
                    200,
                    json_data={"id": sim_id, "status": "COMPLETE", "alpha": child_to_alpha[sim_id]},
                )

        if method == "GET" and "/alphas/" in url:
            alpha_id = url.rsplit("/", 1)[-1]
            return FakeResponse(200, json_data=_alpha_payload(alpha_id))

        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        validator=validator,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        dispatched = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    assert dispatched == 1
    assert len(learned_calls) == 1
    assert learned_calls[0][0] == "rank(bad_col_99)"
    assert "bad_col_99" in validator.blocked_fields
    assert db.get_compiler_rules(rule_type="unknown_field", field_id="bad_col_99")
    assert db.count_queue(status="FAILED") == 1
    assert db.count_queue(status="DONE") == 3
    assert len(db.get_simulations()) == 3
    assert quota.get_status().submitted_count == 3


# ---------------------------------------------------------------------------
# 6. Crash recovery (recover_in_flight)
# ---------------------------------------------------------------------------


def test_crash_recovery_resumes_in_flight_without_resubmit_and_requeues_dispatching(
    tmp_path: Path,
) -> None:
    db = AlphaDB(tmp_path / "recovery.db")
    db.enqueue_candidates(
        [
            _make_candidate(
                0,
                status="IN_FLIGHT",
                progress_url="https://api.worldquantbrain.com/simulations/sim-rec-multi",
                batch_id="USA-batch-multi",
            ),
            _make_candidate(
                1,
                status="IN_FLIGHT",
                progress_url="https://api.worldquantbrain.com/simulations/sim-rec-multi",
                batch_id="USA-batch-multi",
            ),
            _make_candidate(
                2,
                status="IN_FLIGHT",
                progress_url="https://api.worldquantbrain.com/simulations/sim-rec-single",
                batch_id="USA-batch-single",
            ),
            _make_candidate(3, status="DISPATCHING", progress_url="", batch_id=""),
            _make_candidate(4, status="DISPATCHING", progress_url="", batch_id=""),
        ]
    )
    settings = _make_settings(max_slots=3)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)
    qual_queue: queue.Queue[str] = queue.Queue()

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            raise AssertionError("submit_simulation() must NEVER be called during recover_in_flight()")
        if method == "GET" and url.endswith("/simulations/sim-rec-multi"):
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-rec-multi",
                    "status": "COMPLETE",
                    "children": ["rec-c0", "rec-c1"],
                },
            )
        if method == "GET" and url.endswith("/simulations/rec-c0"):
            return FakeResponse(200, json_data={"id": "rec-c0", "status": "COMPLETE", "alpha": "REC_0"})
        if method == "GET" and url.endswith("/simulations/rec-c1"):
            return FakeResponse(200, json_data={"id": "rec-c1", "status": "COMPLETE", "alpha": "REC_1"})
        if method == "GET" and url.endswith("/simulations/sim-rec-single"):
            return FakeResponse(
                200,
                json_data={"id": "sim-rec-single", "status": "COMPLETE", "alpha": "REC_2"},
            )
        if method == "GET" and url.endswith("/alphas/REC_0"):
            return FakeResponse(
                200,
                json_data=_alpha_payload(
                    "REC_0",
                    checks=[
                        {"name": "LOW_SHARPE", "result": "FAIL"},
                        {"name": "LOW_FITNESS", "result": "PASS"},
                    ],
                ),
            )
        if method == "GET" and url.endswith("/alphas/REC_1"):
            return FakeResponse(200, json_data=_alpha_payload("REC_1"))
        if method == "GET" and url.endswith("/alphas/REC_2"):
            return FakeResponse(
                200,
                json_data=_alpha_payload(
                    "REC_2",
                    checks=[
                        {"name": "LOW_TURNOVER", "result": "WARNING"},
                        {"name": "LOW_SHARPE", "result": "PASS"},
                    ],
                ),
            )
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        qualification_queue=qual_queue,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    recovered = dispatcher.recover_in_flight()

    assert recovered == 2
    assert not [c for c in session.calls if c[0] == "POST" and c[1].endswith("/simulations")]
    assert db.count_queue(status="IN_FLIGHT") == 0
    assert db.count_queue(status="DISPATCHING") == 0
    assert db.count_queue(status="PENDING") == 2
    assert db.count_queue(status="DONE") == 3

    sims = db.get_simulations()
    assert {sim.alpha_id for sim in sims} == {"REC_0", "REC_1", "REC_2"}
    rec_0 = db.get_simulation("REC_0")
    rec_2 = db.get_simulation("REC_2")
    assert rec_0 is not None and rec_0.fail_count == 1 and rec_0.warn_count == 0
    assert rec_2 is not None and rec_2.fail_count == 0 and rec_2.warn_count == 1

    queued_ids = {qual_queue.get_nowait() for _ in range(3)}
    assert queued_ids == {"REC_0", "REC_1", "REC_2"}
    assert qual_queue.empty()
    assert quota.get_status().submitted_count == 0
    assert quota.get_status().completed_count == 3


# ---------------------------------------------------------------------------
# 7. Daily 5,000 quota hard-stop & 3-batch circuit breaker cooldown
# ---------------------------------------------------------------------------


def test_daily_5000_quota_hard_stop(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "quota_stop.db")
    settings = _make_settings(max_slots=3, multi_sim_limit=10, daily_sim_limit=5000)
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)

    # Seed quota right below the 5,000 daily ceiling (3 remaining).
    db.increment_daily_quota_submitted(
        quota.get_ny_date(),
        4997,
        warning_threshold=4000,
        daily_limit=5000,
    )
    db.enqueue_candidates(
        [_make_candidate(i, region="USA", universe="TOP3000", delay=1) for i in range(5)]
    )

    post_batches: list[list[dict[str, Any]]] = []

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST" and url.endswith("/simulations"):
            payload = kwargs["json"]
            assert isinstance(payload, list)
            post_batches.append(payload)
            return FakeResponse(
                201,
                headers={"Location": "https://api.worldquantbrain.com/simulations/sim-q-3"},
            )
        if method == "GET" and url.endswith("/simulations/sim-q-3"):
            return FakeResponse(
                200,
                json_data={
                    "id": "sim-q-3",
                    "status": "COMPLETE",
                    "children": ["q-c0", "q-c1", "q-c2"],
                },
            )
        for idx in range(3):
            if method == "GET" and url.endswith(f"/simulations/q-c{idx}"):
                return FakeResponse(
                    200,
                    json_data={"id": f"q-c{idx}", "status": "COMPLETE", "alpha": f"Q_A{idx}"},
                )
            if method == "GET" and url.endswith(f"/alphas/Q_A{idx}"):
                return FakeResponse(200, json_data=_alpha_payload(f"Q_A{idx}"))
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        dispatched_first = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
        dispatched_second = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    assert dispatched_first == 1
    assert dispatched_second == 0
    assert len(post_batches) == 1
    assert len(post_batches[0]) == 3

    status = quota.get_status()
    assert status.submitted_count == 5000
    assert status.remaining == 0
    assert status.can_submit is False
    assert quota.can_submit(1) is False
    assert db.count_queue(status="DONE") == 3
    assert db.count_queue(status="PENDING") == 2


def test_circuit_breaker_three_consecutive_failures_cooldown_and_alert(tmp_path: Path) -> None:
    db = AlphaDB(tmp_path / "circuit_breaker.db")
    settings = _make_settings(
        max_slots=2,
        circuit_breaker_threshold=3,
        region_cooldown_s=1800.0,
        markets={
            "USA": {
                "default_universe": "TOP3000",
                "universes": ["TOP3000"],
                "slots_required": 1,
                "delay": 1,
            },
            "EUR": {
                "default_universe": "TOP2500",
                "universes": ["TOP2500"],
                "slots_required": 1,
                "delay": 1,
            },
        },
    )
    slot_pool = SlotPool.from_settings(settings)
    quota = DailyQuotaManager.from_settings(db, settings, now_fn=_fixed_now_ny)
    alerts: list[str] = []

    sim_seq = 0
    sim_regions: dict[str, str] = {}

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        nonlocal sim_seq
        if method == "POST" and url.endswith("/simulations"):
            payload = kwargs["json"]
            first = payload[0] if isinstance(payload, list) else payload
            region = str(first["settings"]["region"])
            sim_seq += 1
            sim_id = f"cb-sim-{sim_seq}"
            sim_regions[sim_id] = region
            return FakeResponse(
                201,
                headers={"Location": f"https://api.worldquantbrain.com/simulations/{sim_id}"},
            )
        if method == "GET" and "/simulations/" in url:
            sim_id = url.rsplit("/", 1)[-1]
            region = sim_regions[sim_id]
            if region == "USA":
                return FakeResponse(
                    200,
                    json_data={
                        "id": sim_id,
                        "status": "ERROR",
                        "message": "Simulated engine crash",
                    },
                )
            return FakeResponse(
                200,
                json_data={"id": sim_id, "status": "COMPLETE", "alpha": "EUR_SURVIVOR_1"},
            )
        if method == "GET" and url.endswith("/alphas/EUR_SURVIVOR_1"):
            return FakeResponse(200, json_data=_alpha_payload("EUR_SURVIVOR_1"))
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    session = FakeSession(_handler)
    client = _make_client(tmp_path, session)
    dispatcher = SimulationDispatcher(
        db=db,
        client=client,
        quota=quota,
        slot_pool=slot_pool,
        settings=settings,
        alert_callback=alerts.append,
        poll_interval_s=0.02,
        acquire_timeout_s=0.5,
    )

    try:
        # Three consecutive 0-success batches for USA trip the breaker.
        for idx in range(3):
            db.enqueue_candidates([_make_candidate(idx, region="USA", universe="TOP3000", delay=1)])
            assert dispatcher.dispatch_once() == 1
            _drain_workers(dispatcher)
            if idx < 2:
                assert dispatcher.get_region_cooldowns() == {}
                assert alerts == []

        cooldowns = dispatcher.get_region_cooldowns()
        assert "USA" in cooldowns
        assert cooldowns["USA"] > time.time()
        assert len(alerts) == 1
        assert "USA" in alerts[0]
        assert "3" in alerts[0]

        # USA must now be skipped while EUR continues dispatching normally.
        db.enqueue_candidates(
            [
                _make_candidate(99, region="USA", universe="TOP3000", delay=1),
                _make_candidate(100, region="EUR", universe="TOP2500", delay=1),
            ]
        )
        post_count_before = len([c for c in session.calls if c[0] == "POST"])
        dispatched_after = dispatcher.dispatch_once()
        _drain_workers(dispatcher)
    finally:
        dispatcher.stop(drain_timeout_s=5.0)

    post_calls_after = [c for c in session.calls if c[0] == "POST"][post_count_before:]
    assert dispatched_after == 1
    assert len(post_calls_after) == 1
    assert post_calls_after[0][2]["json"]["settings"]["region"] == "EUR"
    assert db.count_queue(status="PENDING", region="USA") == 1
    assert db.count_queue(status="DONE", region="EUR") == 1
    assert db.get_simulation("EUR_SURVIVOR_1") is not None
