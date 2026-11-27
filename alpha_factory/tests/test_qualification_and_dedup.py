"""Phase 4 tests for qualification gates, correlation fetchers, dedup vault, and submitter."""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest
import requests

from alpha_factory.brain.auth import BrainAuthManager
from alpha_factory.brain.client import BrainClient
from alpha_factory.qualification.checks import (
    check_pnl_flatline,
    evaluate_gate1,
    evaluate_gate2,
    parse_checks,
)
from alpha_factory.qualification.correlations import (
    fetch_power_pool_corr,
    fetch_prod_corr,
    fetch_self_corr,
)
from alpha_factory.qualification.dedup_vault import (
    CorrelationDedupVault,
    compute_composite_score,
    pairwise_effective_corr,
)
from alpha_factory.qualification.submitter import SequentialSubmitter
from alpha_factory.storage.db import NY_TZ, AlphaDB
from alpha_factory.storage.models import SimulationRecord, VaultEntry
from alpha_factory.storage.pnl_store import PnlStore


# ---------------------------------------------------------------------------
# Offline & Fast Execution Guards + Fake HTTP Session
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offline_and_fast_guards(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Block all real network I/O and make time.sleep instantaneous."""
    sleep_calls: list[float] = []
    lock = threading.Lock()

    def _fake_sleep(seconds: float) -> None:
        with lock:
            sleep_calls.append(float(seconds))

    def _forbid_network(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Real network call attempted in offline qualification test")

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    monkeypatch.setattr(requests.Session, "request", _forbid_network)
    monkeypatch.setattr(requests, "request", _forbid_network)
    monkeypatch.setattr(requests, "get", _forbid_network)
    monkeypatch.setattr(requests, "post", _forbid_network)
    monkeypatch.setattr(requests, "patch", _forbid_network)
    monkeypatch.setattr(requests, "delete", _forbid_network)
    return sleep_calls


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

    def patch(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("DELETE", url, **kwargs)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _make_client(
    tmp_path: Path,
    handler: Callable[[str, str, dict[str, Any]], FakeResponse],
) -> tuple[BrainClient, FakeSession]:
    auth = BrainAuthManager(
        api_url="https://api.worldquantbrain.com",
        credentials_path=tmp_path / "creds.json",
        session_cache_path=tmp_path / "session.json",
        email="tester@example.com",
        password="secret",
    )
    session = FakeSession(handler)
    client = BrainClient(
        auth_manager=auth,
        session=session,  # type: ignore[arg-type]
        api_url="https://api.worldquantbrain.com",
        default_interval_s=0.01,
        max_wait_s=120.0,
        recordset_max_wait_s=120.0,
        max_http_retries=2,
    )
    return client, session


def _make_dates(n_days: int = 504) -> np.ndarray:
    start = dt.date(2023, 1, 2).toordinal()
    return np.arange(start, start + n_days, dtype=np.int64)


def _orthonormal_vectors(n_days: int = 504, count: int = 4, seed: int = 42) -> list[np.ndarray]:
    """Generate zero-mean unit-norm mutually orthogonal daily vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(n_days, count))
    raw = raw - raw.mean(axis=0, keepdims=True)
    # Remove any mean component along the constant vector before QR so columns stay zero-mean.
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


def _make_cumulative_pnl_payload(
    dates: np.ndarray,
    daily_returns: np.ndarray,
) -> dict[str, Any]:
    cum = np.cumsum(daily_returns)
    records = [
        [dt.date.fromordinal(int(d)).isoformat(), float(v)]
        for d, v in zip(dates, cum)
    ]
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


def _make_daily_pnl_payload(
    dates: np.ndarray,
    daily_returns: np.ndarray,
) -> dict[str, Any]:
    records = [
        [dt.date.fromordinal(int(d)).isoformat(), float(v)]
        for d, v in zip(dates, daily_returns)
    ]
    return {
        "schema": {
            "name": "daily-pnl",
            "properties": [
                {"name": "date", "type": "date"},
                {"name": "pnl", "type": "decimal"},
            ],
        },
        "records": records,
    }


def _make_sim_record(
    alpha_id: str,
    *,
    sharpe: float = 2.0,
    fitness: float = 1.5,
    turnover: float = 0.20,
    prod_corr_max: float = 0.20,
    region: str = "USA",
    delay: int = 1,
    pyramid_matches_json: str = "[]",
) -> SimulationRecord:
    return SimulationRecord(
        alpha_id=alpha_id,
        config_hash=f"cfg-{alpha_id}",
        ast_fingerprint=f"fp-{alpha_id}",
        expression=f"rank(ts_mean(close, 5)) + {alpha_id}",
        region=region,
        universe="TOP3000",
        delay=delay,
        status="COMPLETE",
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        prod_corr_max=prod_corr_max,
        pyramid_matches_json=pyramid_matches_json,
    )


def _healthy_is_checks() -> list[dict[str, Any]]:
    return [
        {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.5},
        {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 2.0},
        {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.2},
        {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.2},
        {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
        {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "limit": 0.75, "value": 1.25},
        {"name": "MATCHES_PYRAMID", "result": "WARNING", "matches": [{"name": "Momentum"}]},
    ]


# ---------------------------------------------------------------------------
# 1. Gate 1 & Real BRAIN is.checks Regression Tests
# ---------------------------------------------------------------------------


def test_gate1_rejects_brain_check_fail_despite_strong_metrics() -> None:
    """Even with strong Sharpe/Fitness/Turnover/Margin, is.checks FAIL MUST reject."""
    payload = {
        "id": "alpha_sub_univ_fail",
        "is": {
            "sharpe": 2.5,
            "fitness": 2.0,
            "turnover": 0.2,
            "margin": 0.002,
            "returns": 0.14,
            "longCount": 140,
            "shortCount": 135,
            "checks": [
                {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.5},
                {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 2.0},
                {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.2},
                {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.2},
                {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
                {
                    "name": "LOW_SUB_UNIVERSE_SHARPE",
                    "result": "FAIL",
                    "limit": 0.75,
                    "value": 0.41,
                },
                {"name": "MATCHES_PYRAMID", "result": "WARNING", "pyramids": [{"name": "Value"}]},
            ],
        },
    }

    res = evaluate_gate1(payload)

    assert res.alpha_id == "alpha_sub_univ_fail"
    assert res.passed is False
    assert res.needs_review is False
    assert "LOW_SUB_UNIVERSE_SHARPE" in res.fail_reasons
    assert "MATCHES_PYRAMID" in res.warn_reasons
    assert res.pyramid_matches == ["Value"]


def test_gate1_sanity_too_good_to_be_true_and_threshold_checks() -> None:
    """Rejects extreme Sharpe=8.5 or returns>0.50 as SANITY_TOO_GOOD_TO_BE_TRUE."""
    extreme_sharpe_payload = {
        "id": "alpha_too_good_sharpe",
        "is": {
            "sharpe": 8.5,
            "fitness": 2.0,
            "turnover": 0.2,
            "margin": 0.002,
            "returns": 0.18,
            "longCount": 120,
            "shortCount": 120,
            "checks": _healthy_is_checks(),
        },
    }
    res_sharpe = evaluate_gate1(extreme_sharpe_payload)
    assert res_sharpe.passed is False
    assert res_sharpe.needs_review is True
    assert "SANITY_TOO_GOOD_TO_BE_TRUE" in res_sharpe.fail_reasons

    extreme_returns_payload = {
        "id": "alpha_too_good_returns",
        "is": {
            "sharpe": 2.5,
            "fitness": 2.0,
            "turnover": 0.2,
            "margin": 0.002,
            "returns": 0.65,
            "longCount": 120,
            "shortCount": 120,
            "checks": _healthy_is_checks(),
        },
    }
    res_returns = evaluate_gate1(extreme_returns_payload)
    assert res_returns.passed is False
    assert res_returns.needs_review is True
    assert "SANITY_TOO_GOOD_TO_BE_TRUE" in res_returns.fail_reasons

    clean_payload = {
        "id": "alpha_clean_g1",
        "is": {
            "sharpe": 2.5,
            "fitness": 2.0,
            "turnover": 0.2,
            "margin": 0.002,
            "returns": 0.15,
            "longCount": 120,
            "shortCount": 120,
            "checks": _healthy_is_checks(),
        },
    }
    res_clean = evaluate_gate1(clean_payload)
    assert res_clean.passed is True
    assert res_clean.needs_review is False
    assert res_clean.fail_reasons == []
    assert res_clean.pyramid_matches == ["Momentum"]

    parsed = parse_checks(
        [
            {"name": "C1", "result": "FAILED"},
            {"name": "C2", "result": "WARN"},
        ]
    )
    assert [(c.name, c.result) for c in parsed] == [("C1", "FAIL"), ("C2", "WARNING")]

    threshold_fail_payload = {
        "id": "alpha_thresh_fail",
        "is": {
            "sharpe": 1.2,
            "fitness": 0.8,
            "turnover": 0.80,
            "margin": 0.0002,
            "returns": 0.08,
            "longCount": 50,
            "shortCount": 0,
            "checks": [],
        },
    }
    res_thresh = evaluate_gate1(threshold_fail_payload)
    assert res_thresh.passed is False
    assert any("LOW_SHARPE" in r for r in res_thresh.fail_reasons)
    assert any("LOW_FITNESS" in r for r in res_thresh.fail_reasons)
    assert any("HIGH_TURNOVER" in r for r in res_thresh.fail_reasons)
    assert any("LOW_MARGIN" in r for r in res_thresh.fail_reasons)
    assert any("NO_SHORT_POSITIONS" in r for r in res_thresh.fail_reasons)


# ---------------------------------------------------------------------------
# 2. PnL Flatline Check Tests
# ---------------------------------------------------------------------------


def test_check_pnl_flatline_rejects_dead_tail_and_zero_heavy_and_passes_healthy() -> None:
    """Rejects dead-tail PnL and zero-heavy (>30% zero) PnL; passes healthy returns."""
    rng = np.random.default_rng(123)

    # Dead-tail (all zeros in trailing 252 days, overall zero ratio = 252/1000 = 0.252 <= 0.30)
    dead_tail_zeros = np.concatenate(
        [rng.normal(0.0005, 0.01, size=748), np.zeros(252, dtype=np.float64)]
    )
    ok_dead_zero, reason_dead_zero = check_pnl_flatline(dead_tail_zeros)
    assert ok_dead_zero is False
    assert "TAIL_ZERO_RETURN_RATIO" in reason_dead_zero

    # Dead-tail (constant non-zero value in trailing 252 days -> zero variance)
    dead_tail_const = np.concatenate(
        [rng.normal(0.0005, 0.01, size=248), np.full(252, 0.0004, dtype=np.float64)]
    )
    ok_dead_var, reason_dead_var = check_pnl_flatline(dead_tail_const)
    assert ok_dead_var is False
    assert "TAIL_ZERO_VARIANCE" in reason_dead_var

    # Zero-heavy (>30% zeros spread across the series so tail zero ratio <= 0.85)
    zero_heavy = rng.normal(0.0004, 0.01, size=500)
    zero_heavy[::2] = 0.0  # 50% zeros > 30%
    ok_zero_heavy, reason_zero_heavy = check_pnl_flatline(zero_heavy)
    assert ok_zero_heavy is False
    assert "HIGH_ZERO_RETURN_RATIO" in reason_zero_heavy

    # Insufficient days and non-finite guards
    ok_short, reason_short = check_pnl_flatline(np.ones(10, dtype=np.float64) * 0.01)
    assert ok_short is False
    assert "INSUFFICIENT_PNL_DAYS" in reason_short

    bad_nan = rng.normal(0.0003, 0.01, size=50)
    bad_nan[12] = np.nan
    ok_nan, reason_nan = check_pnl_flatline(bad_nan)
    assert ok_nan is False
    assert reason_nan == "NON_FINITE_RETURNS"

    # Healthy daily return series (<30% zeros, active tail)
    healthy = rng.normal(0.0006, 0.01, size=500)
    healthy[:10] = 0.0  # 2% zeros
    ok_healthy, reason_healthy = check_pnl_flatline(healthy)
    assert ok_healthy is True
    assert reason_healthy == "OK"


# ---------------------------------------------------------------------------
# 3. Gate 2 & Correlation Fetcher Tests
# ---------------------------------------------------------------------------


def test_gate2_passes_clean_alpha_polling_and_returns_pnl(
    tmp_path: Path,
    _offline_and_fast_guards: list[float],
) -> None:
    """Polls /check, /correlations/self, /correlations/prod, /recordsets/pnl and passes clean alpha."""
    dates = _make_dates(300)
    rng = np.random.default_rng(7)
    daily_returns = rng.normal(0.0007, 0.01, size=300)
    pnl_payload = _make_cumulative_pnl_payload(dates, daily_returns)

    check_calls = 0

    def _handler(method: str, url: str, _kwargs: dict[str, Any]) -> FakeResponse:
        nonlocal check_calls
        if url.endswith("/alphas/a_clean/check"):
            check_calls += 1
            if check_calls == 1:
                return FakeResponse(status_code=200, headers={"Retry-After": "1"}, json_data={})
            return FakeResponse(
                status_code=200,
                json_data={"is": {"checks": _healthy_is_checks()}},
            )
        if url.endswith("/alphas/a_clean/correlations/self"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "selfCorrelation",
                        "properties": [
                            {"name": "id", "type": "text"},
                            {"name": "correlation", "type": "decimal"},
                        ],
                    },
                    "records": [["old_alpha_1", 0.31], ["old_alpha_2", 0.48]],
                    "max": 0.48,
                },
            )
        if url.endswith("/alphas/a_clean/correlations/prod"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "prodCorrelation",
                        "properties": [
                            {"name": "min", "type": "decimal"},
                            {"name": "max", "type": "decimal"},
                            {"name": "alphas", "type": "integer"},
                        ],
                    },
                    "records": [
                        [-1.0, 0.5, 42],
                        [0.5, 0.6, 3],
                        [0.6, 0.7, 0],
                    ],
                    "max": 0.54,
                },
            )
        if url.endswith("/alphas/a_clean/recordsets/pnl"):
            return FakeResponse(status_code=200, json_data=pnl_payload)
        raise AssertionError(f"Unexpected call: {method} {url}")

    client, session = _make_client(tmp_path, _handler)
    res, pnl_pair = evaluate_gate2(client, "a_clean")

    assert res.passed is True
    assert res.fail_reasons == []
    assert res.self_corr_max == pytest.approx(0.48)
    assert res.prod_corr_max == pytest.approx(0.54)
    assert res.prod_risk_tier == "MEDIUM"
    assert res.pyramid_matches == ["Momentum"]
    assert pnl_pair is not None
    ret_dates, ret_vals = pnl_pair
    assert ret_dates.dtype == np.int64
    assert ret_vals.dtype == np.float64
    assert np.array_equal(ret_dates, dates)
    assert np.allclose(ret_vals, daily_returns, atol=1e-12)
    assert any(s == 1.0 for s in _offline_and_fast_guards)
    assert [c[1].split("worldquantbrain.com")[-1] for c in session.calls] == [
        "/alphas/a_clean/check",
        "/alphas/a_clean/check",
        "/alphas/a_clean/correlations/self",
        "/alphas/a_clean/correlations/prod",
        "/alphas/a_clean/recordsets/pnl",
    ]


def test_gate2_daily_pnl_fallback_on_404(tmp_path: Path) -> None:
    """Falls back to /recordsets/daily-pnl when /recordsets/pnl returns HTTP 404."""
    dates = _make_dates(260)
    rng = np.random.default_rng(19)
    daily_returns = rng.normal(0.0005, 0.01, size=260)
    daily_payload = _make_daily_pnl_payload(dates, daily_returns)

    def _handler(method: str, url: str, _kwargs: dict[str, Any]) -> FakeResponse:
        if url.endswith("/alphas/a_fb/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if url.endswith("/alphas/a_fb/correlations/self"):
            return FakeResponse(status_code=204)
        if url.endswith("/alphas/a_fb/correlations/prod"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "prodCorrelation",
                        "properties": [
                            {"name": "min", "type": "decimal"},
                            {"name": "max", "type": "decimal"},
                            {"name": "alphas", "type": "integer"},
                        ],
                    },
                    "records": [[-1.0, 0.4, 15], [0.4, 0.7, 0]],
                    "max": 0.35,
                },
            )
        if url.endswith("/alphas/a_fb/recordsets/pnl"):
            return FakeResponse(status_code=404, text="Not Found")
        if url.endswith("/alphas/a_fb/recordsets/daily-pnl"):
            return FakeResponse(status_code=200, json_data=daily_payload)
        raise AssertionError(f"Unexpected call: {method} {url}")

    client, session = _make_client(tmp_path, _handler)
    res, pnl_pair = evaluate_gate2(client, "a_fb")

    assert res.passed is True
    assert res.self_corr_max == 0.0
    assert res.prod_corr_max == pytest.approx(0.35)
    assert res.prod_risk_tier == "LOW"
    assert pnl_pair is not None
    ret_dates, ret_vals = pnl_pair
    assert np.array_equal(ret_dates, dates)
    assert np.allclose(ret_vals, daily_returns, atol=1e-12)
    assert any(c[1].endswith("/recordsets/daily-pnl") for c in session.calls)


def test_gate2_rejects_check_fail_self_corr_and_prod_corr(tmp_path: Path) -> None:
    """Rejects when /check has FAIL, self_corr >= 0.70, prod_corr >= 0.70, or power-pool >= 0.50."""
    # Case 1: /check FAIL short-circuits before correlation endpoints
    def _fail_check_handler(_m: str, url: str, _k: dict[str, Any]) -> FakeResponse:
        if url.endswith("/check"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "is": {
                        "checks": [
                            {"name": "CONCENTRATED_WEIGHT", "result": "FAIL"},
                        ]
                    }
                },
            )
        raise AssertionError(f"Should not call {url} after /check FAIL")

    c1, _ = _make_client(tmp_path, _fail_check_handler)
    r1, p1 = evaluate_gate2(c1, "a_chk_fail")
    assert r1.passed is False
    assert p1 is None
    assert "CONCENTRATED_WEIGHT" in r1.fail_reasons

    # Case 2: self_corr == 0.70 (>= 0.70 threshold) short-circuits before prod_corr
    def _fail_self_handler(_m: str, url: str, _k: dict[str, Any]) -> FakeResponse:
        if url.endswith("/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if url.endswith("/correlations/self"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "selfCorrelation",
                        "properties": [
                            {"name": "id", "type": "text"},
                            {"name": "correlation", "type": "decimal"},
                        ],
                    },
                    "records": [["peer_x", 0.70]],
                    "max": 0.70,
                },
            )
        raise AssertionError(f"Should not call {url} after self_corr >= 0.70")

    c2, _ = _make_client(tmp_path, _fail_self_handler)
    assert fetch_self_corr(c2, "a_self_fail") == pytest.approx(0.70)
    r2, p2 = evaluate_gate2(c2, "a_self_fail")
    assert r2.passed is False
    assert p2 is None
    assert r2.self_corr_max == pytest.approx(0.70)
    assert any("SELF_CORRELATION" in reason for reason in r2.fail_reasons)

    # Case 3: prod_corr >= 0.70 short-circuits before /recordsets/pnl
    def _fail_prod_handler(_m: str, url: str, _k: dict[str, Any]) -> FakeResponse:
        if url.endswith("/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if url.endswith("/correlations/self"):
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.25})
        if url.endswith("/correlations/prod"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "prodCorrelation",
                        "properties": [
                            {"name": "min", "type": "decimal"},
                            {"name": "max", "type": "decimal"},
                            {"name": "alphas", "type": "integer"},
                        ],
                    },
                    "records": [[-1.0, 0.5, 50], [0.7, 1.0, 2]],
                    "max": 0.76,
                },
            )
        raise AssertionError(f"Should not call {url} after prod_corr >= 0.70")

    c3, _ = _make_client(tmp_path, _fail_prod_handler)
    prod_val, prod_tier = fetch_prod_corr(c3, "a_prod_fail")
    assert prod_val == pytest.approx(0.76)
    assert prod_tier == "HIGH"
    r3, p3 = evaluate_gate2(c3, "a_prod_fail")
    assert r3.passed is False
    assert p3 is None
    assert r3.prod_corr_max == pytest.approx(0.76)
    assert r3.prod_risk_tier == "HIGH"
    assert any("PROD_CORRELATION" in reason for reason in r3.fail_reasons)

    # Case 4: power-pool check enabled and failing (>= 0.50)
    def _fail_pp_handler(_m: str, url: str, _k: dict[str, Any]) -> FakeResponse:
        if url.endswith("/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if url.endswith("/correlations/self"):
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.20})
        if url.endswith("/correlations/prod"):
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.30})
        if url.endswith("/correlations/power-pool"):
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "powerPoolCorrelation",
                        "properties": [
                            {"name": "id", "type": "text"},
                            {"name": "correlation", "type": "decimal"},
                        ],
                    },
                    "records": [["pp1", 0.58]],
                    "max": 0.58,
                },
            )
        raise AssertionError(f"Unexpected call: {url}")

    c4, _ = _make_client(tmp_path, _fail_pp_handler)
    assert fetch_power_pool_corr(c4, "a_pp_fail") == pytest.approx(0.58)
    r4, p4 = evaluate_gate2(c4, "a_pp_fail", qual_cfg={"check_power_pool": True})
    assert r4.passed is False
    assert p4 is None
    assert r4.power_pool_corr_max == pytest.approx(0.58)
    assert any("POWER_POOL_CORRELATION" in reason for reason in r4.fail_reasons)

    # Case 5: Gate 2 rejects flatlined PnL
    flat_dates = _make_dates(300)
    flat_returns = np.concatenate(
        [np.full(48, 0.002, dtype=np.float64), np.zeros(252, dtype=np.float64)]
    )
    flat_pnl_payload = _make_daily_pnl_payload(flat_dates, flat_returns)

    def _fail_flat_handler(_m: str, url: str, _k: dict[str, Any]) -> FakeResponse:
        if url.endswith("/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if url.endswith("/correlations/self"):
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.10})
        if url.endswith("/correlations/prod"):
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.20})
        if url.endswith("/recordsets/pnl"):
            return FakeResponse(status_code=200, json_data={"records": []})
        if url.endswith("/recordsets/daily-pnl"):
            return FakeResponse(status_code=200, json_data=flat_pnl_payload)
        raise AssertionError(f"Unexpected call: {url}")

    c5, _ = _make_client(tmp_path, _fail_flat_handler)
    r5, p5 = evaluate_gate2(c5, "a_flat_fail")
    assert r5.passed is False
    assert p5 is not None  # non-empty dates returned alongside flatline rejection
    assert any("PNL_FLATLINE" in reason for reason in r5.fail_reasons)


# ---------------------------------------------------------------------------
# 4. CorrelationDedupVault Tests
# ---------------------------------------------------------------------------


def test_vault_correlated_pair_keeps_higher_score_and_marks_lower_corr_dup(
    tmp_path: Path,
) -> None:
    """Two alphas with |r| = 0.85: higher-scored is KEPT, lower-scored is CORR_DUP."""
    db = AlphaDB(tmp_path / "vault_pair.db")
    pnl_store = PnlStore(tmp_path / "pnl_pair")
    patch_calls: list[tuple[str, Any]] = []

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "PATCH" and "/alphas/" in url:
            patch_calls.append((url.rsplit("/", 1)[-1], kwargs.get("json")))
            return FakeResponse(status_code=200, json_data={})
        raise AssertionError(f"Unexpected call: {method} {url}")

    client, _ = _make_client(tmp_path, _handler)
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=client, tag_on_brain=True)

    dates = _make_dates(504)
    u1, u2, u3 = _orthonormal_vectors(504, count=3, seed=101)
    ret_winner = _correlated_vector(u1, u2, 1.0)
    ret_dup = _correlated_vector(u1, u3, -0.85)  # negative -0.85 => |r| == 0.85

    assert pairwise_effective_corr(dates, ret_winner, dates, ret_dup) == pytest.approx(
        0.85, abs=1e-6
    )

    sim_winner = _make_sim_record("alpha_win", sharpe=3.0, fitness=2.2)
    sim_dup = _make_sim_record("alpha_dup", sharpe=1.8, fitness=1.2)
    assert compute_composite_score(sim_winner) > compute_composite_score(sim_dup)

    e_win = vault.admit(sim_winner, dates, ret_winner)
    e_dup = vault.admit(sim_dup, dates, ret_dup)

    assert e_win.vault_status == "KEPT"
    assert e_win.duplicate_of is None
    assert e_dup.vault_status == "CORR_DUP"
    assert e_dup.duplicate_of == "alpha_win"
    assert e_dup.max_local_corr == pytest.approx(0.85, abs=1e-6)
    assert e_dup.max_local_corr_peer == "alpha_win"

    assert ("alpha_win", {"color": "GREEN"}) in patch_calls
    assert ("alpha_dup", {"color": "YELLOW"}) in patch_calls


def test_vault_better_alpha_replaces_older_kept_with_hysteresis(tmp_path: Path) -> None:
    """Better alpha arriving later (new_score >= old_score + 0.05) replaces older KEPT."""
    db = AlphaDB(tmp_path / "vault_replace.db")
    pnl_store = PnlStore(tmp_path / "pnl_replace")
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=None, tag_on_brain=False)

    dates = _make_dates(504)
    u1, u2, u3, u4 = _orthonormal_vectors(504, count=4, seed=202)
    ret_old = _correlated_vector(u1, u2, 1.0)
    ret_mid = _correlated_vector(u1, u3, 0.85)
    ret_new = _correlated_vector(u1, u4, 0.85)

    # old_score = 2.0 + 0.5*2.0 - 0.40 = 2.60
    sim_old = _make_sim_record("alpha_old", fitness=2.0, sharpe=2.0)
    # mid_score = 2.0 + 0.5*2.06 - 0.40 = 2.63 (< old_score + 0.05 -> must NOT replace)
    sim_mid = _make_sim_record("alpha_mid", fitness=2.0, sharpe=2.06)
    # new_score = 2.0 + 0.5*2.10 - 0.40 = 2.65 (>= old_score + 0.05 -> MUST replace)
    sim_new = _make_sim_record("alpha_new", fitness=2.0, sharpe=2.10)

    e_old = vault.admit(sim_old, dates, ret_old)
    assert e_old.vault_status == "KEPT"

    e_mid = vault.admit(sim_mid, dates, ret_mid)
    assert e_mid.vault_status == "CORR_DUP"
    assert e_mid.duplicate_of == "alpha_old"

    e_new = vault.admit(sim_new, dates, ret_new)
    assert e_new.vault_status == "KEPT"
    assert e_new.duplicate_of is None
    assert e_new.max_local_corr == pytest.approx(0.85, abs=1e-6)

    kept_entries = db.get_vault_entries(region="USA", delay=1, statuses="KEPT")
    assert [e.alpha_id for e in kept_entries] == ["alpha_new"]

    all_entries = {e.alpha_id: e for e in db.get_vault_entries(region="USA", delay=1)}
    # AlphaDB.demote_vault_entry records 'DUPLICATE', which reconcile_clusters normalizes to 'CORR_DUP'.
    assert all_entries["alpha_old"].vault_status in ("CORR_DUP", "DUPLICATE")
    assert all_entries["alpha_old"].duplicate_of == "alpha_new"

    vault.reconcile_clusters("USA", 1)
    reconciled = {e.alpha_id: e for e in db.get_vault_entries(region="USA", delay=1)}
    assert reconciled["alpha_new"].vault_status == "KEPT"
    assert reconciled["alpha_old"].vault_status == "CORR_DUP"
    assert reconciled["alpha_old"].duplicate_of == "alpha_new"


def test_vault_correlated_with_submitted_always_marked_corr_dup(tmp_path: Path) -> None:
    """Alpha correlated (|r|=0.85) with an existing SUBMITTED alpha is ALWAYS CORR_DUP."""
    db = AlphaDB(tmp_path / "vault_sub.db")
    pnl_store = PnlStore(tmp_path / "pnl_sub")
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=None, tag_on_brain=False)

    dates = _make_dates(504)
    u1, u2 = _orthonormal_vectors(504, count=2, seed=303)
    ret_sub = _correlated_vector(u1, u2, 1.0)
    ret_high = _correlated_vector(u1, u2, 0.85)

    pnl_store.save("alpha_submitted", "USA", dates, ret_sub)
    db.upsert_vault_entry(
        VaultEntry(
            alpha_id="alpha_submitted",
            region="USA",
            delay=1,
            expression="rank(close)",
            composite_score=1.20,
            vault_status="SUBMITTED",
            sharpe=1.6,
            fitness=1.0,
            turnover=0.3,
            prod_corr_max=0.3,
        )
    )

    sim_high = _make_sim_record("alpha_super", fitness=5.0, sharpe=5.0, turnover=0.1, prod_corr_max=0.1)
    assert compute_composite_score(sim_high) > 7.0

    entry = vault.admit(sim_high, dates, ret_high)
    assert entry.vault_status == "CORR_DUP"
    assert entry.duplicate_of == "alpha_submitted"
    assert entry.max_local_corr == pytest.approx(0.85, abs=1e-6)

    submitted_entries = db.get_vault_entries(region="USA", delay=1, statuses="SUBMITTED")
    assert [e.alpha_id for e in submitted_entries] == ["alpha_submitted"]


def test_vault_uncorrelated_alphas_both_remain_kept(tmp_path: Path) -> None:
    """Uncorrelated alphas (|r| = 0.20 < 0.65) both remain KEPT."""
    db = AlphaDB(tmp_path / "vault_uncorr.db")
    pnl_store = PnlStore(tmp_path / "pnl_uncorr")
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=None, tag_on_brain=False)

    dates = _make_dates(504)
    u1, u2 = _orthonormal_vectors(504, count=2, seed=404)
    ret_a = _correlated_vector(u1, u2, 1.0)
    ret_b = _correlated_vector(u1, u2, 0.20)

    assert pairwise_effective_corr(dates, ret_a, dates, ret_b) == pytest.approx(0.20, abs=1e-6)

    e_a = vault.admit(_make_sim_record("alpha_u1", sharpe=2.4, fitness=1.8), dates, ret_a)
    e_b = vault.admit(_make_sim_record("alpha_u2", sharpe=2.2, fitness=1.6), dates, ret_b)

    assert e_a.vault_status == "KEPT"
    assert e_b.vault_status == "KEPT"
    assert e_a.duplicate_of is None
    assert e_b.duplicate_of is None
    kept_ids = {e.alpha_id for e in db.get_vault_entries(region="USA", delay=1, statuses="KEPT")}
    assert kept_ids == {"alpha_u1", "alpha_u2"}


def test_vault_reconcile_clusters_promotes_best_uncorrelated_runner_up(
    tmp_path: Path,
) -> None:
    """reconcile_clusters promotes best uncorrelated CORR_DUP runner-up after winner REJECTED."""
    db = AlphaDB(tmp_path / "vault_reconcile.db")
    pnl_store = PnlStore(tmp_path / "pnl_reconcile")
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=None, tag_on_brain=False)

    dates = _make_dates(504)
    u1, u2, u3 = _orthonormal_vectors(504, count=3, seed=505)

    # Construct winner = (u1 + u2)/sqrt(2), so corr(win, r1)=0.707 and corr(win, r3)=0.707,
    # while corr(r1, r3)=0.0. Also r2 has corr=0.85 with r1 and corr=0.813 with win.
    ret_win = _correlated_vector(u1, u2, float(1.0 / np.sqrt(2.0)))
    ret_r1 = _correlated_vector(u1, u2, 1.0)
    ret_r2 = 0.01 * (0.85 * u1 + 0.30 * u2 + float(np.sqrt(1.0 - 0.85**2 - 0.30**2)) * u3) + 0.0002
    ret_r3 = _correlated_vector(u2, u1, 1.0)

    e_win = vault.admit(_make_sim_record("alpha_win", fitness=3.0, sharpe=3.0), dates, ret_win)
    e_r1 = vault.admit(_make_sim_record("alpha_r1", fitness=2.2, sharpe=2.4), dates, ret_r1)
    e_r2 = vault.admit(_make_sim_record("alpha_r2", fitness=1.7, sharpe=2.0), dates, ret_r2)
    e_r3 = vault.admit(_make_sim_record("alpha_r3", fitness=1.4, sharpe=1.8), dates, ret_r3)

    assert e_win.vault_status == "KEPT"
    assert e_r1.vault_status == "CORR_DUP" and e_r1.duplicate_of == "alpha_win"
    assert e_r2.vault_status == "CORR_DUP" and e_r2.duplicate_of == "alpha_win"
    assert e_r3.vault_status == "CORR_DUP" and e_r3.duplicate_of == "alpha_win"

    # Mark winner REJECTED (as submitter does on live check failure).
    db.upsert_vault_entry(
        VaultEntry(
            alpha_id=e_win.alpha_id,
            region=e_win.region,
            delay=e_win.delay,
            expression=e_win.expression,
            composite_score=e_win.composite_score,
            vault_status="REJECTED",
            duplicate_of=None,
            sharpe=e_win.sharpe,
            fitness=e_win.fitness,
            turnover=e_win.turnover,
            prod_corr_max=e_win.prod_corr_max,
        )
    )

    stats = vault.reconcile_clusters("USA", 1)
    assert stats == {"kept": 2, "demoted": 0, "promoted": 2}

    after = {e.alpha_id: e for e in db.get_vault_entries(region="USA", delay=1)}
    assert after["alpha_win"].vault_status == "REJECTED"
    assert after["alpha_r1"].vault_status == "KEPT"
    assert after["alpha_r1"].duplicate_of is None
    assert after["alpha_r2"].vault_status == "CORR_DUP"
    assert after["alpha_r2"].duplicate_of == "alpha_r1"
    assert after["alpha_r3"].vault_status == "KEPT"
    assert after["alpha_r3"].duplicate_of is None


# ---------------------------------------------------------------------------
# 5. SequentialSubmitter & Sync Tests
# ---------------------------------------------------------------------------


def test_sequential_submitter_rechecks_between_submissions_rejects_and_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submits Alpha 1, rejects Alpha 2 on live self_corr=0.78 >= 0.70, and reconciles clusters."""
    db = AlphaDB(tmp_path / "submitter.db")
    pnl_store = PnlStore(tmp_path / "pnl_submitter")

    dates = _make_dates(504)
    u1, u2, u3, u4 = _orthonormal_vectors(504, count=4, seed=606)
    ret_1 = _correlated_vector(u1, u2, 1.0)
    ret_2 = _correlated_vector(u1, u2, 0.20)  # Locally uncorrelated with ret_1
    ret_2_unit = 0.20 * u1 + float(np.sqrt(1.0 - 0.20**2)) * u2
    ret_2_backup = _correlated_vector(ret_2_unit, u3, 0.85)  # corr(ret_2, ret_2_backup)=0.85, corr(ret_1, backup)=0.17
    ret_3 = _correlated_vector(u4, u1, 1.0)  # Strictly orthogonal to ret_1, ret_2, ret_2_backup

    state: dict[str, Any] = {
        "alpha_1_submitted": False,
        "endpoints": [],
        "reconcile_snapshots": [],
    }

    def _handler(method: str, url: str, _kwargs: dict[str, Any]) -> FakeResponse:
        path = url.split("worldquantbrain.com")[-1]
        state["endpoints"].append((method, path))

        if method == "PATCH" and "/alphas/" in path:
            return FakeResponse(status_code=200, json_data={})

        if method == "GET" and path.endswith("/check"):
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})

        if method == "GET" and path == "/alphas/alpha_1/correlations/self":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.30})
        if method == "GET" and path == "/alphas/alpha_1/correlations/prod":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.40})

        if method == "POST" and path == "/alphas/alpha_1/submit":
            state["alpha_1_submitted"] = True
            return FakeResponse(
                status_code=201,
                headers={"Location": "https://api.worldquantbrain.com/alphas/alpha_1/submit"},
            )
        if method == "GET" and path == "/alphas/alpha_1/submit":
            return FakeResponse(status_code=200, json_data={"status": "COMPLETE"})

        if method == "GET" and path == "/alphas/alpha_2/correlations/self":
            # After Alpha 1 was submitted, live BRAIN self-correlation rises to 0.78 >= 0.70.
            corr_val = 0.78 if state["alpha_1_submitted"] else 0.25
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "selfCorrelation",
                        "properties": [
                            {"name": "id", "type": "text"},
                            {"name": "correlation", "type": "decimal"},
                        ],
                    },
                    "records": [["alpha_1", corr_val]],
                    "max": corr_val,
                },
            )
        if method == "POST" and path == "/alphas/alpha_2/submit":
            raise AssertionError("alpha_2 must NEVER be submitted after self_corr=0.78 >= 0.70")

        if method == "GET" and path == "/alphas/alpha_3/correlations/self":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.22})
        if method == "GET" and path == "/alphas/alpha_3/correlations/prod":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.33})
        if method == "POST" and path == "/alphas/alpha_3/submit":
            return FakeResponse(status_code=200, json_data={"status": "COMPLETE"})

        raise AssertionError(f"Unexpected HTTP call: {method} {path}")

    client, _ = _make_client(tmp_path, _handler)
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=client, tag_on_brain=True)

    # Admit Alpha 1 (score ~ 4.10), Alpha 2 (score ~ 3.55), Alpha 2 backup (score ~ 2.70), Alpha 3 (score ~ 3.10)
    assert vault.admit(_make_sim_record("alpha_1", fitness=3.0, sharpe=3.0), dates, ret_1).vault_status == "KEPT"
    assert vault.admit(_make_sim_record("alpha_2", fitness=2.6, sharpe=2.7), dates, ret_2).vault_status == "KEPT"
    assert (
        vault.admit(_make_sim_record("alpha_2_backup", fitness=2.0, sharpe=2.2), dates, ret_2_backup).vault_status
        == "CORR_DUP"
    )
    assert vault.admit(_make_sim_record("alpha_3", fitness=2.3, sharpe=2.4), dates, ret_3).vault_status == "KEPT"

    state["endpoints"].clear()

    orig_reconcile = vault.reconcile_clusters

    def _spy_reconcile(region: str, delay: int) -> dict[str, int]:
        stats = orig_reconcile(region, delay)
        entries = {e.alpha_id: e.vault_status for e in db.get_vault_entries(region=region, delay=delay)}
        state["reconcile_snapshots"].append((stats, entries))
        return stats

    monkeypatch.setattr(vault, "reconcile_clusters", _spy_reconcile)

    fixed_now = dt.datetime(2026, 6, 15, 10, 0, 0, tzinfo=NY_TZ)
    submitter = SequentialSubmitter(
        db=db,
        client=client,
        vault=vault,
        max_submissions_per_day=2,
        now_fn=lambda: fixed_now,
    )

    results = submitter.submit_top_kept()

    assert [r["alpha_id"] for r in results] == ["alpha_1", "alpha_2", "alpha_3"]
    assert results[0]["status"] == "SUBMITTED" and results[0]["submitted"] is True
    assert results[1]["status"] == "REJECTED" and results[1]["submitted"] is False
    assert results[1]["self_corr"] == pytest.approx(0.78)
    assert "self_corr=0.7800>=0.7000" in results[1]["reason"]
    assert results[2]["status"] == "SUBMITTED" and results[2]["submitted"] is True

    # Verify strict sequential ordering: Alpha 1 fully submitted BEFORE Alpha 2 checked,
    # and Alpha 2 rejected before Alpha 3 checked & submitted.
    assert state["endpoints"] == [
        ("GET", "/alphas/alpha_1/check"),
        ("GET", "/alphas/alpha_1/correlations/self"),
        ("GET", "/alphas/alpha_1/correlations/prod"),
        ("POST", "/alphas/alpha_1/submit"),
        ("GET", "/alphas/alpha_1/submit"),
        ("GET", "/alphas/alpha_2/check"),
        ("GET", "/alphas/alpha_2/correlations/self"),
        ("PATCH", "/alphas/alpha_2_backup"),
        ("GET", "/alphas/alpha_3/check"),
        ("GET", "/alphas/alpha_3/correlations/self"),
        ("GET", "/alphas/alpha_3/correlations/prod"),
        ("POST", "/alphas/alpha_3/submit"),
    ]

    # Verify cluster reconciliation ran upon Alpha 2's rejection and promoted alpha_2_backup -> KEPT.
    assert len(state["reconcile_snapshots"]) == 1
    reconcile_stats, snap_statuses = state["reconcile_snapshots"][0]
    assert reconcile_stats["promoted"] == 1
    assert snap_statuses["alpha_2"] == "REJECTED"
    assert snap_statuses["alpha_2_backup"] == "KEPT"

    final_vault = {e.alpha_id: e.vault_status for e in db.get_vault_entries(region="USA", delay=1)}
    assert final_vault == {
        "alpha_1": "SUBMITTED",
        "alpha_2": "REJECTED",
        "alpha_2_backup": "KEPT",
        "alpha_3": "SUBMITTED",
    }

    # Daily limit reached (2/2) -> subsequent call submits nothing.
    assert submitter.submit_top_kept() == []


def test_sequential_submitter_dry_run_and_sync_submitted_alphas(tmp_path: Path) -> None:
    """Covers sync_submitted_alphas pagination/daily-pnl fallback and dry_run non-mutation."""
    db = AlphaDB(tmp_path / "sync_dry.db")
    pnl_store = PnlStore(tmp_path / "pnl_sync_dry")
    dates = _make_dates(260)
    rng = np.random.default_rng(77)
    ret_sub = rng.normal(0.0005, 0.01, size=260)
    ret_kept = rng.normal(0.0006, 0.01, size=260)

    def _handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        path = url.split("worldquantbrain.com")[-1]
        if method == "GET" and path == "/users/self/alphas":
            offset = int((kwargs.get("params") or {}).get("offset", 0))
            if offset == 0:
                return FakeResponse(
                    status_code=200,
                    json_data={
                        "count": 1,
                        "results": [
                            {
                                "id": "live_sub_1",
                                "settings": {"region": "USA", "delay": 1},
                                "regular": {"code": "rank(close)"},
                                "is": {"sharpe": 2.1, "fitness": 1.4, "turnover": 0.25},
                                "pyramids": [{"name": "Momentum"}],
                            }
                        ],
                    },
                )
            return FakeResponse(status_code=200, json_data={"count": 1, "results": []})
        if method == "GET" and path == "/alphas/live_sub_1/recordsets/pnl":
            return FakeResponse(status_code=404, text="Not Found")
        if method == "GET" and path == "/alphas/live_sub_1/recordsets/daily-pnl":
            return FakeResponse(status_code=200, json_data=_make_daily_pnl_payload(dates, ret_sub))
        if method == "GET" and path == "/alphas/kept_dry/check":
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if method == "GET" and path == "/alphas/kept_dry/correlations/self":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.20})
        if method == "GET" and path == "/alphas/kept_dry/correlations/prod":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.30})
        if method == "POST" and path.endswith("/submit"):
            raise AssertionError("dry_run=True must not POST /submit")
        raise AssertionError(f"Unexpected call: {method} {path}")

    client, _ = _make_client(tmp_path, _handler)
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=client, tag_on_brain=False)

    synced = vault.sync_submitted_alphas(region="USA", page_limit=1)
    assert synced == 1
    assert pnl_store.has("live_sub_1", "USA") is True
    sub_entries = db.get_vault_entries(region="USA", delay=1, statuses="SUBMITTED")
    assert [e.alpha_id for e in sub_entries] == ["live_sub_1"]

    assert vault.admit(_make_sim_record("kept_dry", fitness=2.2, sharpe=2.5), dates, ret_kept).vault_status == "KEPT"

    fixed_now = dt.datetime(2026, 6, 15, 11, 0, 0, tzinfo=NY_TZ)
    submitter = SequentialSubmitter(
        db=db,
        client=client,
        vault=vault,
        max_submissions_per_day=2,
        now_fn=lambda: fixed_now,
    )
    dry_results = submitter.submit_top_kept(dry_run=True)
    assert len(dry_results) == 1
    assert dry_results[0]["alpha_id"] == "kept_dry"
    assert dry_results[0]["status"] == "DRY_RUN"
    assert dry_results[0]["submitted"] is False
    assert db.get_vault_entries(region="USA", delay=1, statuses="KEPT")[0].alpha_id == "kept_dry"
    assert db.get_runtime_state("submissions:2026-06-15", "0") == "0"


def test_sequential_submitter_rejects_live_check_fail_and_prod_corr(tmp_path: Path) -> None:
    """Rejects KEPT alphas failing live /check or /correlations/prod without submitting."""
    db = AlphaDB(tmp_path / "sub_rejects.db")
    pnl_store = PnlStore(tmp_path / "pnl_sub_rejects")
    dates = _make_dates(504)
    u1, u2 = _orthonormal_vectors(504, count=2, seed=808)

    def _handler(method: str, url: str, _kwargs: dict[str, Any]) -> FakeResponse:
        path = url.split("worldquantbrain.com")[-1]
        if method == "GET" and path == "/alphas/bad_chk/check":
            return FakeResponse(
                status_code=200,
                json_data={"is": {"checks": [{"name": "LOW_AFTER_COST_SHARPE", "result": "FAIL"}]}},
            )
        if method == "GET" and path == "/alphas/bad_prod/check":
            return FakeResponse(status_code=200, json_data={"is": {"checks": _healthy_is_checks()}})
        if method == "GET" and path == "/alphas/bad_prod/correlations/self":
            return FakeResponse(status_code=200, json_data={"records": [], "max": 0.20})
        if method == "GET" and path == "/alphas/bad_prod/correlations/prod":
            return FakeResponse(
                status_code=200,
                json_data={
                    "schema": {
                        "name": "prodCorrelation",
                        "properties": [
                            {"name": "min", "type": "decimal"},
                            {"name": "max", "type": "decimal"},
                            {"name": "alphas", "type": "integer"},
                        ],
                    },
                    "records": [[0.7, 1.0, 3]],
                    "max": 0.74,
                },
            )
        if method == "POST" and path.endswith("/submit"):
            raise AssertionError("Rejected alphas must never POST /submit")
        raise AssertionError(f"Unexpected call: {method} {path}")

    client, _ = _make_client(tmp_path, _handler)
    vault = CorrelationDedupVault(db=db, pnl_store=pnl_store, client=client, tag_on_brain=False)

    assert vault.admit(_make_sim_record("bad_chk", fitness=3.0, sharpe=3.0), dates, _correlated_vector(u1, u2, 1.0)).vault_status == "KEPT"
    assert vault.admit(_make_sim_record("bad_prod", fitness=2.5, sharpe=2.5), dates, _correlated_vector(u2, u1, 1.0)).vault_status == "KEPT"

    fixed_now = dt.datetime(2026, 6, 15, 12, 0, 0, tzinfo=NY_TZ)
    submitter = SequentialSubmitter(
        db=db,
        client=client,
        vault=vault,
        max_submissions_per_day=2,
        now_fn=lambda: fixed_now,
    )
    results = submitter.submit_top_kept()
    assert [r["alpha_id"] for r in results] == ["bad_chk", "bad_prod"]
    assert results[0]["status"] == "REJECTED"
    assert "LOW_AFTER_COST_SHARPE" in results[0]["reason"]
    assert results[1]["status"] == "REJECTED"
    assert results[1]["prod_corr"] == pytest.approx(0.74)
    assert "prod_corr=0.7400>=0.7000" in results[1]["reason"]
    assert db.get_vault_entries(region="USA", delay=1, statuses="KEPT") == []
    assert db.get_runtime_state("submissions:2026-06-15", "0") == "0"
