"""Phase 1 tests for BrainClient polling, rate-limiting, validation, and BrainAuthManager."""

from __future__ import annotations

import builtins
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import requests

from alpha_factory.brain import client as client_mod
from alpha_factory.brain.auth import (
    AuthenticationError,
    BiometricAuthTimeoutError,
    BrainAPIError,
    BrainAuthManager,
    RateLimitError,
    RecaptchaRequiredError,
    SESSION_TOKEN_HEADER,
    WebAuthnRequiredError,
)
from alpha_factory.brain.client import (
    BrainClient,
    PollingTimeoutError,
    SimulationError,
    SimulationValidationError,
    flatten_validation_errors,
    validate_simulation_payload,
)


class FakeResponse:
    """Minimal in-memory HTTP response stub compatible with requests.Response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        headers: Optional[dict[str, Any]] = None,
        text: str = "",
        url: str = "",
        json_error: Optional[Exception] = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.headers = dict(headers or {})
        self.text = text
        self.url = url
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        if self._json_data is not None:
            return self._json_data
        if self.text:
            return json.loads(self.text)
        raise ValueError("No JSON body available")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    """Thread-safe mocked requests.Session that never performs network I/O."""

    def __init__(
        self,
        handler: Optional[Callable[..., FakeResponse]] = None,
    ) -> None:
        self.headers: dict[str, str] = {}
        self.cookies = requests.cookies.RequestsCookieJar()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._routes: dict[tuple[str, str], list[Any]] = {}
        self._handler = handler

    def enqueue(self, method: str, url_or_suffix: str, *items: Any) -> None:
        key = (method.upper(), url_or_suffix)
        with self._lock:
            self._routes.setdefault(key, []).extend(items)

    def _next_from_routes(self, method: str, url: str) -> Optional[Any]:
        method_u = method.upper()
        with self._lock:
            exact = self._routes.get((method_u, url))
            if exact:
                return exact.pop(0)
            for (route_method, pattern), queue in self._routes.items():
                if route_method == method_u and queue and url.endswith(pattern):
                    return queue.pop(0)
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        with self._lock:
            self.calls.append((method.upper(), url, dict(kwargs)))
        queued = self._next_from_routes(method, url)
        if queued is not None:
            item = queued(method.upper(), url, **kwargs) if callable(queued) else queued
            if isinstance(item, BaseException):
                raise item
            return item
        if self._handler is not None:
            item = self._handler(method.upper(), url, **kwargs)
            if isinstance(item, BaseException):
                raise item
            return item
        raise AssertionError(f"Unexpected HTTP call: {method.upper()} {url}")

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> FakeResponse:
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def sleep_calls(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch time.sleep globally so polling/retry tests execute instantaneously."""
    recorded: list[float] = []
    lock = threading.Lock()

    def _fake_sleep(seconds: float) -> None:
        with lock:
            recorded.append(float(seconds))

    monkeypatch.setattr(time, "sleep", _fake_sleep)
    return recorded


def _make_auth(
    tmp_path: Path,
    *,
    alert_callback: Optional[Callable[[str], Any]] = None,
    biometric_poll_interval_s: float = 15.0,
    biometric_max_wait_s: float = 1800.0,
    email: Optional[str] = "tester@example.com",
    password: Optional[str] = "s3cr3t",
) -> BrainAuthManager:
    return BrainAuthManager(
        api_url="https://api.worldquantbrain.com",
        credentials_path=tmp_path / "platform-brain.json",
        session_cache_path=tmp_path / "alpha-factory-session.json",
        biometric_poll_interval_s=biometric_poll_interval_s,
        biometric_max_wait_s=biometric_max_wait_s,
        alert_callback=alert_callback,
        email=email,
        password=password,
    )


def _make_client(
    tmp_path: Path,
    session: FakeSession,
    *,
    max_wait_s: float = 3600.0,
    default_interval_s: float = 5.0,
    max_http_retries: int = 3,
    auth_manager: Optional[BrainAuthManager] = None,
) -> BrainClient:
    auth = auth_manager or _make_auth(tmp_path)
    return BrainClient(
        auth_manager=auth,
        session=session,  # type: ignore[arg-type]
        max_wait_s=max_wait_s,
        default_interval_s=default_interval_s,
        max_http_retries=max_http_retries,
    )


# ---------------------------------------------------------------------------
# Polling & Retry-After behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("retry_after_val", [0, "0", "0.0", "0.00"])
def test_poll_retry_after_zero_returns_immediately(
    tmp_path: Path,
    sleep_calls: list[float],
    retry_after_val: Any,
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-zero",
        FakeResponse(
            200,
            json_data={"id": "sim-zero", "status": "COMPLETE"},
            headers={"Retry-After": retry_after_val},
        ),
    )
    client = _make_client(tmp_path, session)

    resp = client.poll("/simulations/sim-zero")

    assert resp.json()["status"] == "COMPLETE"
    assert sleep_calls == []
    assert len(session.calls) == 1


def test_poll_absent_retry_after_returns_immediately(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-absent",
        FakeResponse(200, json_data={"id": "sim-absent", "status": "COMPLETE"}),
    )
    client = _make_client(tmp_path, session)

    resp = client.poll("/simulations/sim-absent")

    assert resp.json()["status"] == "COMPLETE"
    assert sleep_calls == []
    assert len(session.calls) == 1


def test_poll_positive_float_retry_after(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-float",
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": "2.5"}),
        FakeResponse(200, json_data={"status": "SIMULATING"}, headers={"retry-after": "1.25"}),
        FakeResponse(200, json_data={"id": "sim-float", "status": "COMPLETE"}),
    )
    client = _make_client(tmp_path, session)

    resp = client.poll("/simulations/sim-float")

    assert resp.json() == {"id": "sim-float", "status": "COMPLETE"}
    assert sleep_calls == [2.5, 1.25]
    assert len(session.calls) == 3


@pytest.mark.parametrize("bad_header", ["-3.5", "not-a-number", "nan", "inf", "-inf"])
def test_poll_negative_or_unparseable_retry_after_uses_fallback(
    tmp_path: Path,
    sleep_calls: list[float],
    bad_header: str,
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-bad-ra",
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": bad_header}),
        FakeResponse(200, json_data={"status": "COMPLETE"}, headers={"Retry-After": "0"}),
    )
    client = _make_client(tmp_path, session, default_interval_s=3.75)

    resp = client.poll("/simulations/sim-bad-ra")

    assert resp.json()["status"] == "COMPLETE"
    assert sleep_calls == [3.75]
    assert len(session.calls) == 2


def test_poll_timeout_ceiling_raises_before_exceeding_max_wait(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-slow",
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": "4.0"}),
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": "4.0"}),
        FakeResponse(200, json_data={"status": "SIMULATING"}, headers={"Retry-After": "3.0"}),
    )
    client = _make_client(tmp_path, session, max_wait_s=10.0)

    with pytest.raises(PollingTimeoutError) as exc_info:
        client.poll("/simulations/sim-slow")

    assert isinstance(exc_info.value, BrainAPIError)
    assert isinstance(exc_info.value, TimeoutError)
    # Third wait (4 + 4 + 3 = 11 > 10) must abort before sleeping 3.0s
    assert sleep_calls == [4.0, 4.0]
    assert len(session.calls) == 3


def test_poll_timeout_ceiling_wall_clock_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def _advancing_sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(time, "sleep", _advancing_sleep)

    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-wall",
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": "5.0"}),
        FakeResponse(200, json_data={"status": "WAITING"}, headers={"Retry-After": "1.0"}),
    )
    client = _make_client(tmp_path, session, max_wait_s=5.0)

    with pytest.raises(PollingTimeoutError):
        client.poll("/simulations/sim-wall")


# ---------------------------------------------------------------------------
# 429 global rate-limit backoff & retry
# ---------------------------------------------------------------------------


def test_429_rate_limit_retry_after_and_global_coordination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    slept: list[float] = []
    lock = threading.Lock()
    first_sleep_started = threading.Event()
    second_sleep_started = threading.Event()

    def _fake_monotonic() -> float:
        with lock:
            return clock[0]

    def _fake_sleep(seconds: float) -> None:
        with lock:
            idx = len(slept)
            slept.append(float(seconds))
        if idx == 0:
            first_sleep_started.set()
            second_sleep_started.wait(timeout=2.0)
        else:
            with lock:
                clock[0] += float(seconds)
            second_sleep_started.set()

    monkeypatch.setattr(time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(time, "sleep", _fake_sleep)

    session = FakeSession()
    session.enqueue(
        "POST",
        "/simulations",
        FakeResponse(429, headers={"Retry-After": "7.5"}),
        FakeResponse(
            201,
            headers={"Location": "https://api.worldquantbrain.com/simulations/sim-429"},
        ),
    )
    session.enqueue(
        "GET",
        "/alphas/alpha-peer",
        FakeResponse(200, json_data={"id": "alpha-peer"}),
    )

    client = _make_client(tmp_path, session, max_http_retries=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_post = pool.submit(
            client.submit_simulation,
            {"type": "REGULAR", "regular": "rank(close)"},
        )
        assert first_sleep_started.wait(timeout=2.0)
        # Peer thread starts while global _rate_limit_until is active; it must sleep before GET.
        fut_peer = pool.submit(client.get_alpha, "alpha-peer")

        location = fut_post.result(timeout=5.0)
        peer_alpha = fut_peer.result(timeout=5.0)

    assert location == "https://api.worldquantbrain.com/simulations/sim-429"
    assert peer_alpha == {"id": "alpha-peer"}
    assert slept == [7.5, 7.5]
    # First HTTP call was the 429 POST; GET /alphas/alpha-peer occurred only after global wait.
    assert session.calls[0][0] == "POST"
    assert any(c[0] == "GET" and c[1].endswith("/alphas/alpha-peer") for c in session.calls[1:])


def test_429_without_retry_after_uses_exponential_backoff_and_raises_on_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    slept: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def _advancing_sleep(seconds: float) -> None:
        slept.append(float(seconds))
        clock[0] += float(seconds)

    monkeypatch.setattr(time, "sleep", _advancing_sleep)
    monkeypatch.setattr(client_mod.random, "uniform", lambda _a, _b: 1.0)

    session = FakeSession()
    session.enqueue(
        "GET",
        "/alphas/a-backoff",
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200, json_data={"id": "a-backoff"}),
    )
    client = _make_client(tmp_path, session, default_interval_s=2.0, max_http_retries=2)

    result = client.get_alpha("a-backoff")
    assert result == {"id": "a-backoff"}
    assert slept == [2.0, 4.0]

    # Exhausting retries raises RateLimitError
    session.enqueue(
        "GET",
        "/alphas/a-exhaust",
        FakeResponse(429, headers={"Retry-After": "3.0"}),
        FakeResponse(429, headers={"Retry-After": "3.0"}),
        FakeResponse(429, headers={"Retry-After": "3.0"}),
    )
    with pytest.raises(RateLimitError) as exc_info:
        client.get_alpha("a-exhaust")
    assert exc_info.value.retry_after == 3.0


def test_poll_429_rate_limit_backoff_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [200.0]
    slept: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def _advancing_sleep(seconds: float) -> None:
        slept.append(float(seconds))
        clock[0] += float(seconds)

    monkeypatch.setattr(time, "sleep", _advancing_sleep)

    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/sim-poll-429",
        FakeResponse(429, headers={"Retry-After": "4.5"}),
        FakeResponse(200, json_data={"id": "sim-poll-429", "status": "COMPLETE"}),
    )
    client = _make_client(tmp_path, session, max_wait_s=60.0, max_http_retries=2)

    resp = client.poll("/simulations/sim-poll-429")
    assert resp.json()["status"] == "COMPLETE"
    assert slept == [4.5]


# ---------------------------------------------------------------------------
# SimulationValidationError & payload list length (2..10)
# ---------------------------------------------------------------------------


def test_submit_simulation_400_flattens_nested_errors(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    nested_body = {
        "settings": {
            "decay": ["must be non-negative"],
            "nested": {"region": "unsupported"},
        },
        "regular": ["syntax error", {"line": 1, "msg": "unexpected token"}],
        "detail": "bad request",
    }
    session = FakeSession()
    session.enqueue("POST", "/simulations", FakeResponse(400, json_data=nested_body))
    client = _make_client(tmp_path, session)

    with pytest.raises(SimulationValidationError) as exc_info:
        client.submit_simulation({"type": "REGULAR", "regular": "bad("})

    expected_errors = [
        "settings.decay: must be non-negative",
        "settings.nested.region: unsupported",
        "regular: syntax error",
        "regular.line: 1",
        "regular.msg: unexpected token",
        "detail: bad request",
    ]
    assert exc_info.value.errors == expected_errors
    assert str(exc_info.value) == "; ".join(expected_errors)
    # 400 validation failures must not retry or sleep
    assert len(session.calls) == 1
    assert sleep_calls == []


def test_flatten_validation_errors_list_and_string() -> None:
    assert flatten_validation_errors(["err1", {"a": ["err2"]}]) == ["err1", "a: err2"]
    assert flatten_validation_errors("plain error") == ["plain error"]


@pytest.mark.parametrize("bad_count", [0, 1, 11])
def test_validate_simulation_payload_rejects_invalid_list_length(
    tmp_path: Path,
    bad_count: int,
) -> None:
    session = FakeSession()
    client = _make_client(tmp_path, session)
    payload = [{"type": "REGULAR", "regular": f"rank(close) + {i}"} for i in range(bad_count)]

    with pytest.raises(ValueError, match="between 2 and 10"):
        validate_simulation_payload(payload)
    with pytest.raises(ValueError, match="between 2 and 10"):
        client.submit_simulation(payload)
    assert session.calls == []


@pytest.mark.parametrize("valid_count", [2, 10])
def test_submit_simulation_accepts_valid_multi_sim_lengths(
    tmp_path: Path,
    valid_count: int,
) -> None:
    session = FakeSession()
    session.enqueue(
        "POST",
        "/simulations",
        FakeResponse(201, headers={"Location": f"https://api.worldquantbrain.com/simulations/batch-{valid_count}"}),
    )
    client = _make_client(tmp_path, session)
    payload = [{"type": "REGULAR", "regular": f"rank(close) + {i}"} for i in range(valid_count)]

    location = client.submit_simulation(payload)
    assert location.endswith(f"/batch-{valid_count}")
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# poll_simulation_children()
# ---------------------------------------------------------------------------


def test_poll_simulation_children_polls_every_child_individually(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/child-1",
        FakeResponse(200, json_data={"status": "SIMULATING"}, headers={"Retry-After": "1.5"}),
        FakeResponse(200, json_data={"id": "child-1", "status": "COMPLETE", "alpha": "A1"}),
    )
    session.enqueue(
        "GET",
        "https://api.worldquantbrain.com/simulations/child-2",
        FakeResponse(
            200,
            json_data={"id": "child-2", "status": "WARNING", "alpha": "A2", "message": "warn"},
        ),
    )
    session.enqueue(
        "GET",
        "/simulations/child-3",
        FakeResponse(200, json_data={"id": "child-3", "status": "COMPLETE", "alpha": "A3"}),
    )
    client = _make_client(tmp_path, session)

    parent_payload = {
        "id": "parent-1",
        "status": "COMPLETE",
        "children": [
            "child-1",
            "https://api.worldquantbrain.com/simulations/child-2",
            "child-3",
        ],
    }
    results = client.poll_simulation_children(parent_payload)

    assert [r["alpha"] for r in results] == ["A1", "A2", "A3"]
    assert [url for _, url, _ in session.calls] == [
        "https://api.worldquantbrain.com/simulations/child-1",
        "https://api.worldquantbrain.com/simulations/child-1",
        "https://api.worldquantbrain.com/simulations/child-2",
        "https://api.worldquantbrain.com/simulations/child-3",
    ]
    assert sleep_calls == [1.5]


def test_poll_simulation_children_mixed_complete_and_error(
    tmp_path: Path,
    sleep_calls: list[float],
) -> None:
    session = FakeSession()
    session.enqueue(
        "GET",
        "/simulations/child-ok",
        FakeResponse(200, json_data={"id": "child-ok", "status": "COMPLETE", "alpha": "A_OK"}),
    )
    session.enqueue(
        "GET",
        "/simulations/child-err",
        FakeResponse(
            200,
            json_data={
                "id": "child-err",
                "status": "ERROR",
                "message": "Undefined field foo",
                "location": {"property": "regular", "line": 1, "start": 5, "end": 8},
            },
        ),
    )
    client = _make_client(tmp_path, session)

    parent_payload = {
        "id": "parent-mixed",
        "status": "COMPLETE",
        "children": ["child-ok", "child-err"],
    }

    with pytest.raises(SimulationError) as exc_info:
        client.poll_simulation_children(parent_payload)

    err = exc_info.value
    assert err.status == "ERROR"
    assert err.simulation_id == "child-err"
    assert "Undefined field foo" in err.message
    assert err.location == {"property": "regular", "line": 1, "start": 5, "end": 8}
    assert [url for _, url, _ in session.calls] == [
        "https://api.worldquantbrain.com/simulations/child-ok",
        "https://api.worldquantbrain.com/simulations/child-err",
    ]
    assert sleep_calls == []


# ---------------------------------------------------------------------------
# BrainAuthManager: 401 coalescing, 204, challenges, headless biometric flow
# ---------------------------------------------------------------------------


def test_auth_single_lock_concurrent_401_reauth_coalescing(tmp_path: Path) -> None:
    num_threads = 5
    barrier = threading.Barrier(num_threads)
    auth_post_count = 0
    count_lock = threading.Lock()

    def _handler(method: str, url: str, **kwargs: Any) -> FakeResponse:
        nonlocal auth_post_count
        if method == "GET" and url.endswith("/alphas/a1"):
            if session.headers.get(SESSION_TOKEN_HEADER) != "fresh-token":
                # Force all 5 threads to observe 401 before any thread re-authenticates.
                barrier.wait(timeout=2.0)
                return FakeResponse(401, json_data={"detail": "token expired"})
            return FakeResponse(200, json_data={"id": "a1", "status": "ACTIVE"})

        if method == "POST" and url.endswith("/authentication"):
            with count_lock:
                auth_post_count += 1
            return FakeResponse(
                201,
                headers={SESSION_TOKEN_HEADER: "fresh-token"},
                json_data={"user": {"id": "user-1"}, "token": {"expiry": 14400}},
            )
        raise AssertionError(f"Unexpected call {method} {url}")

    session = FakeSession(handler=_handler)
    auth = _make_auth(tmp_path)
    client = _make_client(tmp_path, session, auth_manager=auth)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(lambda _: client.get_alpha("a1"), range(num_threads)))

    assert all(r == {"id": "a1", "status": "ACTIVE"} for r in results)
    assert auth_post_count == 1
    assert auth.auth_generation == 1
    alpha_gets = [c for c in session.calls if c[0] == "GET" and c[1].endswith("/alphas/a1")]
    assert len(alpha_gets) == num_threads * 2


def test_auth_204_means_unauthenticated_and_logs_in_from_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAIN_CREDENTIAL_EMAIL", raising=False)
    monkeypatch.delenv("BRAIN_CREDENTIAL_PASSWORD", raising=False)

    creds_path = tmp_path / "platform-brain.json"
    creds_path.write_text(
        json.dumps({"email": "file-user@example.com", "password": "file-password"}),
        encoding="utf-8",
    )
    cache_path = tmp_path / "alpha-factory-session.json"

    auth = BrainAuthManager(
        credentials_path=creds_path,
        session_cache_path=cache_path,
    )
    session = FakeSession()
    session.enqueue("GET", "/authentication", FakeResponse(204))
    assert auth.check_authenticated(session) is None

    session.enqueue("GET", "/authentication", FakeResponse(204))
    session.enqueue(
        "POST",
        "/authentication",
        FakeResponse(
            201,
            headers={SESSION_TOKEN_HEADER: "tok-204"},
            json_data={"user": {"id": "u-204"}, "token": {"expiry": 14400}},
        ),
    )

    info = auth.ensure_authenticated(session)

    assert info["user"]["id"] == "u-204"
    assert session.headers[SESSION_TOKEN_HEADER] == "tok-204"
    assert auth.auth_generation == 1
    assert cache_path.is_file()
    assert (cache_path.stat().st_mode & 0o777) == 0o600
    post_calls = [c for c in session.calls if c[0] == "POST" and c[1].endswith("/authentication")]
    assert len(post_calls) == 1
    assert post_calls[0][2]["auth"] == ("file-user@example.com", "file-password")


@pytest.mark.parametrize(
    "recaptcha_payload",
    [
        {"recaptcha": "required"},
        {"recaptcha": True},
        {"recaptcha": ["reCAPTCHA verification required"]},
    ],
)
def test_auth_recaptcha_required_error(
    tmp_path: Path,
    recaptcha_payload: dict[str, Any],
) -> None:
    auth = _make_auth(tmp_path)
    session = FakeSession()
    session.enqueue("GET", "/authentication", FakeResponse(204))
    session.enqueue("POST", "/authentication", FakeResponse(401, json_data=recaptcha_payload))

    with pytest.raises(RecaptchaRequiredError) as exc_info:
        auth.ensure_authenticated(session)

    assert isinstance(exc_info.value, AuthenticationError)


@pytest.mark.parametrize(
    "webauthn_payload",
    [
        {"publicKey": {"challenge": "xyz"}},
        {"credentialOptions": {"rpId": "worldquantbrain.com"}},
        {"credentials": {"createOptions": {"challenge": "123"}}},
        {"error": "webauthn"},
    ],
)
def test_auth_webauthn_required_error(
    tmp_path: Path,
    webauthn_payload: dict[str, Any],
) -> None:
    auth = _make_auth(tmp_path)
    session = FakeSession()
    session.enqueue("GET", "/authentication", FakeResponse(204))
    session.enqueue("POST", "/authentication", FakeResponse(401, json_data=webauthn_payload))

    with pytest.raises(WebAuthnRequiredError) as exc_info:
        auth.ensure_authenticated(session)

    assert isinstance(exc_info.value, AuthenticationError)
    assert exc_info.value.options == webauthn_payload


def test_auth_biometric_headless_persona_polling_until_201(
    tmp_path: Path,
    sleep_calls: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_input(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("input() must never be called in headless biometric flow")

    monkeypatch.setattr(builtins, "input", _no_input)

    alerted_urls: list[str] = []
    auth = _make_auth(
        tmp_path,
        alert_callback=alerted_urls.append,
        biometric_poll_interval_s=5.0,
        biometric_max_wait_s=60.0,
    )
    session = FakeSession()
    session.enqueue("GET", "/authentication", FakeResponse(204))
    session.enqueue(
        "POST",
        "/authentication",
        FakeResponse(
            401,
            headers={
                "WWW-Authenticate": "Persona",
                "Location": "/authentication/persona?inquiry=inq_123",
            },
            url="https://api.worldquantbrain.com/authentication",
        ),
    )
    verify_url = "https://api.worldquantbrain.com/authentication/persona?inquiry=inq_123"
    session.enqueue(
        "POST",
        verify_url,
        FakeResponse(403),
        FakeResponse(429, headers={"Retry-After": "3.0"}),
        FakeResponse(
            201,
            headers={SESSION_TOKEN_HEADER: "bio-session-tok"},
            json_data={"user": {"id": "bio-user"}, "token": {"expiry": 14400}},
        ),
    )

    result = auth.ensure_authenticated(session)

    assert alerted_urls == [verify_url]
    assert sleep_calls == [5.0, 5.0, 3.0, 5.0]
    assert result["user"]["id"] == "bio-user"
    assert session.headers[SESSION_TOKEN_HEADER] == "bio-session-tok"
    assert auth.auth_generation == 1
    assert (tmp_path / "alpha-factory-session.json").is_file()


def test_auth_biometric_timeout_raises_biometric_auth_timeout_error(
    tmp_path: Path,
    sleep_calls: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _no_input(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("input() must never be called in headless biometric flow")

    monkeypatch.setattr(builtins, "input", _no_input)

    alerted_urls: list[str] = []
    auth = _make_auth(
        tmp_path,
        alert_callback=alerted_urls.append,
        biometric_poll_interval_s=10.0,
        biometric_max_wait_s=25.0,
    )
    verify_url = "https://verify.example.com/persona/inq_timeout"
    session = FakeSession()
    session.enqueue("GET", "/authentication", FakeResponse(204))
    session.enqueue(
        "POST",
        "/authentication",
        FakeResponse(
            401,
            headers={
                "WWW-Authenticate": "inquiry",
                "Location": verify_url,
            },
        ),
    )
    session.enqueue(
        "POST",
        verify_url,
        FakeResponse(403),
        FakeResponse(401),
        FakeResponse(202),
    )

    with pytest.raises(BiometricAuthTimeoutError) as exc_info:
        auth.ensure_authenticated(session)

    assert isinstance(exc_info.value, AuthenticationError)
    assert isinstance(exc_info.value, TimeoutError)
    assert alerted_urls == [verify_url]
    assert sleep_calls == [10.0, 10.0, 5.0]
